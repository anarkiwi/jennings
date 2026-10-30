"""Static self-modifying-code analysis: which instructions can write each code byte, and
the values it can hold, by a flow-sensitive value-set fixpoint over :mod:`.vsa`.

One joined state per instruction address and innermost pending return address; control flow
follows static targets, JSR/BRK return addresses, constant-written code bytes,
constant-installed interrupt vectors and observed dynamic transfers (see ``docs/coverage.md``).
"""

import heapq
from collections import defaultdict
from typing import NamedTuple

import numpy as np

from jennings.coverage._core import EXEC, RETURN, TAKEN
from jennings.coverage.plugin import save_sidecar
from jennings.coverage.vsa import TOP8, TOP16, bits, from_bool, from_values, is_single
from jennings.coverage.vsa import rot, step, template, word
from jennings.opcodes import MEM_MODES, MODE_LEN, OPCODES

__all__ = [
    "VECTORS",
    "Writer",
    "Cell",
    "SmcMap",
    "Obligation",
    "analyze",
    "obligations",
    "obligation_rows",
    "write_obligations",
    "check",
]

_REGS = {"a": 0, "x": 1, "y": 2, "c": 8, "z": 9, "i": 10, "d": 11, "v": 13, "n": 14}
_CTRL = frozenset((0x20, 0x4C, 0x6C))
_MN = tuple(OPCODES[op][0] for op in range(256))
_MODE = tuple(OPCODES[op][1] for op in range(256))
_LEN = tuple(MODE_LEN[m] for m in _MODE)
_PUSH = {"PHA": 1, "PHP": 1, "PLA": -1, "PLP": -1}
VECTORS = (0xFFFE, 0xFFFA, 0x0314, 0x0316, 0x0318)


class Writer(NamedTuple):
    """A store site: every address it may write and every value it may store there."""

    targets: frozenset
    values: frozenset


class Cell(NamedTuple):
    """A written code byte: ``writers`` ``{pc: exact}`` (``exact``: the cell is the store's
    only target), ``values`` it may hold (exact unless ``data_dependent``: a writer stores a
    computed value), ``variants`` it may hold when executed, ``image`` byte, ``classes``."""

    writers: dict
    values: frozenset
    classes: frozenset
    insns: frozenset
    data_dependent: bool
    variants: frozenset
    image: int


class SmcMap(NamedTuple):
    """``starts`` ``{pc: opcodes}``, ``edges``, feasible ``branches``, ``writers``, and the
    unfollowed ``unresolved_stores`` ``[pc]``, ``unresolved_transfers`` ``{pc: targets}``,
    ``unresolved_opcodes`` ``{pc: opcodes}``; ``cells`` ``{addr: Cell}``, ``regs`` per pc;
    ``vectors``, interrupt ``handlers`` ``{entry: vectors}``, the handler ``chain``
    ``{(source entry, vector, handler)}`` and torn-vector ``hazards``
    ``[(first writer, vector, partner writer or None)]``."""

    starts: dict
    edges: set
    branches: dict
    writers: dict
    unresolved_stores: list
    unresolved_transfers: dict
    unresolved_opcodes: dict
    cells: dict
    regs: dict
    vectors: tuple
    handlers: dict
    chain: set
    hazards: list


class _State:
    """Register value sets, bytes written since entry ``ov``, the return addresses ``rets``
    ``{offset above SP: 16-bit set}`` pushed by JSR/BRK/entry and not yet popped, the vector
    bytes ``pend`` ``{(vector, byte): first writers}`` written without their partner, the
    entries ``ctx`` that reach the state and the flag origins ``fo`` ``{flag: (r, m0, m1)}``:
    the values of register ``r`` with which the flag may be 0 and 1."""

    __slots__ = ("regs", "ov", "rets", "pend", "ctx", "fo")

    def __init__(self, regs, ov, rets, pend, ctx, fo=None):
        self.regs, self.ov, self.rets, self.pend, self.ctx = regs, ov, rets, pend, ctx
        self.fo = fo or {}

    def __eq__(self, other):
        return (self.regs, self.ov, self.rets, self.pend, self.ctx, self.fo) == (
            other.regs,
            other.ov,
            other.rets,
            other.pend,
            other.ctx,
            other.fo,
        )

    __hash__ = None


def _join(s, t):
    """Pointwise union; a byte, return address or flag origin absent (or on another register)
    on either side is dropped, pending vector bytes and entries are kept from both."""
    pend = dict(s.pend)
    for k, v in t.pend.items():
        pend[k] = pend.get(k, frozenset()) | v
    fo = {}
    for f, (r, m0, m1) in s.fo.items():
        o = t.fo.get(f)
        if o is not None and o[0] == r:
            fo[f] = (r, m0 | o[1], m1 | o[2])
    return _State(
        tuple(x | y for x, y in zip(s.regs, t.regs)),
        {a: v | t.ov[a] for a, v in s.ov.items() if a in t.ov},
        {o: v | t.rets[o] for o, v in s.rets.items() if o in t.rets},
        pend,
        s.ctx | t.ctx,
        fo,
    )


def _refine(regs, fo, flag, value):
    """``(regs, fo)`` restricted to ``flag == value``: the origin register keeps the values
    giving ``value``, and the other flags of that register the values it still allows;
    ``None`` when no value remains."""
    o = fo.get(flag)
    if o is None:
        return regs, fo
    r, keep = o[0], o[1 + value]
    regs = list(regs)
    regs[r] &= keep
    if not regs[r]:
        return None
    out = dict(fo)
    for f, (q, m0, m1) in fo.items():
        if q == r:
            out[f] = (q, m0 & keep, m1 & keep)
            regs[f] &= bool(m0 & keep) | bool(m1 & keep) << 1
    return tuple(regs), out


def _carry(fo, writes, new):
    """Flag origins after an instruction writing registers ``writes`` with ``new`` origins."""
    out = {f: o for f, o in fo.items() if f not in writes and o[0] not in writes}
    out.update(new)
    return out


def _shift(rets, k, new=None):
    out = {o + k: v for o, v in rets.items() if 0 < o + k < 0xFF}
    if new:
        out.update(new)
    return out


class _Mem:
    """One instruction's view of memory: the state's overlay, else the global value set."""

    __slots__ = ("an", "pc", "node", "ov", "own", "stack", "pend", "ctx", "i0")

    def __init__(self, an, node, st):
        self.an, self.node, self.pc, self.ov = an, node, node[0], st.ov
        self.own, self.stack = False, False
        self.pend, self.ctx, self.i0 = st.pend, st.ctx, bool(st.regs[10] & 1)

    def byte(self, a):
        v = self.ov.get(a)
        if v is None:
            self.an.readers[a].add(self.node)
            return self.an.vs(a)
        return v

    def word(self, lo, hi):
        return word(self.byte(lo), self.byte(hi))

    def load(self, addrs):
        if is_single(addrs):
            return self.byte(addrs.bit_length() - 1)
        out = 0
        for a, v in self.ov.items():
            if addrs >> a & 1:
                out |= v
                addrs &= ~(1 << a)
        return out | self.an.gather(addrs, self.node) if addrs else out

    def store(self, addrs, vals):
        an = self.an
        rec = an.stores.setdefault(self.pc, [0, 0, set()])
        rec[0] |= addrs
        rec[1] |= vals
        if addrs == TOP16:
            an.unresolved.add(self.pc)
            return
        self.stack |= bool(addrs >> 0x100 & TOP8)
        if not self.own:
            self.ov, self.own = dict(self.ov), True
        if (addrs, vals) not in rec[2]:
            rec[2].add((addrs, vals))
            an.scatter(bits(addrs), vals)
        if is_single(addrs):
            self.ov[addrs.bit_length() - 1] = vals
        else:
            for a in [a for a in self.ov if addrs >> a & 1]:
                self.ov[a] |= vals
        if addrs & an.vmask:
            for a in bits(addrs & an.vmask).tolist():
                an.vector_store(self, a)


def _class(op, k):
    if k == 0:
        return "var"
    if op in _CTRL or _MODE[op] == "rel":
        return "ctrl"
    return "imm" if _MODE[op] == "imm" else "addr" if _MODE[op] in MEM_MODES else "var"


def _sets(d):
    return {k: frozenset(bits(m).tolist()) for k, m in d.items()}


class _Analysis:
    def __init__(self, mem, sp, vectors):
        self.image = np.asarray(mem, np.uint8)
        self.sp = sp
        self.vectors = tuple(vectors)
        self.vbase = {(v + k) & 0xFFFF: v for v in self.vectors for k in (0, 1)}
        self.vmask = from_values(set(self.vbase), 2) if self.vbase else 0
        self.handlers = defaultdict(set)
        self.taken = set()
        self.chain = set()
        self.pairs = defaultdict(set)
        self.exposed = set()
        self.written = {}
        self.readers = defaultdict(set)
        self.wide = {}
        self.vsm = np.zeros((0x10000, 0x100), bool)
        self.vsm[np.arange(0x10000), self.image] = True
        self.trust = self.vsm.copy()
        self.page = self.vsm.reshape((0x100, 0x100, 0x100)).any(1)
        self.data = np.zeros(0x10000, bool)
        self.states = {}
        self.work = []
        self.queued = set()
        self.starts = defaultdict(int)
        self.edges = set()
        self.branches = defaultdict(int)
        self.stores = {}
        self.unresolved = set()
        self.transfers = defaultdict(int)
        self.opcodes = defaultdict(int)
        self.executed = defaultdict(int)
        self.observed = defaultdict(int)

    def vs(self, a):
        return (1 << int(self.image[a])) | self.written.get(a, 0)

    def gather(self, addrs, node):
        """Union of the global value sets at the addresses in 16-bit set ``addrs``, read by
        ``node``: whole pages from the page summaries, the rest address by address."""
        raw = np.frombuffer(addrs.to_bytes(0x2000, "little"), np.uint8).reshape(0x100, 0x20)
        used, full = raw.any(1), (raw == 0xFF).all(1)
        self.wide[node] = self.wide.get(node, 0) | addrs
        out = self.page[full].any(0)
        part = np.flatnonzero(used & ~full)
        if part.size:
            offs = np.unpackbits(raw[part], axis=1, bitorder="little")
            out |= self.vsm[((part << 8)[:, None] + np.arange(0x100))[offs.astype(bool)]].any(0)
        return from_bool(out)

    def scatter(self, arr, vals):
        """Add ``vals`` to the global value sets at addresses ``arr``."""
        vb = bits(vals)
        grow = ~self.vsm[arr][:, vb].all(1)
        if is_single(vals):
            grow |= ~self.trust[arr, vb[0]]
            self.trust[arr, vb[0]] = True
        else:
            grow |= ~self.data[arr]
            self.data[arr] = True
        grown = arr[grow]
        if not grown.size:
            return
        self.vsm[grown[:, None], vb] = True
        self.page[(grown >> 8)[:, None], vb] = True
        for a in grown.tolist():
            self.written[a] = self.written.get(a, 0) | vals
            for k in self.readers[a]:
                self._enqueue(k)
        grown = from_values(grown, 2)
        for k, addrs in self.wide.items():
            if addrs & grown:
                self._enqueue(k)

    def _enqueue(self, node):
        if node not in self.queued:
            self.queued.add(node)
            heapq.heappush(self.work, node)

    def add(self, pc, st):
        """Join ``st`` into the state of ``pc`` for its innermost pending return address."""
        node = (pc, st.rets[min(st.rets)] if st.rets else -1)
        old = self.states.get(node)
        if old is not None:
            st = _join(old, st)
            if st == old:
                return
        self.states[node] = st
        self._enqueue(node)

    def seed(self, addr):
        """Enter ``addr`` in an unknown context: any registers, stack pointer and stack."""
        regs = [1] * 16
        for i in _REGS.values():
            regs[i] = TOP8 if i < 3 else 3
        regs[3] = TOP8
        self.add(addr, _State(tuple(regs), {}, {}, {}, frozenset()))

    def entry(self, addr, opts, vector=None):
        """Enter ``addr`` with registers ``opts`` under a ``call`` frame, or an interrupt
        frame whose pushes are a store site of ``vector`` (when given) of any byte."""
        regs = [1] * 16
        for name, i in _REGS.items():
            v = opts.get(name)
            regs[i] = (TOP8 if i < 3 else 3) if v is None else from_values(set(v))
        sp, irq = self.sp, bool(opts.get("irq", False))
        regs[3] = 1 << sp
        ret = RETURN if irq else (RETURN - 1) & 0xFFFF
        frame = [1 << (ret >> 8), 1 << (ret & 0xFF)] + [TOP8] * irq
        ov = {}
        for k, v in enumerate(frame):
            a = 0x100 | ((sp + len(frame) - k) & 0xFF)
            ov[a] = v
            self.scatter(np.array([a]), v)
        if irq:
            regs[10] = 2
        if vector is not None:
            addrs = from_values([0x100 | ((sp + k) & 0xFF) for k in (1, 2, 3)], 2)
            rec = self.stores.setdefault(vector, [0, 0, set()])
            rec[0] |= addrs
            rec[1] |= TOP8
            self.scatter(bits(addrs), TOP8)
        self.add(addr, _State(tuple(regs), ov, {1 + irq: 1 << ret}, {}, frozenset((addr,))))

    def install(self, mem, v, sites):
        """Install the vector at ``v`` as ``mem`` sees it: constant pairs become interrupt
        entries and chain edges from ``mem``'s entries, data-dependent ones unresolved
        transfers of the writers ``sites``."""
        lo, hi = v, (v + 1) & 0xFFFF
        targets = mem.word(lo, hi)
        if self.data[lo] or self.data[hi]:
            for pc in sites:
                self.transfers[pc] |= targets
            return
        for h in bits(targets).tolist():
            self.chain.update((e, v, h) for e in mem.ctx)
            self.handler(h, v)

    def handler(self, h, v):
        self.interrupt(h, v)
        self.handlers[h].add(v)

    def interrupt(self, h, v):
        """Enter ``h`` in interrupt context through vector ``v``."""
        if (h, v) not in self.taken:
            self.taken.add((h, v))
            self.entry(h, {"irq": True}, v)

    def vector_store(self, mem, a):
        """A store to vector byte ``a``: completes a pending pair, updates a pair whose partner
        was written on the path, or leaves the byte pending its partner."""
        v = self.vbase[a]
        k = a - v & 0xFFFF
        mem.pend = dict(mem.pend)
        firsts = mem.pend.pop((v, 1 - k), None)
        if firsts is not None:
            for f in firsts:
                self.pairs[(f, v)].add(mem.pc)
        elif (v + 1 - k) & 0xFFFF not in mem.ov:
            mem.pend[(v, k)] = mem.pend.get((v, k), frozenset()) | {mem.pc}
            if not mem.i0:
                return
            self.exposed.add((mem.pc, v))
        self.install(mem, v, (mem.pc,))

    def release(self, mem, exposed):
        """Install the pending vectors of ``mem`` where an interrupt may be taken (``exposed``)
        or at an exit."""
        for (v, _), firsts in mem.pend.items():
            if exposed:
                self.exposed.update((f, v) for f in firsts)
            self.install(mem, v, firsts)

    def const(self, mem, a):
        """Values of byte ``a`` from the image or constant stores; none once a computed
        value may be stored there."""
        self.readers[a].add(mem.node)
        return 0 if self.data[a] else mem.byte(a) & from_bool(self.trust[a])

    def _word(self, mem, lo, hi):
        return word(self.const(mem, lo), self.const(mem, hi))

    def _follow(self, kind, pc, st, mem):
        """Targets of a ``kind`` transfer built from constants only, and the new ``rets``."""
        rets, a1, a2 = st.rets, (pc + 1) & 0xFFFF, (pc + 2) & 0xFFFF
        if kind == "br":
            offs = (bits(self.const(mem, a1)) ^ 0x80) - 0x80
            return from_values((pc + 2 + offs) & 0xFFFF, 2), rets
        if kind == "jmp":
            return self._word(mem, a1, a2), rets
        if kind == "jsr":
            return self._word(mem, a1, a2), _shift(rets, 2, {1: 1 << a2})
        if kind == "jmpind":
            out = 0
            for p in bits(self._word(mem, a1, a2)).tolist():
                out |= self._word(mem, p, (p & 0xFF00) | ((p + 1) & 0xFF))
            return out, rets
        if kind == "brk":
            return self._word(mem, 0xFFFE, 0xFFFF), _shift(rets, 3, {2: 1 << a2})
        if kind == "rts":
            return rot(rets.get(1, 0), 1, 2), _shift(rets, -2)
        if kind == "rti":
            return rets.get(2, 0), _shift(rets, -3)
        return 0, rets

    def process(self, node):
        pc, st = node[0], self.states[node]
        mem = _Mem(self, node, st)
        if st.pend and mem.i0:
            self.release(mem, True)
        ops = mem.byte(pc)
        self.starts[pc] |= ops
        ok = self.const(mem, pc)
        if ops & ~ok:
            self.opcodes[pc] |= ops & ~ok
        for op in bits(ok).tolist():
            mem = _Mem(self, node, st)
            kind = template(op).ctrl[0]
            self.executed[pc] |= 1 << op
            for k in range(1, _LEN[op]):
                self.executed[(pc + k) & 0xFFFF] |= mem.byte((pc + k) & 0xFFFF)
            origins = {}
            succ = step(op, pc, list(st.regs), mem, origins)
            fo = _carry(st.fo, template(op).writes, origins)
            if kind == "next":
                d = _PUSH.get(_MN[op], 0)
                follow = None
                rets = _shift(st.rets, d) if d or not mem.stack else {}
                if 3 in template(op).writes and not d:
                    rets = {}
            else:
                follow, rets = self._follow(kind, pc, st, mem)
            vjmp = kind == "jmpind" and any(
                p in self.vectors
                for p in bits(self._word(mem, (pc + 1) & 0xFFFF, (pc + 2) & 0xFFFF)).tolist()
            )
            for flag, targets, regs in succ:
                if flag != EXEC:
                    ctrl = template(op).ctrl
                    cut = _refine(regs, fo, ctrl[1][1], ctrl[2] ^ (flag != TAKEN))
                    if cut is None:
                        continue
                    regs, sfo = cut
                    self.branches[pc] |= flag
                else:
                    sfo = fo
                if follow is not None and (kind != "br" or flag == TAKEN):
                    if targets & ~follow:
                        self.transfers[pc] |= targets & ~follow
                    targets &= follow | self.observed.get(pc, 0)
                if kind in ("rts", "rti") and targets >> RETURN & 1:
                    self.edges.add((pc, RETURN))
                    targets &= ~(1 << RETURN)
                    self.release(mem, False)
                out = _State(tuple(regs), mem.ov, rets, mem.pend, mem.ctx, sfo)
                for t in bits(targets).tolist():
                    self.edges.add((pc, t))
                    self.add(
                        t,
                        (
                            _State(out.regs, out.ov, rets, out.pend, frozenset((t,)), sfo)
                            if vjmp
                            else out
                        ),
                    )

    def run(self, progress):
        from tqdm import tqdm  # pylint: disable=import-outside-toplevel

        with tqdm(desc="smc", unit="insn", disable=not progress) as pbar:
            while self.work:
                node = heapq.heappop(self.work)
                self.queued.discard(node)
                self.process(node)
                pbar.total = pbar.n + 1 + len(self.work)
                pbar.update(1)

    def result(self):
        starts = _sets(self.starts)
        cover = defaultdict(set)
        for pc, ops in starts.items():
            for op in ops:
                for k in range(_LEN[op]):
                    cover[(pc + k) & 0xFFFF].add((pc, op, k))
        for a in self.vbase:
            cover[a].add((-1, -1, -1))
        stores = {pc: tv for pc, tv in self.stores.items() if tv[0] != TOP16}
        cells = {}
        for a, ins in cover.items():
            w = {pc: t == 1 << a for pc, (t, _, _) in stores.items() if t >> a & 1}
            if w:
                cells[a] = Cell(
                    w,
                    frozenset(bits(self.vs(a)).tolist()),
                    frozenset("ctrl" if k < 0 else _class(op, k) for _, op, k in ins),
                    frozenset((pc, op) for pc, op, k in ins if k >= 0),
                    bool(self.data[a]),
                    frozenset(bits(self.executed[a]).tolist()),
                    int(self.image[a]),
                )
        names = dict(_REGS, sp=3)
        regs = {}
        for (pc, _), st in self.states.items():
            old = regs.get(pc)
            regs[pc] = st.regs if old is None else tuple(x | y for x, y in zip(old, st.regs))
        return SmcMap(
            starts,
            self.edges,
            dict(self.branches),
            {
                pc: Writer(frozenset(bits(t).tolist()), frozenset(bits(v).tolist()))
                for pc, (t, v, _) in stores.items()
            },
            sorted(self.unresolved),
            _sets(self.transfers),
            _sets(self.opcodes),
            cells,
            {
                pc: {n: frozenset(bits(r[i]).tolist()) for n, i in names.items()}
                for pc, r in regs.items()
            },
            self.vectors,
            {h: frozenset(v) for h, v in self.handlers.items()},
            self.chain,
            sorted(
                (f, v, p)
                for f, v in self.exposed
                for p in sorted(self.pairs.get((f, v), ())) or [None]
            ),
        )

    def fixpoint(self, progress):
        self.run(progress)
        while self.image_handlers():
            self.run(progress)

    def image_handlers(self):
        """Make each vector's image value an interrupt entry once it is analysed code."""
        new = False
        for v in self.vectors:
            h = int(self.image[v]) | int(self.image[(v + 1) & 0xFFFF]) << 8
            if h in self.starts and v not in self.handlers.get(h, ()):
                self.handler(h, v)
                new = True
        return new


def analyze(mem, entries, cov=None, sp=0xFD, vectors=VECTORS, progress=False):
    """:class:`SmcMap` of the program in 64K ``mem`` entered at ``entries``.

    ``entries``: addresses or ``{addr: {"a": values, ..., "irq": bool}}`` (unlisted registers
    may hold anything), run with SP ``sp`` under a ``call`` (or interrupt) frame whose return
    exits. ``cov`` resolves unresolved transfers by its observed ``edges`` (each followed with
    the state at its source) and enters its interrupt ``entries`` in interrupt context; only
    executed addresses and edge/interrupt targets still unreached are then entered in an
    unknown context. Handlers that constant pairs install at ``vectors``, and image vector
    values that are code, are entered in interrupt context.
    """
    an = _Analysis(mem, sp, vectors)
    items = dict(entries) if isinstance(entries, dict) else dict.fromkeys(entries)
    for addr, opts in items.items():
        an.entry(addr, opts or {})
    if cov is not None:
        for src, dst in cov.edges:
            an.observed[src] |= 1 << dst
        for v, h in sorted(cov.entries):
            an.interrupt(h, v)
    an.fixpoint(progress)
    if cov is not None:
        seeds = set(np.flatnonzero(cov.flags & EXEC).tolist())
        seeds |= {t for _, t in cov.edges | cov.entries if t != RETURN}
        for a in sorted(seeds - an.starts.keys()):
            an.seed(a)
        an.fixpoint(progress)
    return an.result()


class Obligation(NamedTuple):
    """Something a run must show: ``kind`` ``"def"`` (``writer``'s value at ``cell`` was
    executed, or consumed from a vector), ``"value"`` (``cell`` was executed holding ``value``),
    ``"entry"`` (handler ``value`` was entered through vector ``cell``) or ``"chain"`` (after
    ``writer`` installed it, handler ``value`` was entered through ``cell``); unused fields -1."""

    kind: str
    cell: int
    writer: int
    value: int


def obligations(smc):
    """All-defs obligations for every written code and vector byte, executed-variant
    obligations for code bytes whose writers store constants only, an entry obligation for
    every static interrupt handler and a chain obligation for every handler chain edge."""
    out = []
    for a, cell in sorted(smc.cells.items()):
        out += [Obligation("def", a, w, -1) for w in sorted(cell.writers)]
        if not cell.data_dependent:
            out += [Obligation("value", a, -1, v) for v in sorted(cell.variants)]
    out += [Obligation("entry", v, -1, h) for h, vs in sorted(smc.handlers.items()) for v in vs]
    out += [Obligation("chain", v, src, h) for src, v, h in sorted(smc.chain)]
    return out


def obligation_rows(smc):
    """``(Obligation, rows)``: sidecar rows ``[kind, cell, a, b]`` any of which satisfies it
    (see ``jennings.coverage.report.held``); an image-byte variant also holds when an
    instruction ``b`` covering the cell ran unwritten, and a chain edge from a source that is
    not a handler on its vector when the handler follows no target or any of that vector's."""
    for ob in obligations(smc):
        if ob.kind == "def":
            yield ob, [[0, ob.cell, ob.writer, -1]]
            continue
        if ob.kind == "entry":
            yield ob, [[3, ob.cell, ob.value, -1]]
            continue
        if ob.kind == "chain":
            if ob.cell in smc.handlers.get(ob.writer, ()):
                prev = [ob.writer]
            else:
                prev = [-1] + sorted(h for h, vs in smc.handlers.items() if ob.cell in vs)
            yield ob, [[4, ob.cell, p, ob.value] for p in prev]
            continue
        rows = [[1, ob.cell, ob.value, -1]]
        cell = smc.cells[ob.cell]
        if ob.value == cell.image:
            rows += [[2, ob.cell, ob.value, s] for s, _ in sorted(cell.insns)]
        yield ob, rows


def write_obligations(smc, path, labels=None):
    """Write an obligation listing to ``path`` (one line per obligation, ``labels``
    ``{address: name}`` naming cells) and its ``.jcov`` sidecar for the coverage.py plugin."""
    labels = labels or {}
    text, lines, table = [], [], []
    for n, (ob, rows) in enumerate(obligation_rows(smc), 1):
        cell = smc.cells.get(ob.cell)
        what = {
            "def": "written by $%04X" % ob.writer,
            "value": "executed as $%02X" % ob.value,
            "entry": "enters $%04X" % ob.value,
            "chain": "$%04X installs $%04X" % (ob.writer, ob.value),
        }[ob.kind]
        kind = "/".join(sorted(cell.classes)) if ob.kind in ("def", "value") else ob.kind
        text.append("$%04X  %-16s %-5s %s" % (ob.cell, labels.get(ob.cell, ""), kind, what))
        lines.append([n, ob.cell])
        table += [[n] + r for r in rows]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(text) + "\n")
    arcs = [[-n, n] for n, _ in lines] + [[n, -n] for n, _ in lines]
    save_sidecar(path, lines=lines, arcs=arcs, transitions=[], obligations=table)
    return path


def check(smc, cov):
    """Dynamic writes to static code and vector bytes that ``smc`` did not predict:
    ``("writer", pc, cell)`` for a writer outside W(cell), ``("value", pc, cell, byte)``
    outside Var(cell)."""
    code = {
        (pc + k) & 0xFFFF for pc, ops in smc.starts.items() for op in ops for k in range(_LEN[op])
    }
    code.update((v + k) & 0xFFFF for v in smc.vectors for k in (0, 1))
    out = []
    for w, c, _ in sorted(cov.defuse):
        if c in code and (c not in smc.cells or w not in smc.cells[c].writers):
            out.append(("writer", w, c))
    for pc, packed in sorted(cov.variants):
        for i in range(_LEN[packed & 0xFF]):
            c, v = (pc + i) & 0xFFFF, (packed >> (8 * i)) & 0xFF
            if c in smc.cells and v not in smc.cells[c].values:
                out.append(("value", pc, c, v))
    return out
