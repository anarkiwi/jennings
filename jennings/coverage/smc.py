"""Static self-modifying-code analysis: which instructions can write each code byte, and
the values it can hold, by a flow-sensitive value-set fixpoint over :mod:`.vsa`.

One joined state per instruction address; control flow follows static targets, JSR/BRK
return addresses and constant-written code bytes only (see ``docs/coverage.md``).
"""

import heapq
from collections import defaultdict
from typing import NamedTuple

import numpy as np

from jennings.coverage import EXEC, RETURN, TAKEN
from jennings.coverage.vsa import TOP8, TOP16, bits, from_bool, from_values, is_single
from jennings.coverage.vsa import rot, step, template, word
from jennings.opcodes import MEM_MODES, MODE_LEN, OPCODES

__all__ = ["Writer", "Cell", "SmcMap", "analyze"]

_REGS = {"a": 0, "x": 1, "y": 2, "c": 8, "z": 9, "i": 10, "d": 11, "v": 13, "n": 14}
_CTRL = frozenset((0x20, 0x4C, 0x6C))
_MN = tuple(OPCODES[op][0] for op in range(256))
_MODE = tuple(OPCODES[op][1] for op in range(256))
_LEN = tuple(MODE_LEN[m] for m in _MODE)
_PUSH = {"PHA": 1, "PHP": 1, "PLA": -1, "PLP": -1}


class Writer(NamedTuple):
    """A store site: every address it may write and every value it may store there."""

    targets: frozenset
    values: frozenset


class Cell(NamedTuple):
    """A written code byte: ``writers`` ``{pc: exact}`` (``exact``: the cell is the store's
    only target), ``values`` it may hold (exact unless ``data_dependent``: a writer stores
    a computed value), ``classes`` and the ``(start, opcode)`` instructions it is part of."""

    writers: dict
    values: frozenset
    classes: frozenset
    insns: frozenset
    data_dependent: bool


class SmcMap(NamedTuple):
    """``starts`` ``{pc: opcodes}``, ``edges``, feasible ``branches``, ``writers``, and the
    unfollowed ``unresolved_stores`` ``[pc]``, ``unresolved_transfers`` ``{pc: targets}``,
    ``unresolved_opcodes`` ``{pc: opcodes}``; ``cells`` ``{addr: Cell}``, ``regs`` per pc."""

    starts: dict
    edges: set
    branches: dict
    writers: dict
    unresolved_stores: list
    unresolved_transfers: dict
    unresolved_opcodes: dict
    cells: dict
    regs: dict


class _State:
    """Register value sets, bytes written since entry ``ov``, and the return addresses
    ``rets`` ``{offset above SP: 16-bit set}`` pushed by JSR/BRK/entry and not yet popped."""

    __slots__ = ("regs", "ov", "rets")

    def __init__(self, regs, ov, rets):
        self.regs, self.ov, self.rets = regs, ov, rets

    def __eq__(self, other):
        return (self.regs, self.ov, self.rets) == (other.regs, other.ov, other.rets)

    __hash__ = None


def _join(s, t):
    """Pointwise union; a byte or return address absent on either side is dropped."""
    return _State(
        tuple(x | y for x, y in zip(s.regs, t.regs)),
        {a: v | t.ov[a] for a, v in s.ov.items() if a in t.ov},
        {o: v | t.rets[o] for o, v in s.rets.items() if o in t.rets},
    )


def _shift(rets, k, new=None):
    out = {o + k: v for o, v in rets.items() if 0 < o + k < 0xFF}
    if new:
        out.update(new)
    return out


class _Mem:
    """One instruction's view of memory: the state's overlay, else the global value set."""

    __slots__ = ("an", "pc", "ov", "own", "stack")

    def __init__(self, an, pc, st):
        self.an, self.pc, self.ov, self.own, self.stack = an, pc, st.ov, False, False

    def byte(self, a):
        v = self.ov.get(a)
        if v is None:
            self.an.readers[a].add(self.pc)
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
        return out | self.an.gather(bits(addrs), self.pc) if addrs else out

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


def _class(op, k):
    if k == 0:
        return "var"
    if op in _CTRL or _MODE[op] == "rel":
        return "ctrl"
    return "imm" if _MODE[op] == "imm" else "addr" if _MODE[op] in MEM_MODES else "var"


def _sets(d):
    return {k: frozenset(bits(m).tolist()) for k, m in d.items()}


class _Analysis:
    def __init__(self, mem, sp):
        self.image = np.asarray(mem, np.uint8)
        self.sp = sp
        self.written = {}
        self.readers = defaultdict(set)
        self.wide = defaultdict(set)
        self.vsm = np.zeros((0x10000, 0x100), bool)
        self.vsm[np.arange(0x10000), self.image] = True
        self.trust = self.vsm.copy()
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

    def vs(self, a):
        return (1 << int(self.image[a])) | self.written.get(a, 0)

    def gather(self, arr, pc):
        """Union of the global value sets at addresses ``arr``, read by ``pc``."""
        for page in np.unique(arr >> 8).tolist():
            self.wide[page].add(pc)
        return from_bool(self.vsm[arr].any(0))

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
        for a in grown.tolist():
            self.written[a] = self.written.get(a, 0) | vals
            for k in self.readers[a]:
                self._enqueue(k)
        for page in np.unique(grown >> 8).tolist():
            for k in self.wide[page]:
                self._enqueue(k)

    def _enqueue(self, pc):
        if pc not in self.queued:
            self.queued.add(pc)
            heapq.heappush(self.work, pc)

    def add(self, pc, st):
        old = self.states.get(pc)
        if old is not None:
            st = _join(old, st)
            if st == old:
                return
        self.states[pc] = st
        self._enqueue(pc)

    def seed(self, addr):
        """Enter ``addr`` in an unknown context: any registers, stack pointer and stack."""
        regs = [1] * 16
        for i in _REGS.values():
            regs[i] = TOP8 if i < 3 else 3
        regs[3] = TOP8
        self.add(addr, _State(tuple(regs), {}, {}))

    def entry(self, addr, opts):
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
        self.add(addr, _State(tuple(regs), ov, {1 + irq: 1 << ret}))

    def const(self, mem, a, pc):
        """Values of byte ``a`` from the image or constant stores; none once a computed
        value may be stored there."""
        self.readers[a].add(pc)
        return 0 if self.data[a] else mem.byte(a) & from_bool(self.trust[a])

    def _word(self, mem, lo, hi, pc):
        return word(self.const(mem, lo, pc), self.const(mem, hi, pc))

    def _follow(self, kind, pc, st, mem):
        """Targets of a ``kind`` transfer built from constants only, and the new ``rets``."""
        rets, a1, a2 = st.rets, (pc + 1) & 0xFFFF, (pc + 2) & 0xFFFF
        if kind == "br":
            offs = (bits(self.const(mem, a1, pc)) ^ 0x80) - 0x80
            return from_values((pc + 2 + offs) & 0xFFFF, 2), rets
        if kind == "jmp":
            return self._word(mem, a1, a2, pc), rets
        if kind == "jsr":
            return self._word(mem, a1, a2, pc), _shift(rets, 2, {1: 1 << a2})
        if kind == "jmpind":
            out = 0
            for p in bits(self._word(mem, a1, a2, pc)).tolist():
                out |= self._word(mem, p, (p & 0xFF00) | ((p + 1) & 0xFF), pc)
            return out, rets
        if kind == "brk":
            return self._word(mem, 0xFFFE, 0xFFFF, pc), _shift(rets, 3, {2: 1 << a2})
        if kind == "rts":
            return rot(rets.get(1, 0), 1, 2), _shift(rets, -2)
        if kind == "rti":
            return rets.get(2, 0), _shift(rets, -3)
        return 0, rets

    def process(self, pc):
        st = self.states[pc]
        mem = _Mem(self, pc, st)
        ops = mem.byte(pc)
        self.starts[pc] |= ops
        ok = self.const(mem, pc, pc)
        if ops & ~ok:
            self.opcodes[pc] |= ops & ~ok
        for op in bits(ok).tolist():
            mem = _Mem(self, pc, st)
            kind = template(op).ctrl[0]
            succ = step(op, pc, list(st.regs), mem)
            if kind == "next":
                d = _PUSH.get(_MN[op], 0)
                follow = None
                rets = _shift(st.rets, d) if d or not mem.stack else {}
                if 3 in template(op).writes and not d:
                    rets = {}
            else:
                follow, rets = self._follow(kind, pc, st, mem)
            for flag, targets, regs in succ:
                if flag != EXEC:
                    self.branches[pc] |= flag
                if follow is not None and (kind != "br" or flag == TAKEN):
                    if targets & ~follow:
                        self.transfers[pc] |= targets & ~follow
                    targets &= follow
                if kind in ("rts", "rti") and targets >> RETURN & 1:
                    self.edges.add((pc, RETURN))
                    targets &= ~(1 << RETURN)
                out = _State(tuple(regs), mem.ov, rets)
                for t in bits(targets).tolist():
                    self.edges.add((pc, t))
                    self.add(t, out)

    def run(self, progress):
        from tqdm import tqdm  # pylint: disable=import-outside-toplevel

        with tqdm(desc="smc", unit="insn", disable=not progress) as pbar:
            while self.work:
                pc = heapq.heappop(self.work)
                self.queued.discard(pc)
                self.process(pc)
                pbar.total = pbar.n + 1 + len(self.work)
                pbar.update(1)

    def result(self):
        starts = _sets(self.starts)
        cover = defaultdict(set)
        for pc, ops in starts.items():
            for op in ops:
                for k in range(_LEN[op]):
                    cover[(pc + k) & 0xFFFF].add((pc, op, k))
        stores = {pc: tv for pc, tv in self.stores.items() if tv[0] != TOP16}
        cells = {}
        for a, ins in cover.items():
            w = {pc: t == 1 << a for pc, (t, _, _) in stores.items() if t >> a & 1}
            if w:
                cells[a] = Cell(
                    w,
                    frozenset(bits(self.vs(a)).tolist()),
                    frozenset(_class(op, k) for _, op, k in ins),
                    frozenset((pc, op) for pc, op, _ in ins),
                    bool(self.data[a]),
                )
        names = dict(_REGS, sp=3)
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
                pc: {n: frozenset(bits(st.regs[i]).tolist()) for n, i in names.items()}
                for pc, st in self.states.items()
            },
        )


def analyze(mem, entries, cov=None, sp=0xFD, progress=False):
    """:class:`SmcMap` of the program in 64K ``mem`` entered at ``entries``.

    ``entries``: addresses or ``{addr: {"a": values, ..., "irq": bool}}`` (unlisted registers
    may hold anything), run with SP ``sp`` under a ``call`` (or interrupt) frame whose return
    exits. ``cov`` adds its executed addresses and edge/interrupt targets as entries.
    """
    an = _Analysis(mem, sp)
    items = dict(entries) if isinstance(entries, dict) else dict.fromkeys(entries)
    if cov is not None:
        seeds = set(np.flatnonzero(cov.flags & EXEC).tolist())
        seeds |= {t for _, t in cov.edges | cov.entries if t != RETURN}
        for a in seeds - items.keys():
            an.seed(a)
    for addr, opts in items.items():
        an.entry(addr, opts or {})
    an.run(progress)
    return an.result()
