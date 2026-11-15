"""Static targets of computed transfers (``JMP (ind)``, RTS/RTI dispatch, jumps and branches
through self-modified operands) read from constant tables: backward paths to the transfer run
forward in :mod:`.vsa` from an unknown state, splitting per index value at indexed loads from
constant memory so a table entry's bytes stay paired (see ``docs/coverage.md``)."""

import numpy as np

from jennings.coverage._core import EXEC, FALLTHRU, OPERAND, TAKEN, WRITE
from jennings.coverage.smc import _carry, _refine
from jennings.coverage.vsa import TOP8, TOP16, bits, from_bool, is_single, step, template, word
from jennings.isa import is_read, is_rmw, is_store, mode

__all__ = ["DEPTH", "Tables"]

DEPTH, PATHS, SPLITS = 16, 32, 1024
_FLAGS = (8, 9, 10, 11, 13, 14)
_INDEX = {"absx": (1, 0xFFFF), "absy": (2, 0xFFFF), "zpx": (1, 0xFF), "zpy": (2, 0xFF)}
_INDEX["indy"] = (2, 0xFFFF)
_ABS = frozenset(("abs", "absx", "absy"))
_ZP = frozenset(("zp", "zpx", "zpy", "indx", "indy"))
_SPAN = frozenset(("absx", "absy", "zpx", "zpy"))
_MODE = tuple(mode(op) for op in range(256))
_LOADS = frozenset(op for op in range(256) if is_read(op) or is_rmw(op))
_STORES = frozenset(op for op in range(256) if is_store(op) or is_rmw(op))
_PUSHES = frozenset(
    op
    for op in range(256)
    if template(op).ctrl[0] == "next"
    and 3 in template(op).writes
    and any(o[0] == "STORE" for o in template(op).ops)
)


def _exposed(op):
    """Registers ``op`` reads before writing them (SP for its stack traffic)."""
    out, done = set(), set()
    for _, dst, srcs in template(op).ops:
        out |= {v[1] for v in srcs if v[0] == "r" and v[1] not in done}
        if dst and dst[0] == "r":
            done.add(dst[1])
    return frozenset(out | ({3} if template(op).ctrl[0] in ("jsr", "rts", "rti", "brk") else set()))


_READS = tuple(_exposed(op) for op in range(256))
_SMC = frozenset(("jmp", "jsr", "br"))
_COMPUTED = frozenset(("jmpind", "rts", "rti"))


class _View:
    """Memory as one path sees it: its overlay ``ov`` (key -1: a store reached any address),
    else the image byte where constant; ``reads``/``writes`` the addresses accessed."""

    __slots__ = ("const", "m", "ov", "own", "reads", "writes")

    def __init__(self, const, m, ov):
        self.const, self.m, self.ov, self.own = const, m, ov, False
        self.reads = self.writes = 0

    def byte(self, a):
        self.reads |= 1 << a
        v = self.ov.get(a)
        if v is not None:
            return v
        return 1 << int(self.m[a]) if self.const[a] and -1 not in self.ov else TOP8

    def word(self, lo, hi):
        return word(self.byte(lo), self.byte(hi))

    def load(self, addrs):
        if is_single(addrs):
            return self.byte(addrs.bit_length() - 1)
        self.reads |= addrs
        if -1 in self.ov:
            return TOP8
        hit = 0
        for a in self.ov:
            hit |= 1 << a
        hit &= addrs
        rest = bits(addrs & ~hit)
        if not self.const[rest].all():
            return TOP8
        out = from_bool(np.bincount(self.m[rest], minlength=0x100) > 0)
        for a in bits(hit).tolist():
            out |= self.ov[a]
        return out

    def store(self, addrs, vals):
        self.writes |= addrs
        if not self.own:
            self.ov, self.own = dict(self.ov), True
        if is_single(addrs):
            self.ov[addrs.bit_length() - 1] = vals
            return
        if addrs.bit_count() > 0x100:
            self.ov = {a: TOP8 for a in self.ov}
            self.ov[-1] = TOP8
            return
        for a in bits(addrs).tolist():
            self.ov[a] = self.byte(a) | vals


def _base(op, pc, view):
    """``(index register, base, wrap)`` of the indexed load ``op``, or None."""
    idx = _INDEX.get(_MODE[op]) if op in _LOADS else None
    if idx is None:
        return None
    a1, a2 = (pc + 1) & 0xFFFF, (pc + 2) & 0xFFFF
    base = view.word(a1, a2) if _MODE[op] in _ABS else view.byte(a1)
    if _MODE[op] == "indy" and is_single(base):
        z = base.bit_length() - 1
        base = view.word(z, (z + 1) & 0xFF)
    return (idx[0], base.bit_length() - 1, idx[1]) if is_single(base) else None


def _pin(regs, fo, r, v):
    """``regs`` and flag origins ``fo`` with register ``r`` fixed to ``v``."""
    regs = list(regs)
    regs[r] = keep = 1 << v
    out = {}
    for f, (q, m0, m1) in fo.items():
        if q == r:
            m0, m1 = m0 & keep, m1 & keep
            regs[f] &= bool(m0) | bool(m1) << 1
        out[f] = (q, m0, m1)
    return regs, out


class Tables:
    """Resolver over memory ``m`` and decoded code ``succ`` (``{address: successors}``) with
    ``owned`` instruction bytes, ``image`` (bool, the program's bytes), the run's ``cov``
    (or None), ``entries`` (where backward paths stop) and ``firsts``, the first byte each
    table load reads (``base`` + least index), shared across resolvers of one listing."""

    def __init__(self, m, succ, owned, image, cov, entries, firsts):
        self.m, self.succ, self.owned, self.image = m, succ, owned, image
        self.entries, self.firsts = entries, firsts
        flags = np.zeros(0x10000, np.uint8) if cov is None else cov.flags
        self.code = (flags & (EXEC | OPERAND)) != 0
        written = (flags & WRITE) != 0
        self.objects = np.zeros(0x10000, bool)
        self.preds = {}
        for a, out in succ.items():
            for f, t in out:
                self.preds.setdefault(t, []).append((a, f))
            op = int(m[a])
            md = _MODE[op]
            if md in _ABS or md in _ZP:
                b = int(m[(a + 1) & 0xFFFF]) | (int(m[(a + 2) & 0xFFFF]) << 8) * (md in _ABS)
                self.objects[b] = True
                if op in _STORES and not flags[a] & EXEC and not md.startswith("ind"):
                    span = 0x100 if md in _SPAN else 1
                    written[(b + np.arange(span)) & (0xFF if md in _ZP else 0xFFFF)] = True
        self.written = written
        self.const = image & ~written
        self.const[0x100:0x200] = False

    def transfers(self):
        """Decoded transfers with computed targets: ``JMP (ind)``, RTS, RTI, and
        ``JMP``/``JSR``/branches with a written operand byte."""
        out = []
        for a in self.succ:
            kind = template(int(self.m[a])).ctrl[0]
            ops = [(a + 1) & 0xFFFF, (a + 2) & 0xFFFF]
            if kind in _COMPUTED or kind in _SMC and self.written[ops].any():
                out.append(a)
        return sorted(out)

    def _paths(self, t):
        """Backward paths ``[(address, flag of the edge to the next)]`` ending at ``t``."""
        out = []

        def walk(path, seen):
            a = path[-1][0]
            ps = []
            if a not in self.entries and len(path) < DEPTH:
                ps = [(p, f) for p, f in self._preds(a) if p not in seen]
            if not ps:
                out.append(path[::-1])
            for p in ps:
                if len(out) > PATHS:
                    return
                walk(path + [p], seen | {p[0]})

        walk([(t, None)], {t})
        if len(out) <= PATHS:
            return out
        path = [(t, None)]
        while path[-1][0] not in self.entries and len(path) < DEPTH:
            ps = self._preds(path[-1][0])
            if len(ps) != 1 or any(ps[0][0] == q for q, _ in path):
                break
            path.append(ps[0])
        return [path[::-1]]

    def _preds(self, a):
        """Static predecessors ``[(address, flag)]`` of ``a`` other than ``JSR`` returns."""
        return [
            (p, f) for p, f in self.preds.get(a, ()) if template(int(self.m[p])).ctrl[0] != "jsr"
        ]

    def _splits(self, op, pc, state, first):
        """``state`` split per value of the index of the indexed load at ``pc`` (whose least
        byte on the unsplit path is ``first``) unless its least byte is written data, the byte
        each split reads appended to its cells."""
        regs, ov, fo, key, cells = state
        idx = _base(op, pc, _View(self.const, self.m, ov))
        if idx is None or -1 in ov:
            return [state]
        r, base, wrap = idx
        vals = bits(regs[r]).tolist()
        addrs = ((base + np.array(vals)) & wrap).tolist()
        lo = addrs[0]
        if len(vals) > 1 and self.written[lo] > self.owned[lo] and lo not in ov:
            return [state]
        own = (pc, base, first)
        out = []
        for v, a in zip(vals, addrs):
            k = a not in ov
            regs2, fo2 = _pin(regs, fo, r, v) if len(vals) > 1 else (regs, fo)
            key2 = key + (v,) if len(vals) > 1 else key
            out.append((regs2, ov, fo2, key2, cells + ((a, own),) * k))
        return out

    def _run(self, path, split=None, trace=None):
        """``[(split key, targets, cells)]`` of the states reaching the transfer ending
        ``path``, split at path indices ``split`` ``{index: first}``; ``cells`` ``((address,
        (pc, base, first)),)`` the bytes table loads read; ``trace`` gets each step's
        ``(reads, writes, first)``, ``first`` the least byte an indexed load may read."""
        regs = [1] * 16
        regs[0] = regs[1] = regs[2] = TOP8
        regs[3] = 1 << 0xFF
        for f in _FLAGS:
            regs[f] = 3
        states, out, split = [(regs, {}, {}, (), ())], [], split or {}
        for i, (pc, flag) in enumerate(path):
            op = int(self.m[pc])
            t = template(op)
            nxt = []
            for regs, ov, fo, key, cells in (
                s
                for st in states
                for s in (self._splits(op, pc, st, split[i]) if i in split else [st])
            ):
                view, origins = _View(self.const, self.m, ov), {}
                succ = step(op, pc, list(regs), view, origins)
                if trace is not None:
                    idx = _base(op, pc, _View(self.const, self.m, ov))
                    first = None
                    if idx is not None and regs[idx[0]]:
                        first = (idx[1] + (regs[idx[0]] & -regs[idx[0]]).bit_length() - 1) & idx[2]
                    trace.append((view.reads, view.writes, first))
                fo = _carry(fo, t.writes, origins)
                if i == len(path) - 1:
                    tg = 0
                    for f, targets, _ in succ:
                        tg |= 0 if f == FALLTHRU else targets
                    out.append((key, tg, cells))
                    continue
                for f, targets, regs2 in succ:
                    cut = (regs2, fo)
                    if f != EXEC:
                        cut = _refine(tuple(regs2), fo, t.ctrl[1][1], t.ctrl[2] ^ (f != TAKEN))
                    if f == flag and targets >> path[i + 1][0] & 1 and cut is not None:
                        nxt.append((list(cut[0]), view.ov, cut[1], key, cells))
            states = nxt
            if len(states) > SPLITS:
                return []
        return out

    def _slice(self, path, trace):
        """``{index: first byte}`` of the indexed loads whose value reaches the path's
        transfer, by a backward slice over registers and the addresses each step accessed
        that keeps the index registers of table loads apart (they choose the entry)."""
        t = int(self.m[path[-1][0]])
        val, adr, out = [_READS[t], trace[-1][0]], [frozenset(), 0], {}
        for i in range(len(path) - 2, -1, -1):
            op = int(self.m[path[i][0]])
            reads, writes, first = trace[i]
            w = template(op).writes
            hit = [bool(w & s[0] or writes & s[1]) for s in (val, adr)]
            idx = {_INDEX[_MODE[op]][0]} if first is not None else set()
            if hit[0] and first is not None:
                out[i] = first
            for s in (val, adr):
                s[0] -= w
                s[1] &= ~writes if is_single(writes) else -1
            if hit[0]:
                val[0] |= _READS[op] - idx
                val[1] |= reads
                adr[0] |= _READS[op] & idx
            if hit[1]:
                adr[0] |= _READS[op]
                adr[1] |= reads
        return out

    def _ends(self, a, own):
        """Why a table load ``own`` ``(pc, base, first)`` cannot read byte ``a``: 0 it can,
        1 code or not constant, 2 another referenced object or table's first byte."""
        if (self.objects[a] or a in self.firsts) and a not in own[1:]:
            return 2
        return int(bool(self.owned[a] or self.code[a] or not self.const[a]))

    def plan(self, t):
        """``[(path, unsplit result, split indices)]`` of the paths to ``t`` that may resolve;
        adds the first bytes of their table loads to ``firsts``."""
        kind = template(int(self.m[t])).ctrl[0]
        out = []
        for path in self._paths(t):
            if kind in ("rts", "rti") and not any(int(self.m[a]) in _PUSHES for a, _ in path):
                continue
            trace = []
            base = self._run(path, trace=trace)
            split = self._slice(path, trace) if len(trace) == len(path) else None
            if split is None or not split and all(tg in (0, TOP16) for _, tg, _ in base):
                continue
            self.firsts.update(split.values())
            out.append((path, base, split))
        return out

    def targets(self, plan):
        """``(targets, cells)`` of a ``plan``: per path, its splits in index order form the
        table from the first whose bytes may all be table bytes, at the stride to the next
        that reads none of the bytes taken, until one reads a byte a table cannot hold or
        one taken; single targets of the taken splits, ``cells`` their table bytes."""
        found, cells = set(), set()
        for path, base, split in plan:
            taken, v0, stride = set(), None, None
            for key, tg, cs in sorted(self._run(path, split) if split else base):
                v = key[0] if key else 0
                over = any(a in taken for a, _ in cs)
                end = max([self._ends(a, own) for a, own in cs] + [2 * over])
                if v0 is None:
                    if end == 1:
                        continue
                    v0 = v
                elif stride is None and over or stride and (v - v0) % stride:
                    continue
                elif stride is None:
                    stride = v - v0
                if end:
                    break
                taken.update(a for a, _ in cs)
                if tg and is_single(tg):
                    found.add(tg.bit_length() - 1)
            cells |= taken
        return sorted(found), sorted(cells)
