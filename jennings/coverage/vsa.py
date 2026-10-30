"""Exact value-set semantics of the lifted 6502 pcode.

A ``w``-byte value is the set of integers it may hold, as a Python int bitmask of ``256**w``
bits. :func:`step` is ``deity_informant.vm.PcodeVM.step`` over such sets (identical on
singletons), reading operand bytes through the lifter's ``prov`` tags.
"""

from functools import lru_cache

import numpy as np
from deity_informant import lift
from deity_informant.lifter import STATUS_BITS

from jennings.coverage._core import EXEC, FALLTHRU, TAKEN

__all__ = [
    "TOP8",
    "TOP16",
    "bits",
    "from_bool",
    "from_values",
    "is_single",
    "rot",
    "word",
    "apply",
    "template",
    "step",
]

TOP8 = (1 << 256) - 1
TOP16 = (1 << 65536) - 1
SP = 3


def bits(m):
    """Sorted ``int64`` array of the values in bitmask ``m``."""
    if not m & (m - 1):
        return np.array([m.bit_length() - 1] if m else [], np.int64)
    raw = np.frombuffer(m.to_bytes((m.bit_length() + 7) >> 3, "little"), np.uint8)
    return np.unpackbits(raw, bitorder="little").nonzero()[0]


def from_bool(b):
    """Bitmask of the true indices of boolean array ``b``."""
    return int.from_bytes(np.packbits(b, bitorder="little").tobytes(), "little")


def from_values(vals, width=1):
    """Bitmask of the integers in ``vals`` (each below ``256**width``)."""
    b = np.zeros(1 << (8 * width), bool)
    b[np.fromiter(vals, np.int64) if isinstance(vals, (set, frozenset)) else vals] = True
    return from_bool(b)


def is_single(m):
    return not m & (m - 1)


def rot(m, k, width=1):
    """``{(v + k) mod 256**width}`` for ``v`` in ``m``."""
    n = 1 << (8 * width)
    k %= n
    return ((m << k) | (m >> (n - k))) & ((1 << n) - 1)


def word(lo, hi):
    """``{l | h << 8}`` for ``l`` in ``lo``, ``h`` in ``hi``."""
    out = 0
    for h in bits(hi).tolist():
        out |= lo << (h << 8)
    return out


def _concrete(mn, a, b, mask, mask0):
    if mn == "INT_ADD":
        return (a + b) & mask
    if mn == "INT_SUB":
        return (a - b) & mask
    if mn == "INT_AND":
        return a & b
    if mn == "INT_OR":
        return a | b
    if mn == "INT_XOR":
        return a ^ b
    if mn == "INT_LEFT":
        return (a << b) & mask
    if mn == "INT_RIGHT":
        return a >> b
    if mn == "INT_EQUAL":
        return (a == b) * 1
    if mn == "INT_NOTEQUAL":
        return (a != b) * 1
    if mn == "INT_LESS":
        return (a < b) * 1
    if mn == "INT_LESSEQUAL":
        return (a <= b) * 1
    if mn == "INT_CARRY":
        return (a + b > mask0) * 1
    raise NotImplementedError(mn)


def apply(mn, out_size, in_sizes, masks):
    """Value set of pcode op ``mn`` (not LOAD/STORE) over input value sets ``masks``."""
    if mn in ("COPY", "INT_ZEXT"):
        return masks[0]
    m0, m1 = masks
    mask, mask0 = (1 << (8 * out_size)) - 1, (1 << (8 * in_sizes[0])) - 1
    if is_single(m0) and is_single(m1):
        return 1 << _concrete(mn, m0.bit_length() - 1, m1.bit_length() - 1, mask, mask0)
    a, b = bits(m0), bits(m1)
    out = np.zeros(1 << (8 * out_size), bool)
    if a.size <= b.size:
        for v in a.tolist():
            out[_concrete(mn, v, b, mask, mask0)] = True
    else:
        for v in b.tolist():
            out[_concrete(mn, a, v, mask, mask0)] = True
    return from_bool(out)


class _Template:
    __slots__ = ("ops", "prov", "ctrl", "length", "writes")

    def __init__(self, op):
        mem = bytearray(5)
        mem[:2] = op, 3
        rec = lift(mem, 0)
        self.ops, self.ctrl, self.length = rec["ops"], rec["ctrl"], rec["len"]
        self.prov = dict(rec["prov"]["ops"])
        mem[4] = 0x5A
        for i, (a, b) in enumerate(zip(self.ops, lift(mem, 0)["ops"])):
            for j, (va, vb) in enumerate(zip(a[2], b[2])):
                if va != vb:
                    self.prov[(i, j)] = ([1], "ptrhi1")
        stk = {SP} if self.ctrl[0] in ("jsr", "rts", "rti", "brk") else set()
        self.writes = frozenset(o[1][1] for o in self.ops if o[1] and o[1][0] == "r") | stk


@lru_cache(maxsize=None)
def template(op):
    """Lifted pcode of ``op`` with every operand-derived constant tagged in ``prov``
    (``ptrhi1``: the zero-page pointer high byte plus one that SHA (zp),Y folds in)."""
    return _Template(op)


def _const(fn, srcs, pc, mem):
    if fn == "word":
        return mem.word((pc + 1) & 0xFFFF, (pc + 2) & 0xFFFF)
    b = mem.byte((pc + srcs[0]) & 0xFFFF)
    if fn == "id":
        return b
    if fn == "hi1":
        return rot(b, 1)
    hi = 0
    for z in bits(b).tolist():
        hi |= mem.byte((z + 1) & 0xFF)
    return rot(hi, 1)


def _stack(sp):
    return sp << 0x100


def _push(regs, mem, vals):
    mem.store(_stack(regs[SP]), vals)
    regs[SP] = rot(regs[SP], -1)


def _pull_word(regs, mem):
    sp = regs[SP]
    w = 0
    for s in bits(sp).tolist():
        w |= mem.word(0x100 | ((s + 1) & 0xFF), 0x100 | ((s + 2) & 0xFF))
    regs[SP] = rot(sp, 2)
    return w


def _status(regs):
    vals = {0x30}
    for i, s in STATUS_BITS:
        vals = {v | f << s for v in vals for f in bits(regs[i]).tolist()}
    return from_values(vals)


def _vector(mem, ptrs):
    if ptrs == TOP16:
        return TOP16
    out = 0
    for p in bits(ptrs).tolist():
        out |= mem.word(p, (p & 0xFF00) | ((p + 1) & 0xFF))
    return out


def step(op, pc, regs, mem):
    """Execute ``op`` at ``pc``; returns successors ``[(flag, targets, regs)]``.

    ``regs`` are 16 value sets indexed like ``PcodeVM.reg`` (updated in place); ``mem``
    offers ``byte(a)``, ``word(lo, hi)``, ``load(addrs)`` and ``store(addrs, values)``.
    ``flag`` is ``EXEC``, ``TAKEN`` or ``FALLTHRU``; ``targets`` is a 16-bit set.
    """
    t = template(op)
    consts = {k: _const(fn, srcs, pc, mem) for k, (srcs, fn) in t.prov.items()}
    uniq = {}
    for i, (mn, out, ins) in enumerate(t.ops):
        vals = [
            (
                consts[(i, j)]
                if (i, j) in consts
                else 1 << vn[1] if vn[0] == "c" else (regs if vn[0] == "r" else uniq)[vn[1]]
            )
            for j, vn in enumerate(ins)
        ]
        if mn == "STORE":
            mem.store(vals[0], vals[1])
            continue
        res = mem.load(vals[0]) if mn == "LOAD" else apply(mn, out[2], [v[2] for v in ins], vals)
        (regs if out[0] == "r" else uniq)[out[1]] = res
    kind = t.ctrl[0]
    nxt = (pc + t.length) & 0xFFFF
    if kind == "next":
        return [(EXEC, 1 << nxt, regs)]
    if kind == "br":
        flag, pol = t.ctrl[1][1], t.ctrl[2]
        out = []
        if regs[flag] >> pol & 1:
            taken = list(regs)
            taken[flag] = 1 << pol
            off = (bits(mem.byte((pc + 1) & 0xFFFF)) ^ 0x80) - 0x80
            out.append((TAKEN, from_values((nxt + off) & 0xFFFF, 2), taken))
        if regs[flag] >> (1 - pol) & 1:
            regs[flag] = 1 << (1 - pol)
            out.append((FALLTHRU, 1 << nxt, regs))
        return out
    if kind in ("jmp", "jsr", "jmpind"):
        tgt = mem.word((pc + 1) & 0xFFFF, (pc + 2) & 0xFFFF)
        if kind == "jsr":
            ret = (pc + 2) & 0xFFFF
            _push(regs, mem, 1 << (ret >> 8))
            _push(regs, mem, 1 << (ret & 0xFF))
        return [(EXEC, _vector(mem, tgt) if kind == "jmpind" else tgt, regs)]
    if kind == "rts":
        return [(EXEC, rot(_pull_word(regs, mem), 1, 2), regs)]
    if kind == "rti":
        regs[SP] = rot(regs[SP], 1)
        p = bits(mem.load(_stack(regs[SP])))
        for i, s in STATUS_BITS:
            regs[i] = from_values((p >> s) & 1)
        return [(EXEC, _pull_word(regs, mem), regs)]
    if kind == "brk":
        ret = (pc + 2) & 0xFFFF
        status = _status(regs)
        _push(regs, mem, 1 << (ret >> 8))
        _push(regs, mem, 1 << (ret & 0xFF))
        _push(regs, mem, status)
        regs[10] = 2
        return [(EXEC, mem.word(0xFFFE, 0xFFFF), regs)]
    return []
