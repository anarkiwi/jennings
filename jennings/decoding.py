"""Instruction decoding, per-opcode roles, data addresses and value-set transfer.

Decoding reads a 64 KiB image directly; the role tables are read off the lifted pcode the MPU
executes, and :func:`transfer` evaluates that pcode over :mod:`jennings.coverage.vsa` masks.
Mode names are those of :data:`jennings.opcodes.MODE_LEN`.
"""

from typing import NamedTuple

import numpy as np
from deity_informant.lifter import STATUS_BITS, bus_only, template

from jennings.coverage.vsa import TOP8, apply, bits, from_bool, from_values, is_single
from jennings.opcodes import MODE_LEN, OPCODES

__all__ = [
    "A",
    "ADDRESSED",
    "C",
    "DTYPE",
    "FLAGS",
    "INDEX",
    "Instruction",
    "KIND",
    "KINDS",
    "KIND_CODE",
    "LENGTH",
    "LOCATIONS",
    "M",
    "MEM_READ",
    "MEM_WRITE",
    "MODE",
    "MODES",
    "MODE_CODE",
    "OPERAND",
    "READS",
    "Roles",
    "SP",
    "STORED",
    "UNKNOWN",
    "WRITES",
    "X",
    "Y",
    "accesses",
    "as_mask",
    "as_values",
    "decode",
    "decode_many",
    "names",
    "roles",
    "transfer",
]

#: Addressing-mode names in code order: ``MODE[op]`` indexes this tuple.
MODES = tuple(MODE_LEN)
MODE_CODE = {name: code for code, name in enumerate(MODES)}

#: Role-mask bit ``i`` is ``LOCATIONS[i]`` (registers and flags at their PcodeVM index).
LOCATIONS = ("A", "X", "Y", "SP", "M", "OPERAND", None, None, "C", "Z", "I", "D", "B", "V", "N")
_BIT = {name: 1 << i for i, name in enumerate(LOCATIONS) if name}
A, X, Y, SP, M, OPERAND, C = (_BIT[n] for n in ("A", "X", "Y", "SP", "M", "OPERAND", "C"))
FLAGS = sum(_BIT[n] for n in "CZIDBVN")
_REGS = A | X | Y | SP
_REG = {i: n for i, n in enumerate(LOCATIONS) if n and _BIT[n] & (_REGS | FLAGS)}
_STATUS = sum(1 << i for i, _ in STATUS_BITS)

#: Instruction kinds in code order: ``KIND[op]`` indexes this tuple.
KINDS = (
    "load",
    "store",
    "rmw",
    "transfer",
    "incdec",
    "alu",
    "compare",
    "branch",
    "jump",
    "call",
    "return",
    "stack",
    "flag",
    "nop",
    "other",
)
KIND_CODE = {name: code for code, name in enumerate(KINDS)}
_CTRL_KIND = {
    "br": "branch",
    "jmp": "jump",
    "jmpind": "jump",
    "jsr": "call",
    "rts": "return",
    "rti": "return",
    "brk": "other",
    "jam": "other",
}
_CTRL_READS = {"jsr": SP, "rts": SP, "rti": SP, "brk": SP | _STATUS}
_CTRL_WRITES = {"jsr": SP, "rts": SP, "rti": SP | _STATUS, "brk": SP | _BIT["I"]}
_INDEXED = {"zpx": X, "absx": X, "indx": X, "zpy": Y, "absy": Y, "indy": Y}


class _Walk(NamedTuple):
    reads: int
    writes: int
    mem_read: bool
    mem_write: bool
    stored: int
    addressed: int
    sources: int
    stack: bool
    loads: frozenset


def _mask(locs):
    return sum(_BIT.get(n, 0) for n in locs)


def _walk(op):
    """Taint walk of ``op``'s non-bus pcode; a load is a pointer if its value reaches an
    address, a stack access if its address comes from ``SP``, else the memory operand."""
    t = template(op)
    bus = bus_only(t.ops)
    regs, uniq, acc = {}, {}, []

    def src(vn, at):
        if at in t.prov:
            return frozenset(("OPERAND",))
        if vn[0] == "c":
            return frozenset()
        if vn[0] == "u":
            return uniq.get(vn[1], frozenset())
        return regs.get(vn[1], frozenset((_REG[vn[1]],)))

    for i, (mn, dst, ins) in enumerate(t.ops):
        if i in bus:
            continue
        if mn == "STORE":
            acc.append((True, i, src(ins[0], (i, 0)), src(ins[1], (i, 1))))
            continue
        if mn == "LOAD":
            acc.append((False, i, src(ins[0], (i, 0)), None))
            val = frozenset((i,))
        else:
            val = frozenset().union(*(src(vn, (i, j)) for j, vn in enumerate(ins)))
        (regs if dst[0] == "r" else uniq)[dst[1]] = val
    reached = frozenset().union(*(addr for _, _, addr, _ in acc))
    stack = [a for a in acc if "SP" in a[2]]
    data = [a for a in acc if "SP" not in a[2] and a[1] not in reached]
    loads = {i for store, i, _, _ in data if not store}

    def res(s):
        return _mask(n for n in s if not isinstance(n, int)) | (M if s & loads else 0)

    stores = [res(v) for store, _, _, v in data if store]
    final = {r: res(v) for r, v in regs.items()}
    reads = 0
    for v in list(final.values()) + stores + [res(v) for store, _, _, v in stack if store]:
        reads |= v
    kind = t.ctrl[0]
    reads |= (1 << t.ctrl[1] if kind == "br" else 0) | _CTRL_READS.get(kind, 0)
    reads |= SP if stack else 0
    addressed = 0
    for _, _, addr, _ in acc:
        addressed |= 0 if "SP" in addr else res(addr)
    return _Walk(
        reads,
        sum(1 << r for r in regs) | _CTRL_WRITES.get(kind, 0),
        bool(loads),
        bool(stores),
        stores[-1] if stores else 0,
        addressed,
        sum(v for r, v in final.items() if 1 << r & _REGS),
        bool(stack),
        frozenset(loads),
    )


def _kind(op, w):
    ctrl = template(op).ctrl[0]
    regw = w.writes & _REGS
    if ctrl in _CTRL_KIND:
        return _CTRL_KIND[ctrl]
    if w.stack:
        return "stack"
    if w.mem_write:
        return "rmw" if w.mem_read else "store"
    if not regw:
        return ("compare" if w.reads else "flag") if w.writes else "nop"
    if not w.sources & ~(M | OPERAND):
        return "load"
    if OPCODES[op][1] == "impl" and not regw & (regw - 1) and not w.sources & (w.sources - 1):
        return "incdec" if w.sources == regw else "transfer"
    return "alu"


_WALKS = [_walk(op) for op in range(256)]

#: uint8 [256]: :data:`MODES` code of each opcode.
MODE = np.array([MODE_CODE[OPCODES[op][1]] for op in range(256)], np.uint8)
#: uint8 [256]: instruction length in bytes.
LENGTH = np.array([MODE_LEN[OPCODES[op][1]] for op in range(256)], np.uint8)
#: uint8 [256]: :data:`KINDS` code of each opcode.
KIND = np.array([KIND_CODE[_kind(op, w)] for op, w in enumerate(_WALKS)], np.uint8)
#: uint16 [256] location masks: read as values, written, data address sources, stored value.
READS = np.array([w.reads for w in _WALKS], np.uint16)
WRITES = np.array([w.writes for w in _WALKS], np.uint16)
ADDRESSED = np.array([w.addressed for w in _WALKS], np.uint16)
STORED = np.array([w.stored for w in _WALKS], np.uint16)
#: uint16 [256]: ``X``, ``Y`` or 0, the index register of the addressing mode.
INDEX = np.array([_INDEXED.get(OPCODES[op][1], 0) for op in range(256)], np.uint16)
#: bool [256]: the instruction reads / writes its data memory operand on the bus.
MEM_READ = np.array([w.mem_read for w in _WALKS])
MEM_WRITE = np.array([w.mem_write for w in _WALKS])
_QUIRK = ADDRESSED & ~INDEX & (A | X | Y)


def names(mask):
    """Location names of role mask ``mask``, in :data:`LOCATIONS` order."""
    return tuple(n for i, n in enumerate(LOCATIONS) if n and int(mask) >> i & 1)


class Roles(NamedTuple):
    mnemonic: str
    mode: str
    kind: str
    reads: tuple
    writes: tuple
    mem_read: bool
    mem_write: bool
    index: str
    stored: tuple


_ROLES = [
    Roles(
        *OPCODES[op],
        KINDS[KIND[op]],
        names(READS[op]),
        names(WRITES[op]),
        bool(MEM_READ[op]),
        bool(MEM_WRITE[op]),
        (names(INDEX[op]) or (None,))[0],
        names(STORED[op]),
    )
    for op in range(256)
]


def roles(op):
    """:class:`Roles` of opcode ``op``: its table entries as names."""
    return _ROLES[op]


class Instruction(NamedTuple):
    pc: int
    opcode: int
    mnemonic: str
    mode: str
    operand: int
    length: int
    text: str


_SYNTAX = {
    "impl": "",
    "acc": " A",
    "imm": " #$%02x",
    "zp": " $%02x",
    "zpx": " $%02x,X",
    "zpy": " $%02x,Y",
    "indx": " ($%02x,X)",
    "indy": " ($%02x),Y",
    "abs": " $%04x",
    "absx": " $%04x,X",
    "absy": " $%04x,Y",
    "ind": " ($%04x)",
    "rel": " $%04x",
}
#: Operand kind per opcode: 0 none, 1 byte, 2 word, 3 branch target.
_OPERAND = np.array(
    [3 if OPCODES[op][1] == "rel" else LENGTH[op] - 1 for op in range(256)], np.int8
)
_INFO = [
    (mn, md, int(LENGTH[op]), int(_OPERAND[op]), mn + _SYNTAX[md])
    for op, (mn, md) in sorted(OPCODES.items())
]


def decode(mem, pc):
    """:class:`Instruction` at ``pc`` of the 64 KiB image ``mem`` (any byte sequence), its
    effective operand a byte, a word, a branch target or None; ``text`` is the
    :class:`~jennings.disassembler.Disassembler`'s."""
    pc &= 0xFFFF
    op = int(mem[pc])
    mn, md, n, k, fmt = _INFO[op]
    if not k:
        return Instruction(pc, op, mn, md, None, n, fmt)
    v = int(mem[(pc + 1) & 0xFFFF])
    if k == 2:
        v |= int(mem[(pc + 2) & 0xFFFF]) << 8
    elif k == 3:
        v = (pc + 2 + (v ^ 0x80) - 0x80) & 0xFFFF
    return Instruction(pc, op, mn, md, v, n, fmt % v)


#: :func:`decode_many` record: ``operand`` is -1 where :func:`decode`'s is None.
DTYPE = np.dtype(
    [("pc", "<u2"), ("opcode", "u1"), ("mode", "u1"), ("operand", "<i4"), ("length", "u1")]
)


def _bytes(mem):
    if isinstance(mem, (bytes, bytearray, memoryview)):
        return np.frombuffer(mem, np.uint8)
    return np.asarray(mem, np.uint8)


def decode_many(mem, starts):
    """:data:`DTYPE` records of the instructions at ``starts`` (indices, or a 65536 bool mask)
    in the image ``mem``, as :func:`decode` but vectorised."""
    m = _bytes(mem)
    starts = np.asarray(starts)
    pcs = np.flatnonzero(starts) if starts.dtype == bool else starts.astype(np.int64) & 0xFFFF
    op = m[pcs]
    b1 = m[(pcs + 1) & 0xFFFF].astype(np.int64)
    k = _OPERAND[op]
    out = np.empty(len(pcs), DTYPE)
    out["pc"], out["opcode"], out["mode"], out["length"] = pcs, op, MODE[op], LENGTH[op]
    out["operand"] = np.select(
        [k == 1, k == 2, k == 3],
        [b1, b1 | m[(pcs + 2) & 0xFFFF].astype(np.int64) << 8, (pcs + 2 + (b1 ^ 0x80) - 0x80)],
        -1,
    ) & np.where(k > 0, 0xFFFF, -1)
    return out


def as_values(v, n=0x100):
    """Sorted distinct values of value set ``v`` below ``n``: None (all), an int bitmask, a
    bool array of length ``n``, or a collection of integers."""
    if v is None:
        return np.arange(n)
    if isinstance(v, (int, np.integer)):
        return bits(int(v))
    if isinstance(v, np.ndarray) and v.dtype == bool:
        return np.flatnonzero(v)
    return np.unique(np.fromiter(v, np.int64) if isinstance(v, (set, frozenset)) else v)


def as_mask(v, width=1):
    """Bitmask (as :mod:`jennings.coverage.vsa`) of value set ``v`` (as :func:`as_values`)."""
    if v is None:
        return (1 << (1 << (8 * width))) - 1
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, np.ndarray) and v.dtype == bool:
        return from_bool(v)
    return from_values(as_values(v, 1 << (8 * width)), width)


def accesses(op, operand, x=None, y=None, words=None, mem=None, a=None):
    """Sorted data addresses ``op`` with ``operand`` may access over value sets ``x``, ``y``,
    ``a`` (None: any) and, indirect, pointer ``words`` or their bytes in ``mem`` (else any),
    with the zero-page and ``$FFFF`` wraps and the ``SHA``/``SHX``/``SHY``/``TAS`` page cross."""
    if not (MEM_READ[op] or MEM_WRITE[op]):
        return np.zeros(0, np.int64)
    md = OPCODES[op][1]
    wrap = 0xFF if md[:2] == "zp" else 0xFFFF
    idx = as_values({X: x, Y: y}[int(INDEX[op])]) if INDEX[op] else np.zeros(1, np.int64)
    if md in ("indx", "indy"):
        if words is not None:
            base = as_values(words, 0x10000)
        elif mem is None:
            base = np.arange(0x10000)
        else:
            m = _bytes(mem)
            p = (operand + (idx if md == "indx" else 0)) & 0xFF
            base = np.unique(m[p] | m[(p + 1) & 0xFF].astype(np.int64) << 8)
        if md == "indx":
            idx = np.zeros(1, np.int64)
    else:
        base = np.array([operand & wrap])
    ea = np.add.outer(base, idx) & wrap
    out = np.zeros(0x10000, bool)
    q = int(_QUIRK[op])
    if q:
        reg = np.full(1, 0xFF)
        for r, v in ((A, a), (X, x), (Y, y)):
            if q & r:
                reg = np.unique(np.bitwise_and.outer(reg, as_values(v)))
        cross = ((ea ^ base[:, None]) & 0xFF00) != 0
        bi, ii = np.nonzero(cross)
        hi = reg[None, :] & ((base[bi, None] >> 8) + 1) & 0xFF
        out[hi << 8 | ea[bi, ii, None] & 0xFF] = True
        ea = ea[~cross]
    out[ea.ravel()] = True
    return np.flatnonzero(out)


class _Unknown:
    __slots__ = ()

    def __repr__(self):
        return "UNKNOWN"


#: A destination (or a whole opcode) :func:`transfer` cannot give exactly.
UNKNOWN = _Unknown()
_OPAQUE = frozenset(KIND_CODE[k] for k in ("branch", "jump", "call", "return", "stack", "other"))


def _slice(t, bus):
    """``(needed op indices, {destination: defining op})`` of ``t``'s value outputs: the
    final A/X/Y/SP writes and the last non-bus store (``"M"``), not through addresses."""
    reg_def, uniq_def, deps, outs = {}, {}, [], {}
    for i, (mn, dst, ins) in enumerate(t.ops):
        if i in bus:
            deps.append(())
            continue
        use = ins[1:] if mn in ("LOAD", "STORE") else ins
        deps.append(
            tuple((uniq_def if vn[0] == "u" else reg_def).get(vn[1]) for vn in use if vn[0] != "c")
        )
        if mn == "STORE":
            outs["M"] = i
        elif dst[0] == "r":
            reg_def[dst[1]] = i
            if 1 << dst[1] & _REGS:
                outs[_REG[dst[1]]] = i
        else:
            uniq_def[dst[1]] = i
    need, todo = set(), list(outs.values())
    while todo:
        i = todo.pop()
        if i is not None and i not in need:
            need.add(i)
            todo.extend(deps[i])
    return need, outs


def transfer(op, a=None, x=None, y=None, m=None, c=None, sp=None, operand=None):
    """``{destination: value-set mask or UNKNOWN}`` of ``op`` (``"A"``, ``"X"``, ``"Y"``,
    ``"SP"``, ``"M"`` the memory operand) over value sets as :func:`as_mask` (None: any), D
    clear; ``m`` is the operand's value (the immediate in ``imm`` mode). Control and stack
    opcodes give UNKNOWN; a destination is UNKNOWN where its pcode joins dependent sets."""
    if KIND[op] in _OPAQUE:
        return UNKNOWN
    t = template(op)
    need, outs = _slice(t, bus_only(t.ops))
    imm = OPCODES[op][1] == "imm"
    if imm and m is None and operand is not None:
        m = {operand & 0xFF}
    leaf = {n: as_mask(v) for n, v in (("A", a), ("X", x), ("Y", y), ("SP", sp), ("C", c))}
    leaf["M"], leaf["D"] = as_mask(m), 1
    if not all(leaf.values()):
        raise ValueError("empty value set")
    regs = {i: (leaf.get(n, 3), frozenset((n,)), True) for i, n in _REG.items()}

    def const(at, size):
        offs, fn = t.prov[at]
        if imm and fn == "id" and offs == (1,):
            return leaf["M"], frozenset(("M",)), True
        if operand is None:
            return (1 << (1 << (8 * size))) - 1, frozenset(("OPERAND",)), True
        b = (operand & 0xFF, operand >> 8 & 0xFF)[offs[0] - 1]
        return 1 << {"word": operand & 0xFFFF, "id": b, "hi1": (b + 1) & 0xFF}[fn], (), True

    uniq, got = {}, {}
    for i, (mn, dst, ins) in enumerate(t.ops):
        if i not in need:
            continue
        vals = [
            (
                const((i, j), vn[2])
                if (i, j) in t.prov
                else (
                    (1 << vn[1], frozenset(), True)
                    if vn[0] == "c"
                    else (regs if vn[0] == "r" else uniq)[vn[1]]
                )
            )
            for j, vn in enumerate(ins)
            if j or mn not in ("LOAD", "STORE")
        ]
        if mn == "STORE":
            got["M"] = vals[0]
            continue
        if mn == "LOAD":
            res = (
                (leaf["M"], frozenset(("M",)), True)
                if i in _WALKS[op].loads
                else (TOP8, frozenset((i,)), True)
            )
        else:
            wide = [v for v in vals if not is_single(v[0])]
            joint = any(p[1] & q[1] for k, p in enumerate(wide) for q in wide[k + 1 :])
            mask = apply(mn, dst[2], [vn[2] for vn in ins], [v[0] for v in vals])
            single = is_single(mask)
            res = (
                mask,
                frozenset() if single else frozenset().union(*(v[1] for v in vals)),
                single or (all(v[2] for v in vals) and not joint),
            )
        (regs if dst[0] == "r" else uniq)[dst[1]] = res
        got.update({k: res for k, d in outs.items() if d == i})
    return {k: v[0] if v[2] else UNKNOWN for k, v in got.items()}
