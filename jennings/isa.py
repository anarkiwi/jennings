"""ISA-level helpers over the canonical opcode table: classification, lengths, the
addresses an instruction touches, the NMOS ALU, and a linear instruction walker
(dependency-free)."""

from jennings.opcodes import MEM_MODES, MODE_LEN, OPCODES

__all__ = [
    "READ_MNEMONICS",
    "RMW_MNEMONICS",
    "STORE_MNEMONICS",
    "adc",
    "addresses",
    "is_branch",
    "is_memory",
    "is_read",
    "is_rmw",
    "is_store",
    "length",
    "mode",
    "mnemonic",
    "pointer_bytes",
    "s8",
    "sbc",
    "walk",
]

# Mnemonic families, by how they use the data bus (legal + NMOS illegal).
READ_MNEMONICS = frozenset("ADC AND BIT CMP CPX CPY EOR LAX LAS LDA LDX LDY ORA SBC".split())
STORE_MNEMONICS = frozenset("STA STX STY SAX SHA SHX SHY TAS".split())
RMW_MNEMONICS = frozenset("ASL LSR ROL ROR INC DEC SLO RLA SRE RRA DCP ISC".split())


def mnemonic(opcode):
    """Mnemonic for ``opcode`` (``KeyError`` outside ``0x00..0xFF``)."""
    return OPCODES[opcode][0]


def mode(opcode):
    """Addressing-mode name for ``opcode`` (``KeyError`` outside ``0x00..0xFF``)."""
    return OPCODES[opcode][1]


def length(opcode):
    """Byte length (1/2/3) of ``opcode``; undefined opcodes keep their column length."""
    return MODE_LEN[OPCODES[opcode][1]]


def is_branch(opcode):
    """True for the conditional (rel) branches."""
    return mode(opcode) == "rel"


def is_read(opcode):
    """True if the instruction reads a data byte from memory without writing it back."""
    return mnemonic(opcode) in READ_MNEMONICS and mode(opcode) in MEM_MODES


def is_store(opcode):
    """True if the instruction writes a register into a data memory byte."""
    return mnemonic(opcode) in STORE_MNEMONICS and mode(opcode) in MEM_MODES


def is_rmw(opcode):
    """True if the instruction performs a memory-byte read-modify-write."""
    return mnemonic(opcode) in RMW_MNEMONICS and mode(opcode) in MEM_MODES


def is_memory(opcode):
    """True if the instruction accesses the data bus (read, write, or both)."""
    return is_read(opcode) or is_store(opcode) or is_rmw(opcode)


def s8(value):
    """Reinterpret a byte as a signed 8-bit integer (``-128..127``)."""
    value &= 0xFF
    return value - 0x100 if value & 0x80 else value


def _signed(x):
    return x - 0x100 if x & 0x80 else x


def adc(a, b, carry, decimal=False, flags=False):
    """NMOS 6502 ADC of ``a + b + carry`` -> ``(byte, carry_out)``, or with ``flags``
    ``(byte, N, V, Z, C)``. ``decimal`` is the D flag: BCD as the NMOS part computes it
    (Bruce Clark's "Decimal Mode" tutorial), N and V from the intermediate sum and Z from
    the binary one."""
    a, b, carry = a & 0xFF, b & 0xFF, carry & 1
    total = a + b + carry
    z = int(total & 0xFF == 0)
    if not decimal:
        r = total & 0xFF
        out = r, r >> 7, int(((a ^ r) & (b ^ r) & 0x80) != 0), z, total >> 8
    else:
        al = (a & 0xF) + (b & 0xF) + carry
        if al >= 0xA:
            al = ((al + 6) & 0xF) + 0x10
        s = (a & 0xF0) + (b & 0xF0) + al
        n, v = (s >> 7) & 1, int(not -128 <= _signed(a & 0xF0) + _signed(b & 0xF0) + al <= 127)
        if s >= 0xA0:
            s += 0x60
        out = s & 0xFF, n, v, z, int(s >= 0x100)
    return out if flags else (out[0], out[4])


def sbc(a, b, carry, decimal=False, flags=False):
    """NMOS 6502 SBC of ``a - b - (1 - carry)`` -> ``(byte, carry_out)``, or with ``flags``
    ``(byte, N, V, Z, C)``. With ``decimal`` the byte is the NMOS BCD difference; the flags
    are always the binary subtraction's."""
    a, b, carry = a & 0xFF, b & 0xFF, carry & 1
    diff = a - b - (1 - carry)
    r = diff & 0xFF
    if decimal:
        al = (a & 0xF) - (b & 0xF) + carry - 1
        if al < 0:
            al = ((al - 6) & 0xF) - 0x10
        s = (a & 0xF0) - (b & 0xF0) + al
        if s < 0:
            s -= 0x60
        byte = s & 0xFF
    else:
        byte = r
    c = int(diff >= 0)
    if not flags:
        return byte, c
    return byte, r >> 7, int(((a ^ b) & (a ^ r) & 0x80) != 0), int(r == 0), c


#: Page-1 bytes pushed (> 0) or pulled (< 0) by the stack instructions.
_STACK = {"PHA": 1, "PHP": 1, "JSR": 2, "BRK": 3, "PLA": -1, "PLP": -1, "RTS": -2, "RTI": -3}
_SH = frozenset(("SHA", "SHX", "SHY", "TAS"))
_INDEX = {"zpx": 1, "zpy": 2, "absx": 1, "absy": 2, "indy": 2}


def pointer_bytes(mem, pc, x=0):
    """``(lo, hi)`` addresses of the pointer the instruction at ``pc`` reads, else None:
    ``(zp,X)`` (``x`` its X) and ``(zp),Y`` a page-zero pair wrapping at ``$FF``, ``JMP
    (ind)`` a pair that stays in the pointer's page (``JMP ($xxFF)`` reads ``$xx00``)."""
    md = OPCODES[int(mem[pc])][1]
    b1 = int(mem[(pc + 1) & 0xFFFF])
    if md in ("indx", "indy"):
        p = (b1 + (x if md == "indx" else 0)) & 0xFF
        return p, (p + 1) & 0xFF
    if md == "ind":
        w = b1 | int(mem[(pc + 2) & 0xFFFF]) << 8
        return w, (w & 0xFF00) | ((w + 1) & 0xFF)
    return None


def addresses(mem, pc, x, y, sp, a=None):
    """The addresses the instruction at ``pc`` reads or writes with registers ``x``, ``y``
    and ``sp``, besides its own bytes: ``{"data": [effective address], "ptr": [pointer
    bytes], "stack": [page-1 bytes]}``. ``data`` is empty for modes without one and for
    ``JMP``/``JSR``; page-zero indexing wraps at ``$FF``, absolute at ``$FFFF``. ``ptr`` is
    :func:`pointer_bytes`, or ``BRK``'s vector ``$FFFE``/``$FFFF``. ``stack`` lists pushes
    from ``sp`` down (``JSR`` 2 bytes, ``BRK`` 3) and pulls from ``sp + 1`` up (``RTS`` 2,
    ``RTI`` 3), wrapping in page 1. On a page cross ``SHA``/``SHX``/``SHY``/``TAS`` store to
    the address whose high byte is the stored ``reg & (H + 1)`` (``H`` the base's high byte);
    ``SHA``/``TAS`` then need ``a``."""
    mn, md = OPCODES[int(mem[pc])]
    ptr = pointer_bytes(mem, pc, x)
    data = []
    if md in MEM_MODES and mn not in ("JMP", "JSR"):
        index = (0, x, y)[_INDEX.get(md, 0)]
        b1 = int(mem[(pc + 1) & 0xFFFF])
        if ptr is not None:
            base = int(mem[ptr[0]]) | int(mem[ptr[1]]) << 8
        elif md[:3] == "abs":
            base = b1 | int(mem[(pc + 2) & 0xFFFF]) << 8
        else:
            base = b1
        ea = (base + index) & (0xFF if md[:2] == "zp" else 0xFFFF)
        if mn in _SH and (ea ^ base) & 0xFF00:
            if mn in ("SHA", "TAS") and a is None:
                raise ValueError("%s crossing a page needs A" % mn)
            reg = {"SHX": x, "SHY": y}.get(mn, (a or 0) & x)
            ea = (reg & ((base >> 8) + 1) & 0xFF) << 8 | ea & 0xFF
        data = [ea]
    n = _STACK.get(mn, 0)
    stack = [0x100 | ((sp - k if n > 0 else sp + 1 + k) & 0xFF) for k in range(abs(n))]
    return {
        "data": data,
        "ptr": [0xFFFE, 0xFFFF] if mn == "BRK" else list(ptr or ()),
        "stack": stack,
    }


def walk(mem, pc=0, stop=(), budget=256):
    """Yield instruction start addresses from ``pc`` in order, stepping by length.

    Yields each start address, inclusive of the first whose opcode is in ``stop``
    (after which it stops), for at most ``budget`` instructions, or until ``pc``
    leaves ``mem``.
    """
    for _ in range(budget):
        if not 0 <= pc < len(mem):
            return
        op = mem[pc]
        yield pc
        if op in stop:
            return
        pc += length(op)
