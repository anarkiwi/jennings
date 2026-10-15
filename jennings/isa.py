"""ISA-level helpers over the canonical opcode table: classification, lengths,
the binary-mode ALU, and a linear instruction walker (dependency-free)."""

from jennings.opcodes import MEM_MODES, MODE_LEN, OPCODES

__all__ = [
    "READ_MNEMONICS",
    "RMW_MNEMONICS",
    "STORE_MNEMONICS",
    "adc",
    "is_branch",
    "is_memory",
    "is_read",
    "is_rmw",
    "is_store",
    "length",
    "mode",
    "mnemonic",
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


def adc(a, b, carry):
    """6502 binary-mode ADC: ``a + b + carry`` -> ``(byte, carry_out)``."""
    total = (a & 0xFF) + (b & 0xFF) + (carry & 1)
    return total & 0xFF, 1 if total > 0xFF else 0


def sbc(a, b, carry):
    """6502 binary-mode SBC: ``a - b - (1 - carry)`` -> ``(byte, carry_out)``."""
    total = (a & 0xFF) - (b & 0xFF) - (1 - (carry & 1))
    return total & 0xFF, 1 if total >= 0 else 0


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
