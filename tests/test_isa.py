"""ISA-level helpers: classification, lengths, ALU, and the instruction walker."""

from jennings import isa
from jennings.opcodes import OPCODES

LEGAL_STORES = {"STA", "STX", "STY"}
LEGAL_RMW = {"ASL", "LSR", "ROL", "ROR", "INC", "DEC"}
LEGAL_READS = {"LDA", "LDX", "LDY", "CMP", "CPX", "CPY", "BIT", "ADC", "AND", "EOR", "ORA", "SBC"}
CONTROL = (0x20, 0x4C, 0x6C, 0x00, 0x40, 0x60, 0xEA)  # JSR/JMP/BRK/RTI/RTS/NOP
REG_ONLY = (0x0A, 0x4A, 0x2A, 0x6A, 0x18, 0x38, 0xAA)  # acc shifts + flag/transfer ops


def test_lengths_cover_all_opcodes_and_are_legal():
    assert {isa.length(op) for op in range(256)} <= {1, 2, 3}
    assert isa.length(0xA9) == 2  # LDA #imm
    assert isa.length(0x8D) == 3  # STA abs
    assert isa.length(0x60) == 1  # RTS
    assert isa.length(0x4C) == 3  # JMP abs
    assert isa.length(0xEA) == 1  # NOP


def test_branches_are_exactly_the_rel_opcodes():
    branch = {op for op in range(256) if isa.is_branch(op)}
    assert branch == {op for op, (_mn, mode) in OPCODES.items() if mode == "rel"}
    assert branch == {0x10, 0x30, 0x50, 0x70, 0x90, 0xB0, 0xD0, 0xF0}


def test_data_ops_are_memory_and_control_ops_are_not():
    for op in (0xA5, 0xAD, 0x85, 0x8D, 0xC6, 0x06, 0x24):
        assert isa.is_memory(op), hex(op)
    for op in CONTROL:
        assert not isa.is_read(op)
        assert not isa.is_store(op)
        assert not isa.is_rmw(op)
    for op in REG_ONLY:
        assert not isa.is_memory(op), hex(op)  # acc shifts/flag ops are register-only


def test_families_are_mutually_exclusive():
    for op in range(256):
        assert (isa.is_read(op) + isa.is_store(op) + isa.is_rmw(op)) <= 1, hex(op)


def test_store_family_includes_illegal_stores_and_only_writers():
    store_mn = {isa.mnemonic(op) for op in range(256) if isa.is_store(op)}
    assert LEGAL_STORES <= store_mn
    for writer in ("SAX", "SHA", "SHX", "SHY", "TAS"):
        assert writer in store_mn
    assert not isa.is_store(0xA9)  # LDA #imm
    assert not isa.is_store(0x0A)  # ASL A


def test_rmw_requires_a_memory_mode():
    assert {isa.mnemonic(op) for op in range(256) if isa.is_rmw(op)} >= LEGAL_RMW
    for mem in (0x06, 0xC6, 0x0E, 0xEE):  # ASL/DEC zp, ASL abs, INC abs
        assert isa.is_rmw(mem), hex(mem)
    for acc in (0x0A, 0x4A, 0x2A, 0x6A):  # acc shifts
        assert not isa.is_rmw(acc), hex(acc)


def test_read_omits_register_only_and_control_reads():
    assert {isa.mnemonic(op) for op in range(256) if isa.is_read(op)} >= LEGAL_READS
    for mem in (0xA5, 0xAD, 0xC5, 0x24):  # LDA/CMP zp, BIT zp
        assert isa.is_read(mem), hex(mem)
    for op in (0xE9, 0xA9, 0xCA, 0x00):  # SBC #, LDA #, DEX, BRK
        assert not isa.is_read(op), hex(op)


def test_s8_rounds_a_byte_to_signed():
    assert isa.s8(0x00) == 0
    assert isa.s8(0x7F) == 127
    assert isa.s8(0x80) == -128
    assert isa.s8(0xFF) == -1


def test_adc_and_sbc_carry_semantics():
    assert isa.adc(0xFF, 0x01, 0) == (0x00, 1)
    assert isa.adc(0x10, 0x20, 1) == (0x31, 0)
    assert isa.sbc(0x00, 0x01, 1) == (0xFF, 0)
    assert isa.sbc(0x05, 0x03, 1) == (0x02, 1)
    assert isa.sbc(0x05, 0x08, 1) == (0xFD, 0)  # borrow: 5-8 negative clears C
    assert isa.sbc(0x05, 0x03, 0) == (0x01, 1)  # C=0 subtracts 1, 5-3-1=1 stays set


def test_walk_stops_at_opcode_and_respects_budget():
    code = bytes([0xA9, 0x01, 0x8D, 0x00, 0xD4, 0x60])
    assert list(isa.walk(code, 0, {0x60})) == [0, 2, 5]
    assert list(isa.walk(bytes([0xEA] * 10), 0, set(), budget=3)) == [0, 1, 2]
    assert list(isa.walk(code, 2, set())) == [2, 5]  # mid-program start


def test_walk_stops_at_end_of_memory():
    assert list(isa.walk(bytes([0xEA] * 3), 0, set())) == [0, 1, 2]
    assert list(isa.walk(bytes([0xEA] * 3), 1, set())) == [1, 2]
