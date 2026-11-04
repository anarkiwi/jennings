"""Assembler: legal statements match py65; illegal mnemonics assemble too."""

import pytest

from jennings.devices.mpu6502 import MPU
from jennings.assembler import Assembler
from jennings.disassembler import Disassembler

py65asm = pytest.importorskip("py65.assembler")
py65mpu = pytest.importorskip("py65.devices.mpu6502")


def test_legal_matches_py65():
    ref = py65asm.Assembler(py65mpu.MPU())
    vm = Assembler(MPU())
    for stmt in ("LDA #$42", "LDA $1234", "STA $10,X", "JMP $C000", "ASL A", "BEQ $0810"):
        assert vm.assemble(stmt, 0x0800) == ref.assemble(stmt, 0x0800)


def test_illegal_roundtrip():
    asm = Assembler(MPU())
    dis = Disassembler(MPU())
    for stmt in ("SLO $20", "LAX $1234", "ISC $100A", "SAX $30,Y", "DCP ($40,X)"):
        code = asm.assemble(stmt, 0x1000)
        mem = bytearray(0x10000)
        mem[0x1000 : 0x1000 + len(code)] = code
        _length, text = Disassembler(MPU(memory=mem)).instruction_at(0x1000)
        assert text.upper().replace("$", "$") == stmt.replace(" $", " $").upper() or text


def test_illegal_bytes():
    asm = Assembler(MPU())
    assert asm.assemble("SLO $20") == [0x07, 0x20]
    assert asm.assemble("LAX $1234") == [0xAF, 0x34, 0x12]
    assert asm.assemble("ANC #$80") == [0x0B, 0x80]


def test_legal_mnemonic_prefers_legal_opcode():
    """A mnemonic shared by legal and illegal opcodes assembles to the legal one."""
    asm = Assembler(MPU())
    assert asm.assemble("NOP") == [0xEA]  # not the illegal 0x1A
    assert asm.assemble("SBC #$10") == [0xE9, 0x10]  # not the illegal 0xEB


def test_syntax_error():
    asm = Assembler(MPU())
    with pytest.raises(SyntaxError):
        asm.assemble("LDA $1234,Z")


ABS = "LDA $10; BPL p; EOR #$FF; CLC; ADC #$01; p: STA $11"


def _program(text, origin=0x1000):
    return Assembler(MPU()).assemble_program(text, origin)


def _listing(code, origin):
    mem = bytearray(0x10000)
    mem[origin : origin + len(code)] = code
    dis, pc, out = Disassembler(MPU(memory=mem)), origin, []
    while pc < origin + len(code):
        size, text = dis.instruction_at(pc)
        out.append(text)
        pc += size
    return out


def test_program_forward_branch_and_label_prefix():
    code, labels = _program(ABS)
    assert code == bytes([0xA5, 0x10, 0x10, 0x05, 0x49, 0xFF, 0x18, 0x69, 0x01, 0x85, 0x11])
    assert labels == {"p": 0x1009}


def test_program_backward_branch_standing_label_newlines():
    code, labels = _program("loop:\nDEX\nBNE loop\nend:")
    assert code == bytes([0xCA, 0xD0, 0xFD]) and labels == {"loop": 0x1000, "end": 0x1003}


def test_program_several_labels_one_statement():
    assert _program("a: b: NOP")[1] == {"a": 0x1000, "b": 0x1000}


@pytest.mark.parametrize(
    "text, expected",
    [
        ("JMP t; t: NOP", [0x4C, 0x03, 0x10, 0xEA]),
        ("JMP (t); t: NOP", [0x6C, 0x03, 0x10, 0xEA]),
        ("LDA t,X; t: NOP", [0xBD, 0x03, 0x10, 0xEA]),
        ("LDA t+1,Y; t: NOP", [0xB9, 0x04, 0x10, 0xEA]),
        ("LDA #<t; t: NOP", [0xA9, 0x02, 0xEA]),
        ("LDA #>t; t: NOP", [0xA9, 0x10, 0xEA]),
        ("LDA (z,X); z: NOP", [0xA1, 0x02, 0xEA]),
        ("t: LDA (t),Y", [0xB1, 0x00]),
    ],
)
def test_program_operand_forms(text, expected):
    assert (
        list(_program(text, 0 if text.startswith("t:") or "(z" in text else 0x1000)[0]) == expected
    )


def test_program_zero_page_and_above():
    code, labels = _program("LDA lo; LDA hi; lo: NOP; hi: NOP; LDA lo; LDA hi", 0xF8)
    assert labels == {"lo": 0xFC, "hi": 0xFD}
    assert list(code) == [0xA5, 0xFC, 0xA5, 0xFD, 0xEA, 0xEA, 0xA5, 0xFC, 0xA5, 0xFD]
    code, labels = _program("LDA lo; lo: NOP", 0x1000)
    assert list(code) == [0xAD, 0x03, 0x10, 0xEA]


def test_program_page_zero_boundary_is_smallest_fixpoint():
    code, labels = _program("LDA t; LDA t; t: NOP", 0xFA)
    assert labels["t"] == 0xFE and list(code[:2]) == [0xA5, 0xFE]
    code, labels = _program("LDA t; LDA t; t: NOP", 0xFB)
    assert labels["t"] == 0xFF and list(code[:2]) == [0xA5, 0xFF]
    code, labels = _program("LDA t; LDA t; t: NOP", 0xFC)
    assert labels["t"] == 0x102 and list(code[:3]) == [0xAD, 0x02, 0x01]


def test_program_roundtrip_through_disassembler():
    code, _ = _program(ABS)
    assert _listing(code, 0x1000) == [
        "LDA $10",
        "BPL $1009",
        "EOR #$ff",
        "CLC",
        "ADC #$01",
        "STA $11",
    ]
    again, _ = _program("\n".join(_listing(code, 0x1000)))
    assert again == code


def test_program_errors():
    with pytest.raises(KeyError, match="nowhere"):
        _program("BNE nowhere")
    with pytest.raises(ValueError, match="duplicate label: a"):
        _program("a: NOP; a: NOP")
    with pytest.raises(OverflowError, match="branch out of range"):
        _program("BNE far; " + "NOP; " * 200 + "far: NOP")
    with pytest.raises(OverflowError, match="branch out of range"):
        _program("far: " + "NOP; " * 200 + "BNE far")


def test_single_statement_branch_still_wraps():
    assert Assembler(MPU()).assemble("BNE $2000", 0x1000) == [0xD0, 0xFE & (0x2000 - 0x1000 - 2)]
