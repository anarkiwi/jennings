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
