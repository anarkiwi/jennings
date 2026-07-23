"""Disassembler: legal output matches py65; illegals decode instead of ???."""

import pytest

from jennings.devices.mpu6502 import MPU
from jennings.disassembler import Disassembler

py65 = pytest.importorskip("py65.disassembler")
py65mpu = pytest.importorskip("py65.devices.mpu6502")


def test_legal_matches_py65():
    seqs = [
        [0xA9, 0x42],
        [0xAD, 0x34, 0x12],
        [0xBD, 0x00, 0x80],
        [0x20, 0x00, 0x20],
        [0x10, 0x7F],
        [0x0A],
        [0x00],
        [0xA1, 0x20],
        [0x91, 0x30],
    ]
    for seq in seqs:
        rmem = bytearray(0x10000)
        rmem[0x1000 : 0x1000 + len(seq)] = seq
        ref = py65.Disassembler(py65mpu.MPU(memory=rmem))
        vmpu = MPU(memory=bytearray(rmem))
        vm = Disassembler(vmpu)
        assert vm.instruction_at(0x1000) == ref.instruction_at(0x1000)


def test_illegals_decode():
    cases = {
        (0xAF, 0x34, 0x12): "LAX $1234",
        (0x07, 0x20): "SLO $20",
        (0xEF, 0x0A, 0x10): "ISC $100a",
        (0x0B, 0x80): "ANC #$80",
        (0x9F, 0x00, 0x20): "SHA $2000,Y",
        (0x02,): "JAM",
        (0x1A,): "NOP",
    }
    for seq, text in cases.items():
        mem = bytearray(0x10000)
        mem[0x1000 : 0x1000 + len(seq)] = seq
        d = Disassembler(MPU(memory=mem))
        length, out = d.instruction_at(0x1000)
        assert out == text and length == len(seq)
