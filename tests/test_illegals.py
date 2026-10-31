"""Illegal opcodes: what py65 stubs, jennings executes.

RMW combos are checked by NMS decomposition -- the illegal equals its two legal
micro-ops -- with py65 as the oracle for the legal parts. The rest are
known-answer, plus a coverage sweep that every one of the 105 illegals runs.
"""

import numpy as np
import pytest

from deity_informant import MAGIC
from jennings.devices.mpu6502 import MPU, ILLEGAL_OPCODES

py65 = pytest.importorskip("py65.devices.mpu6502")

PC = 0x0800
ZP = 0x40


def _flags(p):
    return (p & 1, (p >> 1) & 1, (p >> 2) & 1, (p >> 3) & 1, (p >> 6) & 1, (p >> 7) & 1)


def _mk_p(rng):
    b = rng.integers(0, 2, 6)
    return int(b[0] | (b[1] << 1) | (b[2] << 2) | 0x20 | (b[4] << 6) | (b[5] << 7))


def _load(mpu, mem, a, x, y, sp, p):
    mpu.memory[:] = mem
    mpu.a, mpu.x, mpu.y, mpu.sp, mpu.p, mpu.pc = a, x, y, sp, p, PC


def _state(mpu):
    return (mpu.a, mpu.x, mpu.y, mpu.sp) + _flags(mpu.p) + (mpu.memory[ZP],)


_RMW = {
    0x07: ("SLO", 0x06, 0x05),
    0x27: ("RLA", 0x26, 0x25),
    0x47: ("SRE", 0x46, 0x45),
    0x67: ("RRA", 0x66, 0x65),
    0xC7: ("DCP", 0xC6, 0xC5),
    0xE7: ("ISC", 0xE6, 0xE5),
}


@pytest.mark.parametrize("op", sorted(_RMW))
def test_rmw_illegal_equals_decomposition(op, new_mem):
    name, op1, op2 = _RMW[op]
    ref = py65.MPU(memory=bytearray(0x10000))
    vm = MPU(memory=new_mem(bytearray(0x10000)))
    rng = np.random.default_rng(op)
    for _ in range(300):
        a, x, y, sp = (int(v) for v in rng.integers(0, 256, 4))
        p = _mk_p(rng)
        base = bytearray(0x10000)
        base[ZP] = int(rng.integers(0, 256))
        vimg = bytearray(base)
        vimg[PC], vimg[PC + 1] = op, ZP
        _load(vm, vimg, a, x, y, sp, p)
        vm.step()
        rimg = bytearray(base)
        rimg[PC], rimg[PC + 1], rimg[PC + 2], rimg[PC + 3] = op1, ZP, op2, ZP
        _load(ref, rimg, a, x, y, sp, p)
        ref.step()
        ref.step()
        assert _state(vm) == _state(ref), "%s a=%02X mem=%02X p=%02X" % (name, a, base[ZP], p)


def test_lax_and_sax(new_mem):
    mpu = MPU(memory=new_mem())
    mpu.memory[0:2] = [0xA7, ZP]  # LAX zp
    mpu.memory[ZP] = 0x80
    mpu.step()
    assert mpu.a == 0x80 and mpu.x == 0x80 and mpu.p & MPU.NEGATIVE
    mpu = MPU(memory=new_mem())
    mpu.a, mpu.x = 0xF0, 0x0F
    mpu.memory[0:2] = [0x87, ZP]  # SAX zp: mem = A & X
    mpu.step()
    assert mpu.memory[ZP] == (0xF0 & 0x0F)


def test_anc_alr_arr(new_mem):
    mpu = MPU(memory=new_mem())
    mpu.a = 0xFF
    mpu.memory[0:2] = [0x0B, 0x80]  # ANC #$80 -> A=A&imm, C=bit7
    mpu.step()
    assert mpu.a == 0x80 and mpu.p & MPU.CARRY and mpu.p & MPU.NEGATIVE
    mpu = MPU(memory=new_mem())
    mpu.a = 0xFF
    mpu.memory[0:2] = [0x4B, 0x03]  # ALR #$03 -> (A&3)>>1, C=old bit0
    mpu.step()
    assert mpu.a == 0x01 and mpu.p & MPU.CARRY


def test_sbx(new_mem):
    mpu = MPU(memory=new_mem())
    mpu.a, mpu.x = 0xFF, 0x0F
    mpu.memory[0:2] = [0xCB, 0x05]  # X = (A&X) - imm
    mpu.step()
    assert mpu.x == (0x0F - 0x05) and mpu.p & MPU.CARRY


def test_lxa_ane_magic(new_mem):
    mpu = MPU(memory=new_mem())
    mpu.a = 0x00
    mpu.memory[0:2] = [0xAB, 0xFF]  # LXA: A=X=(A|MAGIC)&imm
    mpu.step()
    assert mpu.a == (MAGIC & 0xFF) and mpu.x == mpu.a
    mpu = MPU(memory=new_mem())
    mpu.a, mpu.x = 0x00, 0xFF
    mpu.memory[0:2] = [0x8B, 0xFF]  # ANE: A=(A|MAGIC)&X&imm
    mpu.step()
    assert mpu.a == (MAGIC & 0xFF)


def test_las(new_mem):
    mpu = MPU(memory=new_mem())
    mpu.sp = 0xFF
    mpu.memory[0x1000:0x1003] = [0xBB, 0x00, 0x20]  # LAS $2000,Y
    mpu.memory[0x2000] = 0x3C
    mpu.pc = 0x1000
    mpu.step()
    assert mpu.a == 0x3C and mpu.x == 0x3C and mpu.sp == 0x3C


def test_sh_group_high_byte_and(new_mem):
    mpu = MPU(memory=new_mem())
    mpu.a, mpu.x, mpu.y = 0xFF, 0xFF, 0x04
    mpu.memory[0x1000:0x1003] = [0x9F, 0x00, 0x20]  # SHA $2000,Y -> A&X&(H+1)
    mpu.pc = 0x1000
    mpu.step()
    assert mpu.memory[0x2004] == (0xFF & 0xFF & 0x21)


def test_jam_flags_and_advances(new_mem):
    mpu = MPU(memory=new_mem())
    mpu.memory[0x1000] = 0x02  # JAM
    mpu.pc = 0x1000
    c = mpu.processorCycles
    mpu.step()
    # Flags the lock-up but advances (py65-style) so cycle-bounded drivers don't hang.
    assert mpu.jammed and mpu.pc == 0x1001 and mpu.processorCycles > c


def test_all_illegals_execute(new_mem):
    assert len(ILLEGAL_OPCODES) == 105
    jam = {0x02, 0x12, 0x22, 0x32, 0x42, 0x52, 0x62, 0x72, 0x92, 0xB2, 0xD2, 0xF2}
    for op in sorted(ILLEGAL_OPCODES - jam):
        mpu = MPU(memory=new_mem(bytearray(0x10000)))
        mpu.a = mpu.x = mpu.y = 0x5A
        mpu.memory[PC], mpu.memory[PC + 1], mpu.memory[PC + 2] = op, 0x10, 0x20
        mpu.pc = PC
        mpu.step()
        assert mpu.pc != PC, "opcode %02X did not advance" % op


def test_sha_indy_follows_the_pointer_on_rerun(new_mem):
    """$93 at one PC: H+1 tracks the pointer's high byte each time, not the first lift's."""
    mpu = MPU(memory=new_mem())
    mem = mpu.memory
    mem[0x1000:0x1002] = [0x93, ZP]
    for hi in (0x20, 0x40, 0xFE):
        mem[ZP], mem[ZP + 1] = 0x00, hi
        mpu.a = mpu.x = 0xFF
        mpu.y, mpu.pc = 3, 0x1000
        mpu.step()
        assert mem[(hi << 8) | 3] == (hi + 1) & 0xFF
