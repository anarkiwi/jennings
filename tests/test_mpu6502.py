"""jennings.MPU vs py65 across every legal opcode: the façade-wiring proof.

Randomise A/X/Y/SP/P (D forced 0 -- jennings, like its deity-informant engine,
does not model BCD) and all 64 KiB, single-step both cores from a fixed PC, and
compare state, memory, next PC, and cycles. Illegals live in test_illegals.
"""

import numpy as np
import pytest

from jennings.devices.mpu6502 import MPU
from jennings.opcodes import OPCODES, MODE_LEN

py65 = pytest.importorskip("py65.devices.mpu6502")

PC = 0x0800
ITERS = 60
NAMES = ("A", "X", "Y", "SP", "C", "Z", "I", "D", "V", "N", "PC")

# py65 charges $CE DEC abs 3; the NMOS 6510 takes 6, like every RMW absolute.
PY65_CYCLE_DEFECT = {0xCE: 3}

RMW_MN = {"ASL", "ROL", "LSR", "ROR", "DEC", "INC", "SLO", "RLA", "SRE", "RRA", "DCP", "ISC"}
RMW_COST = {"zp": 5, "zpx": 6, "abs": 6, "absx": 7, "absy": 7, "indx": 8, "indy": 8}


def _flags(p):
    return (p & 1, (p >> 1) & 1, (p >> 2) & 1, (p >> 3) & 1, (p >> 6) & 1, (p >> 7) & 1)


def _mk_p(rng):
    b = rng.integers(0, 2, 6)
    return int(b[0] | (b[1] << 1) | (b[2] << 2) | 0x20 | (b[4] << 6) | (b[5] << 7))


def _state(mpu):
    return (mpu.a, mpu.x, mpu.y, mpu.sp) + _flags(mpu.p) + (mpu.pc & 0xFFFF,)


def _run(mpu, mem, a, x, y, sp, p):
    mpu.memory[:] = mem
    mpu.a, mpu.x, mpu.y, mpu.sp, mpu.p, mpu.pc = a, x, y, sp, p, PC
    mpu.processorCycles = 0
    mpu.step()
    return _state(mpu), mpu.processorCycles, bytes(mpu.memory)


def _legal_ops():
    ref = py65.MPU(memory=bytearray(0x10000))
    return [
        op
        for op in range(256)
        if getattr(ref.instruct[op], "__name__", "") != "inst_not_implemented"
    ]


def test_legal_opcodes_match_py65(new_mem):
    ref = py65.MPU(memory=bytearray(0x10000))
    vm = MPU(memory=new_mem(bytearray(0x10000)))
    rng = np.random.default_rng(0xC64)
    bad = {}
    for op in _legal_ops():
        for _ in range(ITERS):
            mem = bytearray(bytes(rng.integers(0, 256, 0x10000, dtype=np.uint8)))
            mem[PC] = op
            a, x, y, sp = (int(v) for v in rng.integers(0, 256, 4))
            p = _mk_p(rng)
            mem = bytes(mem)
            try:
                rst, rcyc, rmem = _run(ref, bytearray(mem), a, x, y, sp, p)
            except IndexError:
                continue  # py65 cannot index the top-of-memory wrap; skip both
            vst, vcyc, vmem = _run(vm, bytearray(mem), a, x, y, sp, p)
            rcyc += PY65_CYCLE_DEFECT.get(op, 0)
            if rst == vst and rmem == vmem and rcyc == vcyc:
                continue
            if op not in bad:
                diffs = [
                    "%s ref=%X vm=%X" % (n, rv, vv)
                    for n, rv, vv in zip(NAMES, rst, vst)
                    if rv != vv
                ]
                if rmem != vmem:
                    d = next(i for i in range(len(rmem)) if rmem[i] != vmem[i])
                    diffs.append("mem[%04X] ref=%02X vm=%02X" % (d, rmem[d], vmem[d]))
                if rcyc != vcyc:
                    diffs.append("cyc ref=%d vm=%d" % (rcyc, vcyc))
                bad[op] = "op %02X: %s" % (op, "; ".join(diffs))
    assert not bad, "mismatches:\n" + "\n".join(bad[o] for o in sorted(bad))


def _step_cost(mpu, prog, org=PC):
    """Cycles the MPU charges for a single step of ``prog`` at ``org``."""
    mpu.memory[:] = bytearray(0x10000)
    mpu.memory[org : org + len(prog)] = prog
    mpu.a = mpu.x = mpu.y = 0
    mpu.pc, mpu.processorCycles = org, 0
    mpu.step()
    return mpu.processorCycles


def test_dec_abs_costs_six(new_mem):
    """$CE was 3, inherited from py65; MOS Programming Manual App. B says 6."""
    assert _step_cost(MPU(memory=new_mem(bytearray(0x10000))), [0xCE, 0x34, 0x12]) == 6


def test_rmw_costs_by_addressing_mode(new_mem):
    """Every RMW opcode, legal and illegal, executed: cost is the mode's alone."""
    mpu = MPU(memory=new_mem(bytearray(0x10000)))
    got = {
        "$%02X %s %s"
        % (op, mn, md): (
            _step_cost(mpu, [op, 0x34, 0x12][: MODE_LEN[md]]),
            RMW_COST[md],
        )
        for op, (mn, md) in sorted(OPCODES.items())
        if mn in RMW_MN and md in RMW_COST
    }
    assert len(got) == 66
    assert not {k: v for k, v in got.items() if v[0] != v[1]}


def test_reset_and_registers(new_mem):
    mpu = MPU(memory=new_mem())
    assert mpu.sp == 0xFF and mpu.a == 0 and mpu.pc == 0
    assert mpu.p == (MPU.UNUSED | MPU.BREAK)
    mpu.a, mpu.x, mpu.y, mpu.sp = 0x11, 0x22, 0x33, 0x44
    assert (mpu.a, mpu.x, mpu.y, mpu.sp) == (0x11, 0x22, 0x33, 0x44)
    mpu.reset()
    assert (mpu.a, mpu.x, mpu.y, mpu.sp) == (0, 0, 0, 0xFF)


def test_reset_vector(new_mem):
    mem = bytearray(0x10000)
    mem[0xFFFC], mem[0xFFFD] = 0x00, 0x40
    mpu = MPU(memory=new_mem(mem), pc=None)
    assert mpu.pc == 0x4000


def test_jsr_rts_roundtrip(new_mem):
    mpu = MPU(memory=new_mem())
    mpu.memory[0x1000:0x1003] = [0x20, 0x00, 0x20]
    mpu.memory[0x2000] = 0x60
    mpu.pc = 0x1000
    mpu.step()
    assert mpu.pc == 0x2000 and mpu.sp == 0xFD
    mpu.step()
    assert mpu.pc == 0x1003 and mpu.sp == 0xFF


def test_irq_and_nmi(new_mem):
    mem = bytearray(0x10000)
    mem[0xFFFE], mem[0xFFFF] = 0x00, 0x30
    mem[0xFFFA], mem[0xFFFB] = 0x00, 0x50
    mpu = MPU(memory=new_mem(mem))
    mpu.pc = 0x1234
    mpu.irq()
    assert mpu.pc == 0x3000 and mpu.p & MPU.INTERRUPT
    mpu.irq()
    assert mpu.pc == 0x3000
    mpu.nmi()
    assert mpu.pc == 0x5000


def test_repr(new_mem):
    assert "6510" in repr(MPU(memory=new_mem()))
