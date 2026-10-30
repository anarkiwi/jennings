import subprocess
import sys

import numpy as np
import pytest
from deity_informant import lift
from deity_informant.vm import run_sub

from jennings.assembler import Assembler
from jennings.coverage import (
    EXEC,
    FALLTHRU,
    OPERAND,
    READ,
    TAKEN,
    WRITE,
    CoverageMap,
    CoverageMPU,
    CoverageVM,
)
from jennings.devices.mpu6502 import MPU

ORG = 0x0800


def _load(mpu, statements, org=ORG):
    asm = Assembler(mpu)
    pc = org
    for st in statements:
        code = asm.assemble(st, pc)
        assert code is not None, st
        mpu.memory[pc : pc + len(code)] = code
        pc += len(code)
    return pc


def _mpu(statements, org=ORG, **mem):
    mpu = CoverageMPU(pc=org)
    end = _load(mpu, statements, org)
    for addr, val in mem.items():
        mpu.memory[int(addr[1:], 16)] = val
    return mpu, end


def _run_to(mpu, end, limit=1000):
    for _ in range(limit):
        if mpu.pc == end:
            return mpu
        mpu.step()
    raise AssertionError("did not reach %04X" % end)


BRANCH = ["LDA $10", "BEQ $0806", "LDX #$01", "NOP"]


@pytest.mark.parametrize("value,flag", [(0, TAKEN), (1, FALLTHRU)])
def test_branch_direction(value, flag):
    mpu, _ = _mpu(BRANCH, m10=value)
    _run_to(mpu, 0x0807)
    f = mpu.cov.flags
    assert f[0x0802] == EXEC | flag
    assert bool(f[0x0804] & EXEC) == (flag == FALLTHRU)
    assert f[0x0801] == OPERAND and f[0x0803] == OPERAND
    assert f[0x0010] == READ
    assert not (f[0x0800:0x0807] & (READ | WRITE)).any()


def test_branch_merge_both_directions():
    covs = []
    for value in (0, 1):
        mpu, _ = _mpu(BRANCH, m10=value)
        covs.append(_run_to(mpu, 0x0807).cov)
    total = CoverageMap()
    total |= covs[0]
    total.merge(covs[1])
    assert total.flags[0x0802] == EXEC | TAKEN | FALLTHRU
    assert total.hits[0x0800] == 2 and total.hits[0x0804] == 1


@pytest.mark.parametrize("p,flag", [(MPU.ZERO, TAKEN), (0, FALLTHRU)])
def test_zero_offset_branch(p, flag):
    mpu = CoverageMPU(pc=ORG)
    mpu.memory[ORG : ORG + 2] = [0xF0, 0x00]
    mpu.p = p
    mpu.step()
    assert mpu.pc == ORG + 2
    assert mpu.cov.flags[ORG] == EXEC | flag


def test_loop_hits_and_branch_flags():
    mpu, end = _mpu(["LDX #$03", "DEX", "BNE $0802"])
    _run_to(mpu, end)
    assert mpu.cov.hits[0x0802] == 3 and mpu.cov.hits[0x0800] == 1
    assert mpu.cov.flags[0x0803] == EXEC | TAKEN | FALLTHRU


def test_data_read_write_flags():
    mpu, end = _mpu(
        ["STA $2000", "INC $30", "LDY #$01", "LDA ($40),Y"], m40=0x00, m41=0x30, m30=0x7F
    )
    _run_to(mpu, end)
    f = mpu.cov.flags
    assert f[0x2000] == WRITE
    assert f[0x0030] == READ | WRITE
    assert f[0x0040] == READ and f[0x0041] == READ and f[0x3001] == READ
    assert mpu.a == 0
    code = f[ORG:end]
    assert not (code & (READ | WRITE)).any()
    assert np.count_nonzero(code & EXEC) == 4


def test_pha_pha_rts_dispatch_edge():
    mpu, _ = _mpu(["LDA #$08", "PHA", "LDA #$0F", "PHA", "RTS"])
    mpu.memory[0x0810] = 0xEA
    _run_to(mpu, 0x0810)
    assert mpu.cov.edges == {(0x0806, 0x0810)}
    assert mpu.cov.flags[0x01FF] == WRITE and mpu.cov.flags[0x01FE] == WRITE


def test_jmp_indirect_edge():
    mpu, _ = _mpu(["JMP ($3000)"])
    mpu.memory[0x3000:0x3002] = [0x00, 0x09]
    mpu.step()
    assert mpu.pc == 0x0900
    assert mpu.cov.edges == {(ORG, 0x0900)}
    assert mpu.cov.flags[ORG + 1] == OPERAND and mpu.cov.flags[ORG + 2] == OPERAND
    assert mpu.cov.flags[0x3000] == READ and mpu.cov.flags[0x3001] == READ


def test_jmp_indirect_page_wrap_pointer_read():
    mpu, _ = _mpu(["JMP ($30FF)"])
    mpu.memory[0x30FF] = 0x00
    mpu.memory[0x3000] = 0x09
    mpu.step()
    assert mpu.pc == 0x0900
    assert mpu.cov.flags[0x30FF] == READ and mpu.cov.flags[0x3000] == READ
    assert not mpu.cov.flags[0x3100]


def test_jsr_rts():
    mpu, end = _mpu(["JSR $0900", "NOP"])
    mpu.memory[0x0900] = 0x60
    _run_to(mpu, end)
    assert mpu.cov.edges == {(0x0900, 0x0803)}
    assert mpu.cov.flags[0x0900] == EXEC and mpu.cov.hits[0x0900] == 1


def test_operand_wraps():
    mpu = CoverageMPU(pc=0xFFFF)
    mpu.memory[0xFFFF] = 0x4C
    mpu.step()
    assert mpu.pc == 0
    f = mpu.cov.flags
    assert f[0xFFFF] == EXEC and f[0x0000] == OPERAND and f[0x0001] == OPERAND


def _vectors(mpu):
    mpu.memory[0xFFFE:0x10000] = [0x00, 0x0A]
    mpu.memory[0xFFFA:0xFFFC] = [0x00, 0x0B]
    mpu.memory[0x0A00] = 0x40
    mpu.memory[0x0B00] = 0x40
    mpu.memory[ORG] = 0xEA


def test_irq_taken_and_masked():
    mpu = CoverageMPU(pc=ORG)
    _vectors(mpu)
    mpu.p = MPU.INTERRUPT
    mpu.irq()
    assert mpu.pc == ORG and not mpu.cov.entries
    mpu.p = 0
    mpu.irq()
    assert mpu.pc == 0x0A00
    assert mpu.cov.entries == {(MPU.IRQ, 0x0A00)}
    assert mpu.cov.flags[0xFFFE] == READ and mpu.cov.flags[0xFFFF] == READ
    mpu.step()
    assert mpu.pc == ORG
    assert mpu.cov.edges == {(0x0A00, ORG)}


def test_nmi():
    mpu = CoverageMPU(pc=ORG)
    _vectors(mpu)
    mpu.p = MPU.INTERRUPT
    mpu.nmi()
    mpu.step()
    assert mpu.cov.entries == {(MPU.NMI, 0x0B00)}
    assert mpu.cov.flags[0xFFFA] & READ and mpu.cov.flags[0xFFFB] & READ
    assert mpu.cov.edges == {(0x0B00, ORG)}


def test_brk_edge():
    mpu = CoverageMPU(pc=ORG)
    _vectors(mpu)
    mpu.memory[ORG] = 0x00
    mpu.step()
    assert mpu.cov.edges == {(ORG, 0x0A00)}
    assert mpu.cov.flags[0xFFFE] & READ and mpu.cov.flags[0xFFFF] & READ


def test_jam_and_reset_keeps_map():
    mpu = CoverageMPU(pc=ORG)
    mpu.memory[ORG] = 0x02
    mpu.step()
    assert mpu.jammed
    assert mpu.cov.flags[ORG] == EXEC and mpu.cov.hits[ORG] == 1
    cov = mpu.cov
    mpu.reset()
    assert mpu.cov is cov and cov.hits[ORG] == 1
    fresh = CoverageMap()
    mpu.cov = fresh
    assert mpu._vm.cov is fresh


def test_save_load_roundtrip(tmp_path):
    mpu = CoverageMPU(pc=ORG)
    _vectors(mpu)
    _load(mpu, ["LDA #$08", "PHA", "LDA #$0F", "PHA", "RTS"])
    mpu.memory[0x0810] = 0xEA
    _run_to(mpu, 0x0810)
    mpu.irq()
    path = tmp_path / "cov.npz"
    mpu.cov.save(path)
    loaded = CoverageMap.load(path)
    assert loaded == mpu.cov
    assert loaded.edges == {(0x0806, 0x0810)} and loaded.entries == {(MPU.IRQ, 0x0A00)}
    empty = tmp_path / "empty"
    CoverageMap().save(empty)
    assert CoverageMap.load(tmp_path / "empty.npz") == CoverageMap()
    assert loaded != CoverageMap() and loaded != 0


def test_matches_mpu_trace():
    mem = np.random.default_rng(1).integers(0, 256, 0x10000).tolist()
    ref, cov = MPU(memory=list(mem), pc=ORG), CoverageMPU(memory=list(mem), pc=ORG)
    for _ in range(3000):
        for m in (ref, cov):
            m.step()
            if m.pc & 0x1F == 0:
                m.irq()
        assert (ref.a, ref.x, ref.y, ref.sp, ref.p, ref.pc, ref.processorCycles) == (
            cov.a,
            cov.x,
            cov.y,
            cov.sp,
            cov.p,
            cov.pc,
            cov.processorCycles,
        )
    assert ref.memory == cov.memory
    assert int(cov.cov.hits.sum()) == 3000
    assert cov.cov.entries


def test_run_sub_driver():
    mem = bytearray(0x10000)
    mem[0x1000:0x1005] = bytes([0xA5, 0x10, 0x85, 0x11, 0x60])
    mem[0x10] = 0x42
    vm = CoverageVM(mem)
    run_sub(vm, 0x1000, {}, lift)
    f = vm.cov.flags
    assert vm.mem[0x11] == 0x42 and vm.cur_pc == 0x1004
    assert f[0x10] == READ and f[0x11] == WRITE
    assert [f[a] for a in range(0x1000, 0x1005)] == [EXEC, OPERAND, EXEC, OPERAND, EXEC]
    assert vm.cov.edges == {(0x1004, 0x0002)}


def test_core_does_not_import_numpy():
    code = "import sys, jennings.devices.mpu6502; assert 'numpy' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_coverage_plugin_import_is_light():
    code = "import sys, jennings.coverage.pytest_plugin; assert 'numpy' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
    import jennings.coverage as package  # pylint: disable=import-outside-toplevel

    assert package.CoverageMap is CoverageMap
    with pytest.raises(AttributeError):
        package.missing  # pylint: disable=pointless-statement
