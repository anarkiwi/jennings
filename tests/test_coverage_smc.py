from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from deity_informant import PcodeVM, lift

from jennings.assembler import Assembler
from jennings.coverage import (
    EXEC,
    FALLTHRU,
    RETURN,
    TAKEN,
    CoverageMap,
    CoverageMPU,
    CoverageVM,
    call,
    vsa,
)
from jennings.coverage.elf import line_starts, load_elf
from jennings.coverage.smc import Writer, analyze
from jennings.devices.mpu6502 import MPU

ELF = Path(__file__).parent / "fixtures" / "sidplayer" / "sidplayer.prg.elf"
ORG = 0x0800
HANDLERS = ("fx_off", "fx_vibrato", "fx_slide", "fx_pulse")


class _Concrete:
    """``vsa.step`` memory over a concrete bytearray (singleton sets only)."""

    def __init__(self, mem):
        self.m = mem

    def byte(self, a):
        return 1 << self.m[a]

    def word(self, lo, hi):
        return 1 << (self.m[lo] | self.m[hi] << 8)

    def load(self, addrs):
        assert vsa.is_single(addrs)
        return 1 << self.m[addrs.bit_length() - 1]

    def store(self, addrs, vals):
        assert vsa.is_single(addrs) and vsa.is_single(vals)
        self.m[addrs.bit_length() - 1] = vals.bit_length() - 1


@pytest.mark.parametrize("op", range(256))
def test_singletons_match_pcode_vm(op):
    rng = np.random.default_rng(op)
    for _ in range(24):
        mem = bytearray(rng.integers(0, 256, 0x10000, dtype=np.uint8).tobytes())
        pc = int(rng.integers(0, 0x10000))
        mem[pc] = op
        reg = [0] * 16
        reg[:4] = rng.integers(0, 256, 4).tolist()
        for i in (8, 9, 10, 11, 13, 14):
            reg[i] = int(rng.integers(0, 2))
        vm = PcodeVM(mem)
        vm.volatile, vm.reg = False, list(reg)
        try:
            nxt = vm.step(pc, {}, lift)
        except RuntimeError:
            nxt = None
        cm = _Concrete(bytearray(mem))
        out = vsa.step(op, pc, [1 << r for r in reg], cm)
        if nxt is None:
            assert out == []
            continue
        assert len(out) == 1
        _, targets, regs = out[0]
        assert targets == 1 << nxt
        assert regs == [1 << r for r in vm.reg]
        assert cm.m == vm.mem


def test_set_ops():
    ten = vsa.from_values(range(10))
    assert vsa.apply("INT_ADD", 1, [1, 1], [ten, 1 << 250]).bit_count() == 10
    assert vsa.bits(vsa.apply("INT_ADD", 1, [1, 1], [ten, 1 << 250])).tolist()[:6] == [
        0,
        1,
        2,
        3,
        250,
        251,
    ]
    assert vsa.apply("INT_LESS", 1, [1, 1], [ten, 1 << 5]) == 3
    assert vsa.apply("INT_SUB", 1, [1, 1], [1 << 3, ten]) == vsa.from_values(
        [(3 - v) & 0xFF for v in range(10)]
    )
    assert vsa.rot(1 << 255, 1) == 1 and vsa.rot(1, 1, 2) == 2
    assert vsa.word(0b110, 0b11) == vsa.from_values([1, 2, 257, 258], 2)
    with pytest.raises(NotImplementedError):
        vsa.apply("INT_MULT", 1, [1, 1], [1, 1])


def _prog(statements, org=ORG, **extra):
    mem = np.zeros(0x10000, np.uint8)
    asm = Assembler(MPU())
    pc = org
    labels = {}
    for st in statements:
        if st.endswith(":"):
            labels[st[:-1]] = pc
            continue
        code = asm.assemble(st, pc)
        assert code is not None, st
        mem[pc : pc + len(code)] = code
        pc += len(code)
    for addr, val in extra.items():
        mem[int(addr[1:], 16)] = val
    return mem, labels


def test_jmp_operand_toggle():
    mem, _ = _prog(
        [
            "CMP #$00",
            "BEQ $080B",
            "LDA #$20",
            "STA $0811",
            "BNE $0810",
            "LDA #$28",
            "STA $0811",
            "JMP $0830",
        ]
    )
    mem[0x0820] = mem[0x0828] = 0x60
    smc = analyze(mem, [ORG])
    assert {b for a, b in smc.edges if a == 0x0810} == {0x0820, 0x0828}
    cell = smc.cells[0x0811]
    assert cell.writers == {0x0806: True, 0x080D: True} and not cell.data_dependent
    assert cell.classes == {"ctrl"} and cell.values == {0x30, 0x20, 0x28}
    assert cell.insns == {(0x0810, 0x4C)} and 0x0812 not in smc.cells
    assert smc.writers[0x0806] == Writer(frozenset({0x0811}), frozenset({0x20}))
    assert smc.branches[0x0802] == TAKEN | FALLTHRU and smc.branches[0x0809] == TAKEN
    assert not smc.unresolved_transfers and not smc.unresolved_stores


def test_bit_jmp_opcode_toggle():
    mem, _ = _prog(
        [
            "AND #$01",
            "BEQ $080B",
            "LDA #$4C",
            "STA $0810",
            "BNE $0810",
            "LDA #$2C",
            "STA $0810",
            "JMP $0811",
            "RTS",
        ]
    )
    smc = analyze(mem, [ORG])
    assert smc.starts[0x0810] == {0x2C, 0x4C} and smc.starts[0x0811] == {0x11}
    cell = smc.cells[0x0810]
    assert cell.classes == {"var"} and cell.values == {0x2C, 0x4C}
    assert cell.writers == {0x0806: True, 0x080D: True} and not cell.data_dependent
    assert {(0x0810, 0x0813), (0x0810, 0x0811), (0x0811, 0x0813), (0x0813, RETURN)} <= smc.edges
    assert not smc.unresolved_opcodes


def test_inc_immediate_counter():
    mem, _ = _prog(["INC $0804", "LDA #$00", "RTS"])
    smc = analyze(mem, [ORG])
    cell = smc.cells[0x0804]
    assert cell.classes == {"imm"} and cell.values == set(range(256)) and cell.data_dependent
    assert cell.writers == {ORG: True}
    assert smc.regs[0x0805]["a"] == set(range(256))
    assert smc.regs[0x0805]["z"] == {0, 1}


def test_data_dependent_code_not_followed():
    mem, _ = _prog(["STA $0807", "STA $0808", "JMP $0900"])
    smc = analyze(mem, [ORG])
    assert smc.unresolved_transfers[0x0806] == set(range(0x10000))
    assert not any(a == 0x0806 for a, _ in smc.edges)
    mem, _ = _prog(["STA $0803", "NOP"])
    smc = analyze(mem, [ORG])
    assert smc.unresolved_opcodes[0x0803] == set(range(256))
    assert 0x0804 not in smc.starts and smc.cells[0x0803].data_dependent


def test_indexed_load_reads_overlay():
    mem, _ = _prog(["LDA #$07", "STA $0901", "LDY $0900,X", "RTS"])
    mem[0x0902] = 9
    smc = analyze(mem, {ORG: {"x": {0, 1, 2}}}, progress=True)
    assert smc.regs[0x0808]["y"] == {0, 7, 9}


def test_unknown_pointer_store_unresolved():
    mem, _ = _prog(["STA $FB", "STX $FC", "STA ($FB),Y", "RTS"])
    smc = analyze(mem, [ORG])
    assert smc.unresolved_stores == [0x0804]
    assert 0x0804 not in smc.writers and not smc.cells
    mem, _ = _prog(["STA $FB", "STX $FC", "JMP ($00FB)"])
    smc = analyze(mem, [ORG])
    assert smc.unresolved_transfers[0x0804] == set(range(0x10000))


def test_singleton_flag_branch():
    mem, _ = _prog(["CLC", "BCC $0805", "SEC", "BCC $0805", "RTS"])
    smc = analyze(mem, {ORG: {"a": {1}, "x": {2}, "y": {3}}})
    assert smc.branches == {0x0801: TAKEN}
    assert 0x0803 not in smc.starts and (0x0801, 0x0805) in smc.edges
    mem, _ = _prog(["SEC", "BCC $0805", "NOP", "NOP", "RTS"])
    assert analyze(mem, [ORG]).branches == {0x0801: FALLTHRU}


def test_brk_rti_and_indirect():
    mem, _ = _prog(["BRK", "NOP", "JMP ($0900)"], **{"_FFFE": 0x00, "_FFFF": 0x0A})
    mem[0x0A00] = 0x40
    mem[0x0900:0x0902] = [0x10, 0x08]
    mem[0x0810] = 0x60
    smc = analyze(mem, [ORG])
    assert {(ORG, 0x0A00), (0x0A00, 0x0802), (0x0802, 0x0810), (0x0810, RETURN)} <= smc.edges
    assert smc.writers[ORG].targets == {0x01FD, 0x01FC, 0x01FB}
    irq, _ = _prog(["INC $10", "RTI"])
    smc = analyze(irq, {ORG: {"irq": True}})
    assert (0x0802, RETURN) in smc.edges and smc.regs[ORG]["i"] == {1}


def test_return_dispatch_unresolved_and_seeded():
    mem, _ = _prog(["LDA #$08", "PHA", "LDA #$0F", "PHA", "RTS", "JSR $0820", "RTS"])
    mem[0x0810] = 0x60
    mem[0x0820:0x0823] = [0x48, 0x68, 0x60]
    smc = analyze(mem, [ORG, 0x0807])
    assert smc.unresolved_transfers == {0x0806: {0x0810}}
    assert 0x0810 not in smc.starts
    assert {(0x0822, 0x080A), (0x080A, RETURN)} <= smc.edges
    cov = CoverageMap()
    cov.flags[0x0810] = EXEC
    smc = analyze(mem, [ORG], cov=cov)
    assert 0x0810 in smc.starts and 0x0810 in smc.unresolved_transfers


class _RecVM(CoverageVM):
    __slots__ = ("trace", "writes", "moves")

    def __init__(self, mem_bytes, cov=None):
        super().__init__(mem_bytes, cov)
        self.trace, self.writes, self.moves = set(), set(), set()

    def _wr(self, addr, val, sz):
        for i in range(sz):
            self.writes.add((self.cur_pc, (addr + i) & 0xFFFF, (val >> (8 * i)) & 0xFF))
        CoverageVM._wr(self, addr, val, sz)

    def step(self, pc, cache, lifter):
        self.trace.add((pc, self.mem[pc], tuple(self.reg)))
        nxt = CoverageVM.step(self, pc, cache, lifter)
        self.moves.add((pc, nxt))
        return nxt


class _RecMPU(CoverageMPU):
    VM = _RecVM


@pytest.fixture(scope="module", name="fixture")
def _fixture():
    mem, syms = load_elf(ELF)
    runs, cov = [], CoverageMap()
    for song in (0, 1):
        mpu = _RecMPU(memory=mem.tolist())
        call(mpu, syms["init"], a=song)
        for _ in range(300):
            call(mpu, syms["play"])
        runs.append(mpu._vm)
        cov |= mpu.cov
    smc = analyze(mem, {syms["init"]: {"a": {0, 1}}, syms["play"]: {}}, cov=cov)
    return mem, syms, smc, runs


def test_fixture_dynamic_within_static(fixture):
    _, _, smc, runs = fixture
    names = {"a": 0, "x": 1, "y": 2, "sp": 3, "c": 8, "z": 9, "i": 10, "d": 11, "v": 13, "n": 14}
    code = {
        (pc + k) & 0xFFFF
        for pc, ops in smc.starts.items()
        for op in ops
        for k in range(vsa.template(op).length)
    }
    edges = smc.edges | {(a, t) for a, ts in smc.unresolved_transfers.items() for t in ts}
    for vm in runs:
        for pc, op, reg in vm.trace:
            assert op in smc.starts[pc]
            for n, i in names.items():
                assert reg[i] in smc.regs[pc][n], (hex(pc), n)
        for pc, addr, val in vm.writes:
            if addr in code:
                cell = smc.cells[addr]
                assert pc in cell.writers and val in cell.values
        assert vm.moves <= edges


def test_fixture_cells(fixture):
    mem, syms, smc, runs = fixture
    observed = defaultdict(set)
    for vm in runs:
        for _, addr, val in vm.writes:
            observed[addr].add(val)
    cases = {
        syms["seq_rd"] + 1: syms["next_pattern"] + 3,
        syms["seq_rd"] + 2: syms["next_pattern"] + 9,
        syms["pat_rd"] + 1: syms["fetch"] + 3,
        syms["pat_rd"] + 2: syms["fetch"] + 9,
        syms["sidw"] + 1: syms["voice"] + 3,
    }
    assert set(smc.cells) == set(cases)
    for addr, writer in cases.items():
        cell = smc.cells[addr]
        assert cell.writers == {writer: True} and cell.classes == {"addr"}
        assert cell.data_dependent and observed[addr] | {int(mem[addr])} <= cell.values
    rts = syms["dispatch"] + 14
    assert {syms[h] for h in HANDLERS} <= smc.unresolved_transfers[rts]
    assert not smc.unresolved_stores and not smc.unresolved_opcodes
    lines = line_starts(ELF, mem)
    assert set(smc.starts) <= set(lines)
