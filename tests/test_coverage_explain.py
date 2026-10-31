from pathlib import Path

import numpy as np
import pytest

from jennings.assembler import Assembler
from jennings.coverage import FALLTHRU, TAKEN, CoverageMPU, call
from jennings.coverage.cfg import LEN
from jennings.coverage.elf import load_elf
from jennings.coverage.explain import (
    Constant,
    TraceMPU,
    explain,
    format_explanations,
    slice_of,
)
from jennings.coverage.smc import analyze
from jennings.dataflow import flow
from jennings.devices.mpu6502 import MPU

ELF = Path(__file__).parent / "fixtures" / "sidplayer" / "sidplayer.prg.elf"
HANDLERS = ("fx_off", "fx_vibrato", "fx_slide", "fx_pulse")
ORG = 0x0800


def _insn(mem, start, end, op, operand=None):
    """Address of the first instruction ``op`` (with ``operand``) decoded from ``start``."""
    a = start
    while a < end:
        n = int(LEN[mem[a]])
        word = int(mem[a + 1]) | int(mem[a + 2]) << 8 if n == 3 else int(mem[a + 1])
        if mem[a] == op and operand in (None, word):
            return a
        a += n
    raise LookupError(hex(op))


@pytest.fixture(scope="module", name="song0")
def _song0():
    mem, syms = load_elf(ELF)

    def run(mpu):
        mpu.memory = mem.tolist()
        call(mpu, syms["init"], a=0)
        for _ in range(300):
            call(mpu, syms["play"])

    mpu = CoverageMPU()
    run(mpu)
    entries = {syms["init"]: {"a": {0, 1}}, syms["play"]: {}}
    entries.update({syms[h]: {} for h in HANDLERS})
    smc = analyze(mem, entries, cov=mpu.cov)
    return mem, syms, run, mpu.cov, explain(run, mpu.cov, smcmap=smc, symbols=syms)


def test_slide_branch_ends_in_init_constant(song0):
    mem, syms, _, cov, out = song0
    slide = syms["slide"]
    beq = _insn(mem, syms["voice"], syms["vibrato"], 0xBD, slide) + 3
    store = _insn(mem, syms["init_voice"], syms["next_pattern"], 0x9D, slide)
    zero = _insn(mem, syms["init_voice"], store, 0xA9, 0)
    e = next(e for e in out if e.branch == beq)
    assert cov.flags[beq] & (TAKEN | FALLTHRU) == TAKEN == e.direction
    assert Constant(zero, "A") in e.constants and e.writers[slide] == {store}
    assert Constant(syms["voices"], "X") in e.constants and not e.inputs
    assert e.alternatives[slide] == {syms["fx_slide"]}
    text = str(out)
    assert (
        "voice+9: only taken" in text
        and "slide set by init_voice+%d" % (store - syms["init_voice"]) in text
    )
    assert "fx_slide" in text.split("slide also written by")[1].splitlines()[0]


def test_vibrato_depth_slice_ends_in_song0_data(song0):
    mem, syms, run, _, _ = song0
    mpu = TraceMPU()
    run(mpu)
    trace = mpu.trace
    beq = syms["vibrato"] + 3
    depth = syms["vib_depth"] + 1
    i = next(i for i, r in enumerate(trace) if r[0] == beq and trace[i - 1][2] == (depth,))
    inputs, _, pcs, writers = slice_of(trace, i, initial=mpu._vm.initial)
    pat, seq = syms["pat_tab"] + 2 * int(mem[syms["seq1"]]), syms["song_seq"] + 2
    data = [syms["pat2"] + 3, pat, pat + 1, syms["seq1"], seq, seq + 1]
    assert {a: v for a, v in inputs.items() if isinstance(a, int)} == {a: int(mem[a]) for a in data}
    assert inputs["A"] == 0 and mem[syms["pat2"] + 2] == 0x91 and inputs[data[0]] == 0x30
    assert writers[depth] == {syms["fx_vibrato"]} and syms["dispatch"] in pcs
    assert writers[syms["pat_rd"] + 1] == {syms["fetch"] + 3}


def _assemble(statements, **image):
    asm, labels, mem = Assembler(MPU()), {}, np.zeros(0x10000, np.uint8)
    pc = ORG
    for st in statements:
        if st.endswith(":"):
            labels[st[:-1]] = pc
            continue
        code = asm.assemble(st, pc)
        assert code is not None, st
        mem[pc : pc + len(code)] = code
        pc += len(code)
    for k, v in image.items():
        mem[int(k[1:], 16)] = v
    labels["end"] = pc
    return mem, labels


def _trace(mem, end, hook=None, mpu=None):
    mpu = mpu or TraceMPU(memory=mem.tolist(), pc=ORG)
    for _ in range(64):
        if mpu.pc == end:
            return mpu
        if hook:
            hook(mpu)
        mpu.step()
    raise AssertionError("did not reach $%04X" % end)


def _last(mpu, pc):
    return max(i for i, r in enumerate(mpu.trace) if r[0] == pc)


def test_slice_registers_and_flags():
    mem, lb = _assemble(["LDX #$03", "DEX", "TXA", "CMP #$02", "br:", "BEQ $0808"])
    mpu = _trace(mem, lb["end"])
    inputs, consts, pcs, writers = slice_of(mpu.trace, _last(mpu, lb["br"]))
    assert not inputs and not writers
    assert consts == {Constant(ORG, "X")} and pcs == {ORG, ORG + 2, ORG + 3, ORG + 4, lb["br"]}


def test_slice_memory_inputs_and_writers():
    mem, lb = _assemble(
        ["CLC", "LDA $10", "ADC $11", "st:", "STA $20", "LDA $20", "br:", "BNE $080B"],
        _10=3,
        _11=4,
    )
    mpu = _trace(mem, lb["end"])
    inputs, consts, _, writers = slice_of(mpu.trace, _last(mpu, lb["br"]))
    assert inputs == {0x10: 3, 0x11: 4} and consts == {Constant(ORG, "C")}
    assert writers == {0x20: {lb["st"]}}


def test_slice_follows_written_operands_only():
    mem, lb = _assemble(
        ["LDA #$34", "w:", "STA $0806", "ld:", "LDA $1200", "br:", "BEQ $080A"], _1234=0
    )
    mpu = _trace(mem, lb["end"])
    inputs, consts, _, writers = slice_of(mpu.trace, _last(mpu, lb["br"]))
    assert inputs == {0x1234: 0} and writers == {lb["ld"] + 1: {lb["w"]}}
    assert consts == {Constant(ORG, "A")}


def test_slice_driver_and_initial_registers():
    mem, lb = _assemble(["NOP", "TXA", "STY $30", "LDA $30", "br:", "BEQ $0808"])

    def poke(mpu):
        if mpu.pc == ORG + 1:
            mpu.y = 9

    mpu = TraceMPU(memory=mem.tolist(), pc=ORG)
    mpu.x = 7
    _trace(mem, lb["end"], poke, mpu)
    _, consts, _, _ = slice_of(mpu.trace, _last(mpu, lb["br"]))
    assert consts == {Constant(None, "Y")}
    inputs, _, _, _ = slice_of(mpu.trace, _last(mpu, ORG + 1), ("A",), mpu._vm.initial)
    assert inputs == {"X": 7}


def test_slice_through_interrupt_restores_flags():
    mem, lb = _assemble(
        ["LDA #$00", "main:", "NOP", "br:", "BEQ $0805"],
        _0900=0xA9,
        _0901=0x01,
        _0902=0x40,
        _FFFF=0x09,
    )
    mpu = TraceMPU(memory=mem.tolist(), pc=ORG)
    mpu.step()
    mpu.irq()
    mpu.irq()
    _trace(mem, lb["br"] + 2, mpu=mpu)
    inputs, consts, pcs, _ = slice_of(mpu.trace, _last(mpu, lb["br"]))
    assert consts == {Constant(ORG, "Z"), Constant(ORG, "N")} and 0x0900 not in pcs
    assert 0x0902 in pcs and "SP" in inputs


def test_explain_hand_program_text():
    mem, lb = _assemble(["LDA $10", "br:", "BEQ $0805", "NOP"], _10=5)

    def run(mpu):
        mpu.memory = mem.tolist()
        mpu.pc = ORG
        _trace(mem, lb["end"], mpu=mpu)

    mpu = CoverageMPU()
    run(mpu)
    out = explain(run, mpu.cov, symbols={"start": ORG, "val": 0x10}, progress=True)
    assert [(e.branch, e.direction, e.inputs) for e in out] == [(lb["br"], FALLTHRU, {0x10: 5})]
    assert str(out) == "start+2: only fallthrough (record 1)\n  input    val = $05\n"
    assert format_explanations([]) == ""
    assert "$0010" in format_explanations(out)


def test_flow_operands():
    mem = bytearray(0x10000)
    mem[ORG : ORG + 3] = bytes([0xB9, 0x34, 0x12])
    ops = {1: "lo", 2: "hi"}
    assert flow(mem, ORG, [0x1235], (), ops)["A"] == {"Y", "lo", "hi", 0x1235}
    mem[ORG : ORG + 2] = bytes([0xA9, 0x05])
    assert flow(mem, ORG, (), (), ops)["A"] == {"lo"}
    mem[ORG : ORG + 3] = bytes([0x8D, 0x34, 0x12])
    assert flow(mem, ORG, (), [0x1234], ops) == {0x1234: {"A"}}
