"""jennings.memo: the memoising orbit against the brute-force orbit it must equal."""

import random

import numpy as np
import pytest

from helpers import image
from jennings import memo
from jennings.devices.mpu6502 import REG, NativeMPU, Remap
from jennings.memory import NATIVE, NativeMemory

pytestmark = pytest.mark.skipif(not NATIVE, reason="native core not built")

PLAY = 0x1000
MAX_CYCLES = 100_000


def machine(ram):
    return NativeMPU(memory=NativeMemory(bytes(ram)))


def agree(ram, limit, max_cycles=MAX_CYCLES, **kw):
    """The brute-force orbit of ``ram`` as ``(end, states, accesses, edges)`` and the memo's
    stats per grain, after asserting each grain's memo orbit equals it and leaves the memory
    and registers it leaves."""
    brute = machine(ram)
    end, states, acc, edges = brute.orbit(PLAY, 0, max_cycles, limit, **kw)
    want = end, states, acc.tolist(), edges.tolist()
    stats = {}
    for grain in memo.GRAINS:
        mpu = machine(ram)
        got = memo.orbit(mpu, PLAY, 0, max_cycles, limit, grain=grain, **kw)
        assert (got[0], got[1], got[2].tolist(), got[3].tolist()) == want, grain
        assert bytes(mpu.memory) == bytes(brute.memory), grain
        regs = [(m.a, m.x, m.y, m.sp, m.p, m.pc, m.processorCycles) for m in (mpu, brute)]
        assert regs[0] == regs[1], grain
        assert got[4]["dropped"] == 0, grain
        stats[grain] = got[4]
    return want, stats


COUNTERS = """
    inc c3
    lda c3
    cmp #3
    bne +
    lda #0
    sta c3
+   dec c5
    bpl +
    lda #4
    sta c5
+   rts
c3 !byte 0
c5 !byte 4
"""


def test_independent_counters_close_at_their_common_period():
    ram, _ = image(COUNTERS)
    (end, states, _, _), stats = agree(ram, 1000)
    # The first call's return address is state too, so the cycle is entered after one call.
    assert (end, states) == ("closed", 16)
    assert stats["call"]["hits"] + stats["call"]["misses"] == stats["call"]["calls"]
    assert stats["call"]["calls"] == 16 + 1


def test_keep_and_cells_choose_the_state_that_must_repeat():
    ram, s = image(COUNTERS)
    assert agree(ram, 1000, keep=[range(s["c3"], s["c3"] + 1)])[0][:2] == ("closed", 3)
    assert agree(ram, 1000, cells=[s["c5"]])[0][:2] == ("closed", 5)
    assert agree(ram, 1000, cells=[s["c3"], s["c5"], REG + 1])[0][:2] == ("closed", 15)


VOICES = """
    ldx #2
voice
    jsr tick
    dex
    bpl voice
    inc frame
    rts
tick
    dec count,x
    bpl +
    lda period,x
    sta count,x
    inc note,x
    lda note,x
    and #3
    sta note,x
    pha
    jsr emit
    pla
    sta last,x
+   rts
emit
    tay
    lda freq,y
    sta $d400,x
    rts
count  !byte 0, 0, 0
period !byte 6, 10, 12
note   !byte 0, 0, 0
last   !byte 0, 0, 0
frame  !byte 0
freq   !byte 10, 20, 30, 40
"""


def test_voices_sharing_code_realign_long_after_each_repeats():
    ram, _ = image(VOICES)
    (end, states, _, _), stats = agree(ram, 300_000, volatile=[range(0xD000, 0xE000)])
    # Voices repeat every 28, 44 and 52 calls and the frame counter every 256.
    assert (end, states) == ("closed", 256256 + 1)
    assert stats["loop"]["entries"] < stats["call"]["entries"]
    assert stats["sub"]["steps"] * 5 < stats["call"]["steps"]
    assert stats["split"]["hits"] > 3 * stats["split"]["calls"]


def test_volatile_state_is_not_state_but_ends_at_its_load():
    ram, _ = image(VOICES.replace("sta $d400,x", "sta $d400,x\n    lda $d41b"))
    (end, states, acc, _), _ = agree(ram, 1000, volatile=[range(0xD000, 0xE000)])
    assert (end, states) == ("volatile", 1)
    assert any(k & 0x1FFFF == 0xD41B for k in acc)


SELF_MODIFIED = """
    inc operand+1
    lda operand+1
    and #7
    sta operand+1
operand
    lda #0
    sta out
    lda opcode
    eor #$20
    sta opcode
opcode
    inx
    stx seen
    inc slow
    bne +
    inc slow+1
+   rts
out  !byte 0
seen !byte 0
slow !word 0
"""


def test_self_modified_operands_and_opcodes_are_read_as_state():
    ram, s = image(SELF_MODIFIED)
    # INX ($E8) and INY ($C8) alternate, so X steps every other call: 512 calls to repeat.
    (end, states, _, edges), stats = agree(ram, 5000, keep=[range(PLAY, s["slow"])])
    assert (end, states) == ("closed", 512)
    assert s["opcode"] << 16 | (s["opcode"] + 1) in edges
    assert all(st["flushes"] > 0 for st in stats.values())
    assert agree(ram, 300)[0][:2] == ("open", 301)


def test_a_counter_wider_than_the_limit_stays_open():
    ram, _ = image("""
    inc n
    bne +
    inc n+1
    bne +
    inc n+2
+   rts
n !byte 0, 0, 0
""")
    (end, states, _, edges), stats = agree(ram, 700)
    assert (end, states) == ("open", 701)
    assert len(edges) == 5
    # A call that does not carry reads only the low byte, so its unit is replayed.
    call = stats["call"]
    assert call["hits"] + call["misses"] == 700 and call["hits"] > 0
    assert call["roots"] == [(PLAY, call["live"])]


def test_registers_left_by_the_call_before_are_read_as_state():
    ram, s = image("""
    inx
    bne +
    iny
+   stx out
    sty out+1
    php
    pla
    sta out+2
    rts
out !byte 0, 0, 0
""")
    assert agree(ram, 70_000)[0][:2] == ("closed", 65536 + 1)
    assert agree(ram, 1000, keep=[range(s["out"] + 1, s["out"] + 2)])[0][:2] == ("closed", 1)
    assert agree(ram, 1000, cells=[REG + 1])[0][:2] == ("closed", 256)


def test_a_jam_and_a_call_that_does_not_return_end_as_the_orbit_does():
    ram, _ = image("    inc n\n    !byte $02\n    rts\nn !byte 0")
    assert agree(ram, 10)[0][:2] == ("jam", 1)
    ram, _ = image("    inc n\n    lda n\n    cmp #3\n    bne +\n-   jmp -\n+   rts\nn !byte 0")
    (end, states, _, _), _ = agree(ram, 6, max_cycles=500)
    assert (end, states) == ("open", 7)


def test_a_stack_pointer_that_moves_between_calls_empties_the_memo():
    ram, _ = image(
        "    inc n\n    lda n\n    and #3\n    bne +\n    pla\n    pla\n+   rts\nn !byte 0"
    )
    (end, states, _, _), stats = agree(ram, 40)
    assert (end, states) == ("open", 41)
    assert all(st["flushes"] >= 1 for st in stats.values())


def test_remapped_accesses_are_those_the_orbit_takes():
    ram, s = image("""
    ldx idx
    lda table,x
    sta out,x
    inx
    txa
    and #7
    sta idx
    lda (ptr),y
    sta out+8
    rts
idx   !byte 0
table !byte 1, 2, 3, 4, 5, 6, 7, 8
out   !fill 9, 0
""".replace("ptr", "$10"))
    ram[0x10:0x12] = s["table"] & 0xFF, s["table"] >> 8
    rng = np.random.default_rng(3)
    tables = {}
    for pc in (PLAY + 3, PLAY + 6, PLAY + 16):
        t = np.full(0x10000, -1, np.int32)
        at = np.arange(s["table"], s["out"] + 9)
        t[at] = rng.choice([-1, -2, s["table"], s["idx"]], at.size)
        t[[0x10, 0x11]] = -2
        tables[pc] = t
    remap = Remap(tables, [0xB1])
    (end, states, acc, _), _ = agree(ram, 100, remap=remap)
    assert end == "closed" and states > 8
    assert (PLAY + 6) << 17 | 1 << 16 | s["out"] in acc


def _soup(rng):
    """A routine of random instructions over a few cells: counters, indexed tables, stack use,
    subroutines, forward branches and stores into its own operands and opcodes."""
    cells = ["c0", "c1", "c2", "c3"]
    plain = [
        "inc {c}",
        "dec {c}",
        "lda {c}",
        "sta {c}",
        "adc {c}",
        "eor {c}",
        "ldx {c}",
        "ldy {c}",
        "and #{n}",
        "lda t,x",
        "sta t,x",
        "inc t,x",
        "lda t,y",
        "txa",
        "tay",
        "inx",
        "dey",
        "clc",
        "sec",
        "pha\n    lda {c}\n    sta {d}\n    pla",
        "php\n    inc {d}\n    plp",
        "jsr sub{k}",
        "sta imm{k}+1",
        "lda #$e8\n    eor flip\n    sta flip",
        "cpx #{n}",
        "asl {c}",
        "rol {d}",
    ]
    lines, skips = [], 0
    for i in range(rng.randrange(8, 30)):
        if rng.random() < 0.2:
            lines.append(
                f"    {rng.choice(['bne', 'beq', 'bcc', 'bcs', 'bpl', 'bmi'])} skip{skips}"
            )
            lines.append(f"    {rng.choice(plain)}")
            lines.append(f"skip{skips}")
            skips += 1
        else:
            lines.append(f"    {rng.choice(plain)}")
        if i % 7 == 3:
            lines.append("    txa\n    and #3\n    tax\n    tya\n    and #3\n    tay")
    lines += ["imm0", "    lda #1", "flip", "    inx", "    rts"]
    for k in (0, 1):
        lines += [f"sub{k}", f"imm{k + 1}", "    lda #2", "    and #3", "    tax"]
        lines += ["    inc t,x", "    jsr leaf" if k else "    nop", "    rts"]
    lines += ["leaf", "    dec c3", "    rts"]
    lines += [f"{c} !byte {rng.randrange(4)}" for c in cells] + ["t !fill 260, 3"]
    src = "\n".join(lines)
    fill = {
        "c": lambda: rng.choice(cells),
        "d": lambda: rng.choice(cells),
        "n": lambda: str(rng.choice([1, 3, 7])),
        "k": lambda: str(rng.randrange(2)),
    }
    for name, pick in fill.items():
        while "{" + name + "}" in src:
            src = src.replace("{" + name + "}", pick(), 1)
    return src.replace("sta imm2+1", "sta imm0+1")


@pytest.mark.parametrize("seed", range(60))
def test_random_routines_agree_with_the_orbit(seed):
    ram, s = image(_soup(random.Random(seed)))
    kw = [{}, {"keep": [range(s["c0"], s["c0"] + 4)]}, {"cells": [s["c1"], REG + 1, REG + 8]}]
    (end, _, _, _), stats = agree(ram, 400, **kw[seed % 3])
    assert end in ("closed", "open")
    assert stats["loop"]["hits"] + stats["loop"]["misses"] > 0


def test_random_routines_replay_units():
    hits = 0
    for seed in range(20):
        ram, _ = image(_soup(random.Random(seed)))
        mpu = machine(ram)
        hits += memo.orbit(mpu, PLAY, 0, MAX_CYCLES, 400)[4]["hits"]
    assert hits > 1000


def test_crowded_units_are_cut_where_their_reads_part():
    ram, _ = image("""
    inc a
    lda a
    and #63
    sta a
    nop
    inc b
    lda b
    cmp #61
    bne +
    lda #0
    sta b
+   rts
a !byte 0
b !byte 0
""")
    (end, states, _, _), stats = agree(ram, 10_000)
    assert (end, states) == ("closed", 64 * 61 + 1)
    assert stats["loop"]["splits"] == 0 and stats["split"]["splits"] > 0
    assert stats["split"]["live"] < 200 < stats["loop"]["live"]
    assert stats["split"]["fanout"]


def test_memory_with_a_watch_is_refused():
    ram, _ = image(COUNTERS)
    mpu = machine(ram)
    mpu.memory.subscribe_to_read([0x10], lambda addr: None)
    with pytest.raises(ValueError, match="nothing mapped"):
        memo.orbit(mpu, PLAY, 0, MAX_CYCLES, 10)
    with pytest.raises(ValueError):
        mpu._core.memo_orbit(PLAY, 0, MAX_CYCLES, 10, grain=4)
