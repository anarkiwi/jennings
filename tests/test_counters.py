"""jennings.coverage.counters: cells stepped by a constant, and state nothing follows."""

import sys
from pathlib import Path

import numpy as np
import pytest

import test_components as T
from jennings import memo
from jennings.coverage import components as C
from jennings.coverage import counters as K
from jennings.devices.mpu6502 import NativeMPU, periods_of
from jennings.memory import NATIVE, NativeMemory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import memo_orbit  # noqa: E402  pylint: disable=wrong-import-position

pytestmark = pytest.mark.skipif(not NATIVE, reason="native core not built")

PLAY = T.PLAY

STEPPED = """
    inc up
    dec down
    lda by3
    sec
    sbc #3
    sta by3
    clc
    lda by4
    adc #4
    sta by4
    sec
    lda by5
    adc #4
    sta by5
    lda zp
    clc
    sbc #1
    sta zp
    inc twice
    inc twice
    lda up
    sta $d400
    rts
up    !byte 0
down  !byte 0
by3   !byte 0
by4   !byte 1
by5   !byte 0
twice !byte 0
zp = $fb
"""


def measured(ram, cell, calls=600):
    """The period of ``cell`` over ``calls`` calls, by its values."""
    rows = np.ascontiguousarray(C.trajectory(ram, PLAY, [cell], calls))
    return periods_of(rows, [[0]])[0][0][1]


def test_cells_every_store_steps_by_a_constant_are_counters_with_their_periods():
    ram, s = T.image(STEPPED)
    found = {c.cell: c for c in K.recognise(ram, PLAY, 64)}
    want = {"up": (1, 256), "down": (-1, 256), "by3": (-3, 256), "by4": (4, 64), "by5": (5, 256)}
    want["zp"] = (-2, 128)
    assert sorted(found) == sorted({s[n] for n in want} | {s["twice"]})
    for name, (step, period) in want.items():
        got = found[s[name]]
        assert [st for _, st in got.steps] == [step], name
        assert got.period == period == measured(ram, s[name]), name
    twice = found[s["twice"]]
    assert [st for _, st in twice.steps] == [1, 1] and twice.period == 128
    assert twice.period == measured(ram, s["twice"])
    assert found[s["up"]].use == K.READ and found[s["down"]].use == K.UNREAD
    assert [kind for _, kind in found[s["up"]].sites] == [C.VALUE], "the byte put out"
    assert found[s["down"]].sites == found[s["down"]].indexed == ()


USED = """
    inc beat
    lda beat
    cmp #5
    bne +
    inc hits
+   inc i
    ldx i
    lda tab,x
    sta $d400
    lda gate
    and #1
    beq +
    inc gated
+   inc gate
    lda loose
    adc #1
    sta loose
    inc lap
    lda lap
    cmp #3
    bne +
    lda #0
    sta lap
+   ldx #2
-   inc each,x
    dex
    bpl -
    lda gate
    jmp mid
    lda jumped
    clc
mid adc #2
    sta jumped
    rts
beat  !byte 0
hits  !byte 0
i     !byte 0
gate  !byte 0
gated !byte 0
loose !byte 0
lap   !byte 0
each  !fill 3, 0
jumped !byte 0
tab   !fill 256, 0
"""


def test_a_counter_s_use_is_read_off_what_its_value_taints():
    ram, s = T.image(USED)
    found = {c.cell: c for c in K.recognise(ram, PLAY, 600)}
    assert sorted(found) == sorted(s[n] for n in ("beat", "hits", "i", "gate", "gated"))
    assert found[s["beat"]].use == K.COMPARED and found[s["beat"]].period == 256
    assert {kind for _, kind in found[s["beat"]].sites} == {C.CONTROL, C.VALUE}, "and a carry"
    assert found[s["hits"]].use == K.UNREAD and found[s["hits"]].period is None
    assert found[s["gated"]].period is None, "stepped in some calls only"
    index = found[s["i"]]
    assert index.use == K.INDEX and index.period == 256
    ((site, first, step, count),) = index.indexed
    assert (first, step, count) == (s["tab"], 1, 256) and (site, C.LOAD) in index.sites
    assert s["loose"] not in found, "the carry into its ADC is not a constant"
    assert s["lap"] not in found, "one of its stores is of a constant"
    assert s["each"] not in found, "its store site stores to other cells too"
    assert s["jumped"] not in found, "the bytes before its store are a counter's, the flow is not"


def test_a_progression_is_of_equal_steps():
    assert K.progression([5, 8, 11]) == (5, 3, 3) and K.progression([7]) == (7, 0, 1)
    assert K.progression([5, 8, 12]) is None and K.progression([]) is None


SLOW = """
    inc slow
    lda slow
    sta $d400
    inc slower
    bne +
    inc slowest
+   lda slowest
    sta $d401
    ldx pos
    lda tab,x
    inx
    cpx #3
    bne +
    ldx #0
+   stx pos
    rts
slow    !byte 0
slower  !byte 0
slowest !byte 0
pos     !byte 0
tab     !byte 1, 2, 3
"""


def left(ram, orbit):
    mpu = NativeMPU(memory=NativeMemory(bytes(ram)))
    end, states, acc, edges, *_ = orbit(mpu)
    return (end, states, acc.tolist(), edges.tolist()), memo_orbit.left(mpu)


def test_state_nothing_follows_is_set_apart_and_the_orbit_is_the_plain_one():
    ram, s = T.image(SLOW)
    assert K.loose(ram, PLAY, 600) == (s["slow"], s["slowest"])
    assert K.loose(ram, PLAY, 64) == (s["slow"],), "the other is not stored to in 64 calls"
    assert s["slower"] in C.learn(ram, PLAY, 64).feed, "a branch follows it"
    cells = [s[n] for n in ("slow", "slower", "slowest", "pos")]
    ram_state = {
        "keep": [range(0xD000), range(0xE000, 0x10000)],
        "volatile": [range(0xD000, 0xE000)],
    }
    for state in ({"cells": cells}, ram_state):
        for limit in (50_000, 250_000):
            plain = left(ram, lambda m, n=limit, kw=state: m.orbit(PLAY, 0, 1000, n, **kw))
            auto = left(
                ram,
                lambda m, n=limit, kw=state: memo.orbit(m, PLAY, 0, 1000, n, fibre=memo.AUTO, **kw),
            )
            assert auto == plain
    assert plain[0][:2] == ("closed", 3 * 65536 + 1), "one more: the outputs are first written"
    mpu = NativeMPU(memory=NativeMemory(bytes(ram)))
    stats = memo.orbit(mpu, PLAY, 0, 1000, 250_000, fibre=memo.AUTO, **ram_state)[4]
    assert stats["tape_period"] == 768 and stats["tape_passes"] > 0
    given = memo.orbit(
        NativeMPU(memory=NativeMemory(bytes(ram))),
        PLAY,
        0,
        1000,
        250_000,
        fibre=[s["slow"], s["slowest"]],
        **ram_state,
    )[4]
    assert given == stats, "the fibre chosen is the one given by hand"


def test_the_tool_takes_the_fibre_it_is_not_given(tmp_path, capsys):
    ram, _ = T.image(SLOW)
    path = tmp_path / "slow.bin"
    path.write_bytes(bytes(ram))
    args = [str(path), "--pc", "0x1000", "--limit", "250000", "--volatile", "0xd000:0xe000"]
    assert memo_orbit.main(args + ["--fibre", "auto", "--brute"]) == 0
    rows = capsys.readouterr().out.splitlines()
    auto = next(r for r in rows if "fibre auto" in r)
    assert "same True" in auto and "states 196609" in auto and "tape_period 768" in auto
