"""jennings.memo: the memoising orbit against the brute-force orbit it must equal."""

import random
import sys
from pathlib import Path

import numpy as np
import pytest

from helpers import COUNTERS, image
from jennings import memo
from jennings.coverage.elf import load_elf
from jennings.devices.mpu6502 import REG, NativeMPU, Remap
from jennings.memory import NATIVE, NativeMemory

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import memo_orbit  # noqa: E402  pylint: disable=wrong-import-position

pytestmark = pytest.mark.skipif(not NATIVE, reason="native core not built")

PLAY = 0x1000
MAX_CYCLES = 100_000
#: The state :mod:`memo_orbit` takes: all RAM but the I/O area, which is volatile.
RAM = {"keep": [range(0xD000), range(0xE000, 0x10000)], "volatile": [range(0xD000, 0xE000)]}


def machine(ram):
    return NativeMPU(memory=NativeMemory(bytes(ram)))


def found(mpu, got):
    """What an orbit of ``mpu`` that returned ``got`` found and left."""
    return (got[0], got[1], got[2].tolist(), got[3].tolist()), memo_orbit.left(mpu)


def brute_force(ram, limit, max_cycles=MAX_CYCLES, pc=PLAY, **kw):
    """What the brute-force orbit of ``ram`` finds and leaves (:func:`found`)."""
    mpu = machine(ram)
    return found(mpu, mpu.orbit(pc, 0, max_cycles, limit, **kw))


def agree(ram, limit, max_cycles=MAX_CYCLES, **kw):
    """The brute-force orbit of ``ram`` as ``(end, states, accesses, edges)`` and the memo's
    stats per grain, after asserting each grain's memo orbit equals it and leaves the memory
    and registers it leaves."""
    want = brute_force(ram, limit, max_cycles, **kw)
    stats = {}
    for grain in memo.GRAINS:
        mpu = machine(ram)
        got = memo.orbit(mpu, PLAY, 0, max_cycles, limit, grain=grain, **kw)
        assert found(mpu, got) == want, grain
        assert got[4]["dropped"] == 0, grain
        stats[grain] = got[4]
    return want[0], stats


def apart(ram, limit, fibres, grains=memo.GRAINS, max_cycles=MAX_CYCLES, pc=PLAY, **kw):
    """The memo's stats per fibre and grain with each of ``fibres`` set apart, after
    asserting each orbit finds and leaves what the brute-force orbit of ``ram`` does."""
    want = brute_force(ram, limit, max_cycles, pc, **kw)
    stats = {}
    for fibre in fibres:
        for grain in grains:
            mpu = machine(ram)
            got = memo.orbit(mpu, pc, 0, max_cycles, limit, grain, list(fibre), **kw)
            assert found(mpu, got)[0][:2] == want[0][:2], (grain, fibre)
            assert found(mpu, got) == want, (grain, fibre)
            stats[tuple(fibre), grain] = got[4]
    return want[0], stats


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


def test_a_slow_counter_set_apart_leaves_a_tape_of_the_voices_period():
    ram, s = image(VOICES)
    frame = [s["frame"]]
    fibres = [frame, frame + [s["note"]], [s["count"]], [0x5000], [s["frame"], 0x5000, 0xD400]]
    (end, states, _, _), stats = apart(ram, 300_000, fibres, ["sub", "split"], **RAM)
    assert (end, states) == ("closed", 256256 + 1)
    for grain in ("sub", "split"):
        # The voices repeat every 4004 calls and the frame counter is all a tape leaves to run.
        tape = stats[tuple(frame), grain]
        assert (tape["tape_period"], tape["tape_lost"]) == (4004, 0), grain
        assert tape["tape_calls"] > 240_000 and tape["tape_passes"] > 60, grain
        assert tape["tape_learned"] <= 256 + 8 and tape["tape_failed"] == 0, grain
        assert tape["hits"] * 8 < stats[(0x5000,), grain]["hits"], grain
        # A cell never written is no fibre, so the state closes before any base repeats.
        assert stats[(0x5000,), grain]["tape_from"] == 0, grain
        assert stats[(s["frame"], 0x5000, 0xD400), grain]["tape_calls"] > 240_000, grain
    # A voice's count set apart leaves a base that takes the frame counter to repeat.
    assert stats[(s["count"],), "split"]["tape_calls"] == 0


def test_no_fibre_is_the_orbit_without_a_tape():
    ram, s = image(VOICES)
    runs = []
    for fibre in (None, [], [s["frame"]]):
        mpu = machine(ram)
        got = memo.orbit(mpu, PLAY, 0, MAX_CYCLES, 20_000, "split", fibre, **RAM)
        runs.append((found(mpu, got), got[4]))
    assert runs[0] == runs[1] and runs[0][0] == runs[2][0]
    assert runs[0][1]["tape_period"] == 0 and runs[2][1]["tape_calls"] > 0
    with pytest.raises(OverflowError):
        memo.orbit(machine(ram), PLAY, 0, MAX_CYCLES, 10, fibre=[0x10000])


@pytest.mark.parametrize("seed", range(40))
def test_random_routines_agree_with_the_orbit_whatever_is_set_apart(seed):
    ram, s = image(_soup(random.Random(seed)))
    cells = [s["c0"], s["c1"], s["c2"], s["c3"]]
    table = list(range(s["t"], s["t"] + 260))
    fibres = [cells[:1], cells[1:3], table[:3], [cells[3], 0x4000], cells + table]
    fibres += [[0x1FE, 0x1FF, s["imm0"] + 1], range(0x10000)]
    kw = [{}, {"keep": [range(s["c0"], s["c0"] + 4)]}, {"cells": [s["c1"], REG + 1, REG + 8]}]
    apart(ram, 400, fibres, **kw[seed % 3])


def test_random_routines_record_tapes_and_give_them_up():
    counts = {"tape_learned": 0, "tape_failed": 0, "tape_lost": 0}
    counts |= {"tape_retaped": 0, "tape_passes": 0}
    for seed in range(40):
        ram, s = image(_soup(random.Random(seed)))
        for fibre in ([s["c0"]], [s["c1"], s["c2"]], [s["c3"], s["t"]]):
            stats = memo.orbit(machine(ram), PLAY, 0, MAX_CYCLES, 400, "split", fibre)[4]
            counts = {k: v + stats[k] for k, v in counts.items()}
    assert all(v > 0 for v in counts.values()), counts


DELAY = """
    jsr slow
    jsr wait
    rts
slow
    inc f
    lda f
    and #7
    sta f
    tax
-   dex
    bpl -
    rts
wait
    inc b
    lda b
    cmp #3
    bne +
    lda #0
    sta b
+   tay
-   dey
    bpl -
    rts
f !byte 0
b !byte 0
"""


@pytest.mark.parametrize("max_cycles", range(100, 126, 2))
def test_a_call_cut_short_by_the_cycle_limit_is_cut_short_under_a_tape(max_cycles):
    ram, s = image(DELAY)
    _, stats = apart(ram, 300, [[s["f"]], [s["b"]]], ["sub"], max_cycles)
    assert stats[(s["f"],), "sub"]["tape_period"] == 3


def test_a_call_past_the_cycle_limit_is_not_run_by_the_tape():
    ram, s = image(DELAY)
    (end, states, _, _), stats = apart(ram, 300, [[s["f"]]], ["sub"], 112)
    # The longest call's last RTS straddles the limit, so it returns and the state closes.
    assert (end, states) == ("closed", 25)
    assert stats[(s["f"],), "sub"]["tape_limited"] == 1
    (end, _, _, _), stats = apart(ram, 300, [[s["f"]]], ["sub"], 108)
    assert end == "open" and stats[(s["f"],), "sub"]["tape_limited"] > 0


def test_a_fibre_under_a_remap_and_with_registers_as_cells():
    ram, s = image("""
    jsr step
    jsr tock
    ldx idx
    lda table,x
    sta out,x
    rts
step
    inc idx
    lda idx
    and #7
    sta idx
    rts
tock
    dec slow
    bpl +
    lda #4
    sta slow
+   rts
idx   !byte 0
slow  !byte 0
table !byte 1, 2, 3, 4, 5, 6, 7, 8
out   !fill 9, 0
""")
    table = np.full(0x10000, -1, np.int32)
    table[np.arange(s["table"], s["table"] + 8)] = s["table"]
    remap = Remap({s["step"] - 7: table}, [])
    cells = list(range(0x100, 0x200)) + list(range(s["idx"], s["out"] + 9))
    cells += [REG + r for r in (0, 1, 2, 3, 8, 9, 10, 11, 13, 14)]
    for kw in ({"remap": remap}, {"cells": cells}, {"cells": cells, "remap": remap}):
        (end, states, _, _), stats = apart(ram, 1000, [[s["slow"]], [s["idx"]]], ["sub"], **kw)
        assert end == "closed" and states >= 40, kw
        assert stats[(s["slow"],), "sub"]["tape_known"] > 0, kw


SIDPLAYER = Path(__file__).parent / "fixtures" / "sidplayer" / "sidplayer.prg.elf"


@pytest.mark.parametrize("song", [0, 1])
def test_a_play_routine_closes_alike_whatever_is_set_apart(song):
    mem, syms = load_elf(SIDPLAYER)
    mpu = machine(mem.tolist())
    assert mpu.trajectory(syms["init"], song, 10 * MAX_CYCLES, 1, [0], None)[0] == "sp"
    ram = bytes(mpu.memory)
    voices = {name: [syms[name] + v for v in range(3)] for name in ("pat_pos", "seq_pos")}
    voices |= {name: [syms[name] + v for v in range(3)] for name in ("dur_left", "vib_phase")}
    fibres = [[syms["tick"]], voices["dur_left"], voices["vib_phase"], voices["seq_pos"]]
    fibres += [voices["seq_pos"] + voices["pat_pos"], [syms["tmp"], syms["frame"]]]
    fibres += [range(syms["speed"], syms["speed"] + 0x60), range(0x10000)]
    (end, states, _, _), stats = apart(ram, 5000, fibres, pc=syms["play"], **RAM)
    assert (end, states) == ("closed", (2065, 977)[song])
    # Its durations set apart, the rest repeats within a few calls at times, but not for long.
    tape = stats[tuple(voices["dur_left"]), "split"]
    assert tape["tape_from"] > 0 and tape["tape_lost"] > 0


def test_the_tool_checks_each_fibre_against_the_orbit_without_one(tmp_path, capsys):
    ram, s = image(VOICES)
    path = tmp_path / "voices.bin"
    path.write_bytes(ram)
    args = ["--pc", "0x1000", "--volatile", "0xd000:0xe000", "--limit", "9000", "--brute"]
    args += ["--fibre", f"{s['frame']:#x}", "--fibre", f"{s['count']:#x},0x5000", str(path)]
    assert memo_orbit.main(args) == 0
    rows = [line.split() for line in capsys.readouterr().out.splitlines()[1:-1]]
    assert [row[1] for row in rows] == ["end", "fibre", "fibre"]
    assert all(row[-2:] == ["same", "True"] for row in rows)
    assert int(rows[1][rows[1].index("tape_calls") + 1]) > 0
