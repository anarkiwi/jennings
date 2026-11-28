"""jennings.coverage.segments: calls cut into segments, and certificates by their cycles."""

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import test_components as T
from jennings.coverage import certificate as V
from jennings.coverage import components as C
from jennings.coverage import segments as S
from jennings.devices.mpu6502 import REG, Remap, periods_of
from jennings.memory import NATIVE

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import cover_compare  # noqa: E402  pylint: disable=wrong-import-position
import cover_segments  # noqa: E402  pylint: disable=wrong-import-position
import cover_sites  # noqa: E402  pylint: disable=wrong-import-position
import cover_verify  # noqa: E402  pylint: disable=wrong-import-position

pytestmark = pytest.mark.skipif(not NATIVE, reason="native core not built")

PLAY = T.PLAY
X, SP, D = REG + 1, REG + 3, REG + 11

HEAD = """
    ldx #1
loop inc pos,x
    lda pos,x
    cmp len,x
    bne +
    lda #0
    sta pos,x
+   tay
    lda tab,y
    sta $d400,x
"""
TAIL = """
    dex
    bpl loop
out rts
pos !byte 0, 0
len !byte {a}, {b}
peer !byte 1, 0
skew !byte 0, 0
tab !fill 128, 0
"""
#: Two voices, each stepping its own position through its own length.
VOICES = HEAD + TAIL
#: As :data:`VOICES`, each voice also putting out a table byte at the other's position.
PEER = HEAD + """
    ldy peer,x
    lda pos,y
    cmp #200
    beq out
    tay
    lda tab,y
    sta $d402,x
""" + TAIL
#: As :data:`VOICES`, each voice's own ``skew`` turning over where the other's position is odd.
LEANS = HEAD + """
    ldy peer,x
    lda pos,y
    and #1
    beq +
    lda skew,x
    eor #1
    sta skew,x
+
""" + TAIL


def image(src, a=7, b=11):
    return T.image(src.format(a=a, b=b))


def brute(ram, calls):
    acc, edges = T.seen(ram, calls)
    return acc, edges


keys = cover_compare.claimed


def test_voices_of_coprime_periods_are_certified_each_by_its_own_cycle():
    ram, s = image(VOICES, 67, 71)
    cert = V.certify(ram, PLAY, 4000, learn=64)
    assert isinstance(cert, V.Segmented)
    assert cert.cuts == (s["loop"],) and cert.starts == (PLAY, s["loop"], s["loop"])
    assert (cert.lead, cert.free, cert.values) == (0, (), ())
    assert cert.cycles == ((0, 1), (0, 71), (0, 67)) and cert.calls == 71
    assert cert.feed == (s["pos"], s["pos"] + 1)
    assert (cert.preperiod, cert.period) == (0, 67 * 71)
    assert V.verify(ram, cert) == V.Verdict(True, calls=71)
    assert keys(cert) == brute(ram, 67 * 71), "71 calls show what the joint cycle does"
    assert V.from_dict(json.loads(json.dumps(cert.to_dict()))) == cert
    closed = V.closure(ram, cert, 100_000)
    assert closed.ok and closed.calls == 67 * 71 + 1, "one more: the output is first written"


def test_the_joint_certificate_is_kept_where_the_feed_locations_repeat_within_the_limit():
    ram, _ = image(VOICES, 67, 71)
    cert = V.certify(ram, PLAY, 5000, learn=64)
    assert isinstance(cert, V.Certificate) and cert.calls == 67 * 71
    assert V.from_dict(cert.to_dict()) == cert
    ram, _ = image(VOICES)
    short = V.certify(ram, PLAY, 60)
    assert isinstance(short, V.Segmented) and short.calls == 11 and V.verify(ram, short).ok


def test_never_repeats_is_a_proof_only_where_every_entry_hashes_apart():
    ram, s = image(VOICES)
    cuts, free, values, found = S.propose(ram, PLAY, 60)
    assert (cuts, free, values, found.calls) == ((s["loop"],), (), {}, 11)
    pos = (s["pos"], s["pos"] + 1)
    assert S.never_repeats(found, pos, 76)
    assert not S.never_repeats(found, pos, 77), "calls 0 and 77 start alike"
    assert S.never_repeats(found, pos[:1], 6) and not S.never_repeats(found, pos[:1], 7)
    assert not S.never_repeats(found, (s["len"], X), 1), "no state among them: no finding"
    late = S.examine(ram, PLAY, 3 + 11, cuts, lead=3)
    assert late.cycles == ((3, 1), (3, 11), (3, 7)) and late.rows.shape == (15, 2)
    assert S.never_repeats(late, pos, 76) and not S.never_repeats(late, pos, 77)


def test_a_segment_s_feed_is_its_own_held_or_shared():
    ram, s = image(VOICES)
    found = S.examine(ram, PLAY, 22, (s["loop"],))
    parts, pos = found.parts, s["pos"]
    assert parts.own == (frozenset(), frozenset({pos + 1}), frozenset({pos}))
    assert parts.shared == parts.free == parts.deps == (frozenset(),) * 3
    assert parts.groups == (frozenset({0}), frozenset({1}), frozenset({2}))
    assert parts.cells == ((), (pos + 1,), (pos,))
    assert parts.held[0] == {SP: 0xFF}
    assert parts.held[1] == {X: 1, SP: 0xFD, S.BASE: 0xFF}
    assert parts.held[2] == {X: 0, SP: 0xFD, S.BASE: 0xFF, 0x1FE: 1, 0x1FF: 0}
    assert found.cycles == ((0, 1), (0, 11), (0, 7)) and found.closed and found.calls == 11
    assert found.feed == (pos, pos + 1) and found.rows.shape == (23, 2)
    assert found.trace.runs == (22, 22, 22) and found.trace.forks == 0
    whole = S.examine(ram, PLAY, 22, ())
    assert whole.cycles == (None,) and not whole.closed and whole.calls is None
    with pytest.raises(ValueError, match="leading calls"):
        S.cut(ram, PLAY, 4, (), lead=5)


def test_a_voice_putting_out_another_s_state_is_free_of_it():
    ram, s = image(PEER)
    cert = V.certify(ram, PLAY, 60)
    pos = s["pos"]
    assert isinstance(cert, V.Segmented)
    assert cert.free == ((1, pos), (2, pos + 1)), "each reads the other's position"
    assert cert.values == ((pos, tuple(range(7))), (pos + 1, tuple(range(11))))
    assert cert.cycles == ((0, 1), (0, 11), (0, 7)) and cert.calls == 11
    assert V.verify(ram, cert).ok
    assert keys(cert) == brute(ram, 77)
    found = S.examine(ram, PLAY, 11, cert.cuts, free=cert.free, values=dict(cert.values))
    assert found.parts.free == (frozenset(), frozenset({pos}), frozenset({pos + 1}))
    assert found.trace.forks == 11 * (6 + 10) and found.trace.runs == (11, 11 * 7, 11 * 11)


def test_free_locations_are_held_to_their_values_and_to_what_the_calls_do():
    ram, s = image(PEER)
    cert = V.certify(ram, PLAY, 60)
    pos = s["pos"]
    bound = V.verify(ram, dataclasses.replace(cert, free=()))
    assert bound.failed == "repeat", "each voice then depends on the other"
    assert "segment 1 (from $%04X)" % s["loop"] in bound.detail
    fewer = ((pos, tuple(range(6))), cert.values[1])
    got = V.verify(ram, dataclasses.replace(cert, values=fewer))
    assert got.failed == "values" and got.detail == (
        "$%04X holds $06 where a run ends, not one of its values" % pos
    )
    far = ((pos, tuple(range(7)) + (99,)), cert.values[1])
    got = V.verify(ram, dataclasses.replace(cert, values=far))
    assert got.failed == "accesses" and "no call makes" in got.detail
    away = ((pos, tuple(range(7)) + (200,)), cert.values[1])
    got = V.verify(ram, dataclasses.replace(cert, values=away))
    assert got.failed == "segments" and got.detail == (
        "a run of segment 1 in call 0 with its free locations otherwise ends elsewhere"
    )


def test_a_voice_whose_state_leans_on_another_s_depends_on_it():
    ram, s = image(LEANS)
    pos, skew, cuts = s["pos"], s["skew"], (s["loop"],)
    found = S.examine(ram, PLAY, 40, cuts)
    assert found.parts.shared == (frozenset(), frozenset({pos}), frozenset({pos + 1}))
    assert found.parts.deps == (frozenset(), frozenset({2}), frozenset({1}))
    assert found.parts.groups[1] == found.parts.groups[2] == frozenset({1, 2})
    assert found.parts.cells[1] == (pos, pos + 1, skew, skew + 1)
    assert found.cycles[1] is None
    free, values = ((1, pos), (2, pos + 1)), {pos: range(7), pos + 1: range(11)}
    with pytest.raises(S.Refused, match="otherwise leaves \\$%04X otherwise" % (skew + 1)) as e:
        S.examine(ram, PLAY, 40, cuts, free=free, values=values)
    assert e.value.failed == "autonomy"
    got = V.certify(ram, PLAY, 60)
    assert isinstance(got, V.Open) and got.segments.startswith("repeat: the state of segment 0")
    assert got == dataclasses.replace(got, segments=""), "the reason is not compared"
    joint = V.certify(ram, PLAY, 77 * 4)
    assert isinstance(joint, V.Certificate)


def test_a_state_that_does_not_go_on_as_its_first_repeat_has_it_is_refused(monkeypatch):
    ram, s = image(LEANS)
    monkeypatch.setattr(S, "_loose", lambda *_: None)
    free = ((1, s["pos"]), (2, s["pos"] + 1))
    with pytest.raises(S.Refused, match="but not after call") as e:
        S.examine(ram, PLAY, 60, (s["loop"],), free=free)
    assert e.value.failed == "autonomy" and e.value.call is not None


SCRATCH = """
    ldx #1
loop lda tmp
    sta $d401,x
    inc pos,x
    lda pos,x
    and #3
    sta pos,x
    {leave}
    dex
    bpl loop
    rts
pos !byte 0, 0
tmp !byte {tmp}
"""


@pytest.mark.parametrize(
    "leave, tmp, held",
    [
        ("lda #5\n sta tmp", 5, True),
        ("lda #5\n sta tmp", 9, False),
        ("bne +\n lda #5\n sta tmp\n+", 5, True),
        ("bne +\n lda #5\n sta tmp\n+", 9, False),
        ("sta tmp", 5, False),
    ],
    ids=["always", "always, another at first", "at times", "at times, another at first", "varying"],
)
def test_a_shared_scratch_cell_is_held_only_where_it_is_proved_one_value(leave, tmp, held):
    ram, s = T.image(SCRATCH.format(leave=leave, tmp=tmp))
    found = S.examine(ram, PLAY, 16, (s["loop"],))
    cell = s["tmp"]
    if held:
        assert found.parts.held[1][cell] == found.parts.held[2][cell] == 5
        assert found.parts.deps == (frozenset(),) * 3 and found.calls == 4
    elif leave.startswith("lda"):
        assert found.parts.shared == (frozenset(), frozenset({cell}), frozenset())
        assert found.parts.held[2][cell] == 5, "the voice before always leaves it"
    else:
        assert found.parts.shared[1] == found.parts.shared[2] == frozenset({cell})
        assert found.parts.deps == (frozenset(), frozenset({2}), frozenset({1}))
    later = S.examine(ram, PLAY, 24, (s["loop"],), lead=8)
    assert (cell in later.parts.held[1]) == (leave != "sta tmp"), "5 once it has been stored"


STALE = """
    inc p1
    lda p1
    and #1
    beq +
    lda #{first}
    sta tmp
+   nop
v2  lda p1
    and #1
    beq +
    lda #5
    sta tmp
+   nop
v3  lda p1
    and #1
    {skip} +
    lda tmp
    sta $d401
+   {leave}
    rts
p1  !byte 0
tmp !byte {tmp}
"""


def test_a_cell_read_as_one_value_only_as_the_calls_fell_out_is_not_held():
    """The last part reads 5 in every call seen, but only as the parts before it kept step."""
    always = {"first": 9, "skip": "bne", "leave": "lda #5\n sta tmp", "tmp": 5}
    ram, s = T.image(STALE.format(**always))
    cuts, cell = (s["v2"], s["v3"]), s["tmp"]
    found = S.examine(ram, PLAY, 16, cuts)
    assert found.trace.feed[(found.trace.feed[:, :2] == (2, cell)).all(axis=1)][0, 2] == 5
    assert cell in found.parts.shared[2], "the first part leaves 9 at times"
    assert found.parts.deps[2] == frozenset({0, 1}) and cell not in found.parts.held[2]
    at_times = {"first": 5, "skip": "beq", "leave": "nop"}
    ram, s = T.image(STALE.format(tmp=9, **at_times))
    found = S.examine(ram, PLAY, 16, (s["v2"], s["v3"]))
    assert s["tmp"] in found.parts.shared[2], "9 until a part first stores 5"
    ram, s = T.image(STALE.format(tmp=5, **at_times))
    found = S.examine(ram, PLAY, 16, (s["v2"], s["v3"]))
    assert found.parts.held[2][s["tmp"]] == 5 and s["tmp"] not in found.parts.shared[2]


READS_IO = HEAD + "    lda $d41b\n    sta $d402,x\n" + TAIL
READS_BACK = HEAD + "    lda $d400,x\n    sta $d402,x\n" + TAIL


def test_a_segment_reading_a_volatile_address_before_writing_it_is_refused():
    ram, s = image(READS_IO)
    with pytest.raises(S.Refused, match="reads volatile \\$D41B before writing it") as e:
        S.examine(ram, PLAY, 22, (s["loop"],), volatile=((0xD000, 0xE000),))
    assert e.value.failed == "volatile"
    assert S.examine(ram, PLAY, 22, (s["loop"],)).calls == 11, "nothing is volatile unless given"
    got = V.certify(ram, PLAY, 60)
    assert isinstance(got, V.Open)
    assert got.segments == (
        "volatile: segment 0 (from $%04X) reads volatile $D41B before writing it" % PLAY
    )
    cert = V.certify(ram, PLAY, 60, exclude=(C.STACK,))
    assert isinstance(cert, V.Segmented) and cert.volatile == ()
    got = V.verify(ram, dataclasses.replace(cert, volatile=((0xD400, 0xD420),)))
    assert got.failed == "volatile" and not got.ok
    ram, s = image(READS_BACK)
    cert = V.certify(ram, PLAY, 60, exclude=((0x100, 0x200), (0xD000, 0xE000)))
    assert cert.volatile == ((0xD000, 0xE000),)
    assert V.verify(ram, cert) == V.Verdict(True, calls=11, assumed=(0xD400, 0xD401))


WAVERS = """
    ldx n
loop inc pos,x
    dex
    bpl loop
    lda n
    eor #1
    sta n
    rts
n   !byte 0
pos !byte 0, 0
"""


def test_calls_not_cut_alike_are_refused():
    ram, s = T.image(WAVERS)
    with pytest.raises(S.Refused) as e:
        S.examine(ram, PLAY, 8, (s["loop"],))
    assert e.value.failed == "segments" and e.value.call == 1
    assert e.value.detail == (
        "where call 0 returns after 2 segments, call 1 is at $%04X" % s["loop"]
    )
    with pytest.raises(S.Refused, match="where call 1 is at .* after 2 segments, call 2 returns"):
        S.examine(ram, PLAY, 8, (s["loop"],), lead=1)
    many = T.image("    ldx #0\nloop dex\n    bne loop\n    rts\n")
    with pytest.raises(S.Refused, match="more than 256 segments"):
        S.examine(many[0], PLAY, 2, (many[1]["loop"],))
    assert S.examine(ram, PLAY, 600, ()).calls == 512, "uncut, the whole state repeats"
    whole = S.cut(ram, PLAY, 8, ())
    assert s["loop"] not in whole.arrivals and whole.arrivals[PLAY] == 1
    assert S.candidates(whole, ram) == []


ONCE = """
    lda once
    bne +
    inc once
    rts
+""" + HEAD + TAIL + "once !byte 0\n"


def test_calls_before_the_segments_settle_are_left_whole():
    ram, s = image(ONCE)
    cert = V.certify(ram, PLAY, 60)
    assert isinstance(cert, V.Segmented) and cert.lead == 3 and cert.calls == 3 + 11
    assert cert.cycles == ((3, 1), (3, 11), (3, 7))
    assert V.verify(ram, cert).ok and keys(cert) == brute(ram, 78)
    got = V.verify(ram, dataclasses.replace(cert, lead=5))
    assert got.failed == "input", "a preperiod is from the leading calls on"
    cycles = ((0, 1), (1, 11), (1, 7))
    got = V.verify(ram, dataclasses.replace(cert, lead=0, cycles=cycles))
    assert got.failed == "segments" and "call 1 is at $%04X" % s["loop"] in got.detail
    cycles = ((1, 1), (1, 11), (1, 7))
    got = V.verify(ram, dataclasses.replace(cert, lead=1, cycles=cycles, calls=12))
    assert got.ok, "one call is enough"


def test_a_tampered_certificate_fails_the_condition_it_breaks():
    ram, s = image(VOICES)
    cert = V.certify(ram, PLAY, 60)

    def failed(**changes):
        got = V.verify(ram, dataclasses.replace(cert, **changes))
        assert not got.ok
        return got.failed, got.detail

    assert failed(cycles=((0, 1), (0, 10), (0, 7)))[0] == "repeat"
    why = failed(cycles=((0, 1), (1, 11), (0, 7)))
    assert why == (
        "repeat",
        "the state of segment 1 (from $%04X) ($%04X) repeats at the entries of calls 0 and 11"
        % (s["loop"], s["pos"] + 1),
    )
    assert failed(calls=10)[1].endswith("repeats at no entry within the calls")
    assert failed(starts=(PLAY, s["loop"], PLAY))[0] == "segments"
    assert failed(cuts=(), cycles=((0, 77),), starts=(PLAY,))[0] == "repeat"
    assert failed(cuts=())[0] == "segments"
    assert failed(cycles=cert.cycles[:2])[0] == "input"
    assert failed(feed=cert.feed[:1]) == (
        "repeat",
        "$%04X is of a segment's state or of the feed, not both" % (s["pos"] + 1),
    )
    assert failed(edges=cert.edges[1:])[0] == "edges"
    key = next(k for k, at in cert.accesses.items() if len(at) > 2)
    fewer = {**cert.accesses, key: cert.accesses[key][:-1]}
    assert failed(accesses=fewer)[0] == "accesses"
    assert failed(state="0" * 64)[0] == "input"
    assert failed(lead=11)[0] == failed(cycles=((0, 1), (0, 0), (0, 7)))[0] == "input"
    more = dataclasses.replace(cert, edges=cert.edges + ((1, 2),))
    assert V.verify(ram, more) == V.Verdict(True, calls=11, edges=((1, 2),))
    assert V.verify(ram, cert, taint="classes").ok, "a taint is not used"


def test_segments_whose_longest_runs_could_pass_the_cycle_limit_are_refused():
    ram, s = image(VOICES)
    most = sum(S.cut(ram, PLAY, 11, (s["loop"],)).cycles)
    cert = V.certify(ram, PLAY, 60, max_cycles=most)
    assert V.verify(ram, cert).ok
    got = V.verify(ram, dataclasses.replace(cert, max_cycles=most - 1))
    assert got.failed == "cycles", "no call of the 11 has both voices at their longest"
    assert got.detail == (
        "the segments' longest runs take %d cycles together, a call at most %d" % (most, most - 1)
    )
    got = V.verify(ram, dataclasses.replace(cert, max_cycles=20))
    assert got.failed == "replay" and "'cycles'" in got.detail


def test_a_certificate_by_segments_under_a_remap_is_verified_under_it():
    ram, s = image(VOICES)
    table = np.full(0x10000, -1, np.int32)
    table[s["tab"] : s["tab"] + 128] = -2
    remap = Remap({s["loop"] + 16: table})
    rid = V.remap_digest({s["loop"] + 16: table})
    cert = V.certify(ram, PLAY, 60, remap=remap, remap_id=rid)
    assert isinstance(cert, V.Segmented) and cert.remap == rid
    assert V.verify(ram, cert, remap=remap, remap_id=rid).ok
    assert V.verify(ram, cert).failed == "input"


IDLE = """
    inc $02
    lda $02
    sta $d400
    inc lo
    bne +
    inc hi
+   ldx hi
    lda $0400,x
    ldy mod
    lda tab,y
    iny
    cpy #3
    bne +
    ldy #0
+   sty mod
    rts
lo  !byte 0
hi  !byte 0
mod !byte 0
tab !byte 1, 2, 3
"""


def test_no_cover_point_is_reported_right_where_the_first_persistent_cell_feeds_nothing():
    ram, s = T.image(IDLE)
    got = V.certify(ram, PLAY, 2000, learn=600)
    assert isinstance(got, V.Open) and got.feed == (s["lo"], s["hi"], s["mod"])
    assert got.unrepeated == ((s["hi"],),) and got.period == 768
    rows = np.zeros((40, 3), np.uint8)
    rows[:, 1], rows[:, 2] = np.arange(40) % 5, np.arange(40) % 8
    assert periods_of(rows, [[1], [2], [1, 2]]) == [((0, 5), None), ((0, 8), None), (None, None)]
    by_column = np.asfortranarray(rows)
    assert periods_of(by_column, [[2]]) == periods_of(rows, [[2]]) == [((0, 8), None)]


def test_the_tools_cut_list_and_certify_state_dumps(tmp_path, capsys):
    ram, s = image(PEER)
    path = tmp_path / "state.bin"
    path.write_bytes(bytes(ram))
    assert cover_segments.main([str(path), "--pc", "1000", "--limit", "60"]) == 0
    out = capsys.readouterr().out
    assert "candidates $%04X" % s["loop"] in out and "closes at 11" in out
    assert "| 1 | $%04X |" % s["loop"] in out and "$%04X | 1 |" % s["pos"] in out
    cut_at = ["--cut", "%X" % s["loop"], "--calls", "5"]
    assert cover_segments.main([str(path), "--pc", "1000", *cut_at]) == 1
    assert "closes at None" in capsys.readouterr().out
    assert cover_segments.main([str(path), "--pc", "1000", *cut_at, "--lead", "5"]) == 1
    assert "refused: replay" in capsys.readouterr().out
    assert cover_sites.main([str(path), "--pc", "1000", "D402", "--calls", "8"]) == 0
    out = capsys.readouterr().out
    assert "$D402:\n store at $" in out and "STA $d402,X" in out
    assert cover_verify.main([str(path), "--pc", "1000", "--limit", "60", "--save"]) == 0
    assert "| 11 |" in capsys.readouterr().out
    saved = V.from_dict(json.loads((tmp_path / "state.bin.cert.json").read_text()))
    assert isinstance(saved, V.Segmented) and V.verify(ram, saved).ok
    args = [str(path), "--pc", "1000", "--limit", "60", "--segments", "--closure", "500"]
    assert cover_verify.main(args) == 0
    assert "| closed | 78 |" in capsys.readouterr().out
    assert cover_compare.main([str(path), "--pc", "1000", "--limit", "60", "--orbit", "500"]) == 0
    out = capsys.readouterr().out
    assert "certify: Segmented, 11 calls" in out and "the certificates claim the same" in out
    assert "orbit: closed after 78 calls" in out
    assert "segments: the orbit adds 0 accesses and 0 edges, and lacks 0 and 0" in out
    (tmp_path / "leans.bin").write_bytes(bytes(image(LEANS)[0]))
    assert cover_compare.main([str(tmp_path / "leans.bin"), "--pc", "1000", "--limit", "60"]) == 0
    assert "segments: repeat: the state of segment 0" in capsys.readouterr().out
