"""jennings.coverage.certificate: a cover claim's certificate, its verifier and closure."""

import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pytest

import test_components as T
from jennings.coverage import certificate as V
from jennings.coverage import components as C
from jennings.devices.mpu6502 import Learner, Remap
from jennings.memory import NATIVE

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import cover_verify  # noqa: E402  pylint: disable=wrong-import-position

pytestmark = pytest.mark.skipif(not NATIVE, reason="native core not built")

PLAY = T.PLAY
SOURCES = {
    "COUNTERS": 15,
    "GATED": 4,
    "SHARED": 4,
    "SMC": 8,
    "SCRATCH": None,
    "TABLE": 4,
    "SWEEP": 768,
    "CARRY": None,
    "GATE": 4,
    "LATE": 768,
    "COMBO": None,
    "SONG": 8,
    "REPLAY": 2,
    "VOICES": None,
}


def certified(src, **kw):
    ram, syms = T.image(src)
    cert = V.certify(ram, PLAY, 5000, **kw)
    assert isinstance(cert, V.Certificate)
    return ram, syms, cert


@pytest.mark.parametrize("name", SOURCES)
def test_a_certificate_of_cover_calls_is_accepted_and_closes(name):
    ram, _, cert = certified(getattr(T, name), learn=64)
    comps = C.learn(ram, PLAY, max(cert.calls, 64))
    calls, comps = C.cover_calls(ram, PLAY, comps, 5000)
    assert cert.calls == calls and cert.feed == tuple(sorted(comps.feed))
    assert SOURCES[name] in (None, calls)
    acc, edges = T.seen(ram, calls)
    assert set(V._access_keys(cert.accesses).tolist()) == acc
    assert set(V._edge_keys(cert.edges).tolist()) == edges
    assert V.Certificate.from_dict(json.loads(json.dumps(cert.to_dict()))) == cert
    by_call = V.verify(ram, cert)
    assert by_call == V.Verdict(True, calls=calls), by_call.detail
    assert V.verify(ram, cert, taint="classes") == by_call
    closed = V.closure(ram, cert, 100_000)
    assert (
        closed.ok and closed.calls >= calls and closed == V.Closure("closed", closed.calls, {}, ())
    )


def test_a_wrong_period_or_preperiod_is_no_repeat():
    ram, s, cert = certified(T.COUNTERS)
    assert (cert.preperiod, cert.period) == (0, 15)
    short = V.verify(ram, dataclasses.replace(cert, period=5))
    assert short.failed == "repeat" and "$%04X differs" % s["c3"] in short.detail
    assert "calls 0 and 5" in short.detail and not short.ok
    late = V.verify(ram, dataclasses.replace(cert, preperiod=1))
    assert late.failed == "repeat" and "already repeat at the entry of call 15" in late.detail
    twice = V.verify(ram, dataclasses.replace(cert, period=30))
    assert twice.failed == "repeat" and "already repeat at the entry of call 15" in twice.detail


TRANSIENT = """
    lda once
    bne +
    inc once
    sta $0400
+   ldx i
    lda tab,x
    inx
    txa
    and #1
    sta i
    rts
once !byte 0
i    !byte 0
tab  !byte 1, 2
"""


def test_a_preperiod_is_the_entry_the_feed_locations_return_to():
    ram, _, cert = certified(TRANSIENT)
    assert (cert.preperiod, cert.period) == (1, 2)
    assert V.verify(ram, cert).ok
    assert V.verify(ram, dataclasses.replace(cert, preperiod=0, period=3)).failed == "repeat"


def test_an_access_or_edge_the_certificate_lacks_fails_and_one_never_made_is_reported():
    ram, s, cert = certified(T.SONG)
    key = next(k for k, at in cert.accesses.items() if len(at) == 8)
    assert key[1] == C.LOAD and cert.accesses[key][0] == s["tab"]
    fewer = {**cert.accesses, key: cert.accesses[key][:-1]}
    got = V.verify(ram, dataclasses.replace(cert, accesses=fewer))
    assert got.failed == "accesses"
    assert got.detail == "the load at $%04X of $%04X is not certified" % (key[0], s["tab"] + 7)
    without = {k: v for k, v in cert.accesses.items() if k[1] != C.STORE}
    got = V.verify(ram, dataclasses.replace(cert, accesses=without))
    assert got.failed == "accesses" and got.detail.startswith("the store at")
    got = V.verify(ram, dataclasses.replace(cert, edges=cert.edges[1:]))
    assert got.failed == "edges"
    assert got.detail == "the edge from $%04X to $%04X is not certified" % cert.edges[0]
    more = {**cert.accesses, key: cert.accesses[key] + (0x3000,), (0x2000, C.STORE): (5,)}
    got = V.verify(ram, dataclasses.replace(cert, accesses=more, edges=cert.edges + ((1, 2),)))
    assert got.ok and got.edges == ((1, 2),)
    assert got.accesses == {key: (0x3000,), (0x2000, C.STORE): (5,)}


def test_a_feed_set_missing_what_taints_an_address_or_a_branch_is_rejected():
    ram, s, cert = certified(T.TABLE)
    assert cert.feed == (s["idx"],)
    first = V.certify(ram, PLAY, 1, learn=1)
    assert isinstance(first, V.Open) and first.calls == 2
    one = {k: v[:1] for k, v in cert.accesses.items()}
    blind = dataclasses.replace(cert, feed=(), period=1, accesses=one)
    for taint in ("calls", "classes"):
        got = V.verify(ram, blind, taint=taint)
        assert got.failed == "address" and not got.ok
        assert "the address of the load at $%04X is tainted by $%04X" % (PLAY + 3, s["idx"]) in (
            got.detail
        )
    ram, s, cert = certified(T.SWEEP)
    assert cert.feed == (s["pos"], s["sweep"])
    acc, edges = T.seen(ram, 256)
    sweep = dataclasses.replace(
        cert,
        feed=(s["sweep"],),
        period=256,
        accesses=V._accesses(np.array(sorted(acc))),
        edges=V._edges(np.array(sorted(edges))),
    )
    got = V.verify(ram, sweep)
    assert got.failed == "control" and "is tainted by $%04X, outside" % s["pos"] in got.detail
    assert got.detail.startswith("where control goes at $")


DRIVEN = """
    ldx k
    lda tab,x
    sta $0400
    lda k
    clc
    adc j
    and #3
    sta k
    lda j
    eor #1
    sta j
    rts
k   !byte 0
j   !byte 0
tab !byte 1, 2, 3, 4
"""


def test_a_feed_location_depending_on_state_outside_the_feed_set_is_not_autonomous():
    ram, s, cert = certified(DRIVEN)
    assert cert.feed == (s["k"], s["j"]) and V.verify(ram, cert).ok
    acc, edges = T.seen(ram, 1)
    alone = dataclasses.replace(
        cert,
        feed=(s["k"],),
        preperiod=0,
        period=1,
        accesses=V._accesses(np.array(sorted(acc))),
        edges=V._edges(np.array(sorted(edges))),
    )
    for taint in ("calls", "classes"):
        got = V.verify(ram, alone, taint=taint)
        assert got.failed == "autonomy"
        assert got.detail == "$%04X is tainted by $%04X, outside the feed set" % (s["k"], s["j"])


IO_STATE = """
    ldx $d400
    lda tab,x
    sta $0400
    inx
    txa
    and #3
    sta $d400
    rts
tab !byte 1, 2, 3, 4
"""


def test_excluded_state_that_taints_an_address_is_reported_as_assumed():
    ram, _, cert = certified(IO_STATE)
    assert (
        cert.feed == () and cert.calls == 1 and cert.exclude == ((0x100, 0x200), (0xD000, 0xE000))
    )
    got = V.verify(ram, cert)
    assert got.ok and got.assumed == (0xD400,)
    closed = V.closure(ram, cert, 1000)
    assert not closed.ok and closed.end == "closed" and closed.calls == 5
    assert [len(v) for v in closed.accesses.values()] == [3], "the table bytes the claim missed"
    ram, _, cert = certified(IO_STATE, exclude=[(0x100, 0x200)])
    assert cert.feed == (0xD400,) and cert.calls == 4
    assert V.verify(ram, cert) == V.Verdict(True, calls=4)
    assert V.closure(ram, cert, 1000).ok
    with pytest.raises(ValueError, match="step 1"):
        V.certify(ram, PLAY, 100, exclude=[range(0, 16, 2)])


def test_another_state_or_remap_is_refused_as_input():
    ram, s, cert = certified(T.COUNTERS)
    other = bytearray(ram)
    other[0x8000] ^= 1
    got = V.verify(other, cert)
    assert got == V.Verdict(False, "input", "the state is not the one certified")
    m = C.machine(ram)
    m.x = 7
    assert V.verify(m, cert).failed == "input" and V.state_digest(m) != cert.state
    m.x, m.a = 0, 9
    assert V.verify(m, cert).ok, "A is set at each call"
    assert V.verify(ram, dataclasses.replace(cert, period=0)).failed == "input"
    table = np.full(0x10000, -1, np.int32)
    table[s["c5"]] = -2
    tables = {PLAY: table}
    remap, digest = Remap(tables), V.remap_digest(tables)
    assert V.verify(ram, cert, remap=remap, remap_id=digest).detail == (
        "the remap is not the one certified"
    )
    with pytest.raises(ValueError, match="remap_id"):
        V.verify(ram, cert, remap=remap)
    with pytest.raises(ValueError, match="the state is not"):
        V.closure(other, cert, 100)


def test_a_certificate_under_a_remap_is_verified_under_it():
    ram, s = T.image(T.SONG)
    table = np.full(0x10000, -1, np.int32)
    table[s["tab"] + 4 : s["tab"] + 8] = s["tab"]
    load = next(pc for pc in range(PLAY, PLAY + 8) if ram[pc] == 0xBD)
    tables = {load: table}
    remap, digest = Remap(tables), V.remap_digest(tables)
    assert digest != V.remap_digest(tables, [0xB1]) != V.remap_digest({load + 1: table})
    cert = V.certify(ram, PLAY, 1000, remap=remap, remap_id=digest)
    assert cert.remap == digest and cert.calls == 8
    assert cert.accesses[(load, C.LOAD)] == tuple(range(s["tab"], s["tab"] + 8))
    for taint in ("calls", "classes"):
        assert V.verify(ram, cert, remap=remap, remap_id=digest, taint=taint).ok
    assert V.verify(ram, cert).failed == "input"
    assert V.closure(ram, cert, 1000, remap=remap, remap_id=digest).ok


def test_a_call_that_does_not_return_fails_the_replay():
    ram, _, cert = certified(T.COUNTERS)
    got = V.verify(ram, dataclasses.replace(cert, max_cycles=5))
    assert got.failed == "replay" and got.detail == "call 0 ended at 'cycles', not its RTS"
    ram[0x3000] = 0x02
    cert = dataclasses.replace(cert, state=V.state_digest(ram), pc=0x3000)
    got = V.verify(ram, cert)
    assert got.failed == "replay" and got.detail == "call 0 ended as 'jam'"
    assert V.closure(ram, cert, 100).end == "jam"


def test_a_taint_of_other_calls_than_those_replayed_is_refused(monkeypatch):
    ram, _, cert = certified(T.COUNTERS)
    monkeypatch.setitem(V._TAINT, "calls", lambda *_: Learner({}, []))
    got = V.verify(ram, cert)
    assert got.failed == "replay" and "not the calls replayed" in got.detail


WIDE = """
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


def test_no_cover_point_says_what_has_not_repeated():
    ram, s = T.image(WIDE)
    got = V.certify(ram, PLAY, 2000, learn=600)
    assert isinstance(got, V.Open) and got.calls == 2001
    assert got.feed == (s["lo"], s["hi"], s["mod"])
    assert got.unrepeated == ((s["hi"],),), "lo and mod repeat, hi with lo does not"
    assert got.period == 768, "lo's 256 and mod's 3"
    ram, s = T.image(T.SWEEP)
    got = V.certify(ram, PLAY, 500)
    assert got == V.Open(501, (s["pos"], s["sweep"]), (), 768), "each repeats, together not yet"
    assert V.certify(ram, PLAY, 768).calls == 768


def test_a_late_dependence_is_certified_after_relearning():
    ram, s, cert = certified(T.LATE, learn=50)
    assert cert.feed == (s["m"], s["k"]) and cert.calls == 768
    assert V.verify(ram, cert).ok


STALLS = """
    inc n
    lda n
    cmp #200
    bne +
-   jmp -
+   rts
n   !byte 0
"""


def test_certifying_a_call_that_never_returns_raises():
    with pytest.raises(RuntimeError, match="'cycles'"):
        V.certify(T.image("loop jmp loop\n")[0], PLAY, 100, max_cycles=1000)
    with pytest.raises(RuntimeError, match="ended at 'cycles', not its RTS"):
        V.certify(T.image(STALLS)[0], PLAY, 1000, max_cycles=1000, learn=16)


def test_closure_runs_in_stages_to_its_limit_and_takes_in_new_stores(monkeypatch):
    ram, s, cert = certified(T.SWEEP)
    monkeypatch.setattr(C, "CHUNK", 16)
    monkeypatch.setattr(V, "_STAGE", 2)
    still = V.closure(ram, cert, 700)
    assert still == V.Closure("open", 700, {}, ()) and not still.ok
    acc, edges = T.seen(ram, 3)
    early = dataclasses.replace(
        cert,
        period=3,
        accesses=V._accesses(np.array(sorted(a for a in acc if not a >> 16 & 1))),
        edges=V._edges(np.array(sorted(edges))),
    )
    got = V.closure(ram, early, 5000)
    assert got.end == "closed" and got.calls == 769 and not got.ok and not got.edges
    assert sum(len(v) for k, v in got.accesses.items() if k[1] == C.LOAD) == 256 - 3
    assert {k[1] for k in got.accesses} == {C.LOAD, C.STORE}
    assert s["sweep"] in {a for k, v in got.accesses.items() if k[1] == C.STORE for a in v}


def test_the_tool_certifies_and_verifies_state_dumps(tmp_path, capsys):
    paths = []
    for name in ("COUNTERS", "SWEEP"):
        paths.append(str(tmp_path / name))
        Path(paths[-1]).write_bytes(T.image(getattr(T, name))[0])
    args = ["--pc", "%x" % PLAY, "--taint", "calls", "--taint", "classes", "--save"]
    assert cover_verify.main(args + ["--closure", "5000", "--limit", "700"] + paths) == 0
    rows = [[c.strip() for c in r.split("|")[1:-1]] for r in capsys.readouterr().out.splitlines()]
    assert rows[0][:2] == ["state", "calls"] and len(rows) == 4
    assert rows[2][1] == "15" and rows[2][4:7] == ["ok", "0", "0"] and rows[2][8] == "ok"
    assert rows[2][-4:-2] == ["closed", "16"] and rows[2][-1] == "0"
    assert rows[3][1] == "open at 701" and rows[3][-1] == "768"
    saved = V.Certificate.from_dict(json.loads(Path(paths[0] + ".cert.json").read_text()))
    assert V.verify(Path(paths[0]).read_bytes(), saved).ok
    assert cover_verify.main(["--pc", "%x" % PLAY, "--exclude", "100:200", paths[0]]) == 0
    assert "| 15 |" in capsys.readouterr().out


@pytest.mark.parametrize("name", ["COUNTERS", "SMC", "SONG", "VOICES"])
def test_tainting_every_call_in_runs_is_tainting_them_in_one(name, monkeypatch):
    """The calls cut into several runs, each tainted from its own entry, learn what one
    learner over all of them does, and the verdict is the same."""
    ram, _, cert = certified(getattr(T, name))
    args = (cert.pc, cert.acc, cert.max_cycles)
    acc = np.asarray(C.machine(ram).orbit(*args, cert.calls)[2], np.int64)
    stores = acc[acc >> 16 & 1 == 1] & 0xFFFF
    written = np.union1d(stores, np.arange(C.STACK.start, C.STACK.stop)).tolist()
    one = Learner({}, written)
    C.machine(ram).learn(one, *args, cert.calls)
    monkeypatch.setattr(C, "CHUNK", max(1, cert.calls // 5))
    monkeypatch.setattr(V.os, "cpu_count", lambda: 4)
    got = V._taint_calls(ram, cert, written, None).learned()
    for key, kind in V._LEARNED.items():
        want = np.unique(np.frombuffer(one.learned()[key], kind))
        assert np.array_equal(np.frombuffer(got[key], kind), want), key
    assert V.verify(ram, cert).ok
