"""The parts of jennings.coverage.certificate and segments that need no native core."""

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from jennings.coverage import certificate as V
from jennings.coverage import components as C
from jennings.coverage import segments as S

RAM = bytes(0x10000)
DIGEST = hashlib.sha256(RAM).hexdigest()


def joint(**kw):
    held = {
        "state": DIGEST,
        "pc": 0x1000,
        "acc": 0,
        "max_cycles": 1000,
        "exclude": ((0x100, 0x200),),
        "remap": None,
        "feed": (0x2000, 0x2001),
        "preperiod": 2,
        "period": 5,
        "accesses": {(0x1000, C.LOAD): (0x2000, 0x2001), (0x1003, C.STORE): (0xD400,)},
        "edges": ((0x1000, 0x1003),),
    }
    return V.Certificate(**{**held, **kw})


def segmented(**kw):
    held = {
        "state": DIGEST,
        "pc": 0x1000,
        "acc": 0,
        "max_cycles": 1000,
        "remap": None,
        "volatile": ((0xD000, 0xE000),),
        "cuts": (0x1010,),
        "lead": 1,
        "free": ((1, 0x2002),),
        "values": ((0x2002, (0, 1)),),
        "starts": (0x1000, 0x1010),
        "cycles": ((1, 3), (2, 4)),
        "feed": (0x2000,),
        "calls": 14,
        "accesses": {(0x1000, C.LOAD): (0x2000,)},
        "edges": ((0x1000, 0x1010),),
    }
    return V.Segmented(**{**held, **kw})


@pytest.mark.parametrize("cert", [joint(), segmented()])
def test_certificates_go_to_json_values_and_back(cert):
    d = cert.to_dict()
    assert all(isinstance(v, list) for k, v in d.items() if k in ("feed", "edges", "accesses"))
    assert V.from_dict(d) == cert


def test_a_joint_and_a_segmented_certificate_count_their_calls():
    assert joint().calls == 7
    seg = segmented()
    assert (seg.preperiod, seg.period) == (2, 12)


def test_what_does_not_match_the_state_and_remap_is_said(monkeypatch):
    monkeypatch.setattr(V, "state_digest", lambda state: hashlib.sha256(state).hexdigest())
    assert V._mismatch(RAM, joint(), None) is None
    assert V._mismatch(RAM, segmented(), None) is None
    other = bytearray(RAM)
    other[0x2000] = 1
    assert "state" in V._mismatch(bytes(other), joint(), None)
    assert "remap" in V._mismatch(RAM, joint(), "x")
    assert "period" in V._mismatch(RAM, joint(period=0), None)
    assert "cycle per segment" in V._mismatch(RAM, segmented(cycles=((1, 3),)), None)
    assert "cut call" in V._mismatch(RAM, segmented(lead=14), None)
    assert "preperiod" in V._mismatch(RAM, segmented(cycles=((0, 3), (2, 4))), None)


def test_access_and_edge_keys_go_both_ways():
    acc = joint().accesses
    keys = V._access_keys(acc)
    assert keys.tolist() == sorted(keys.tolist())
    assert V._accesses(keys) == acc
    assert V._access_keys({}).size == 0
    edges = ((0x1000, 0x1003), (0x1003, 0x1000))
    assert V._edges(V._edge_keys(edges)) == tuple(sorted(edges))
    assert V._grouped([1, 2, 1 << 4 | 3], 4) == {0: [1, 2], 1: [3]}


def test_spans_remaps_and_repeats():
    assert V._spans([range(4, 8), (0, 2)]) == ((0, 2), (4, 8))
    with pytest.raises(ValueError):
        V._spans([range(0, 8, 2)])
    with pytest.raises(ValueError):
        V._remap_id(None, "x")
    assert V._remap_id(None, None) is None
    assert V.remap_digest({1: [0, 1]}) == V.remap_digest({1: np.array([0, 1])})
    assert V.remap_digest({1: [0, 1]}) != V.remap_digest({1: [1, 0]})
    rows = np.array([[0], [1], [2], [1]], np.uint8)
    assert V._first_repeat(rows, 3) == 1
    assert V._volatile(((0, 0x300),)) == ((0, 0x100), (0x200, 0x300))


def test_spent_calls_grow_their_budget_until_it_is_refused():
    asked = []

    def afford(calls):
        asked.append(calls)
        if calls > 40:
            raise S.Refused("budget", "the repeat is found")
        return 2 * max(calls, 8)

    spent = S.Spent(afford)
    assert (spent.budget, spent.left(5)) == (16, 11), "the runs left under the budget"
    assert spent.left(20) == 27 and asked == [0, 26], "grown past the calls run"
    with pytest.raises(S.Refused) as e:
        spent.left(30)
    assert e.value.failed == "budget"
    free = S.Spent()
    assert (free.left(10**9), free.calls) == (0, 10**9), "no budget: any runs"
    stop = S.Spent(lambda calls: 4 if calls < 4 else None)
    assert stop.left(10) == 0, "a budget given up is any calls"


def test_a_found_shows_no_two_entries_alike_only_where_its_cycles_do():
    rows = np.array([[0, 5], [1, 6], [2, 7], [0, 8], [1, 5], [2, 6]], np.uint8)
    parts = SimpleNamespace(cells=((0x10,), (0x11,)))
    found = SimpleNamespace(feed=(0x10, 0x11), parts=parts, cycles=((0, 3), (0, 4)), rows=rows)
    assert S.never_repeats(found, (0x10, 0x11), 11)
    assert not S.never_repeats(found, (0x10, 0x11), 12), "the joint cycle is 12"
    assert not S.never_repeats(found, (0x10,), 3)
    assert S._by_cycle(found, (0x11,)) == {(0, 4): [1]}
    alike = S._hashed(np.array([[1, 2], [1, 2], [2, 1]], np.uint8))
    assert alike[0] == alike[1] != alike[2]


def test_a_refusal_is_its_own_unclosed_reason():
    e = S.Refused("budget", "more", 3)
    assert (e.failed, e.detail, e.call, str(e)) == ("budget", "more", 3, "more")
    assert S._unclosed(e, 10) is e
