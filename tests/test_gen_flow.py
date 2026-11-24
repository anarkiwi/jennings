"""crates/jennings-core/src/flow_gen.rs is the generator's output for the installed lifter."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import gen_flow  # pylint: disable=wrong-import-position
from helpers import check_flags_a_stale_file, check_regeneration_is_a_noop


def test_regeneration_is_a_noop():
    check_regeneration_is_a_noop(gen_flow)


def test_sta_indy_stores_a_through_its_pointer_bytes_and_y():
    kind, n, loads, stores, outs, la, sa, vals = gen_flow.opcode_flow(0x91)
    assert (kind, n, loads, stores) == ("Next", 2, 2, 1)
    assert outs == [(gen_flow.STORE0, 1)]
    o1, y = 1 << gen_flow.O1, 1 << 2
    assert la == [o1, o1] and sa == [o1 | y | 3 << gen_flow.LOAD0]
    assert vals == outs


def test_explicit_values_drop_load_addresses():
    load = lambda k: 1 << gen_flow.LOAD0 + k
    assert gen_flow.opcode_flow(0xB1)[7] == [(0, load(2)), (9, load(2)), (14, load(2))]
    inc = gen_flow.opcode_flow(0xFE)[7]
    assert inc == [(9, load(0)), (14, load(0)), (gen_flow.STORE0, load(0))]
    for op in (0x9E, 0x9F):
        f = gen_flow.opcode_flow(op)
        assert f[7] == f[4] and f[4][0][1] & 1 << gen_flow.O2, "SHX/SHA store the high byte"


def test_a_branch_maps_its_flag_to_the_condition():
    assert gen_flow.opcode_flow(0xB0)[:5] == ("Branch", 2, 0, 0, [(gen_flow.PCOUT, 1 << 8)])
    assert gen_flow.opcode_flow(0x20)[:5] == ("Jsr", 3, 0, 0, [])


def test_check_flags_a_stale_file(monkeypatch):
    check_flags_a_stale_file(gen_flow, monkeypatch, "missing-stale-flow.rs")
