"""crates/jennings-core/src/flow_gen.rs is the generator's output for the installed lifter."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import gen_flow  # pylint: disable=wrong-import-position


def test_regeneration_is_a_noop():
    assert gen_flow.OUT.read_text() == gen_flow.generate()
    assert gen_flow.main(["--check"]) == 0


def test_sta_indy_stores_a_through_its_pointer_bytes_and_y():
    kind, n, loads, stores, outs, la, sa = gen_flow.opcode_flow(0x91)
    assert (kind, n, loads, stores) == ("Next", 2, 2, 1)
    assert outs == [(gen_flow.STORE0, 1)]
    o1, y = 1 << gen_flow.O1, 1 << 2
    assert la == [o1, o1] and sa == [o1 | y | 3 << gen_flow.LOAD0]


def test_a_branch_maps_its_flag_to_the_condition():
    assert gen_flow.opcode_flow(0xB0)[:5] == ("Branch", 2, 0, 0, [(gen_flow.PCOUT, 1 << 8)])
    assert gen_flow.opcode_flow(0x20)[:5] == ("Jsr", 3, 0, 0, [])


def test_check_flags_a_stale_file(monkeypatch):
    monkeypatch.setattr(gen_flow, "OUT", gen_flow.OUT.with_name("missing-stale-flow.rs"))
    gen_flow.OUT.write_text("stale")
    try:
        assert gen_flow.main(["--check"]) == 1
    finally:
        gen_flow.OUT.unlink()
