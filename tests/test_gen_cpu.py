"""crates/jennings-core/src/cpu_gen.rs is the generator's output for the installed lifter."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import gen_cpu  # pylint: disable=wrong-import-position


def test_regeneration_is_a_noop():
    assert gen_cpu.OUT.read_text() == gen_cpu.generate()
    assert gen_cpu.main(["--check"]) == 0


def test_every_opcode_has_a_generated_fn():
    src = gen_cpu.OUT.read_text()
    for op in range(256):
        assert "fn op_%02x<" % op in src and "0x%02x => op_%02x(" % (op, op) in src


def test_sha_indy_has_no_operand_bound_h_plus_1():
    """$93's H+1 is P-Code on the loaded pointer, so only the zp operand is bound."""
    assert set(gen_cpu._Op(0x93).consts.values()) == {"b1", "((b1 + 1) & 0xff)"}


def test_unexplained_constant_is_an_error(monkeypatch):
    o = gen_cpu._Op(0xA9)
    rec = o.lifts[1][0]
    i = next(i for i, (_m, _o, ins) in enumerate(rec["ops"]) if any(v[0] == "c" for v in ins))
    j = next(j for j, v in enumerate(rec["ops"][i][2]) if v[0] == "c")
    rec["ops"][i][2][j] = ["c", rec["ops"][i][2][j][1] ^ 1, 1]
    o.rec["prov"]["ops"].pop((i, j), None)
    with pytest.raises(NotImplementedError):
        o._bind_consts()
    monkeypatch.setattr(gen_cpu, "OUT", gen_cpu.OUT.with_name("missing-stale.rs"))
    gen_cpu.OUT.write_text("stale")
    try:
        assert gen_cpu.main(["--check"]) == 1
    finally:
        gen_cpu.OUT.unlink()
