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


def test_sha_indy_binds_the_loaded_pointer():
    """$93's H+1 is read by the lifter from the pointer; the generator binds the P-Code LOAD."""
    o = gen_cpu._Op(0x93)
    assert any(e.startswith("((u") for e in o.consts.values())


def test_unexplained_constant_is_an_error(monkeypatch):
    o = gen_cpu._Op(0x93)
    with pytest.raises(NotImplementedError):
        o._lift_time_read([1, 2])
    monkeypatch.setattr(gen_cpu, "OUT", gen_cpu.OUT.with_name("missing-stale.rs"))
    gen_cpu.OUT.write_text("stale")
    try:
        assert gen_cpu.main(["--check"]) == 1
    finally:
        gen_cpu.OUT.unlink()
