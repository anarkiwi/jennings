"""crates/jennings-core/src/cpu_gen.rs is the generator's output for the installed lifter."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import gen_cpu  # pylint: disable=wrong-import-position
from helpers import check_flags_a_stale_file, check_regeneration_is_a_noop


def test_regeneration_is_a_noop():
    check_regeneration_is_a_noop(gen_cpu)


def test_every_opcode_has_a_generated_fn():
    src = gen_cpu.OUT.read_text()
    for op in range(256):
        assert "fn op_%02x<" % op in src and "0x%02x => op_%02x(" % (op, op) in src


def test_sha_indy_has_no_operand_bound_h_plus_1():
    """$93's H+1 is P-Code on the loaded pointer, so only the zp operand is bound."""
    assert set(gen_cpu._Op(0x93).consts.values()) == {"b1", "((b1 + 1) & 0xff)"}


def test_check_flags_a_stale_file(monkeypatch):
    check_flags_a_stale_file(gen_cpu, monkeypatch, "missing-stale.rs")
