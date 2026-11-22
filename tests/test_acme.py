"""jennings.acme against the files acme 0.97 writes (tests/acme_cases.py)."""

import os
import subprocess
import sys

import pytest

from acme_cases import ERRORS, EXPECTED, SOURCES, UNSUPPORTED
from jennings.acme import Acme, AcmeError, assemble, main, mnemonics, read_symbol_list


@pytest.mark.parametrize("name", sorted(SOURCES))
def test_to_directive(name):
    """Without -o, !to names the file and picks the format; the symbol list matches."""
    acme = Acme([("src.asm", SOURCES[name])]).run()
    want = {k: v for k, v in EXPECTED[name].items() if k not in ("labels", "out")}
    assert ({acme.outname: acme.output()} if acme.outname else {}) == want
    assert acme.symbol_list() == EXPECTED[name]["labels"]


@pytest.mark.parametrize("name", sorted(SOURCES))
def test_read_symbol_list_inverts_symbol_list(name):
    """The defined symbols, values as the list prints them (32-bit), size suffixes dropped."""
    acme = Acme([("src.asm", SOURCES[name])]).run()
    want = {n: v & 0xFFFFFFFF for n, v in acme.symbols().items() if v is not None}
    assert read_symbol_list(EXPECTED[name]["labels"]) == want


def test_read_symbol_list_forms():
    text = "\tb+2\t= $3\n\tz\t= ?\t; ?\n  name=$C000\t; unused\nnot a line\n"
    assert read_symbol_list(text) == {"b": 3, "name": 0xC000}


@pytest.mark.parametrize("name", sorted(SOURCES))
def test_outfile_overrides_to(name):
    """-o wins over !to, whose format is then ignored: the output stays plain."""
    acme = Acme([("src.asm", SOURCES[name])], outname="out").run()
    assert (acme.outname, acme.output()) == ("out", EXPECTED[name]["out"])


@pytest.mark.parametrize("source", sorted(ERRORS))
def test_errors(source):
    with pytest.raises(AcmeError) as info:
        Acme([("e.asm", source)]).run()
    assert ERRORS[source] in str(info.value)
    assert str(info.value).startswith("e.asm:")


@pytest.mark.parametrize("source", sorted(UNSUPPORTED))
def test_unsupported(source):
    """ACME features outside the subset fail loudly instead of assembling wrongly."""
    with pytest.raises(AcmeError, match="Unsupported: " + UNSUPPORTED[source]):
        Acme([("u.asm", source)]).run()


def test_assemble():
    """The API returns the output file bytes and the global symbols in list order."""
    data, symbols = assemble(SOURCES["player"])
    assert data == EXPECTED["player"]["out"]
    assert list(symbols) == ["ADDR", "init", "_init", "TRACK", "frc", "tab", "play"]
    assert symbols["frc"] == 0x1023 and symbols["TRACK"] == 0xF0


def test_formats():
    """-f selects cbm/apple headers; a !to format wins over it, as in ACME."""
    text = "*=$c000\n!by 1, 2\n"
    assert assemble(text, fmt="cbm")[0] == b"\x00\xc0\x01\x02"
    assert assemble(text, fmt="apple")[0] == b"\x00\xc0\x02\x00\x01\x02"
    assert assemble('!to "x", plain\n' + text, fmt="cbm")[0] == b"\x01\x02"
    assert assemble('!to "x"\n' + text, fmt="plain")[0] == b"\x01\x02"
    with pytest.raises(AcmeError, match="Unknown output format"):
        Acme([("s", text)], fmt="d64")


def test_cpu():
    """The 6502 set has no illegals; 6510 adds them with ACME's ASR alias of ALR."""
    assert "lax" not in mnemonics("6502") and mnemonics("6510")["lax"]["abs"] == {1: 0xA7, 2: 0xAF}
    assert mnemonics("6510")["asr"] is mnemonics("6510")["alr"]
    assert mnemonics("NMOS6502") is mnemonics("6510")
    assert assemble("*=0\nlax $12\n", cpu="6510")[0] == b"\xa7\x12"
    with pytest.raises(AcmeError, match="Unsupported: !cpu 65816"):
        mnemonics("65816")


def test_multiple_sources():
    """Sources are assembled in order every pass, sharing symbols."""
    acme = Acme([("a", "*=$1000\njmp b\n"), ("b", "b nop\n")], outname="o").run()
    assert acme.output() == b"\x4c\x03\x10\xea"
    assert acme.image() == (0x1000, b"\x4c\x03\x10\xea")


def test_passes():
    """Passes repeat while undefined results remain, as many as acme -v2 reports."""
    acme = Acme([("s", "*=0\n!by a\na = b\nb = c\nc = 1\n")]).run()
    assert acme.passno == 4 and acme.output() == b"\x01"
    acme = Acme([("s", "*=0\nc = b\nb = a\na = 1\n!by c\n")]).run()
    assert acme.passno == 3


def test_cli(tmp_path, monkeypatch, capsys):
    src = tmp_path / "p.asm"
    src.write_text(SOURCES["player"], encoding="latin-1")
    out, labels = tmp_path / "p.sid", tmp_path / "p.lbl"
    assert main(["-l", str(labels), "-o", str(out), str(src)]) == 0
    assert out.read_bytes() == EXPECTED["player"]["out"]
    assert labels.read_text(encoding="latin-1") == EXPECTED["player"]["labels"]
    monkeypatch.chdir(tmp_path)
    assert main(["--symbollist", "q.lbl", str(src)]) == 0
    assert (tmp_path / "player.sid").read_bytes() == EXPECTED["player"]["player.sid"]
    src.write_text("x = 1\n", encoding="latin-1")
    assert main([str(src)]) == 0
    assert "No output file specified" in capsys.readouterr().err
    src.write_text("nop\n", encoding="latin-1")
    assert main(["-o", str(out), str(src)]) == 1
    assert "Program counter undefined" in capsys.readouterr().err
    assert main([str(tmp_path / "missing.asm")]) == 1


def test_module_entry(tmp_path):
    """``python -m jennings.acme`` is the same command line."""
    src = tmp_path / "p.asm"
    src.write_text("*=$1000\nrts\n", encoding="ascii")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    subprocess.run(
        [sys.executable, "-m", "jennings.acme", "-f", "cbm", "-o", "p.prg", "p.asm"],
        cwd=tmp_path,
        check=True,
        env=env,
    )
    assert (tmp_path / "p.prg").read_bytes() == b"\x00\x10\x60"
