import runpy
import struct
import sys
from pathlib import Path

import pytest

from jennings import cli
from jennings.acme import Acme
from jennings.assembler import Assembler
from jennings.devices.mpu6502 import MPU
from jennings.image import Image, read_labels, sid_header
from jennings.memory import NATIVE
from jennings.symbols import symbols

PRG = Path(__file__).parent / "fixtures" / "sidplayer" / "sidplayer.prg"
ENGINES = ["python", "native"] if NATIVE else ["python"]


def _main(capsys, status, *argv):
    assert cli.main([str(a) for a in argv]) == status
    return capsys.readouterr()


def run(capsys, *argv):
    return _main(capsys, 0, *argv).out


def fail(capsys, *argv):
    return _main(capsys, 1, *argv).err


def assemble(org, src):
    return Assembler(MPU()).assemble_program(src, org)[0]


def psid(payload, load, init, play, version=2, start=1):
    head = struct.pack(
        ">4sHHHHHHHI32s32s32s",
        b"PSID",
        version,
        0x7C if version > 1 else 0x76,
        load,
        init,
        play,
        3,
        start,
        0,
        b"Tune",
        b"Someone",
        b"2026 Nobody",
    )
    if version > 1:
        head += struct.pack(">HBBBB", 0x14, 0, 0, 0, 0)
    return head + payload


@pytest.fixture
def sid(tmp_path):
    prg = PRG.read_bytes()
    path = tmp_path / "tune.sid"
    path.write_bytes(psid(prg, 0, 0x1000, 0x1003))
    return path


@pytest.fixture
def drive(tmp_path):
    img = bytearray(0x800)
    for org, src in (
        (0x500, "wait: LDA $1C00; AND #$10; BEQ wait; RTS"),
        (
            0x544,
            "JSR $0500; LDA #$08; ORA $1C00; STA $1C00; INC $1C00; JSR $C100;"
            "BIT $1800; BMI out; JMP $F969; out: RTS",
        ),
        (0x600, "JMP $0544"),
    ):
        code = assemble(org, src)
        img[org : org + len(code)] = code
    img[0x5B8:0x5C1] = b"PROTECTED"
    path = tmp_path / "drive.bin"
    path.write_bytes(img)
    return path


@pytest.fixture
def odd(tmp_path):
    """Illegal opcodes, accumulator mode, a JAM, a non-canonical NOP and an abs-encoded
    zero-page operand in a .prg at $C000."""
    code = assemble(0xC000, "ASL A; LAX $10; SLO ($20),Y; ANC #$0F; RTS")
    code += bytes([0xBD, 0x10, 0x00, 0x12, 0x1A, 0x0C, 0x34, 0x12, 0xAD, 0x10, 0x00]) + b"HI!\x8d!"
    path = tmp_path / "odd.prg"
    path.write_bytes(b"\x00\xc0" + code)
    return path, code


def roundtrip(capsys, *argv):
    src = run(capsys, "dis", *argv, "--source")
    acme = Acme([("out.a", src)], cpu="6510").run()
    return src, acme


def test_image_formats(tmp_path, sid):
    img = Image().load(str(PRG))
    assert (img.format, img.lo, img.hi) == ("prg", 0x1000, 0x138A)
    assert img.mem[0x1000:0x1003] == b"\x4c\x06\x10"
    raw = tmp_path / "rom.bin"
    raw.write_bytes(bytes(range(16)))
    img = Image().load(str(raw) + "@$e000").load(str(raw) + "@0xe008")
    assert img.format == "raw" and img.merged() == [(0xE000, 0xE018)] and img.mem[0xE009] == 1
    assert Image().load(str(raw), base=0x800).spans == [(0x800, 0x810)]
    assert Image().load(str(raw)).spans == [(0, 16)]
    weird = tmp_path / "a@b.bin"
    weird.write_bytes(b"\x01")
    assert Image().load(str(weird)).spans == [(0, 1)]
    assert Image().load(str(PRG), fmt="raw").lo == 0
    img = Image().load(str(sid))
    assert (
        img.format == "sid" and img.lo == 0x1000 and img.labels == {"INIT": 0x1000, "PLAY": 0x1003}
    )
    assert img.sid["name"] == "Tune" and img.sid["flags"] == 0x14 and img.sid["songs"] == 3
    with pytest.raises(ValueError):
        Image().load(str(raw) + "@fff8")


def test_sid_header_rules():
    h = sid_header(psid(b"\x00\x20\x60", 0, 0, 0, version=1))
    assert (h["load"], h["init"], h["play"], h["data"]) == (0x2000, 0x2000, 0, b"\x60")
    assert "flags" not in h
    h = sid_header(psid(b"\x60", 0x3000, 0x3000, 0x3000))
    assert (h["load"], h["data"]) == (0x3000, b"\x60")


def test_sid_play_label_omitted(tmp_path):
    path = tmp_path / "t.sid"
    path.write_bytes(psid(b"\x60", 0x3000, 0x3000, 0x3000))
    assert Image().load(str(path)).labels == {"INIT": 0x3000}


def test_read_labels_formats():
    acme = Acme([("t", "!cpu 6510\n* = $1000\nstart nop\nzp = $fb\n")]).run().symbol_list()
    text = acme + "al C:c100 .romcall\nal 1c00 via\nport=0x0001\nbank = $01\njunk\n"
    got = read_labels(text)
    assert got == {
        "start": 0x1000,
        "zp": 0xFB,
        "romcall": 0xC100,
        "via": 0x1C00,
        "port": 1,
        "bank": 1,
    }


def test_symbols_sets():
    c64 = symbols("c64")
    assert c64["VIC_EC"] == 0xD020 and c64["VIC_M7C"] == 0xD02E and c64["SID_MODEVOL"] == 0xD418
    assert c64["CIA2_CRB"] == 0xDD0F and c64["CHROUT"] == 0xFFD2 and c64["CINV"] == 0x0314
    d = symbols("1541")
    assert d["VIA2_PB"] == 0x1C00 and d["VIA1_IER"] == 0x180E and d["SECTOR5"] == 0x11
    assert d["BUF4"] == 0x700 and d["JOB5"] == 5
    assert symbols("1571")["WD_DATA"] == 0x2003 and symbols("1571")["CIA_ICR"] == 0x400D
    assert symbols("1581")["WD_STATUS"] == 0x6000 and "VIA2_PB" not in symbols("1581")


def test_dis_linear(capsys, tmp_path):
    out = run(capsys, "dis", PRG, "1000+3", "1006:100a")
    assert out.splitlines() == [
        "1000  4C 06 10         JMP $1006",
        "1003  4C 72 11         JMP $1172",
        "1006  AA               TAX",
        "",
        "1006  AA               TAX",
        "1007  BD E5 12         LDA $12e5,X",
    ]
    out = run(capsys, "dis", PRG, "1000", "-n", "2", "--no-bytes", "-D", "INIT=1000")
    assert out.splitlines() == ["1000  INIT:  JMP $1006", "1003         JMP $1172"]
    labels = tmp_path / "l.txt"
    labels.write_text("al C:1006 .start\nspeed = $138a\n")
    out = run(capsys, "dis", PRG, "1000-100d", "--labels", labels, "--auto-labels")
    assert "1000  4C 06 10         JMP start" in out
    assert "1003  4C 72 11         JMP $1172" in out
    assert "1006  AA        start: TAX" in out and "STA speed" in out
    out = run(capsys, "dis", PRG, "1000:1002")
    assert out.splitlines() == [
        '1000                   .byte $4C  ; "L"',
        "1001                   .byte $06",
    ]
    assert (
        run(capsys, "dis", PRG, "1000:1003", "--auto-labels").splitlines()[0].endswith("JMP $1006")
    )


def test_dis_default_span_and_text(capsys, odd):
    path, code = odd
    out = run(capsys, "dis", path)
    assert "C000  0A               ASL A" in out and "LAX $10" in out and "SLO ($20),Y" in out
    assert "C008  BD 10 00         LDA $0010,X" in out and "C00B  12               JAM" in out
    assert "C014  49 21            EOR #$21" in out and "C016                   .byte $8D" in out
    assert out.splitlines()[-1] == 'C017                   .byte $21  ; "!"'
    assert len(out.splitlines()) == 14


def test_source_roundtrip_linear(capsys, odd):
    path, code = odd
    src, acme = roundtrip(capsys, path, "-D", "zp=10")
    assert "        ASL\n" in src and "LDA+2 zp,X" in src and "        LAX zp" in src
    assert "        LDA+2 zp\n" in src and "        NOP $1234\n" in src
    assert "!byte $12" in src and "!byte $1A" in src and "zp = $10\n" in src
    assert acme.image() == (0xC000, code)


def test_source_roundtrip_sidplayer(capsys):
    prg = PRG.read_bytes()
    for extra in ((), ("-f", "-e", "1000", "-e", "1003", "--resolve")):
        src, acme = roundtrip(capsys, PRG, *extra)
        assert acme.image() == (0x1000, prg[2:])
    assert "L1006\n        TAX" in src and "        JMP L1006" in src


def test_dis_follow_sid(capsys, sid):
    out = run(capsys, "dis", sid, "-f", "--resolve").splitlines()
    assert out[:3] == [
        "1000  4C 06 10  INIT:  JMP L1006",
        "1003  4C 72 11  PLAY:  JMP L1172",
        "1006  AA        L1006: TAX",
    ]
    assert any(".byte" in line for line in out)
    lean = run(capsys, "dis", sid, "1000:1006", "-f", "--no-bytes").splitlines()
    assert lean[0] == "1000  INIT:  JMP L1006"


def test_dis_follow_drive_dump(capsys, drive):
    out = run(capsys, "dis", drive, "-f", "-e", "600", "--symbols", "1541")
    lines = out.splitlines()
    assert "0600  4C 44 05  BUF3:  JMP L0544" in lines
    assert "0500  AD 00 1C  BUF2:  LDA VIA2_PB" in lines
    assert "0552  20 00 C1         JSR XC100  ; external" in lines
    assert "055A  4C 69 F9         JMP XF969  ; external" in lines
    assert (
        '05B8                   .byte $50,$52,$4F,$54,$45,$43,$54,$45,$44  ; "PROTECTED"' in lines
    )
    assert not any(line.startswith("05B8") and "BVC" in line for line in lines)
    rows = run(capsys, "dis", drive, "-f", "-e", "c100").splitlines()
    assert rows[0].startswith("0000 ") and all(".byte" in row for row in rows)
    labels = run(capsys, "dis", drive, "-f", "-e", "600", "-D", "rom=c100")
    assert "JSR rom  ; external" in labels
    src, acme = roundtrip(capsys, drive, "-f", "-e", "600", "--symbols", "1541")
    assert '!byte $50,$52,$4F,$54,$45,$43,$54,$45,$44  ; "PROTECTED"' in src
    assert "XC100 = $C100" in src and "VIA2_PB = $1C00" in src and "        STA VIA2_PB" in src
    assert acme.image() == (0, drive.read_bytes())


def test_dis_follow_needs_numpy(capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "jennings.coverage.listing", None)
    assert "--follow needs numpy" in fail(capsys, "dis", PRG, "-f")


def test_hex(capsys, tmp_path):
    raw = tmp_path / "s.bin"
    raw.write_bytes(b"\x41\x61\xc1\x01\x20\x7f" + bytes(range(0x30, 0x3A)) + b"\xff")
    out = run(capsys, "hex", str(raw) + "@400").splitlines()
    assert out[0] == "0400  41 61 C1 01 20 7F 30 31 32 33 34 35 36 37 38 39  Aa.. .0123456789"
    assert out[1] == "0410  FF" + " " * 45 + "  ."
    assert run(capsys, "hex", raw, "0+4", "--petscii").split("  ")[-1] == "aAA.\n"
    assert run(capsys, "hex", raw, "0:4", "--screen").split("  ")[-1] == "A.Aa\n"
    out = run(capsys, "hex", raw, "2", "0:1", "-n", "3", "-w", "2").splitlines()
    assert out == ["0002  C1 01  ..", "0004  20      ", "", "0000  41     A"]


def test_find(capsys, odd):
    path, _ = odd
    assert run(capsys, "find", path, "13 ?? 0B").splitlines() == ["C003"]
    assert run(capsys, "find", path, "a7", "10", "--dis").splitlines() == ["C001  LAX $10"]
    assert run(capsys, "find", path, "--text", "HI").splitlines() == ["C013"]
    assert run(capsys, "find", path, "12??").splitlines() == ["C00B", "C00F"]
    assert "bad pattern" in fail(capsys, "find", path, "1")
    assert "PATTERN or --text" in fail(capsys, "find", path)


def test_xref(capsys, drive):
    out = run(capsys, "xref", drive, "1c00", "--symbols", "1541")
    assert out.splitlines() == [
        "0500  read    LDA VIA2_PB",
        "0549  read    ORA VIA2_PB",
        "054C  write   STA VIA2_PB",
        "054F  rmw     INC VIA2_PB",
    ]
    out = run(
        capsys,
        "xref",
        drive,
        "--range",
        "VIA1_PB:VIA2_PB+10",
        "-f",
        "-e",
        "600",
        "--symbols",
        "1541",
    )
    assert "0555  read    BIT VIA1_PB" in out and out.count("\n") == 5
    out = run(capsys, "xref", drive, "0500", "c100", "f969", "0544", "055d", "-e", "600")
    assert out.splitlines() == [
        "0505  branch  BEQ $0500",
        "0544  call    JSR $0500",
        "0552  call    JSR XC100",
        "0558  branch  BMI $055d",
        "055A  jump    JMP XF969",
        "0600  jump    JMP $0544",
    ]
    assert "ADDR or --range" in fail(capsys, "xref", drive)


def test_xref_ref_kind(capsys, odd):
    path, _ = odd
    assert run(capsys, "xref", path, "1234").splitlines() == ["C00D  ref     NOP $1234"]


def test_asm(capsys, tmp_path):
    out = run(capsys, "asm", "loop: LDA #$01; STA $D020; BNE loop", "--org", "c000")
    assert out.splitlines() == ["C000  A9 01 8D 20 D0 D0 F9", "loop = $C000"]
    prg, raw = tmp_path / "o.prg", tmp_path / "o.bin"
    run(capsys, "asm", "SLO $20", "--org", "$0801", "-o", prg)
    run(capsys, "asm", "SLO $20", "-o", raw)
    assert prg.read_bytes() == b"\x01\x08\x07\x20" and raw.read_bytes() == b"\x07\x20"
    assert "assembly failed" in fail(capsys, "asm", "FOO $10")


@pytest.mark.parametrize("engine", ENGINES)
def test_run_calls(capsys, tmp_path, engine):
    raw = tmp_path / "r.bin"
    code = assemble(0x1000, "STX $2000; STY $2001; STA $2002; INX; RTS")
    raw.write_bytes(code.ljust(16, b"\xea") + assemble(0x1010, "JMP $1010"))
    out = run(
        capsys,
        "run",
        str(raw) + "@1000",
        "--engine",
        engine,
        "--call",
        "1000:7:5:9",
        "--call",
        "1010",
        "--max-cycles",
        "30",
        "--dump",
        "2000+3",
    ).splitlines()
    assert out[0] == "1000 $1000  20 cycles  A=07 X=06 Y=09 SP=FF P=30"
    assert out[1].startswith("1010 $1010  3") and "(budget)" in out[1]
    assert "changed 01FE:01FF  01" in out and "changed 2000:2003  05 09 07" in out
    assert out[-1] == "2000  05 09 07"


@pytest.mark.parametrize("engine", ENGINES)
def test_run_sid(capsys, sid, engine):
    out = run(capsys, "run", sid, "--engine", engine, "--init", "--play", "50").splitlines()
    assert out[0] == "INIT $1000  525 cycles  A=0F X=03 Y=06 SP=FF P=31"
    assert out[1].startswith("PLAY $1003 x50  ")
    assert "changed D418:D419  0F" in out


def test_run_engines_agree(capsys, sid):
    outs = [run(capsys, "run", sid, "--engine", e, "--init", "1", "--play", "20") for e in ENGINES]
    assert len(set(outs)) == 1 and "Y=0C" in outs[0].splitlines()[0]


def test_run_trace(capsys, sid):
    out = run(capsys, "run", sid, "--call", "INIT:0", "--trace").splitlines()
    assert out[0] == "  1000  JMP $1006        A=00 X=00 Y=00 SP=FD P=30"
    assert out[1].startswith("  1006  TAX") and out[2].startswith("  1007  LDA $12e5,X  ")
    assert "INIT $1000  525 cycles" in "\n".join(out)


def test_run_errors(capsys, odd):
    path, _ = odd
    assert "nothing to run" in fail(capsys, "run", path)
    assert "unknown label: 'PLAY'" in fail(capsys, "run", path, "--play", "1")


def test_info(capsys, sid, tmp_path):
    out = run(capsys, "info", sid)
    assert "format  sid" in out and "span    1000:138A  (906 bytes)" in out
    assert "name    Tune" in out and "init    $1000" in out and "play    $1003" in out
    rom = tmp_path / "rom.bin"
    rom.write_bytes(bytes(0x1FFA) + b"\x01\xfe\x02\xfe\x03\xfe")
    out = run(capsys, "info", rom, "--base", "e000")
    assert "NMI     $FE01" in out and "RESET   $FE02" in out and "IRQ     $FE03" in out
    assert "NMI" not in run(capsys, "info", PRG, "--overlay", str(rom) + "@0")


def test_errors(capsys, tmp_path):
    assert "bad address or unknown label" in fail(capsys, "dis", PRG, "nolabel")
    assert "bad address" in fail(capsys, "dis", PRG, "10 00")
    assert "out of range" in fail(capsys, "dis", PRG, "ffff+2:10001")
    assert "NAME=ADDR" in fail(capsys, "dis", PRG, "-D", "x")
    assert "decimal N" in fail(capsys, "dis", PRG, "1000+x")
    short = tmp_path / "short.sid"
    short.write_bytes(b"PSID\x00\x02")
    assert "truncated SID header" in fail(capsys, "info", short)
    assert "No such file" in fail(capsys, "dis", tmp_path / "missing.prg")


def test_module_entry(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["jennings", "dis", str(PRG), "1000+1"])
    with pytest.raises(SystemExit) as e:
        runpy.run_path(str(Path(cli.__file__).with_name("__main__.py")), run_name="__main__")
    assert e.value.code == 0 and "JMP $1006" in capsys.readouterr().out


def test_listing_names_and_inside():
    from jennings.coverage.listing import Listing  # pylint: disable=import-outside-toplevel

    mem = bytearray(0x10000)
    mem[0x1000:0x1007] = assemble(0x1000, "JSR $2000; JMP $1000; RTS")
    mem[0x2000] = 0x60
    whole = Listing.disassemble(mem, [0x1000], names={0x2000: "sub"})
    assert (
        "1000  20 00 20  L1000: JSR sub" in whole.source and "2000  60        sub:" in whole.source
    )
    bounded = Listing.disassemble(mem, [0x1000], inside=[(0x1000, 0x1006)])
    assert bounded.lines[:, 1].tolist() == [0x1000, 0x1003]
