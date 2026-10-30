import re
import shutil
from pathlib import Path

import coverage
import numpy as np
import pytest

from jennings.coverage import RETURN, TAKEN, CoverageMPU, call
from jennings.coverage.elf import line_starts, load_elf, sidecars, write_sidecars
from jennings.coverage.plugin import load_sidecar
from jennings.coverage.report import write_data
from jennings.devices.mpu6502 import MPU

FIX = Path(__file__).parent / "fixtures" / "sidplayer"
ELF = FIX / "sidplayer.prg.elf"
FRAMES = 300
HANDLERS = ("fx_off", "fx_vibrato", "fx_slide", "fx_pulse")


def _asm():
    """``(instruction lines, {label: line of its first instruction})`` parsed from player.s."""
    text = (FIX / "player.s").read_text().split(".section .rodata")[0].splitlines()
    code, labels, pending = [], {}, []
    for n, row in enumerate(text, 1):
        body = row.split(";")[0]
        m = re.match(r"^(\w+):", body)
        if m:
            pending.append(m.group(1))
        elif body.startswith((" ", "\t")) and body.strip() and not body.strip().startswith("."):
            code.append(n)
            labels.update(dict.fromkeys(pending, n))
            pending = []
    return code, labels


def _find(name, needle):
    rows = (FIX / name).read_text().splitlines()
    return next(n for n, r in enumerate(rows, 1) if needle in r)


CODE, LABELS = _asm()


@pytest.fixture(scope="module", name="image")
def _image():
    return load_elf(ELF)


@pytest.fixture(scope="module", name="tables")
def _tables():
    return {Path(s).name: t for s, t in sidecars(ELF, ["init", "play"]).items()}


@pytest.fixture(name="work")
def _work(tmp_path):
    for name in ("player.s", "freq.c", ELF.name):
        shutil.copy(FIX / name, tmp_path)
    return tmp_path


def _line_of(table, address):
    lines = table["lines"]
    return int(lines[lines[:, 1] == address, 0][0])


def _arcs(table):
    return set(map(tuple, table["arcs"].tolist()))


def test_load_elf_matches_prg(image):
    mem, syms = image
    prg = (FIX / "sidplayer.prg").read_bytes()
    load = prg[0] | prg[1] << 8
    assert load == syms["init"] == 0x1000 and syms["play"] == 0x1003
    assert bytes(mem[load : load + len(prg) - 2]) == prg[2:]
    assert not mem[load + len(prg) - 2 :].any()
    jmp = mem[syms["init"] : syms["init"] + 3].tobytes()
    assert jmp[0] == 0x4C and int.from_bytes(jmp[1:], "little") == syms["start"]
    assert syms["SID"] == 0xD400 and syms["__rc0"] == 0xE0
    assert syms["note_freq"] > syms["sidw"] and "ld-temp.o" not in syms


def test_asm_lines_match_labels(tables):
    asm = tables["player.s"]
    assert sorted(set(asm["lines"][:, 0].tolist())) == CODE
    assert np.unique(asm["lines"][:, 0]).size == len(asm["lines"])
    _, syms = load_elf(ELF)
    for label, line in LABELS.items():
        assert _line_of(asm, syms[label]) == line, label


def test_asm_arcs(tables):
    asm = tables["player.s"]
    arcs = _arcs(asm)
    line = LABELS.__getitem__
    beq = _find("player.s", "beq     seq_rd")
    assert {(beq, line("seq_rd")), (beq, line("seq_have"))} <= arcs
    for name in ("init", "play", "next_pattern", "getb", "fetch", "dispatch", "sidw") + HANDLERS:
        assert (-line(name), line(name)) in arcs, name
    assert (-line("voice"), line("voice")) not in arcs
    rts = _find("player.s", "rts")
    assert (rts, -rts) in arcs and not any(a == rts and b > 0 for a, b in arcs)
    jsr = _find("player.s", "jsr     note_freq")
    assert (jsr, jsr + 1) in arcs and (jsr, -jsr) not in arcs


def test_c_lines_and_loops(tables, image):
    c = tables["freq.c"]
    _, syms = image
    head = _find("freq.c", "note_freq(")
    count = _find("freq.c", "while (note < base)")
    shift = _find("freq.c", "f >>= 1")
    ret = _find("freq.c", "return f")
    assert _line_of(c, syms["note_freq"]) == head
    per_line = dict(zip(*np.unique(c["lines"][:, 0], return_counts=True)))
    assert min(per_line[count], per_line[shift]) > 2
    arcs = _arcs(c)
    assert {(-head, head), (count, head), (shift, -shift), (shift, ret), (ret, -ret)} <= arcs
    assert not any(a == b for a, b in arcs)
    loops = c["transitions"]
    assert (loops[:, 2] == shift).any() and (loops[(loops[:, 3] == -shift), 1] == TAKEN).all()


def test_root_resolution_and_write(work):
    srcs = write_sidecars(work / ELF.name, ["play"], root=work / "missing")
    assert srcs == []
    (work / "freq.c").unlink()
    srcs = write_sidecars(work / ELF.name, ["play"])
    assert srcs == [str(work / "player.s")]
    side = load_sidecar(srcs[0])
    assert side["lines"].shape[1] == 2 and side["transitions"].shape[1] == 4
    starts = line_starts(work / ELF.name, load_elf(ELF)[0], root="/src")
    assert {s for s, _ in starts.values()} == {"/src/player.s", "/src/freq.c"}


def test_call_runs_to_rts():
    for cls in (MPU, CoverageMPU):
        mpu = cls()
        mpu.memory[0x0900:0x0907] = [0x20, 0x06, 0x09, 0xE8, 0xC8, 0x60, 0xE8]
        mpu.memory[0x0907] = 0x60
        sp = mpu.sp
        assert call(mpu, 0x0900, a=7, x=1, y=2) is mpu
        assert (mpu.a, mpu.x, mpu.y, mpu.pc, mpu.sp) == (7, 3, 3, RETURN, sp)
    assert mpu.cov.hits[0x0900:0x0908].tolist() == [1, 0, 0, 1, 1, 1, 1, 1]
    mpu.memory[0x0A00:0x0A03] = [0x4C, 0x00, 0x0A]
    with pytest.raises(RuntimeError, match=r"\$0A00 did not return within 50 steps"):
        call(mpu, 0x0A00, max_steps=50)


def _play(song, frames=FRAMES):
    mem, syms = load_elf(ELF)
    mpu = CoverageMPU(memory=mem.tolist())
    call(mpu, syms["init"], a=song)
    for _ in range(frames):
        call(mpu, syms["play"])
    return mpu, syms


def _report(work, data):
    rc = work / "rc"
    rc.write_text("[run]\nplugins = jennings.coverage.plugin\nbranch = True\n")
    c = coverage.Coverage(data_file=str(data), config_file=str(rc))
    c.load()
    return c


def test_sid_registers_follow_song():
    mpu, _ = _play(1, 40)
    sid = mpu.memory[0xD400:0xD419]
    assert sid[24] == 0x0F
    assert [sid[4] & 0xF0, sid[11] & 0xF0, sid[18] & 0xF0] == [0x40, 0x20, 0x80]


def test_end_to_end_two_songs(work):
    elf = work / ELF.name
    srcs = write_sidecars(elf, ["init", "play"])
    asm, cfile = srcs[1], srcs[0]
    line = LABELS.__getitem__
    feature = _find("player.s", "bne     glide")
    slide_only = set(range(line("fx_slide"), line("fx_slide") + 2)) | {line("slide_down")}
    data = work / ".coverage-6502"
    for song in (0, 1):
        mpu, _ = _play(song)
        write_data(mpu.cov, srcs, data_file=str(data), context="song%d" % song, suffix=True)
        if song == 0:
            write_data(mpu.cov, srcs, data_file=str(work / "song0"))
            c = _report(work, work / "song0")
            _, _, _, missing, _ = c.analysis2(asm)
            assert slide_only <= set(missing)
            assert {line(h) for h in ("fx_off", "fx_vibrato")}.isdisjoint(missing)
            assert c._analyze(asm).missing_branch_arcs()[feature] == [line("glide")]
    c = _report(work, data)
    c.combine([str(work)])
    c.save()
    c = _report(work, data)
    _, statements, _, missing, _ = c.analysis2(asm)
    assert slide_only.isdisjoint(missing) and set(missing) < set(statements)
    assert {line(h) for h in HANDLERS}.isdisjoint(missing)
    assert feature not in c._analyze(asm).missing_branch_arcs()
    assert not c.analysis2(cfile)[3]
    assert c.get_data().measured_contexts() == {"song0", "song1"}
    assert c.html_report(directory=str(work / "html")) > 80
    page = next((work / "html").glob("*player_s.html")).read_text()
    assert "jsr     note_freq" in page and "dispatch:" in page
