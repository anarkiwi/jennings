import io
import os
import subprocess
import sys
from pathlib import Path

import coverage
import numpy as np
import pytest

import jennings
from jennings.assembler import Assembler
from jennings.coverage import CoverageMap, CoverageMPU
from jennings.coverage.listing import Listing
from jennings.coverage.plugin import PLUGIN_NAME, FileReporter6502, Plugin6502, load_sidecar
from jennings.coverage.report import executed_arcs, write_data
from jennings.devices.mpu6502 import MPU

ORG = 0x0800
BRANCH = ["LDA $10", "BEQ $0806", "LDX #$01", "NOP"]


def _mpu(statements, org=ORG, **mem):
    mpu = CoverageMPU(pc=org)
    asm, pc = Assembler(mpu), org
    for st in statements:
        code = asm.assemble(st, pc)
        mpu.memory[pc : pc + len(code)] = code
        pc += len(code)
    for addr, val in mem.items():
        mpu.memory[int(addr[1:], 16)] = val
    return mpu


def _run_to(mpu, end, limit=1000):
    for _ in range(limit):
        if mpu.pc == end:
            return mpu.cov
        mpu.step()
    raise AssertionError("did not reach %04X" % end)


def _branch_run(value):
    mpu = _mpu(BRANCH, m10=value)
    return mpu, _run_to(mpu, 0x0807)


def _write(tmp_path, mpu, cov, entries=(ORG,), name="prog.s", **kw):
    listing = Listing.disassemble(mpu.memory, entries, cov, **kw)
    src = tmp_path / name
    listing.write(src)
    return listing, str(src)


def _cov(tmp_path, data_file):
    rc = tmp_path / "rc"
    rc.write_text("[run]\nplugins = jennings.coverage.plugin\nbranch = True\n")
    c = coverage.Coverage(data_file=str(data_file), config_file=str(rc))
    c.load()
    return c


def _analysis(tmp_path, data_file, src):
    c = _cov(tmp_path, data_file)
    _, statements, _, missing, _ = c.analysis2(src)
    return c, statements, missing, c.branch_stats(src)


def test_partial_branch(tmp_path):
    mpu, cov = _branch_run(0)
    listing, src = _write(tmp_path, mpu, cov)
    assert [listing.line(a) for a in (0x800, 0x802, 0x804, 0x806, 0x807)] == [1, 2, 3, 4, 5]
    assert "BEQ L0806" in listing.source and "L0806: NOP" in listing.source
    data = tmp_path / "d"
    write_data(cov, src, data_file=str(data))
    c, statements, missing, branches = _analysis(tmp_path, data, src)
    assert statements == [1, 2, 3, 4, 5] and missing == [3, 5]
    assert branches == {2: (2, 1)}
    out = io.StringIO()
    c.report(file=out, show_missing=True)
    row = next(r for r in out.getvalue().splitlines() if "prog.s" in r)
    assert row.split()[1:] == ["5", "2", "2", "1", "57%", "3,", "5"]
    fr = FileReporter6502(src)
    assert fr.missing_arc_description(2, 3) == "line 2 ($0802) didn't jump to line 3 ($0804)"
    assert fr.exit_counts() == {1: 1, 2: 2, 3: 1, 4: 1, 5: 1}
    assert fr.arc_description(5, -5) == "exit"


def test_combine_union_and_contexts(tmp_path):
    data = tmp_path / "d"
    for value in (0, 1):
        mpu, cov = _branch_run(value)
        _, src = _write(tmp_path, mpu, cov)
        write_data(cov, [src], data_file=str(data), context="m10=%d" % value, suffix=True)
    assert len(list(tmp_path.glob("d.*"))) == 2
    c = _cov(tmp_path, data)
    c.combine([str(tmp_path)])
    c.save()
    c, _, missing, branches = _analysis(tmp_path, data, src)
    assert missing == [5] and branches == {2: (2, 2)}
    assert c.get_data().measured_contexts() == {"m10=0", "m10=1"}
    assert c.get_data().file_tracer(src) == PLUGIN_NAME


def test_cli_report_fail_under(tmp_path):
    mpu, cov = _branch_run(0)
    _, src = _write(tmp_path, mpu, cov)
    write_data(cov, src, data_file=str(tmp_path / ".coverage-6502"))
    (tmp_path / ".coveragerc").write_text(
        "[run]\nplugins = jennings.coverage.plugin\ndata_file = .coverage-6502\nbranch = True\n"
    )
    run = [sys.executable, "-m", "coverage", "report", "-m"]
    env = dict(os.environ, PYTHONPATH=str(Path(jennings.__file__).parents[1]))
    kw = {"cwd": tmp_path, "env": env, "capture_output": True, "text": True}
    ok = subprocess.run(run + ["--fail-under=50"], check=False, **kw)
    assert ok.returncode == 0, ok.stderr
    assert ok.stdout.splitlines()[2].split()[1:] == ["5", "2", "2", "1", "57%", "3,", "5"]
    bad = subprocess.run(run + ["--fail-under=90"], check=False, **kw)
    assert bad.returncode == 2


def test_static_discovery_of_unexecuted_code(tmp_path):
    mpu = _mpu(["LDA $10", "BEQ $0807", "JSR $0900", "NOP"], m10=0)
    mpu.memory[0x0900:0x0902] = [0xE8, 0x60]
    cov = _run_to(mpu, 0x0808)
    listing, src = _write(tmp_path, mpu, cov)
    callee = listing.line(0x0900)
    assert listing.line(0x0804) == 3 and callee == 6 and listing.line(0x0901) == 7
    assert [-callee, callee] in listing.arcs.tolist() and [3, 4] in listing.arcs.tolist()
    assert "JSR L0900" in listing.source and "L0900: INX" in listing.source
    write_data(cov, src, data_file=str(tmp_path / "d"))
    _, _, missing, branches = _analysis(tmp_path, tmp_path / "d", src)
    assert missing == [3, 5, 6, 7] and branches == {2: (2, 1)}


def test_rts_dispatch_target_from_edges(tmp_path):
    mpu = _mpu(["LDA #$08", "PHA", "LDA #$0F", "PHA", "RTS"])
    mpu.memory[0x0810] = 0xEA
    cov = _run_to(mpu, 0x0810)
    assert not cov.flags[0x0810]
    assert Listing.disassemble(mpu.memory, [ORG]).line(0x0810) is None
    listing, src = _write(tmp_path, mpu, cov)
    rts, nop = listing.line(0x0806), listing.line(0x0810)
    assert nop is not None and [-nop, nop] in listing.arcs.tolist()
    assert (rts, -rts) in executed_arcs(cov, src) and (rts, nop) not in executed_arcs(cov, src)
    mpu.step()
    assert (rts, nop) in executed_arcs(cov, src)


def test_interrupt_entry_and_jmp_indirect(tmp_path):
    mpu = _mpu(["CLI", "JMP ($3000)"])
    mpu.memory[0x3000:0x3002] = [0x00, 0x09]
    mpu.memory[0xFFFE:0x10000] = [0x00, 0x0A]
    mpu.memory[0x0A00] = 0x40
    mpu.memory[0x0900] = 0xEA
    mpu.step()
    mpu.irq()
    _run_to(mpu, 0x0901)
    listing, src = _write(tmp_path, mpu, mpu.cov)
    rti, jmp, tgt = listing.line(0x0A00), listing.line(0x0801), listing.line(0x0900)
    assert {(-rti, rti), (rti, -rti), (jmp, tgt), (rti, jmp)} <= executed_arcs(mpu.cov, src)
    assert listing.source.splitlines()[jmp - 1].endswith("JMP ($3000)")


def test_byte_rows_not_executable(tmp_path):
    mpu, cov = _branch_run(1)
    mpu.memory[0x0808:0x0812] = range(1, 11)
    listing, src = _write(tmp_path, mpu, cov, ranges=[(0x0800, 0x0811), (0x0810, 0x0813)])
    rows = listing.source.splitlines()
    assert rows[5:] == [
        "0808                   .byte $01,$02,$03,$04,$05,$06,$07,$08",
        "0810                   .byte $09,$0A,$00,$00",
    ]
    assert listing.line(0x0808) is None and listing.lines[:, 0].tolist() == [1, 2, 3, 4, 5]
    write_data(cov, src, data_file=str(tmp_path / "d"))
    _, statements, missing, _ = _analysis(tmp_path, tmp_path / "d", src)
    assert statements == [1, 2, 3, 4, 5] and missing == [5]


def test_illegal_opcode_text():
    mpu = _mpu(["LAX $10", "SLO $2000,X", "ISC ($20),Y", "JAM"])
    listing = Listing.disassemble(mpu.memory[: ORG + 0x10], [ORG])
    texts = [r[23:] for r in listing.source.splitlines()]
    assert texts == ["LAX $10", "SLO $2000,X", "ISC ($20),Y", "JAM"]
    assert listing.arcs.tolist() == [[-1, 1], [1, 2], [2, 3], [3, 4], [4, -4]]


def test_overlapping_starts_prefer_executed():
    prog = ["LDA $10", "BEQ $0805"]
    covs = []
    for value in (0, 1):
        mpu = _mpu(prog, m10=value)
        mpu.memory[0x0804:0x0808] = [0x2C, 0xA9, 0x02, 0xEA]
        covs.append(_run_to(mpu, 0x0808))
    one = Listing.disassemble(mpu.memory, [ORG], covs[0])
    assert one.line(0x0804) is None and one.line(0x0805) == 3
    assert one.arcs.tolist() == [[-1, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, -5]]
    both = Listing.disassemble(mpu.memory, [ORG], CoverageMap().merge(covs[0]).merge(covs[1]))
    assert [both.line(a) for a in (0x0804, 0x0805, 0x0807)] == [3, 4, 5]
    assert "BIT $02a9" in both.source and "L0805: LDA #$02" in both.source


def test_branch_to_self(tmp_path):
    mpu = CoverageMPU(pc=ORG)
    mpu.memory[ORG : ORG + 3] = [0x50, 0xFE, 0x02]
    mpu.p |= MPU.OVERFLOW
    mpu.step()
    mpu.step()
    listing, src = _write(tmp_path, mpu, mpu.cov)
    assert listing.arcs.tolist() == [[-1, 1], [1, -1], [1, 2], [2, -2]]
    write_data(mpu.cov, src, data_file=str(tmp_path / "d"))
    c, _, missing, branches = _analysis(tmp_path, tmp_path / "d", src)
    assert not missing and branches == {1: (2, 1)}
    fr = FileReporter6502(src)
    assert fr.missing_arc_description(1, -1) == "line 1 ($0800) didn't jump to line 1 ($0800)"
    c.html_report(directory=str(tmp_path / "html"))
    assert (
        "line 1 ($0800) didn't jump to line 1 ($0800)"
        in next((tmp_path / "html").glob("*prog_s.html")).read_text()
    )


def test_html_xml_lcov(tmp_path):
    mpu, cov = _branch_run(0)
    _, src = _write(tmp_path, mpu, cov)
    write_data(cov, src, data_file=str(tmp_path / "d"))
    c = _cov(tmp_path, tmp_path / "d")
    assert c.html_report(directory=str(tmp_path / "html")) == pytest.approx(400 / 7)
    page = next((tmp_path / "html").glob("*prog_s.html")).read_text()
    assert "line 2 ($0802) didn't jump to line 3 ($0804)" in page
    c.xml_report(outfile=str(tmp_path / "c.xml"))
    assert 'condition-coverage="50% (1/2)"' in (tmp_path / "c.xml").read_text()
    c.lcov_report(outfile=str(tmp_path / "c.lcov"))
    lcov = (tmp_path / "c.lcov").read_text()
    assert "BRDA:2,0,jump to line 3 ($0804),0" in lcov
    assert "BRDA:2,0,jump to line 4 ($0806),1" in lcov


def test_empty_run_touches_file(tmp_path):
    mpu, _ = _branch_run(0)
    _, src = _write(tmp_path, mpu, None)
    write_data(CoverageMap(), src, data_file=str(tmp_path / "d"))
    _, statements, missing, _ = _analysis(tmp_path, tmp_path / "d", src)
    assert statements == missing == [1, 2, 3, 4, 5]


def test_sidecar_and_plugin():
    plugin = Plugin6502()
    assert plugin.file_tracer("x.py") is None
    assert PLUGIN_NAME == "jennings.coverage.plugin.Plugin6502"


def test_sidecar_roundtrip(tmp_path):
    mpu, cov = _branch_run(0)
    listing, src = _write(tmp_path, mpu, cov)
    side = load_sidecar(src)
    assert np.array_equal(side["lines"], listing.lines)
    assert np.array_equal(side["arcs"], listing.arcs)
    assert np.array_equal(side["transitions"], listing.transitions)
    assert (tmp_path / "prog.s").read_text() == listing.source


def test_jmp_absolute_and_self_jump():
    mpu = _mpu(["JMP $0805", "NOP", "NOP", "JMP $0805"])
    listing = Listing.disassemble(mpu.memory, [ORG])
    assert listing.lines.tolist() == [[1, 0x0800], [2, 0x0805]]
    assert listing.arcs.tolist() == [[-1, 1], [1, 2], [2, -2]]
    assert listing.source.splitlines()[1].endswith("L0805: JMP L0805")
