import os
from pathlib import Path

import coverage
import pytest
from coverage import CoverageData

import jennings
from jennings.assembler import Assembler
from jennings.coverage import CoverageMPU
from jennings.coverage.listing import Listing
from jennings.coverage.plugin import PLUGIN_NAME
from jennings.coverage.pytest_plugin import Cov6502
from jennings.coverage.report import executed_arcs

pytest_plugins = ["pytester"]

ORG, END = 0x0800, 0x0807
BRANCH = ["LDA $10", "BEQ $0806", "LDX #$01", "NOP"]

INNER = """
import pytest
from jennings.coverage import CoverageMPU

ORG, END, CODE, SRC = {org}, {end}, {code!r}, {src!r}


def run(mpu, value):
    mpu.memory[ORG : ORG + len(CODE)] = CODE
    mpu.memory[0x10] = value
    while mpu.pc != END:
        mpu.step()
    return mpu


@pytest.mark.xdist_group("a")
@pytest.mark.parametrize("value", [0, 1])
def test_branch(cov6502, value):
    run(cov6502.mpu(pc=ORG), value)
    cov6502.sources(SRC)


@pytest.mark.xdist_group("b")
def test_merge(cov6502):
    mpu = run(cov6502.adopt(CoverageMPU(pc=ORG)), 0)
    cov6502.adopt(mpu.cov)
    run(cov6502.mpu(pc=ORG), 1)
    cov6502.sources(SRC)


@pytest.mark.xdist_group("b")
def test_no_sources(cov6502):
    run(cov6502.mpu(pc=ORG), 0)
"""


def _code():
    mpu, code, pc = CoverageMPU(pc=ORG), [], ORG
    asm = Assembler(mpu)
    for st in BRANCH:
        code += asm.assemble(st, pc)
        pc = ORG + len(code)
    return code


def _run(value):
    mpu = CoverageMPU(pc=ORG)
    code = _code()
    mpu.memory[ORG : ORG + len(code)] = code
    mpu.memory[0x10] = value
    while mpu.pc != END:
        mpu.step()
    return mpu


@pytest.fixture(name="project")
def _project(pytester):
    src = pytester.path / "prog.s"
    Listing.disassemble(_run(0).memory, [ORG]).write(src)
    pytester.makepyfile(test_inner=INNER.format(org=ORG, end=END, code=_code(), src=str(src)))
    (pytester.path / ".coveragerc").write_text(
        "[run]\nplugins = jennings.coverage.plugin\nbranch = True\n"
    )
    return pytester, str(src)


def _contexts(path, src):
    data = CoverageData(str(path))
    data.read()
    out = {}
    for ctx in data.measured_contexts():
        data.set_query_context(ctx)
        out[ctx] = (data.lines(src), data.arcs(src))
    return out


def _combined(pytester):
    p = pytester.path
    c = coverage.Coverage(data_file=str(p / ".coverage-6502"), config_file=str(p / ".coveragerc"))
    c.combine([str(p)])
    c.save()
    return c


def _expect(src):
    a0, a1 = (executed_arcs(_run(v).cov, src) for v in (0, 1))
    branch = "test_inner.py::test_branch[%d]"
    return {
        branch % 0: ({1, 2, 4}, a0),
        branch % 1: ({1, 2, 3, 4}, a1),
        "test_inner.py::test_merge": ({1, 2, 3, 4}, a0 | a1),
    }


def _check(c, src):
    data = c.get_data()
    assert data.file_tracer(src) == PLUGIN_NAME
    _, statements, _, missing, _ = c.analysis2(src)
    assert statements == [1, 2, 3, 4, 5] and missing == [5]
    assert c.branch_stats(src) == {2: (2, 2)}


def test_inert_without_option(project):
    pytester, _ = project
    pytester.runpytest_inprocess().assert_outcomes(passed=4)
    assert not list(pytester.path.glob(".coverage-6502*"))


def test_contexts_per_test(project):
    pytester, src = project
    stale = pytester.path / ".coverage-6502.gw7"
    stale.write_bytes(b"")
    pytester.runpytest_inprocess("--cov6502").assert_outcomes(passed=4)
    files = sorted(p.name for p in pytester.path.glob(".coverage-6502*"))
    assert files == [".coverage-6502.main"]
    got = _contexts(pytester.path / files[0], src)
    assert {k: (set(lines), set(arcs)) for k, (lines, arcs) in got.items()} == _expect(src)
    _check(_combined(pytester), src)


def test_custom_data_file(project):
    pytester, src = project
    pytester.runpytest_inprocess("--cov6502=sub.dat", "-k", "merge").assert_outcomes(passed=1)
    assert set(_contexts(pytester.path / "sub.dat.main", src)) == {"test_inner.py::test_merge"}


def test_xdist_workers_combine(project, monkeypatch):
    pytester, src = project
    root = str(Path(jennings.__file__).parents[1])
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(filter(None, (root, os.getenv("PYTHONPATH")))))
    result = pytester.runpytest_subprocess("--cov6502", "-n", "2", "--dist", "loadgroup")
    result.assert_outcomes(passed=4)
    workers = sorted(p.name for p in pytester.path.glob(".coverage-6502.*"))
    assert workers == [".coverage-6502.gw0", ".coverage-6502.gw1"]
    c = _combined(pytester)
    assert not list(pytester.path.glob(".coverage-6502.*"))
    assert c.get_data().measured_contexts() == set(_expect(src))
    _check(c, src)


def test_fixture_object():
    rec = Cov6502()
    a, b = _run(0), _run(1)
    assert rec.adopt(a) is a and rec.adopt(b._vm) is b._vm and rec.adopt(a.cov) is a.cov
    assert rec.sources(Path("x.s"), "y.s").paths == ["x.s", "y.s"]
    merged = rec.coverage()
    assert merged == _run(0).cov.merge(_run(1).cov)
    assert isinstance(rec.mpu(pc=ORG), CoverageMPU) and len(rec.recorded) == 4
