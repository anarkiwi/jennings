"""pytest plugin recording 6502 coverage per test. ``--cov6502[=DATAFILE]`` writes each test's
arcs for the sources it names, with the test's node id as the coverage.py context, to
``DATAFILE.<worker>`` (``gwN`` under xdist, ``main`` otherwise) for ``coverage combine``.
The ``cov6502`` fixture creates or adopts the MPUs to record and names the reported sources."""

# pylint: disable=import-outside-toplevel

import glob
import os

import pytest

__all__ = ["DATA_FILE", "Cov6502", "cov6502"]

DATA_FILE = ".coverage-6502"
_DATA = pytest.StashKey()


def pytest_addoption(parser):
    parser.getgroup("cov6502", "6502 coverage").addoption(
        "--cov6502",
        nargs="?",
        const=DATA_FILE,
        default=None,
        metavar="DATAFILE",
        help="record 6502 coverage per test to DATAFILE.<worker> (default %s)" % DATA_FILE,
    )


def pytest_configure(config):
    if config.option.cov6502:
        config.option.cov6502 = os.path.abspath(config.option.cov6502)


def pytest_sessionstart(session):
    """Outside xdist workers, remove the parallel data files of earlier runs."""
    data_file = session.config.option.cov6502
    if data_file and not hasattr(session.config, "workerinput"):
        for path in glob.glob(glob.escape(data_file) + ".*"):
            os.remove(path)


class Cov6502:
    """Coverage maps of the MPUs a test creates or adopts, and the sources to report them on."""

    def __init__(self):
        self.recorded = []
        self.paths = []

    def mpu(self, **kwargs):
        """A new ``CoverageMPU(**kwargs)``, recorded."""
        from jennings.coverage._core import CoverageMPU

        return self.adopt(CoverageMPU(**kwargs))

    def adopt(self, obj):
        """Record ``obj`` (a ``CoverageMPU``, ``CoverageVM`` or ``CoverageMap``); its map is
        read at teardown, so a replaced ``obj.cov`` is honoured."""
        self.recorded.append(obj)
        return obj

    def sources(self, *paths):
        """Report on the sidecar-backed ``paths``."""
        self.paths.extend(map(os.fspath, paths))
        return self

    def coverage(self):
        """The union of the recorded coverage maps."""
        from jennings.coverage._core import CoverageMap

        maps = {id(m): m for m in (getattr(o, "cov", o) for o in self.recorded)}
        merged = CoverageMap()
        for m in maps.values():
            merged.merge(m)
        return merged


def _data(config):
    """This process's ``CoverageData``, erased on first use."""
    if _DATA not in config.stash:
        from coverage import CoverageData

        worker = getattr(config, "workerinput", {}).get("workerid", "main")
        config.stash[_DATA] = CoverageData(basename=config.option.cov6502, suffix=worker)
    return config.stash[_DATA]


def pytest_unconfigure(config):
    if _DATA in config.stash:
        config.stash[_DATA].close()


@pytest.fixture
def cov6502(request):
    """``Cov6502`` whose arcs are added under the test's node id at teardown."""
    rec = Cov6502()
    yield rec
    if request.config.option.cov6502 and rec.paths:
        from jennings.coverage.report import write_data

        write_data(
            rec.coverage(), rec.paths, context=request.node.nodeid, data=_data(request.config)
        )
