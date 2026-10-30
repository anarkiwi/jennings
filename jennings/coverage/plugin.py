"""coverage.py plugin reporting 6502 sources from their ``<source>.jcov`` sidecars
(``lines`` [[line, address]], static ``arcs`` [[from, to]] and ``transitions``
[[address, flag mask, from, to]]; see docs/coverage.md). Enable with
``[run] plugins = jennings.coverage.plugin``."""

import json
from collections import Counter

import coverage
import numpy as np

from jennings.coverage import FALLTHRU, TAKEN

__all__ = [
    "PLUGIN_NAME",
    "SIDECAR",
    "Plugin6502",
    "FileReporter6502",
    "load_sidecar",
    "save_sidecar",
]

SIDECAR = ".jcov"
_WIDTH = {"lines": 2, "arcs": 2, "transitions": 4}


def save_sidecar(source, **tables):
    """Write ``tables`` (``lines``, ``arcs``, ``transitions``) next to ``source``."""
    with open(str(source) + SIDECAR, "w", encoding="utf-8") as f:
        json.dump({k: np.asarray(tables[k]).tolist() for k in _WIDTH}, f)


def load_sidecar(source):
    """Read the sidecar of ``source`` as ``{name: int64 array}``."""
    with open(str(source) + SIDECAR, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: np.array(raw[k], np.int64).reshape(-1, w) for k, w in _WIDTH.items()}


class FileReporter6502(coverage.FileReporter):
    def __init__(self, filename):
        super().__init__(filename)
        side = load_sidecar(filename)
        lines = side["lines"]
        self._addr = dict(zip(lines[::-1, 0].tolist(), lines[::-1, 1].tolist()))
        self._arcs = set(map(tuple, side["arcs"].tolist()))
        t = side["transitions"]
        loop = ((t[:, 1] & (TAKEN | FALLTHRU)) != 0) & (t[:, 3] == -t[:, 2])
        self._loops = set(map(tuple, t[loop, 2:].tolist()))

    def lines(self):
        return set(self._addr)

    def arcs(self):
        return self._arcs

    def exit_counts(self):
        return Counter(a for a, _ in self._arcs if a > 0)

    def _at(self, line):
        return "line %d ($%04X)" % (line, self._addr[line])

    def arc_description(self, start, end):
        if end < 0 and (start, end) not in self._loops:
            return "exit"
        return "jump to " + self._at(abs(end))

    def missing_arc_description(self, start, end, executed_arcs=None):
        return "%s didn't %s" % (self._at(start), self.arc_description(start, end))


class Plugin6502(coverage.CoveragePlugin):
    """File-tracer plugin that never traces Python and reports sidecar-backed sources."""

    def file_reporter(self, filename):
        return FileReporter6502(filename)


PLUGIN_NAME = "%s.%s" % (__name__, Plugin6502.__name__)


def coverage_init(reg, options):
    reg.add_file_tracer(Plugin6502())
