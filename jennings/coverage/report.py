"""Convert a ``CoverageMap`` into coverage.py arc data for sidecar-backed 6502 sources."""

import os

import numpy as np
from coverage import CoverageData

from jennings.coverage import EXEC
from jennings.coverage.plugin import PLUGIN_NAME, load_sidecar

__all__ = ["executed_arcs", "write_data"]


def executed_arcs(cov, source):
    """Arcs of ``source`` executed in ``cov``: sidecar transitions whose address has the
    transition's flag, plus dynamic ``edges`` between mapped lines, keeping only arcs into
    executed lines or exits."""
    side = load_sidecar(source)
    t, lines = side["transitions"], side["lines"]
    arcs = t[(cov.flags[t[:, 0]] & t[:, 1]) != 0, 2:]
    if cov.edges:
        line_at = np.zeros(0x10000, np.int64)
        line_at[lines[:, 1]] = lines[:, 0]
        e = line_at[np.array(sorted(cov.edges), np.int64).reshape(-1, 2)]
        arcs = np.concatenate((arcs, e[(e != 0).all(1)]))
    ran = np.zeros(lines[:, 0].max(initial=0) + 1, bool)
    ran[lines[(cov.flags[lines[:, 1]] & EXEC) != 0, 0]] = True
    to = arcs[:, 1]
    return set(map(tuple, arcs[(to < 0) | ran[np.maximum(to, 0)]].tolist()))


def write_data(cov, sources, data_file=".coverage-6502", context=None, suffix=None):
    """Add ``cov``'s executed arcs for ``sources`` (a path or iterable of paths) to the
    coverage.py data file; ``suffix=True`` writes a per-process file for ``coverage combine``."""
    if isinstance(sources, (str, os.PathLike)):
        sources = [sources]
    arcs = {os.path.abspath(s): executed_arcs(cov, s) for s in sources}
    data = CoverageData(basename=data_file, suffix=suffix)
    data.set_context(context)
    data.add_arcs(arcs)
    data.touch_files(list(arcs), PLUGIN_NAME)
    data.write()
    return data
