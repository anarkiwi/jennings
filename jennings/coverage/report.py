"""Convert a ``CoverageMap`` into coverage.py arc data for sidecar-backed 6502 sources."""

import os

import numpy as np
from coverage import CoverageData

from jennings.coverage._core import EXEC
from jennings.coverage.cfg import LEN
from jennings.coverage.plugin import PLUGIN_NAME, load_sidecar

__all__ = ["executed_arcs", "held", "write_data"]


def held(rows, cov):
    """Which obligation ``rows`` ``[kind, cell, a, b]`` hold in ``cov``: ``0`` writer ``a``'s
    value at ``cell`` was executed or consumed, ``1`` ``cell`` was executed holding byte ``a``
    after a code write, ``2`` the instruction at ``b`` ran with none of its bytes written, ``3``
    target ``a`` was taken through pointer ``cell``, ``4`` target ``b`` followed ``a`` there."""
    defs = {(w, c) for w, c, _ in cov.defuse}
    taken = {(p, t) for p, _, t in cov.chains}
    execd = {
        ((pc + i) & 0xFFFF, (packed >> (8 * i)) & 0xFF)
        for pc, packed in cov.variants
        for i in range(LEN[packed & 0xFF])
    }
    rows = np.asarray(rows, np.int64).reshape(-1, 4)
    return np.array(
        [
            (k == 0 and (a, c) in defs)
            or (k == 1 and (c, a) in execd)
            or bool(k == 2 and cov.image[b])
            or (k == 3 and (c, a) in taken)
            or (k == 4 and (c, a, b) in cov.chains)
            for k, c, a, b in rows.tolist()
        ],
        bool,
    )


def executed_arcs(cov, source):
    """Arcs of ``source`` executed in ``cov``: sidecar transitions whose address has the
    transition's flag, plus dynamic ``edges`` between mapped lines, keeping only arcs into
    executed lines or exits; for an obligation listing, the lines with a row that holds."""
    side = load_sidecar(source)
    t, lines, ob = side["transitions"], side["lines"], side["obligations"]
    if ob.size:
        done = np.unique(ob[held(ob[:, 1:], cov), 0])
        return set(map(tuple, np.concatenate((np.c_[-done, done], np.c_[done, -done])).tolist()))
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


def write_data(cov, sources, data_file=".coverage-6502", context=None, suffix=None, data=None):
    """Add ``cov``'s executed arcs for ``sources`` (a path or iterable of paths) to the
    coverage.py data file; ``suffix=True`` writes a per-process file for ``coverage combine``.
    A new file is erased on first write; pass an open ``CoverageData`` as ``data`` to append."""
    if isinstance(sources, (str, os.PathLike)):
        sources = [sources]
    arcs = {os.path.abspath(s): executed_arcs(cov, s) for s in sources}
    if data is None:
        data = CoverageData(basename=data_file, suffix=suffix)
    data.set_context(context)
    data.add_arcs(arcs)
    data.touch_files(list(arcs), PLUGIN_NAME)
    data.write()
    return data
