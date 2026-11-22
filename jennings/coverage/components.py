"""Autonomous components of a repeatedly called, input-free routine, their cycles, and the calls
that cover everything it can do.

:func:`learn` taints a call's state with the state before it natively, :func:`cycles` measures
each closure's first repeat on a native :func:`trajectory`, :func:`cover_calls` combines them.
"""

import math
from dataclasses import dataclass, field, replace

import networkx as nx
import numpy as np
from tqdm import tqdm

from jennings.coverage.cfg import control_dependence
from jennings.dataflow import REGISTERS
from jennings.devices.mpu6502 import REG, Learner, NativeMPU
from jennings.memory import NativeMemory

__all__ = [
    "REG",
    "STACK",
    "IO",
    "LOAD",
    "STORE",
    "CONTROL",
    "Components",
    "Cycles",
    "location",
    "machine",
    "learn",
    "closures",
    "trajectory",
    "cycles",
    "widen",
    "cover_calls",
]

STACK = range(0x100, 0x200)
IO = range(0xD000, 0xE000)
MAX_CYCLES = 1_000_000
CHUNK = 4096
LOAD, STORE, CONTROL = "r", "w", "b"
_KINDS = (LOAD, STORE, CONTROL)


def location(loc):
    """``loc`` as text: ``$hhhh`` for an address, the register name for :data:`REG` + index."""
    return REGISTERS.get(loc - REG, "r%d" % (loc - REG)) if loc >= REG else "$%04X" % loc


@dataclass(frozen=True)
class Components:
    """What :func:`learn` found over ``calls`` calls; locations as :func:`location` names them.

    Attributes:
        persistent: the locations some call reads before writing and some call writes.
        deps: each location's entry locations its end-of-call value depends on.
        closures: autonomous sets, each a sink SCC of ``deps`` over ``persistent`` with its
            ancestors.
        feed: the persistent locations tainting a site, with their ancestors.
        feeding: the indices of the closures containing a location that taints a site.
        edges: ``(pc, next)`` of consecutive instructions within a call.
        sites: ``(pc, kind)`` to the locations tainting the address of a load (:data:`LOAD`) or
            store (:data:`STORE`) there, or where control goes from it (:data:`CONTROL`).
        calls: the calls learned from.
    """

    persistent: tuple
    deps: dict
    closures: tuple
    feed: frozenset
    feeding: tuple
    edges: frozenset = field(repr=False)
    sites: dict = field(repr=False)
    calls: int = 0


@dataclass(frozen=True)
class Cycles:
    """First repeats of each closure and of the feed set over a trajectory, and conflicts.

    Attributes:
        closures: closure index to ``(preperiod, period)``, None if it does not repeat.
        feed: the feed set's ``(preperiod, period)`` or None.
        conflicts: closure index or ``"feed"`` to ``(t1, t2, cells)``: entries equal on it with
            unequal successors, and the persistent cells outside it that differ between them.
        calls: the calls the trajectory ran.
    """

    closures: dict
    feed: tuple
    conflicts: dict
    calls: int

    @property
    def orbit(self):
        """``(preperiod, period)`` of all the closures together, None if one does not repeat."""
        if any(c is None for c in self.closures.values()):
            return None
        mu = max((c[0] for c in self.closures.values()), default=0)
        return mu, math.lcm(*(c[1] for c in self.closures.values()))


def machine(state):
    """A :class:`NativeMPU` on a flat copy of ``state``: a :class:`NativeMPU` (RAM and
    registers) or 64 KiB of RAM (bytes-like or :class:`NativeMemory`)."""
    ram = bytes(state.memory if isinstance(state, NativeMPU) else state)
    if len(ram) != 0x10000:
        raise ValueError("state holds 64 KiB of RAM")
    m = NativeMPU(memory=NativeMemory(ram))
    if isinstance(state, NativeMPU):
        m.a, m.x, m.y, m.sp, m.p = state.a, state.x, state.y, state.sp, state.p
    return m


def _drive(run, calls, desc):
    """``run(n)`` over ``calls`` calls in chunks, with progress; raises if a call fails."""
    with tqdm(
        total=calls, desc=desc, unit="call", disable=calls <= CHUNK, mininterval=1
    ) as progress:
        done = 0
        while done < calls:
            n = min(CHUNK, calls - done)
            stop = run(n)
            if stop != "sp":
                raise RuntimeError("a call after %d ended at %r, not its RTS" % (done, stop))
            done += n
            progress.update(n)


def _excluded(loc, exclude):
    return loc < REG and any(loc in r for r in exclude)


def closures(deps, cells):
    """``(closures, graph)``: each sink SCC of the ``deps`` graph (``src -> loc``) over
    ``cells`` with its ancestors, smallest first, and the graph."""
    cells = set(cells)
    g = nx.DiGraph()
    g.add_nodes_from(cells)
    g.add_edges_from((s, c) for c in cells for s in deps.get(c, ()) if s in cells and s != c)
    cond = nx.condensation(g)
    out = []
    for n in cond:
        if cond.out_degree(n) == 0:
            members = set(cond.nodes[n]["members"])
            for a in nx.ancestors(cond, n):
                members |= cond.nodes[a]["members"]
            out.append(frozenset(members))
    return tuple(sorted(out, key=lambda c: (len(c), min(c)))), g


def _with_ancestors(g, cells):
    out = set(cells)
    for c in cells:
        out |= nx.ancestors(g, c)
    return frozenset(out)


def _feeding(cls, tainted):
    return tuple(i for i, c in enumerate(cls) if c & tainted)


def _tainted(comps):
    return set().union(*comps.sites.values()) & set(comps.persistent)


def learn(state, pc, calls, *, acc=0, max_cycles=MAX_CYCLES, remap=None, exclude=(STACK, IO)):
    """:class:`Components` of ``calls`` calls of ``pc`` with ``A = acc`` from ``state`` (see
    :func:`machine`), ``remap`` (a :class:`~jennings.devices.mpu6502.Remap`) applied; addresses
    in ``exclude`` are never persistent."""
    m = machine(state)
    end, _states, acc_log, edge_keys = m.orbit(pc, acc, max_cycles, calls, remap=remap)
    if end == "jam":
        raise RuntimeError("a call jammed")
    keys = np.asarray(acc_log, np.uint64)
    written = np.unique(keys[(keys >> 16) & 1 == 1] & 0xFFFF).tolist()
    ek = np.asarray(edge_keys, np.uint64)
    pairs = list(zip((ek >> 16).tolist(), (ek & 0xFFFF).tolist()))
    exits = {b for _a, b in pairs} - {a for a, _b in pairs}
    cd = control_dependence(pairs) if exits else {}
    learner = Learner({p: sorted(b) for p, b in cd.items()}, written)
    m = machine(state)
    _drive(lambda n: m.learn(learner, pc, acc, max_cycles, n, remap), calls, "learn")
    pers, dep, edg, sit = learner.learned()
    persistent = tuple(
        loc for loc in np.frombuffer(pers, np.uint32).tolist() if not _excluded(loc, exclude)
    )
    d = np.frombuffer(dep, np.uint64)
    deps = {}
    for loc, src in zip((d >> 32).tolist(), (d & 0xFFFFFFFF).tolist()):
        deps.setdefault(loc, set()).add(src)
    s = np.frombuffer(sit, np.uint64)
    sites = {}
    for key, loc in zip((s >> 17).tolist(), (s & 0x1FFFF).tolist()):
        sites.setdefault((key >> 2, _KINDS[key & 3]), set()).add(loc)
    e = np.frombuffer(edg, np.uint64)
    cls, g = closures(deps, persistent)
    comps = Components(
        persistent=persistent,
        deps={k: frozenset(v) for k, v in deps.items()},
        closures=cls,
        feed=frozenset(),
        feeding=(),
        edges=frozenset(zip((e >> 16).tolist(), (e & 0xFFFF).tolist())),
        sites={k: frozenset(v) for k, v in sites.items()},
        calls=calls,
    )
    tainted = _tainted(comps)
    return replace(comps, feed=_with_ancestors(g, tainted), feeding=_feeding(cls, tainted))


def trajectory(state, pc, cells, calls, *, acc=0, max_cycles=MAX_CYCLES, remap=None):
    """uint8 ``[calls + 1, len(cells)]``: the values of the locations ``cells`` at the entry of
    each of ``calls + 1`` calls of ``pc`` from ``state`` (see :func:`learn`)."""
    m = machine(state)
    rows = []

    def run(k):
        stop, r = m.trajectory(pc, acc, max_cycles, k, cells, remap)
        rows.append(r)
        return stop

    _drive(run, calls + 1, "trajectory")
    return np.concatenate(rows)


def _ids(rows, cols):
    """Per row, the index of the first row equal to it on ``cols``."""
    if not cols:
        return np.zeros(len(rows), np.int64)
    sub = np.ascontiguousarray(rows[:, cols])
    v = sub.view(np.dtype((np.void, sub.shape[1]))).ravel()
    _, first, inv = np.unique(v, return_index=True, return_inverse=True)
    return first[inv.ravel()]


def _cycle(f):
    """``(preperiod, period)`` of the first repeat in the first-index array ``f``, or None."""
    t = np.flatnonzero(f < np.arange(len(f)))
    return None if t.size == 0 else (int(f[t[0]]), int(t[0] - f[t[0]]))


def _conflict(rows, f, cols):
    """``(t1, t2, columns)`` of the first entries equal on ``cols`` with unequal successors."""
    n = len(f) - 1
    bad = np.flatnonzero(f[f[:n] + 1] != f[1 : n + 1]) if n > 0 else np.empty(0, np.int64)
    if bad.size == 0:
        return None
    t1, t2 = int(f[bad[0]]), int(bad[0])
    return t1, t2, set(np.flatnonzero(rows[t1] != rows[t2]).tolist()) - set(cols)


def _measure(rows, comps):
    col = {c: i for i, c in enumerate(comps.persistent)}
    out, conflicts = {}, {}
    for key, cells in [*enumerate(comps.closures), ("feed", comps.feed)]:
        cols = sorted(col[c] for c in cells)
        f = _ids(rows, cols)
        bad = _conflict(rows, f, cols)
        if bad is not None:
            conflicts[key] = (bad[0], bad[1], frozenset(comps.persistent[i] for i in bad[2]))
        out[key] = _cycle(f) if cols else (0, 1)
    feed = out.pop("feed")
    return Cycles(closures=out, feed=feed, conflicts=conflicts, calls=len(rows) - 1)


def cycles(state, pc, comps, limit, *, acc=0, max_cycles=MAX_CYCLES, remap=None, rows=None):
    """:class:`Cycles` of ``comps`` over a :func:`trajectory` of ``limit`` calls (or ``rows``,
    one over ``comps.persistent``)."""
    if rows is None:
        rows = trajectory(
            state, pc, comps.persistent, limit, acc=acc, max_cycles=max_cycles, remap=remap
        )
    return _measure(rows, comps)


def widen(comps, cyc):
    """``comps`` with each conflicting closure (and the feed set) joined by the cells that told
    its conflicting entries apart and their ancestors; closures inside another are dropped."""
    _, g = closures(comps.deps, comps.persistent)
    cls, feed = list(comps.closures), comps.feed
    for key, (_t1, _t2, cells) in cyc.conflicts.items():
        if key == "feed":
            feed = feed | _with_ancestors(g, cells)
        else:
            cls[key] = cls[key] | _with_ancestors(g, cells)
    cls = sorted(set(cls), key=lambda c: (len(c), min(c)))
    cls = tuple(c for c in cls if not any(c < d for d in cls))
    return replace(comps, closures=cls, feed=feed, feeding=_feeding(cls, _tainted(comps)))


def cover_calls(state, pc, comps, limit, *, acc=0, max_cycles=MAX_CYCLES, remap=None):
    """``(calls, comps)``: the calls from ``state`` after which every data access and control
    edge has been seen (the feed set's preperiod plus period; None if it does not repeat within
    ``limit``), with ``comps`` widened until autonomous on the trajectory."""
    rows = trajectory(
        state, pc, comps.persistent, limit, acc=acc, max_cycles=max_cycles, remap=remap
    )
    while True:
        cyc = _measure(rows, comps)
        if not cyc.conflicts:
            break
        wider = widen(comps, cyc)
        if wider == comps:
            raise RuntimeError("conflicts no persistent cell explains: %r" % cyc.conflicts)
        comps = wider
    return (None if cyc.feed is None else cyc.feed[0] + cyc.feed[1]), comps
