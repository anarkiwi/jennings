"""Autonomous components of a repeatedly called, input-free routine, their cycles, and the calls
that cover everything it can do.

:func:`learn` taints a call's state with the state before it natively, :func:`cycles` measures
each closure's first repeat on a native :func:`trajectory`, :func:`cover_calls` combines them.
:func:`explicit` gives the data-only flows, value apart from address, constants by where read.
"""

import math
from dataclasses import dataclass, field, replace

import networkx as nx
import numpy as np
from tqdm import tqdm

from jennings.coverage.cfg import control_dependence
from jennings.dataflow import REGISTERS
from jennings.devices.mpu6502 import CONST, REG, Learner, NativeMPU
from jennings.memory import NativeMemory

__all__ = [
    "REG",
    "CONST",
    "STACK",
    "IO",
    "LOAD",
    "STORE",
    "CONTROL",
    "VALUE",
    "Components",
    "Cycles",
    "Flows",
    "location",
    "machine",
    "control",
    "learn",
    "explicit",
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
LOAD, STORE, CONTROL, VALUE = "r", "w", "b", "v"
_KINDS = (LOAD, STORE, CONTROL, VALUE)
_BITS = 18


def location(loc):
    """``loc`` as text: ``$hhhh`` for an address, the register name for :data:`REG` + index,
    ``#$hhhh`` for :data:`CONST` + pc, a constant loaded at ``pc``."""
    if loc >= CONST:
        return "#$%04X" % (loc - CONST)
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
            store (:data:`STORE`) there, where control goes from it (:data:`CONTROL`), or the
            byte a data store there stores (:data:`VALUE`).
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


@dataclass(frozen=True)
class Flows:
    """What :func:`explicit` found over ``calls`` calls, by data alone.

    Attributes:
        deps: each location written to the entry locations (and :data:`CONST` + pc constants)
            its end-of-call value depends on by data, itself included.
        sites: ``(pc, kind)`` as :attr:`Components.sites`, each taint explicit.
        calls: the calls learned from.
    """

    deps: dict
    sites: dict = field(repr=False)
    calls: int = 0


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
    sites = (v for (_pc, kind), v in comps.sites.items() if kind != VALUE)
    return set().union(*sites) & set(comps.persistent)


def _pairs(edges):
    e = np.frombuffer(edges, np.uint64)
    return list(zip((e >> 16).tolist(), (e & 0xFFFF).tolist()))


def control(state, pc, calls, *, acc=0, max_cycles=MAX_CYCLES, remap=None):
    """``(control, written)`` of ``calls`` calls of ``pc`` from ``state``: ``{pc: branches}`` it
    is directly control dependent on (:func:`~jennings.coverage.cfg.control_dependence`, each
    call's last instruction leading to the exit), and the memory the calls write."""
    learner = Learner({}, [])
    m = machine(state)
    _drive(lambda n: m.learn(learner, pc, acc, max_cycles, n, remap), calls, "edges")
    got = learner.learned()
    exits = np.frombuffer(got["exits"], np.uint16).tolist()
    cd = control_dependence(_pairs(got["edges"]) + [(x, -1) for x in exits])
    return {p: sorted(b) for p, b in cd.items()}, np.frombuffer(got["written"], np.uint16).tolist()


def _learned(state, pc, calls, cd, written, explicit_, acc, max_cycles, remap):
    """The native ``Learner`` result of ``calls`` calls and its ``(deps, sites)``: ``{loc: set}``
    self included, ``{(pc, kind): set}``."""
    learner = Learner(cd, written, explicit_)
    m = machine(state)
    desc = "explicit" if explicit_ else "learn"
    _drive(lambda n: m.learn(learner, pc, acc, max_cycles, n, remap), calls, desc)
    got = learner.learned()
    d = np.frombuffer(got["deps"], np.uint64)
    deps = {}
    for loc, src in zip((d >> 32).tolist(), (d & 0xFFFFFFFF).tolist()):
        deps.setdefault(loc, set()).add(src)
    s = np.frombuffer(got["sites"], np.uint64)
    sites = {}
    for key, loc in zip((s >> _BITS).tolist(), (s & ((1 << _BITS) - 1)).tolist()):
        sites.setdefault((key >> 2, _KINDS[key & 3]), set()).add(loc)
    return got, deps, {k: frozenset(v) for k, v in sites.items()}


def learn(state, pc, calls, *, acc=0, max_cycles=MAX_CYCLES, remap=None, exclude=(STACK, IO)):
    """:class:`Components` of ``calls`` calls of ``pc`` with ``A = acc`` from ``state`` (see
    :func:`machine`), ``remap`` (a :class:`~jennings.devices.mpu6502.Remap`) applied; addresses
    in ``exclude`` are never persistent."""
    cd, written = control(state, pc, calls, acc=acc, max_cycles=max_cycles, remap=remap)
    got, deps, sites = _learned(state, pc, calls, cd, written, False, acc, max_cycles, remap)
    persistent = tuple(
        loc
        for loc in np.frombuffer(got["persistent"], np.uint32).tolist()
        if not _excluded(loc, exclude)
    )
    deps = {k: v - {k} for k, v in deps.items() if v - {k}}
    cls, g = closures(deps, persistent)
    comps = Components(
        persistent=persistent,
        deps={k: frozenset(v) for k, v in deps.items()},
        closures=cls,
        feed=frozenset(),
        feeding=(),
        edges=frozenset(_pairs(got["edges"])),
        sites=sites,
        calls=calls,
    )
    tainted = _tainted(comps)
    return replace(comps, feed=_with_ancestors(g, tainted), feeding=_feeding(cls, tainted))


def explicit(state, pc, calls, *, acc=0, max_cycles=MAX_CYCLES, remap=None):
    """:class:`Flows` of ``calls`` calls of ``pc`` (as :func:`learn`): the explicit flows, each
    load's value apart from its address, no control; a data load of a location no call writes
    at ``pc`` is the constant :data:`CONST` + ``pc``."""
    _cd, written = control(state, pc, calls, acc=acc, max_cycles=max_cycles, remap=remap)
    _got, deps, sites = _learned(state, pc, calls, {}, written, True, acc, max_cycles, remap)
    return Flows(deps={k: frozenset(v) for k, v in deps.items()}, sites=sites, calls=calls)


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


def _first(sub):
    """Per row of ``sub``, the index of the first row equal to it."""
    v = sub.view(np.dtype((np.void, sub.shape[1]))).ravel()
    _, first, inv = np.unique(v, return_index=True, return_inverse=True)
    return first[inv.ravel()]


def _first_repeat(sub, start=4096):
    """``(preperiod, period)`` of the first row of ``sub`` equal to an earlier one, or None;
    searched on prefixes growing fourfold, so it costs the repeat's position, not ``sub``'s."""
    n = start
    while True:
        m = min(n, len(sub))
        f = _first(sub[:m])
        t = np.flatnonzero(f < np.arange(m))
        if t.size:
            return int(f[t[0]]), int(t[0] - f[t[0]])
        if m == len(sub):
            return None
        n *= 4


def _measure(rows, comps):
    """:class:`Cycles` of ``comps`` on ``rows``: a projection repeating first at
    ``(mu, lam)`` is autonomous on ``rows`` iff it has period ``lam`` from ``mu`` on."""
    col = {c: i for i, c in enumerate(comps.persistent)}
    out, conflicts = {}, {}
    for key, cells in [*enumerate(comps.closures), ("feed", comps.feed)]:
        cols = sorted(col[c] for c in cells)
        if not cols:
            out[key] = (0, 1)
            continue
        sub = np.ascontiguousarray(rows[:, cols])
        out[key] = cyc = _first_repeat(sub)
        if cyc is None:
            continue
        mu, lam = cyc
        bad = np.flatnonzero((sub[mu + lam :] != sub[mu : len(sub) - lam]).any(axis=1))
        if bad.size:
            t1 = mu + int(bad[0]) - 1
            diff = set(np.flatnonzero(rows[t1] != rows[t1 + lam]).tolist()) - set(cols)
            conflicts[key] = (t1, t1 + lam, frozenset(comps.persistent[i] for i in diff))
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


def _autonomous(rows, comps):
    """``(cycles, comps)``: ``comps`` widened until no closure conflicts on ``rows``."""
    while True:
        cyc = _measure(rows, comps)
        if not cyc.conflicts:
            return cyc, comps
        wider = widen(comps, cyc)
        if wider == comps:
            raise RuntimeError("conflicts no persistent cell explains: %r" % cyc.conflicts)
        comps = wider


def cover_calls(
    state, pc, comps, limit, *, acc=0, max_cycles=MAX_CYCLES, remap=None, exclude=(STACK, IO)
):
    """``(calls, comps)``: the calls from ``state`` after which every data access and control
    edge has been seen (the feed set's preperiod plus period; None if it does not repeat within
    ``limit``), ``comps`` relearned over at least ``calls`` calls and widened until autonomous."""
    kw = {"acc": acc, "max_cycles": max_cycles, "remap": remap}
    rows = None
    while True:
        if rows is None or rows.shape[1] != len(comps.persistent):
            rows = trajectory(state, pc, comps.persistent, limit, **kw)
        cyc, comps = _autonomous(rows, comps)
        calls = None if cyc.feed is None else cyc.feed[0] + cyc.feed[1]
        if calls is None or calls <= comps.calls:
            return calls, comps
        wider = learn(state, pc, calls, exclude=exclude, **kw)
        if wider.persistent != comps.persistent:
            rows = None
        comps = wider
