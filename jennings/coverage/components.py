"""Autonomous components of a repeatedly called, input-free routine, their cycles, and the calls
that cover everything it can do.

:func:`census` runs the calls once, sorting them into classes by what a learner sees of each;
:func:`learn` taints each class's first call with the state before it natively, :func:`cycles`
measures each closure's first repeat on a :func:`trajectory`, :func:`cover_calls` combines them.
:func:`explicit` gives the data-only flows, value apart from address, constants by where read.
"""

import math
from dataclasses import dataclass, field, replace

import networkx as nx
import numpy as np
from tqdm import tqdm

from jennings.coverage.cfg import control_dependence
from jennings.dataflow import REGISTERS
from jennings.devices.mpu6502 import CONST, REG, Census, Learner, NativeMPU, periods_of
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
    "ACCESSES",
    "SITE_BITS",
    "Components",
    "Cycles",
    "Flows",
    "location",
    "machine",
    "control",
    "census",
    "Calls",
    "learn",
    "explicit",
    "condensed",
    "closures",
    "trajectory",
    "cycles",
    "widen",
    "cover_calls",
    "cover",
    "drive",
    "edge_pairs",
    "excluded",
    "spans",
]

STACK = range(0x100, 0x200)
IO = range(0xD000, 0xE000)
MAX_CYCLES = 1_000_000
CHUNK = 4096
LOAD, STORE, CONTROL, VALUE = "r", "w", "b", "v"
#: A census access's kind, by its number in ``pc << 19 | kind << 16 | address``.
ACCESSES = ("load", "stack load", "store", "stack store", "read", "write")
_KINDS = (LOAD, STORE, CONTROL, VALUE)
#: A learner site key is ``(pc << 2 | kind) << SITE_BITS | loc``.
SITE_BITS = 18
_PER = ("stored", "addressed", "gated")


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
        stored: each memory location a data store reaches to the explicit taints of the bytes
            stored there, over every store.
        addressed: as ``stored``, the taints of those stores' addresses.
        gated: as ``stored``, the conditions those stores are under (transitively) and those
            of the branches its store sites are control dependent on that a call ran.
        calls: the calls learned from.
    """

    deps: dict
    sites: dict = field(repr=False)
    stored: dict = field(default_factory=dict, repr=False)
    addressed: dict = field(default_factory=dict, repr=False)
    gated: dict = field(default_factory=dict, repr=False)
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


def drive(run, calls, desc, update=None):
    """``run(n)`` over ``calls`` calls in chunks of at most :data:`CHUNK`, each returning how
    its last call stopped; raises :class:`RuntimeError` where that is not ``"sp"`` (its RTS).

    Progress is a bar named ``desc``, or ``update(n)`` after each chunk where given."""

    def chunks(update):
        done = 0
        while done < calls:
            n = min(CHUNK, calls - done)
            stop = run(n)
            if stop != "sp":
                raise RuntimeError("a call after %d ended at %r, not its RTS" % (done, stop))
            done += n
            update(n)

    if update is not None:
        chunks(update)
        return
    with tqdm(
        total=calls, desc=desc, unit="call", disable=calls <= CHUNK, mininterval=1
    ) as progress:
        chunks(progress.update)


def excluded(locs, bounds):
    """Whether each of the locations ``locs`` is an address in one of the half-open
    ``(lo, hi)`` ``bounds`` (as :func:`spans` gives them)."""
    locs = np.asarray(locs, np.int64)
    out = np.zeros(locs.shape, bool)
    for lo, hi in bounds:
        out |= (locs >= lo) & (locs < hi) & (locs < REG)
    return out


def spans(exclude):
    """``exclude`` (``range`` objects or ``(lo, hi)`` pairs) as sorted half-open ``(lo, hi)``
    tuples; a range of another step than 1 as one span per address in it."""
    out = []
    for r in exclude:
        if isinstance(r, range):
            out.extend([(r.start, r.stop)] if r.step == 1 else [(a, a + 1) for a in r])
        else:
            out.append((int(r[0]), int(r[1])))
    return tuple(sorted(out))


def condensed(deps, cells):
    """``(cond, sets, graph)``: the condensation of the ``deps`` graph (``src -> loc``) over
    ``cells``, each of its nodes' members with its ancestors', and the graph."""
    cells = set(cells)
    g = nx.DiGraph()
    g.add_nodes_from(cells)
    g.add_edges_from((s, c) for c in cells for s in deps.get(c, ()) if s in cells and s != c)
    cond = nx.condensation(g)
    sets = {
        n: frozenset(cond.nodes[n]["members"]).union(
            *(cond.nodes[a]["members"] for a in nx.ancestors(cond, n))
        )
        for n in cond
    }
    return cond, sets, g


def closures(deps, cells):
    """``(closures, graph)``: each sink SCC of the ``deps`` graph (``src -> loc``) over
    ``cells`` with its ancestors, smallest first, and the graph."""
    cond, sets, g = condensed(deps, cells)
    out = [sets[n] for n in cond if cond.out_degree(n) == 0]
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


def edge_pairs(keys):
    """``(pc, next)`` of edge keys ``pc << 16 | next`` (an array, or uint64 bytes)."""
    e = np.frombuffer(keys, np.uint64) if isinstance(keys, bytes) else np.asarray(keys, np.int64)
    return list(zip((e >> 16).tolist(), (e & 0xFFFF).tolist()))


def census(state, pc, calls, cells=(), *, acc=0, max_cycles=MAX_CYCLES, remap=None):
    """A :class:`~jennings.devices.mpu6502.Census` of ``calls`` calls of ``pc`` with ``A = acc``
    from ``state`` (see :func:`machine`), ``remap`` applied, ``cells`` recorded at each entry."""
    return Calls(state, pc, cells, acc=acc, max_cycles=max_cycles, remap=remap).run(calls).census


class Calls:
    """The calls of ``pc`` with ``A = acc`` from ``state`` (see :func:`machine`), ``remap``
    applied: a :func:`census` recording ``cells`` that runs on as more calls are asked for."""

    # pylint: disable-next=too-many-arguments
    def __init__(self, state, pc, cells=(), *, acc=0, max_cycles=MAX_CYCLES, remap=None):
        self.state, self.pc, self.cells = state, pc, tuple(cells)
        self.kw = {"acc": acc, "max_cycles": max_cycles, "remap": remap}
        self.census, self.machine = Census(list(self.cells)), machine(state)
        self.failed = None

    @property
    def calls(self):
        """The calls run."""
        return self.census.calls

    def run(self, calls):
        """These calls run on to ``calls``; raises (as :func:`census`) where a call fails, the
        census then holding it as its last, and on any call asked for after it."""
        if self.failed is not None and calls > self.calls:
            raise self.failed
        m, kw = self.machine, self.kw
        try:
            drive(
                lambda n: m.census(
                    self.census, self.pc, kw["acc"], kw["max_cycles"], n, kw["remap"]
                ),
                calls - self.calls,
                "census",
            )
        except RuntimeError as e:
            self.failed = e
            raise
        return self

    def rows(self, calls=None, cells=None):
        """uint8 ``[calls, len(cells)]``: the ``cells`` (of these calls' cells; all where None)
        at the entry of each of the first ``calls`` calls run (all where None), by column."""
        rows = _rows(self.census, self.cells)[:calls]
        if cells is None or tuple(cells) == self.cells:
            return rows
        col = {c: i for i, c in enumerate(self.cells)}
        return rows[:, [col[c] for c in cells]]

    def holds(self, cells):
        """Whether these calls record every one of ``cells``."""
        return set(cells) <= set(self.cells)

    def repeat(self, cells, limit, start=1):
        """The calls before the first entry at which the ``cells`` (of these calls' cells)
        repeat an earlier entry (its preperiod plus period; 1 for no cells), running on from
        ``start`` calls, doubling, to the ``limit + 1`` that show a repeat within ``limit``;
        None where none does. A call that fails stops the run there (:attr:`failed`)."""
        col = {c: i for i, c in enumerate(self.cells)}
        cols = sorted(col[c] for c in cells)
        n = min(max(start, 1), limit + 1)
        while True:
            try:
                self.run(max(n, self.calls))
            except RuntimeError:
                pass
            first = periods_of(self.rows(), [cols])[0][0] if cols else (0, 1)
            if first is not None and sum(first) < self.calls:
                return sum(first)
            if self.failed is not None or self.calls > limit:
                return None
            n = min(2 * self.calls, limit + 1)

    def seen(self, name, calls):
        """``(items, at)``: the items of :meth:`~jennings.devices.mpu6502.Census.firsts`
        ``name`` first seen in the first ``calls`` calls, in order of first sight, and the call
        each was first seen in."""
        items, at = self.census.firsts()[name]
        at = np.frombuffer(at, np.uint32)
        n = int(np.searchsorted(at, calls))
        dtype = np.uint64 if name in ("edges", "accesses") else np.uint16
        return np.frombuffer(items, dtype)[:n], at[:n]

    def fork(self, calls, remap):
        """These calls' first ``calls``, run on under ``remap``: ``calls`` ends them or is a
        class's first (:meth:`~jennings.devices.mpu6502.Census.prefix`), and the calls before
        it access no location ``remap`` redirects otherwise than these calls' remap does."""
        cen = self.census.prefix(calls)
        if cen is None:
            raise ValueError("call %d is no class's first" % calls)
        out = Calls(self.state, self.pc, self.cells, **{**self.kw, "remap": remap})
        out.census = cen
        if calls < self.calls:
            out.machine.restore(self.census, self.census.class_of(calls))
        else:
            out.machine, out.failed = machine(self.machine), self.failed
        return out


def _rows(cen, cells):
    """The census' rows of its ``cells`` as uint8 ``[calls, len(cells)]``, by column."""
    return np.frombuffer(cen.columns(), np.uint8).reshape(len(cells), cen.calls).T


def _prefix(cen, calls):
    """``(control, written, seen)`` of the first ``calls`` calls of census ``cen``: as
    :func:`control` gives them, and how many edges, exits and written addresses they saw."""
    got = cen.firsts()

    def first(name, dtype):
        items, at = got[name]
        n = int(np.searchsorted(np.frombuffer(at, np.uint32), calls))
        return np.sort(np.frombuffer(items, dtype)[:n])

    edges, exits, written = (
        first("edges", np.uint64),
        first("exits", np.uint16),
        first("written", np.uint16),
    )
    cd = control_dependence(edge_pairs(edges.tobytes()) + [(x, -1) for x in exits.tolist()])
    seen = (len(edges), len(exits), len(written))
    return {p: sorted(b) for p, b in cd.items()}, written.tolist(), seen


def control(state, pc, calls, *, acc=0, max_cycles=MAX_CYCLES, remap=None):
    """``(control, written)`` of ``calls`` calls of ``pc`` from ``state``: ``{pc: branches}`` it
    is directly control dependent on (:func:`~jennings.coverage.cfg.control_dependence`, each
    call's last instruction leading to the exit), and the memory the calls write."""
    cen = census(state, pc, calls, acc=acc, max_cycles=max_cycles, remap=remap)
    return _prefix(cen, calls)[:2]


def _by_loc(raw):
    """``{loc: set(src)}`` of uint64 ``loc << 32 | src`` bytes."""
    d = np.frombuffer(raw, np.uint64)
    out = {}
    for loc, src in zip((d >> 32).tolist(), (d & 0xFFFFFFFF).tolist()):
        out.setdefault(loc, set()).add(src)
    return out


@dataclass
class _Run:
    """A :class:`Learner` over the first ``calls`` calls of a census, with the ``seen`` of
    :func:`_prefix` it was made with and the machine it runs on."""

    learner: Learner
    machine: NativeMPU
    seen: tuple
    calls: int = 0


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def _learn_on(state, pc, cen, calls, explicit_, kw, run=None):
    """``run`` (a :class:`_Run`) taken on to ``calls`` calls of census ``cen`` of ``pc`` from
    ``state``, or a new one where ``run`` is None or those calls see more than its calls did.

    Learning is a function of each call's class (its instructions, opcodes and the locations
    of its accesses, in order), of the control dependences and of the memory the calls write,
    so only each class's first call is tainted; the next calls extend a learner whose calls
    saw every edge, exit and written address they do, as relearning them all would."""
    cd, written, seen = _prefix(cen, calls)
    if run is None or run.seen != seen:
        run = _Run(Learner(cd, written, explicit_), machine(state), seen)
    acc, max_cycles, remap = kw["acc"], kw["max_cycles"], kw["remap"]
    learner, m = run.learner, run.machine
    drive(
        lambda n: m.learn(learner, pc, acc, max_cycles, n, remap, census=cen),
        calls - run.calls,
        "explicit" if explicit_ else "learn",
    )
    run.calls = calls
    return run


def _sites(got):
    """``{(pc, kind): frozenset}`` of a ``Learner`` result's sites."""
    s = np.frombuffer(got["sites"], np.uint64)
    sites = {}
    for key, loc in zip((s >> SITE_BITS).tolist(), (s & ((1 << SITE_BITS) - 1)).tolist()):
        sites.setdefault((key >> 2, _KINDS[key & 3]), set()).add(loc)
    return {k: frozenset(v) for k, v in sites.items()}


def _components(got, calls, exclude):
    """:class:`Components` of a ``Learner`` result over ``calls`` calls, ``exclude`` never
    persistent."""
    locs = np.frombuffer(got["persistent"], np.uint32)
    persistent = tuple(locs[~excluded(locs, spans(exclude))].tolist())
    deps = {k: v - {k} for k, v in _by_loc(got["deps"]).items() if v - {k}}
    cls, g = closures(deps, persistent)
    comps = Components(
        persistent=persistent,
        deps={k: frozenset(v) for k, v in deps.items()},
        closures=cls,
        feed=frozenset(),
        feeding=(),
        edges=frozenset(edge_pairs(got["edges"])),
        sites=_sites(got),
        calls=calls,
    )
    tainted = _tainted(comps)
    return replace(comps, feed=_with_ancestors(g, tainted), feeding=_feeding(cls, tainted))


def learn(state, pc, calls, *, acc=0, max_cycles=MAX_CYCLES, remap=None, exclude=(STACK, IO)):
    """:class:`Components` of ``calls`` calls of ``pc`` with ``A = acc`` from ``state`` (see
    :func:`machine`), ``remap`` (a :class:`~jennings.devices.mpu6502.Remap`) applied; addresses
    in ``exclude`` are never persistent."""
    kw = {"acc": acc, "max_cycles": max_cycles, "remap": remap}
    cen = census(state, pc, calls, **kw)
    run = _learn_on(state, pc, cen, calls, False, kw)
    return _components(run.learner.learned(), calls, exclude)


def explicit(state, pc, calls, *, acc=0, max_cycles=MAX_CYCLES, remap=None):
    """:class:`Flows` of ``calls`` calls of ``pc`` (as :func:`learn`), by data alone.

    Each load's value is apart from its address; a data load of a location no call writes at
    ``pc`` is the constant :data:`CONST` + ``pc``. Control enters only ``gated``."""
    kw = {"acc": acc, "max_cycles": max_cycles, "remap": remap}
    cen = census(state, pc, calls, **kw)
    got = _learn_on(state, pc, cen, calls, True, kw).learner.learned()
    per = {k: {loc: frozenset(v) for loc, v in _by_loc(got[k]).items()} for k in _PER}
    deps = {k: frozenset(v) for k, v in _by_loc(got["deps"]).items()}
    return Flows(deps=deps, sites=_sites(got), calls=calls, **per)


def trajectory(state, pc, cells, calls, *, acc=0, max_cycles=MAX_CYCLES, remap=None):
    """uint8 ``[calls + 1, len(cells)]``: the values of the locations ``cells`` at the entry of
    each of ``calls + 1`` calls of ``pc`` from ``state`` (see :func:`learn`)."""
    if not cells:
        return np.zeros((calls + 1, 0), np.uint8)
    m = machine(state)
    rows = []

    def run(k):
        stop, r = m.trajectory(pc, acc, max_cycles, k, cells, remap)
        rows.append(r)
        return stop

    drive(run, calls + 1, "trajectory")
    return np.concatenate(rows)


def _measure(rows, comps, known=None):
    """:class:`Cycles` of ``comps`` on ``rows``: a projection repeating first at
    ``(mu, lam)`` is autonomous on ``rows`` iff it has period ``lam`` from ``mu`` on. ``known``
    holds the native :func:`~jennings.devices.mpu6502.periods_of` of column sets measured."""
    known = {} if known is None else known
    if not (rows.flags.c_contiguous or rows.flags.f_contiguous):
        rows = np.ascontiguousarray(rows)
    col = {c: i for i, c in enumerate(comps.persistent)}
    sets = [*enumerate(comps.closures), ("feed", comps.feed)]
    sets = [(key, tuple(sorted(col[c] for c in cells))) for key, cells in sets]
    new = sorted({cols for _, cols in sets if cols and cols not in known})
    known.update(zip(new, periods_of(rows, [list(c) for c in new]) if new else ()))
    out, conflicts = {}, {}
    for key, cols in sets:
        if not cols:
            out[key] = (0, 1)
            continue
        out[key], t1 = known[cols]
        if t1 is not None:
            lam = out[key][1]
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
    return _measure(np.asarray(rows, np.uint8), comps)


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
    known = {}
    while True:
        cyc = _measure(rows, comps, known)
        if not cyc.conflicts:
            return cyc, comps
        wider = widen(comps, cyc)
        if wider == comps:
            raise RuntimeError("conflicts no persistent cell explains: %r" % cyc.conflicts)
        comps = wider


# pylint: disable-next=too-many-arguments,too-many-locals
def cover_calls(
    state,
    pc,
    comps,
    limit,
    *,
    acc=0,
    max_cycles=MAX_CYCLES,
    remap=None,
    exclude=(STACK, IO),
    ran=None,
):
    """``(calls, comps)``: the calls from ``state`` after which every data access and control
    edge has been seen (the feed set's preperiod plus period; None if it does not repeat within
    ``limit``), ``comps`` relearned over at least ``calls`` calls and widened until autonomous.

    One :func:`census` of the ``limit + 1`` calls gives the trajectory and every relearning:
    each a learner over its calls' classes, taken on from the last where no new edge, exit or
    written address comes between. Given ``ran``, :class:`Calls` of these recording
    ``comps.persistent`` among their cells, that census is theirs, run on as far as it needs,
    and so is the trajectory of every relearning's cells it records."""
    kw = {"acc": acc, "max_cycles": max_cycles, "remap": remap}
    if ran is None:
        ran = Calls(state, pc, comps.persistent, **kw)
    cen = ran.run(max(ran.calls, limit + 1)).census
    rows, cells, run = ran.rows(limit + 1, comps.persistent), comps.persistent, None
    while True:
        if cells != comps.persistent:
            cells = comps.persistent
            rows = (
                ran.rows(limit + 1, cells)
                if ran.holds(cells)
                else trajectory(state, pc, cells, limit, **kw)
            )
        cyc, comps = _autonomous(rows, comps)
        calls = None if cyc.feed is None else cyc.feed[0] + cyc.feed[1]
        if calls is None or calls <= comps.calls:
            return calls, comps
        run = _learn_on(state, pc, cen, calls, False, kw, run)
        comps = _components(run.learner.learned(), calls, exclude)


# pylint: disable-next=too-many-arguments
def cover(
    state,
    pc,
    limit,
    *,
    acc=0,
    max_cycles=MAX_CYCLES,
    remap=None,
    exclude=(STACK, IO),
    first=CHUNK,
    start=None,
):
    """``(calls, comps, ran)``: the cover point of ``pc`` called from ``state``, the feed
    locations' first repeat as :func:`cover_calls` settles it, with the :class:`Components`
    and the :class:`Calls` it was measured on.

    The components are learned over ``first`` calls (at most ``limit``) and the feed
    locations' first repeat found on calls recording the persistent locations;
    :func:`cover_calls` is run over twice that plus the calls learned, and the components it
    gives are taken on until it gives the same repeat and feed locations. Where it finds a
    conflict no persistent location explains, the components are learned again over more
    calls. ``calls`` is None where the feed locations do not repeat within ``limit`` calls.

    ``start(cells)``, where given, is :class:`Calls` recording ``cells`` to take the first of
    those runs on from, or None to run them anew. Raises :class:`RuntimeError` where a call
    does not return, or where the feed locations do not settle within ``limit`` calls."""
    kw = {"acc": acc, "max_cycles": max_cycles, "remap": remap}
    learned, seen = min(first, limit), set()
    comps = learn(state, pc, learned, exclude=exclude, **kw)
    while True:
        seen.add((comps.feed, learned))
        ran = start(comps.persistent) if start is not None else None
        start = None
        ran = ran or Calls(state, pc, comps.persistent, **kw)
        calls = ran.repeat(comps.feed, limit, learned)
        if ran.failed is not None:
            raise ran.failed
        if calls is None or not comps.persistent:
            return calls, comps, ran
        try:
            n, wider = cover_calls(
                state, pc, comps, min(limit, 2 * calls + learned), exclude=exclude, ran=ran, **kw
            )
        except RuntimeError:
            if ran.failed is not None or learned >= limit:
                raise
            learned = min(limit, max(8 * learned, 2 * calls))
            comps = learn(state, pc, learned, exclude=exclude, **kw)
            continue
        if n == calls and wider.feed == comps.feed:
            return calls, comps, ran
        learned, comps = max(learned, wider.calls), wider
        if (comps.feed, learned) in seen:
            raise RuntimeError("the feed locations do not settle within %d calls" % limit)
