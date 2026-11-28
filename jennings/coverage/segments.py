"""Calls cut into segments, the state each runs on, and where every segment's state repeats.

:func:`cut` runs the calls, :func:`classify` sorts each segment's feed into its own state,
values held and dependences, :func:`examine` checks that the states repeat and
:func:`propose` looks for cuts under which they do (``docs/coverage.md``).
"""

from dataclasses import dataclass, field
from itertools import chain

import numpy as np
from tqdm import tqdm

from jennings.coverage import components as C
from jennings.devices.mpu6502 import BASE, MAX_FORKS, MAX_SEGMENTS, periods_of

__all__ = [
    "BASE",
    "MAX_SEGMENTS",
    "Trace",
    "Parts",
    "Found",
    "Refused",
    "cut",
    "classify",
    "examine",
    "candidates",
    "propose",
    "never_repeats",
]

_JSR = 0x20
_VARIES = -1
#: The times a proposal's calls are run again as its free locations and their values grow.
_ROUNDS = 12


class Refused(Exception):
    """Why calls cut into segments cannot be certified: ``failed`` names the condition (as
    :attr:`~jennings.coverage.certificate.Verdict.failed`), ``detail`` says it in words and
    ``call`` is the call that broke it, where one did."""

    def __init__(self, failed, detail, call=None):
        super().__init__(detail)
        self.failed, self.detail, self.call = failed, detail, call


@dataclass(frozen=True)
class Trace:
    """What :func:`cut` saw of the calls it cut; ``feed`` and ``stores`` one row per item.

    Attributes:
        lead: the calls run before the first cut one.
        calls: the cut calls run whose segments are the first's.
        stop: ``"sp"`` unless the last call run ended otherwise.
        odd: ``(call, segment, pc, again)`` of the first call cut otherwise than call
            ``lead``: the segment it did not start alike, the PC it started it at (None: it
            returned) and whether the run before was one made again.
        forks: the runs made again with other values of free locations.
        starts: the PC each segment starts at.
        runs: the runs of each segment: one per cut call, and those made again.
        cycles: the most cycles a run of each segment took.
        entry: uint8 per location: its value at the entry of call ``lead``.
        feed: int64 ``[n, 3]``: a segment, a location some run of it read before writing,
            and the value every such read found (-1 where they differ).
        stores: int64 ``[n, 4]``: a segment, a location some run of it wrote, the runs that
            did, and the value each of them left there (-1 where they differ).
        accesses: the access keys (``pc << 17 | store << 16 | address``) of every call run,
            the leading ones too, ascending.
        edges: their edge keys (``pc << 16 | next``), ascending.
        again: ``(accesses, edges)`` as those, of the runs made again.
        arrivals: ``{pc: n}``: the instructions every cut call arrived at ``n`` times.
        values: ``{location: values}``: what each free location held at the entry of call
            ``lead`` and wherever a run ended, ascending.
        differs: ``(segment, free, location)``: a run of the segment made again, the free
            location last given another value, left the location otherwise than the call's.
    """

    lead: int
    calls: int
    stop: str
    odd: tuple | None
    forks: int
    starts: tuple
    runs: tuple
    cycles: tuple
    entry: np.ndarray = field(repr=False)
    feed: np.ndarray = field(repr=False)
    stores: np.ndarray = field(repr=False)
    accesses: np.ndarray = field(repr=False)
    edges: np.ndarray = field(repr=False)
    again: tuple = field(repr=False)
    arrivals: dict = field(repr=False)
    values: dict
    differs: tuple


@dataclass(frozen=True)
class Parts:
    """Each segment's feed by who else stores it (:func:`classify`): one item per segment.

    Attributes:
        own: the feed locations only the segment itself stores.
        held: ``{location: value}``: those other segments store that hold one value whenever
            the segment starts.
        free: those of the rest given as free for the segment.
        shared: the other feed locations some segment stores.
        deps: the other segments storing a ``shared`` location.
        groups: the segment with every segment it depends on, directly or through others.
        cells: the ``own`` and ``shared`` locations of its group, ascending: its state.
    """

    own: tuple
    held: tuple
    free: tuple
    shared: tuple
    deps: tuple
    groups: tuple
    cells: tuple


@dataclass(frozen=True)
class Found:
    """What :func:`examine` found of calls cut into segments.

    Attributes:
        trace, parts: the :class:`Trace` and its :class:`Parts`.
        cycles: per segment the ``(preperiod, period)`` of its state's first repeat at the
            entries of the calls from call ``trace.lead`` on, None where it repeats at none.
        feed: the locations of any segment's state, ascending.
        rows: uint8 ``[n + 1, len(feed)]``: those locations at the entry of each of the ``n``
            calls run, the leading ones too, and after the last.
        assumed: addresses in the volatile ranges a segment loads after storing to them.
    """

    trace: Trace
    parts: Parts
    cycles: tuple
    feed: tuple
    rows: np.ndarray
    assumed: tuple

    @property
    def closed(self):
        """Whether every segment's state repeats within the calls."""
        return all(c is not None for c in self.cycles)

    @property
    def calls(self):
        """The calls after which every segment's state has repeated; None unless closed."""
        return max(sum(c) for c in self.cycles) if self.closed else None


def _unpack(raw):
    """``(segment, location, value)`` columns of packed feed or store items."""
    k = np.frombuffer(raw, np.uint64).astype(np.int64)
    value = np.where(k >> 8 & 1 == 1, _VARIES, k & 0xFF)
    return k >> 32, k >> 9 & 0x7FFFFF, value


def _keys(raw):
    return np.frombuffer(raw, np.uint64).astype(np.int64)


# pylint: disable-next=too-many-arguments
def cut(
    state,
    pc,
    calls,
    cuts,
    *,
    lead=0,
    free=(),
    values=None,
    acc=0,
    max_cycles=C.MAX_CYCLES,
    remap=None,
):
    """The :class:`Trace` of ``calls`` calls of ``pc`` with ``A = acc`` from ``state`` (see
    :func:`~jennings.coverage.components.machine`): the first ``lead`` whole, each of the
    rest cut where it arrives at a PC of ``cuts`` after its first instruction.

    A run that reads a location of ``free`` (``(segment, location)`` pairs) of its segment
    before writing it is made again with each other of its ``values`` (``{location:
    values}``), and so on for what those runs read."""
    if not 0 <= lead <= calls:
        raise ValueError("the leading calls are among the calls")
    known = [(loc, v) for loc, vals in sorted((values or {}).items()) for v in vals]
    core = C.machine(state)._core
    got = core.cut_calls(
        pc, acc, max_cycles, calls - lead, sorted(cuts), remap, lead, sorted(free), known
    )
    stored = np.frombuffer(got["stored"], np.uint32).astype(np.int64)
    seg, loc, value = _unpack(got["stores"])
    arrivals, odd, held = _keys(got["arrivals"]), got["odd"], {}
    for cell, v in got["values"]:
        held.setdefault(cell, []).append(v)
    return Trace(
        lead=lead,
        calls=got["calls"],
        stop=got["stop"],
        odd=odd and (lead + odd[0], *odd[1:]),
        forks=got["forks"],
        starts=tuple(got["starts"]),
        runs=tuple(got["runs"]),
        cycles=tuple(got["cycles"]),
        entry=np.frombuffer(got["entry"], np.uint8),
        feed=np.stack(_unpack(got["feed"]), axis=1),
        stores=np.stack((seg, loc, stored, value), axis=1),
        accesses=_keys(got["accesses"]),
        edges=_keys(got["edges"]),
        again=(_keys(got["fork_accesses"]), _keys(got["fork_edges"])),
        arrivals=dict(zip((arrivals >> 32).tolist(), (arrivals & 0xFFFFFFFF).tolist())),
        values={cell: tuple(vals) for cell, vals in held.items()},
        differs=tuple(got["differs"]),
    )


def _holds(seg, loc, value, storers, trace):
    """Whether ``loc`` holds ``value`` whenever segment ``seg`` starts, by its ``storers``
    (``{segment: (runs, value left)}``) and its value at the first cut call's entry alone.

    Going back from ``seg`` through the segments before it, then the call before's after it,
    each that stores it leaves ``value``, up to one that stores it in every run; or every
    segment that stores it leaves ``value`` and the first cut call starts with it."""
    count = len(trace.starts)
    for other in chain(range(seg - 1, -1, -1), range(count - 1, seg - 1, -1)):
        if other in storers:
            runs, left = storers[other]
            if left != value:
                return False
            if runs == trace.runs[other]:
                return True
    return loc != BASE and int(trace.entry[loc]) == value


def _reach(deps):
    """Each node of ``deps`` (``node -> nodes``, by index) with all it reaches."""
    out = []
    for start in range(len(deps)):
        seen, todo = {start}, [start]
        while todo:
            for nxt in deps[todo.pop()]:
                if nxt not in seen:
                    seen.add(nxt)
                    todo.append(nxt)
        out.append(frozenset(seen))
    return tuple(out)


def classify(trace, free=()):
    """The :class:`Parts` of ``trace``, ``free`` the ``(segment, location)`` pairs it ran with.

    A feed location no segment stores is constant and in none of them. :data:`BASE` is
    stored at each call's start before anything reads it, so it is state at no call's entry
    and is left out of ``cells``."""
    count, free = len(trace.starts), set(free)
    storers = {}
    for seg, loc, runs, value in trace.stores.tolist():
        storers.setdefault(loc, {})[seg] = (runs, value)
    own, loose, shared, deps = ([set() for _ in range(count)] for _ in range(4))
    held = [{} for _ in range(count)]
    fed = trace.feed[np.isin(trace.feed[:, 1], np.fromiter(storers, np.int64, len(storers)))]
    for seg, loc, value in fed.tolist():
        by = storers[loc]
        if set(by) <= {seg}:
            own[seg].add(loc)
        elif value != _VARIES and _holds(seg, loc, value, by, trace):
            held[seg][loc] = value
        elif (seg, loc) in free:
            loose[seg].add(loc)
        else:
            shared[seg].add(loc)
            deps[seg] |= set(by) - {seg}
    groups = _reach(deps)
    cells = tuple(
        tuple(sorted(set().union(*(own[i] | shared[i] for i in g)) - {BASE})) for g in groups
    )
    own, loose, shared, deps = (
        tuple(frozenset(s) for s in part) for part in (own, loose, shared, deps)
    )
    return Parts(own, tuple(held), loose, shared, deps, groups, cells)


def _describe(trace, seg):
    return "segment %d (from $%04X)" % (seg, trace.starts[seg])


def _volatile(trace, volatile):
    """The addresses of ``volatile`` the segments load; raises :class:`Refused` where one
    reads such an address before writing it."""
    at = C.excluded(trace.feed[:, 1], volatile)
    if at.any():
        seg, loc, _ = trace.feed[at][0].tolist()
        how = "%s reads volatile %s before writing it"
        raise Refused("volatile", how % (_describe(trace, seg), C.location(loc)))
    loads = trace.accesses[trace.accesses >> 16 & 1 == 0] & 0xFFFF
    return tuple(np.unique(loads[C.excluded(loads, volatile)]).tolist())


def _replayed(trace, calls, max_cycles):
    """Raises :class:`Refused` unless ``trace`` is of ``calls`` calls that each return within
    ``max_cycles`` whatever their segments' runs, with the same segments in each."""
    if trace.stop != "sp":
        how = "a call after %d ended at %r, not its RTS" % (trace.calls, trace.stop)
        raise Refused("replay", how)
    if trace.forks >= MAX_FORKS:
        raise Refused("replay", "more than %d runs made again" % MAX_FORKS)
    if trace.odd is not None:
        call, seg, pc, again = trace.odd
        if again:
            how = "a run of segment %d in call %d with its free locations otherwise ends elsewhere"
            raise Refused("segments", how % (seg - 1, call))
        if seg >= MAX_SEGMENTS:
            raise Refused("segments", "a call has more than %d segments" % MAX_SEGMENTS)
        was = "returns" if seg >= len(trace.starts) else "is at $%04X" % trace.starts[seg]
        now = "returns" if pc is None else "is at $%04X" % pc
        how = "where call %d %s after %d segments, call %d %s"
        raise Refused("segments", how % (trace.lead, was, seg, call, now), call)
    if trace.calls != calls - trace.lead or trace.calls < 1:
        raise Refused("replay", "%d calls ran cut, not %d" % (trace.calls, calls - trace.lead))
    if sum(trace.cycles) > max_cycles:
        how = "the segments' longest runs take %d cycles together, a call at most %d"
        raise Refused("cycles", how % (sum(trace.cycles), max_cycles))


def _loose(trace, parts, values):
    """Raises :class:`Refused` unless the runs made again bear out the free locations:
    each holds one of its ``values`` wherever a run ends, no such run leaves a location of a
    segment's state otherwise than the call's own, and none makes an access or edge no call
    does."""
    for loc, held in sorted(trace.values.items()):
        more = sorted(set(held) - set(values.get(loc, ())))
        if more:
            how = "%s holds $%02X where a run ends, not one of its values"
            raise Refused("values", how % (C.location(loc), more[0]))
    state = set().union(*parts.cells)
    for seg, free, loc in trace.differs:
        if loc in state:
            how = "a run of %s with %s otherwise leaves %s otherwise"
            how %= (_describe(trace, seg), C.location(free), C.location(loc))
            raise Refused("autonomy", how)
    for name, mine, theirs in zip(
        ("accesses", "edges"), trace.again, (trace.accesses, trace.edges)
    ):
        if np.setdiff1d(mine, theirs).size:
            how = "a run with a free location otherwise makes one of the %s no call makes"
            raise Refused(name, how % name)


# pylint: disable-next=too-many-arguments
def examine(
    state,
    pc,
    calls,
    cuts,
    *,
    lead=0,
    free=(),
    values=None,
    acc=0,
    max_cycles=C.MAX_CYCLES,
    remap=None,
    volatile=(),
):
    """The :class:`Found` of ``calls`` calls of ``pc`` from ``state``, all but the first
    ``lead`` cut at ``cuts`` with ``free`` and ``values`` as :func:`cut` takes them, no
    segment reading an address of ``volatile`` (``(lo, hi)`` pairs) before writing it.

    Raises:
        Refused: ``"replay"`` (a call does not return), ``"segments"`` (the calls are not cut
            alike), ``"volatile"``, ``"cycles"`` (the segments' longest runs together pass
            ``max_cycles``), ``"values"`` (a free location holds another value than given),
            ``"accesses"``, ``"edges"`` (a run made again makes one no call does) or
            ``"autonomy"`` (a state does not go on as its repeat has it, or a run made again
            leaves one otherwise).
    """
    kw = {"acc": acc, "max_cycles": max_cycles, "remap": remap}
    trace = cut(state, pc, calls, cuts, lead=lead, free=free, values=values, **kw)
    _replayed(trace, calls, max_cycles)
    parts = classify(trace, free)
    return _examined(state, pc, trace, parts, (values or {}, volatile), kw)


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def _examined(state, pc, trace, parts, given, kw):
    """The :class:`Found` of a ``trace`` that :func:`_replayed` passes and its ``parts``,
    ``given`` the ``(values, volatile)`` it is held to; raises :class:`Refused` as
    :func:`examine` does."""
    assumed = _volatile(trace, given[1])
    _loose(trace, parts, given[0])
    return _found(state, pc, trace, parts, assumed, kw)


def _found(state, pc, trace, parts, assumed, kw):
    """The :class:`Found` of ``trace`` and its ``parts``: where each segment's state repeats.

    Raises :class:`Refused` (``"autonomy"``) where one does not go on as its repeat has it."""
    lead, calls = trace.lead, trace.lead + trace.calls
    feed = tuple(sorted(set().union(*parts.cells)))
    rows = np.ascontiguousarray(C.trajectory(state, pc, feed, calls, **kw))
    col = {c: i for i, c in enumerate(feed)}
    sets = sorted({cells for cells in parts.cells if cells})
    later = np.ascontiguousarray(rows[lead:])
    got = periods_of(later, [[col[c] for c in cells] for cells in sets]) if sets else ()
    first = {(): ((lead, 1), None)}
    for cells, (at, conflict) in zip(sets, got):
        first[cells] = (at and (lead + at[0], at[1]), conflict)
    for seg, cells in enumerate(parts.cells):
        at, conflict = first[cells]
        if conflict is not None:
            how = "the state of %s repeats at the entries of calls %d and %d but not after call %d"
            how %= (_describe(trace, seg), at[0], sum(at), lead + conflict)
            raise Refused("autonomy", how, lead + conflict)
    cycles = tuple(first[cells][0] for cells in parts.cells)
    return Found(trace, parts, cycles, feed, rows, assumed)


def candidates(trace, state):
    """PCs to cut at, ascending: the instructions every call of ``trace`` (uncut) arrives at
    the same number of times, more than once, by a backward edge or a ``JSR``."""
    ram = np.frombuffer(bytes(C.machine(state).memory), np.uint8)
    src, dst = trace.edges >> 16, trace.edges & 0xFFFF
    entered = dst[(dst <= src) | (ram[src] == _JSR)]
    return sorted(pc for pc in set(entered.tolist()) if trace.arrivals.get(pc, 0) > 1)


def _score(found):
    """How well cuts set states apart: more distinct states first, then fewer segments; least
    for a :class:`Refused`."""
    if isinstance(found, Refused):
        return (-1, 0)
    return (len({cells for cells in found.parts.cells if cells}), -len(found.trace.starts))


def _choose(look, pcs):
    """``(cuts, found)`` of the best-scoring cuts among ``pcs``: each alone, then joined by
    each other while that scores better; ``look(cuts)`` is their :class:`Found` or the
    :class:`Refused` they met."""
    best = ((), look(()))
    for pc in pcs:
        got = look((pc,))
        if _score(got) > _score(best[1]):
            best = ((pc,), got)
    grew = bool(best[0])
    while grew:
        grew = False
        for pc in pcs:
            cuts = tuple(sorted({*best[0], pc}))
            got = look(cuts) if pc not in best[0] else best[1]
            if _score(got) > _score(best[1]):
                best, grew = (cuts, got), True
    return best


def _freed(look, cuts, calls, lead, start=(frozenset(), {})):
    """``(free, values, found)`` of ``calls`` calls cut at ``cuts`` after ``lead``: each
    segment's shared locations made free, from those of ``start`` on, unless a run made again
    then leaves state otherwise. Found is the :class:`Refused` the calls met, if any.

    ``look(cuts, calls, lead, free, values)`` gives their trace and parts, and with a trace
    and parts in place of the calls their :class:`Found`."""
    (free, values), bound = start, set()
    for _ in range(_ROUNDS):
        try:
            trace, parts = look(cuts, calls, lead, free, values)
        except Refused as e:
            return free, values, e
        state = set().union(*parts.cells)
        bound |= {(seg, cell) for seg, cell, loc in trace.differs if loc in state}
        wider = {(seg, loc) for seg, part in enumerate(parts.shared) for loc in part}
        wider = frozenset((free | wider) - bound)
        known = {loc: tuple(sorted({*values.get(loc, ()), *v})) for loc, v in trace.values.items()}
        known = {loc: known.get(loc, ()) for _, loc in wider}
        if (wider, known) == (free, values):
            try:
                return free, values, look(cuts, (trace, parts), lead, free, values)
            except Refused as e:
                return free, values, e
        free, values = wider, known
    return free, values, Refused("values", "the free locations do not settle")


def _unclosed(found, calls):
    """The :class:`Refused` of ``found``, not closed within ``calls`` calls."""
    if isinstance(found, Refused):
        return found
    seg = next(i for i, c in enumerate(found.cycles) if c is None)
    cells = " ".join(C.location(c) for c in found.parts.cells[seg][:8])
    how = "the state of %s (%s) repeats at no entry within %d calls"
    return Refused("repeat", how % (_describe(found.trace, seg), cells, calls))


# pylint: disable-next=too-many-arguments
def propose(
    state, pc, limit, *, acc=0, max_cycles=C.MAX_CYCLES, remap=None, volatile=(), learn=C.CHUNK
):
    """``(cuts, free, values, found)``: cuts and free locations under which every segment's
    state repeats within ``limit`` calls of ``pc`` from ``state``, with their closed
    :class:`Found`. A heuristic: :func:`examine` alone says what holds.

    The cuts are chosen among :func:`candidates` of the later half of ``learn`` calls, which
    double to ``limit``, their first half left whole; then fewer are left whole.

    Raises:
        Refused: what the cuts chosen met, where none is found.
    """
    kw = {"acc": acc, "max_cycles": max_cycles, "remap": remap}

    def look(cuts, calls, lead, free, values):
        if isinstance(calls, tuple):
            return _examined(state, pc, *calls, (values, volatile), kw)
        trace = cut(state, pc, calls, cuts, lead=lead, free=free, values=values, **kw)
        _replayed(trace, calls, max_cycles)
        return trace, classify(trace, free)

    def closed(found):
        return isinstance(found, Found) and found.closed

    calls, tried = min(learn, limit), {}
    whole = cut(state, pc, calls, (), lead=calls // 2, **kw)
    _replayed(whole, calls, max_cycles)

    def first(cuts):
        tried[cuts] = _freed(look, cuts, calls, calls // 2)
        return tried[cuts][2]

    cuts, found = _choose(first, candidates(whole, state))
    free, values = tried[cuts][:2]
    with tqdm(
        total=limit, initial=calls, desc="segments", unit="call", disable=limit <= C.CHUNK
    ) as progress:
        while isinstance(found, Found) and not found.closed and calls < limit:
            calls = min(limit, 2 * calls)
            free, values, found = _freed(look, cuts, calls, calls // 2, (free, values))
            progress.update(calls - progress.n)
    if not closed(found):
        raise _unclosed(found, calls)
    for lead in (0, calls // 16, calls // 8, calls // 4):
        fewer = _freed(look, cuts, calls, lead, (free, values))
        if closed(fewer[2]):
            free, values, found = fewer
            break
    lead = found.trace.lead
    while found.calls < lead + found.trace.calls:
        fewer = _freed(look, cuts, found.calls, lead, (free, values))
        if not closed(fewer[2]):
            break
        free, values, found = fewer
    return cuts, tuple(sorted(free)), values, found


def _hashed(rows):
    """A uint64 per row of ``rows``, equal for equal rows."""
    out = np.full(len(rows), 0x9E3779B97F4A7C15, np.uint64)
    for column in rows.T.astype(np.uint64):
        out = (out ^ column) * np.uint64(0x100000001B3)
        out ^= out >> np.uint64(29)
    return out


def _by_cycle(found, cells):
    """``{(preperiod, period): columns}`` of ``found.rows`` for those of ``cells`` in a
    segment's state, each under the shortest cycle of a state holding it."""
    col = {c: i for i, c in enumerate(found.feed)}
    out, left = {}, set(cells)
    for mine, cycle in sorted(zip(found.parts.cells, found.cycles), key=lambda p: p[1][1]):
        took = left & set(mine)
        if took:
            out.setdefault(cycle, []).extend(sorted(col[c] for c in took))
            left -= took
    return out


def never_repeats(found, cells, limit):
    """Whether ``found`` (closed) proves no two entries of calls 0 to ``limit`` alike on ``cells``.

    Those of ``cells`` in a segment's state are, at the entry of call ``k``, as at its
    cycle's entry. Entries whose hashes of them differ are not alike; False is no finding."""
    calls = np.arange(limit + 1)
    total = np.zeros(limit + 1, np.uint64)
    for n, ((mu, lam), cols) in enumerate(_by_cycle(found, cells).items()):
        entry = np.where(calls < mu + lam, calls, mu + (calls - mu) % lam)
        total += _hashed(found.rows[: mu + lam, cols])[entry] * np.uint64(2 * n + 3)
    return len(np.unique(total)) == limit + 1
