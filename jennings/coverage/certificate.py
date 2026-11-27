"""A cover claim as a certificate, and its check by replay (``docs/coverage.md``).

:func:`certify` records what :func:`~jennings.coverage.components.cover_calls` claims,
:func:`verify` checks it from the certificate alone and :func:`closure` runs on to the first
repeat of the whole state.
"""

import hashlib
import math
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from tqdm import tqdm

from jennings import memo
from jennings.coverage import components as C
from jennings.devices.mpu6502 import REG, Census, Learner, periods_of

__all__ = [
    "Certificate",
    "Open",
    "Verdict",
    "Closure",
    "CARRIED",
    "REGISTERS",
    "TAINTS",
    "state_digest",
    "remap_digest",
    "certify",
    "verify",
    "closure",
]

#: Registers a call hands the next that the taint gives no entry taint: SP and B.
CARRIED = (REG + 3, REG + 12)
#: Every register a call hands the next (A is set at each call): X, Y, SP and the flags.
REGISTERS = tuple(REG + i for i in (1, 2, 3, 8, 9, 10, 11, 12, 13, 14))
_STAGE = 4
_LOAD, _STORE = C.ACCESSES.index("load"), C.ACCESSES.index("store")
_SITE = {0: "address", 1: "address", 2: "control"}
_WHAT = {0: "the address of the load", 1: "the address of the store", 2: "where control goes"}


@dataclass(frozen=True)
class Certificate:
    """What a cover claim asserts for one state and entry.

    Attributes:
        state: :func:`state_digest` of the state the calls start from.
        pc: the entry called.
        acc: A at each call.
        max_cycles: the cycles a call may take.
        exclude: ``(lo, hi)`` half-open address ranges taken to hold no state between calls.
        remap: the :func:`remap_digest` the calls ran under, None for no remap.
        feed: the locations every address and control transfer is a function of, ascending.
        preperiod: the first call whose entry the feed locations repeat at.
        period: the calls from it to that repeat.
        accesses: ``(pc, kind)`` (:data:`~jennings.coverage.components.LOAD` or
            :data:`~jennings.coverage.components.STORE`) to the addresses the instruction's
            data accesses take, ascending, as it names them (before any remap).
        edges: ``(pc, next)`` of consecutive instructions within a call, ascending.
    """

    state: str
    pc: int
    acc: int
    max_cycles: int
    exclude: tuple
    remap: str | None
    feed: tuple
    preperiod: int
    period: int
    accesses: dict = field(repr=False)
    edges: tuple = field(repr=False)

    @property
    def calls(self):
        """The calls claimed to make every access and take every edge."""
        return self.preperiod + self.period

    def to_dict(self):
        """This certificate as JSON-serialisable values."""
        return {
            "state": self.state,
            "pc": self.pc,
            "acc": self.acc,
            "max_cycles": self.max_cycles,
            "exclude": [list(r) for r in self.exclude],
            "remap": self.remap,
            "feed": list(self.feed),
            "preperiod": self.preperiod,
            "period": self.period,
            "accesses": [[pc, kind, list(at)] for (pc, kind), at in sorted(self.accesses.items())],
            "edges": [list(e) for e in self.edges],
        }

    @classmethod
    def from_dict(cls, d):
        """The certificate :meth:`to_dict` gave ``d`` of."""
        return cls(
            state=d["state"],
            pc=d["pc"],
            acc=d["acc"],
            max_cycles=d["max_cycles"],
            exclude=tuple((lo, hi) for lo, hi in d["exclude"]),
            remap=d["remap"],
            feed=tuple(d["feed"]),
            preperiod=d["preperiod"],
            period=d["period"],
            accesses={(pc, kind): tuple(at) for pc, kind, at in d["accesses"]},
            edges=tuple((a, b) for a, b in d["edges"]),
        )


@dataclass(frozen=True)
class Open:
    """Why :func:`certify` found no cover point within its limit.

    Attributes:
        calls: the entries observed, the feed locations alike at no two: their preperiod plus
            period is at least this.
        feed: the feed locations as learned.
        unrepeated: sets of feed locations, each depending on every other of its set, that
            with the locations they depend on repeated at no entry although those did.
        period: the least common multiple of the periods of the sets that did repeat with the
            locations they depend on; the feed locations' period is a multiple of it.
    """

    calls: int
    feed: tuple
    unrepeated: tuple
    period: int


@dataclass(frozen=True)
class Verdict:
    """What :func:`verify` found.

    Attributes:
        ok: whether every condition holds.
        failed: the first condition that does not, None where all do: ``"input"`` (the state
            or settings are not the certificate's), ``"replay"`` (a call did not return),
            ``"repeat"`` (the feed locations at ``preperiod`` and ``preperiod + period`` differ,
            or repeat before), ``"accesses"``, ``"edges"`` (the replay makes one the certificate
            lacks), ``"address"``, ``"control"`` (a location outside the feed set taints one)
            or ``"autonomy"`` (one taints a feed location's end-of-call value).
        detail: that failure in words.
        calls: the calls replayed.
        accesses: ``(pc, kind)`` to the addresses the certificate lists that no replayed call
            takes.
        edges: the edges the certificate lists that no replayed call takes.
        assumed: locations in the certificate's ``exclude`` ranges that some call reads before
            writing, some call writes, and that taint an address, a control transfer or a feed
            location: the claim holds where they carry no state.
    """

    ok: bool
    failed: str | None = None
    detail: str = ""
    calls: int = 0
    accesses: dict = field(default_factory=dict)
    edges: tuple = ()
    assumed: tuple = ()


@dataclass(frozen=True)
class Closure:
    """What :func:`closure` found.

    Attributes:
        end: ``"closed"`` where the whole state repeated, else as
            :meth:`~jennings.devices.mpu6502.NativeMPU.orbit` ends (``"open"`` at the limit).
        calls: the calls run.
        accesses: ``(pc, kind)`` to the addresses taken that the certificate lacks.
        edges: the edges taken that the certificate lacks.
    """

    end: str
    calls: int
    accesses: dict
    edges: tuple

    @property
    def ok(self):
        """Whether the state closed with no access or edge beyond the certificate."""
        return self.end == "closed" and not self.accesses and not self.edges


def state_digest(state):
    """The SHA-256 (hex) of ``state`` (see :func:`~jennings.coverage.components.machine`): its
    RAM, X, Y, SP and P."""
    m = C.machine(state)
    return hashlib.sha256(bytes(m.memory) + bytes([m.x, m.y, m.sp, m.p])).hexdigest()


def remap_digest(tables, pointer_opcodes=()):
    """The SHA-256 (hex) of the arguments of a :class:`~jennings.devices.mpu6502.Remap`."""
    h = hashlib.sha256()
    for pc in sorted(tables):
        h.update(int(pc).to_bytes(2, "little"))
        h.update(np.ascontiguousarray(tables[pc], "<i4").tobytes())
    h.update(bytes(sorted(set(pointer_opcodes))))
    return h.hexdigest()


def _spans(exclude):
    """``exclude`` (``range`` objects of step 1 or ``(lo, hi)`` pairs) as sorted ``(lo, hi)``
    tuples."""
    if any(isinstance(r, range) and r.step != 1 for r in exclude):
        raise ValueError("an excluded range has step 1")
    return C.spans(exclude)


def _remap_id(remap, remap_id):
    if (remap is None) != (remap_id is None):
        raise ValueError("a remap comes with its remap_id (remap_digest), and only then")
    return remap_id


def _first_repeat(rows, calls):
    """The first row of ``rows`` equal to row ``calls``."""
    return int(np.flatnonzero((rows[:calls] == rows[calls]).all(axis=1))[0])


def _grouped(keys, shift):
    """``{key >> shift: [key & mask]}`` of ascending ``keys``."""
    keys = np.asarray(keys, np.int64)
    head = keys >> shift
    start = np.flatnonzero(np.diff(head, prepend=-1))
    ends = np.append(start[1:], len(keys))
    low = keys & ((1 << shift) - 1)
    return {int(head[a]): low[a:b].tolist() for a, b in zip(start, ends)}


def _accesses(keys):
    """``{(pc, kind): addresses}`` of access keys ``pc << 17 | store << 16 | address``."""
    kinds = (C.LOAD, C.STORE)
    return {(k >> 1, kinds[k & 1]): tuple(at) for k, at in _grouped(np.unique(keys), 16).items()}


def _access_keys(accesses):
    """Ascending access keys of ``{(pc, kind): addresses}``."""
    out = [
        (np.int64(pc) << 17 | (kind == C.STORE) << 16) | np.asarray(at, np.int64)
        for (pc, kind), at in accesses.items()
    ]
    return np.unique(np.concatenate(out)) if out else np.zeros(0, np.int64)


def _edge_keys(edges):
    e = np.asarray(edges, np.int64).reshape(-1, 2)
    return np.unique(e[:, 0] << 16 | e[:, 1])


def _edges(keys):
    return tuple(C.edge_pairs(keys))


def _census_accesses(ran, calls):
    """``{(pc, kind): addresses}`` of the data accesses of the first ``calls`` of ``ran``."""
    got = np.unique(ran.seen("accesses", calls)[0].astype(np.int64))
    out = {}
    for number, kind in ((_LOAD, C.LOAD), (_STORE, C.STORE)):
        mine = got[(got >> 16) & 7 == number]
        by_pc = _grouped((mine >> 19) << 16 | mine & 0xFFFF, 16)
        out.update({(pc, kind): tuple(at) for pc, at in by_pc.items()})
    return out


def _open(comps, ran):
    """:class:`Open` of ``comps`` on the entries ``ran`` recorded."""
    rows = np.ascontiguousarray(ran.rows(None, comps.persistent))
    col = {c: i for i, c in enumerate(comps.persistent)}
    g = nx.DiGraph()
    g.add_nodes_from(comps.feed)
    g.add_edges_from(
        (s, c) for c in comps.feed for s in comps.deps.get(c, ()) if s in comps.feed and s != c
    )
    cond = nx.condensation(g)
    nodes = list(cond)
    sets = [
        set(cond.nodes[n]["members"]).union(
            *(cond.nodes[a]["members"] for a in nx.ancestors(cond, n))
        )
        for n in nodes
    ]
    got = periods_of(rows, [sorted(col[c] for c in s) for s in sets]) if sets else ()
    first = {n: p for n, (p, _t) in zip(nodes, got)}
    unrepeated = [
        tuple(sorted(cond.nodes[n]["members"]))
        for n in nodes
        if first[n] is None and all(first[p] is not None for p in cond.predecessors(n))
    ]
    period = math.lcm(*(p[1] for p in first.values() if p is not None))
    return Open(len(rows), tuple(sorted(comps.feed)), tuple(sorted(unrepeated)), period)


# pylint: disable-next=too-many-arguments
def certify(
    state,
    pc,
    limit,
    *,
    acc=0,
    max_cycles=C.MAX_CYCLES,
    remap=None,
    remap_id=None,
    exclude=(C.STACK, C.IO),
    learn=C.CHUNK,
):
    """The :class:`Certificate` of the cover point of ``pc`` called from ``state``, as
    :func:`~jennings.coverage.components.cover` finds it; an :class:`Open` where the feed
    locations do not repeat within ``limit`` calls. Raises :class:`RuntimeError` where a call
    does not return or the feed locations do not settle.

    Args:
        state: as :func:`~jennings.coverage.components.machine` takes it.
        acc, max_cycles, remap, exclude: as
            :func:`~jennings.coverage.components.cover` takes them.
        remap_id: the :func:`remap_digest` of ``remap``, given with it.
        learn: the calls the components are first learned over (``cover``'s ``first``).
    """
    remap_id, spans = _remap_id(remap, remap_id), _spans(exclude)
    calls, comps, ran = C.cover(
        state,
        pc,
        limit,
        acc=acc,
        max_cycles=max_cycles,
        remap=remap,
        exclude=tuple(range(lo, hi) for lo, hi in spans),
        first=learn,
    )
    if calls is None:
        return _open(comps, ran)
    feed = tuple(sorted(comps.feed))
    preperiod = _first_repeat(ran.rows(calls + 1, feed), calls)
    return Certificate(
        state=state_digest(state),
        pc=pc,
        acc=acc,
        max_cycles=max_cycles,
        exclude=spans,
        remap=remap_id,
        feed=feed,
        preperiod=preperiod,
        period=calls - preperiod,
        accesses=_census_accesses(ran, calls),
        edges=_edges(np.unique(ran.seen("edges", calls)[0])),
    )


def _mismatch(state, cert, remap_id):
    """What of ``state`` and ``remap_id`` is not ``cert``'s, None where all is."""
    if state_digest(state) != cert.state:
        return "the state is not the one certified"
    if remap_id != cert.remap:
        return "the remap is not the one certified"
    if cert.preperiod < 0 or cert.period < 1:
        return "a preperiod is at least 0 and a period at least 1"
    return None


#: The element type of each array of a learner's result that is a union over its calls.
_LEARNED = {
    "sites": np.uint64,
    "deps": np.uint64,
    "edges": np.uint64,
    "written": np.uint16,
    "registers_read": np.uint32,
    "registers_written": np.uint32,
}


class _Learned:  # pylint: disable=too-few-public-methods
    """The union of learners' results, as one learner's over all their calls.

    Each array of :data:`_LEARNED` is the union of the learners'. ``persistent`` is not a union
    of theirs, being the registers some call reads before writing that some call, maybe
    another learner's, writes; it holds only registers (the memory some call writes is
    ``written``)."""

    def __init__(self, parts):
        got = {
            k: np.unique(np.concatenate([np.frombuffer(p[k], t) for p in parts]))
            for k, t in _LEARNED.items()
        }
        got["persistent"] = np.intersect1d(got["registers_read"], got["registers_written"])
        self._got = {k: v.tobytes() for k, v in got.items()}

    def learned(self):
        """The arrays of :data:`_LEARNED` and ``persistent``, each sorted."""
        return self._got


def _taint_calls(state, cert, written, remap):
    """Data-only :class:`~jennings.devices.mpu6502.Learner` results over every certified call.

    The calls are cut into one run per processor, each tainted from the entry a plain replay
    reaches it at: a call's taint starts from its own entry, so the runs share nothing."""
    args = (cert.pc, cert.acc, cert.max_cycles)
    jobs = max(1, min(len(os.sched_getaffinity(0)), cert.calls // C.CHUNK))
    cuts = [cert.calls * k // jobs for k in range(jobs + 1)]
    m, starts = C.machine(state), []
    for a, b in zip(cuts, cuts[1:]):
        starts.append(C.machine(m))
        if b < cert.calls:
            C.drive(lambda n: m.trajectory(*args, n, [cert.pc], remap)[0], b - a, "replay")

    with tqdm(
        total=cert.calls, desc="verify", unit="call", disable=cert.calls <= C.CHUNK, mininterval=1
    ) as progress:
        lock = threading.Lock()

        def update(n):
            with lock:
                progress.update(n)

        def taint(k):
            learner, m = Learner({}, written), starts[k]
            C.drive(lambda n: m.learn(learner, *args, n, remap), cuts[k + 1] - cuts[k], "", update)
            return learner.learned()

        with ThreadPoolExecutor(jobs) as pool:
            return _Learned(list(pool.map(taint, range(jobs))))


def _taint_classes(state, cert, written, remap):
    """As :func:`_taint_calls`, over the first call of each class of a census of the calls."""
    cen, m = Census([]), C.machine(state)
    C.drive(
        lambda n: m.census(cen, cert.pc, cert.acc, cert.max_cycles, n, remap),
        cert.calls,
        "classes",
    )
    learner, m = Learner({}, written), C.machine(state)
    for cls in tqdm(
        range(cen.classes), desc="verify", unit="class", disable=cen.classes <= C.CHUNK
    ):
        m.restore(cen, cls)
        m.learn(learner, cert.pc, cert.acc, cert.max_cycles, 1, remap)
    return learner


_TAINT = {"calls": _taint_calls, "classes": _taint_classes}
#: The taints :func:`verify` takes.
TAINTS = tuple(_TAINT)


def _outside(got, cert, written):
    """``(sites, deps, assumed)`` of a learner's result ``got``: what state outside ``cert``'s
    feed set taints.

    Memory is state where ``written`` (some call writes it), a register where the learner has it
    persistent; the rest holds one value at every entry.

    Returns:
        sites: ``(pc << 2 | kind, loc)``, ``loc`` not excluded tainting an address or control.
        deps: ``(loc, src)``, ``src`` not excluded tainting feed location ``loc``.
        assumed: the excluded locations tainting either.
    """
    held = np.array(sorted(set(cert.feed) | set(CARRIED)), np.int64)
    persistent = np.frombuffer(got["persistent"], np.uint32).astype(np.int64)

    def state(loc):
        return np.where(loc < REG, np.isin(loc, written), np.isin(loc, persistent))

    s = np.frombuffer(got["sites"], np.uint64).astype(np.int64)
    key, loc = s >> C.SITE_BITS, s & ((1 << C.SITE_BITS) - 1)
    at = (key & 3 != 3) & state(loc) & ~np.isin(loc, held)
    key, loc = key[at], loc[at]
    d = np.frombuffer(got["deps"], np.uint64).astype(np.int64)
    dst, src = d >> 32, d & 0xFFFFFFFF
    at = np.isin(dst, held) & (src != dst) & state(src) & ~np.isin(src, held)
    dst, src = dst[at], src[at]
    ex_s, ex_d = C.excluded(loc, cert.exclude), C.excluded(src, cert.exclude)
    assumed = np.union1d(loc[ex_s], src[ex_d])
    return (
        list(zip(key[~ex_s].tolist(), loc[~ex_s].tolist())),
        list(zip(dst[~ex_d].tolist(), src[~ex_d].tolist())),
        tuple(assumed.tolist()),
    )


def _replay(state, cert, remap):
    """``(failure, accesses, edges)`` of the certified calls: their data access and edge keys.

    ``failure`` is the ``(condition, detail)`` of a call that does not return or of feed
    locations that do not first repeat at the certified entries, else None."""
    held = sorted(set(cert.feed) | set(CARRIED))
    args = (cert.pc, cert.acc, cert.max_cycles)
    end, states, acc, edges = C.machine(state).orbit(*args, cert.calls, cells=held, remap=remap)
    acc, edges = np.asarray(acc, np.int64), np.asarray(edges, np.int64)
    if end not in ("open", "closed"):
        return ("replay", "call %d ended as %r" % (states - 1, end)), acc, edges
    stop, rows = C.machine(state).trajectory(*args, cert.calls + 1, held, remap)
    if stop != "sp":
        return ("replay", "call %d ended at %r, not its RTS" % (len(rows) - 1, stop)), acc, edges
    differ = np.flatnonzero(rows[cert.preperiod] != rows[cert.calls])
    if differ.size:
        loc = C.location(held[int(differ[0])])
        how = "%s differs at the entries of calls %d and %d" % (loc, cert.preperiod, cert.calls)
        return ("repeat", how), acc, edges
    if states < cert.calls:
        how = "the feed locations already repeat at the entry of call %d" % states
        return ("repeat", how), acc, edges
    return None, acc, edges


def _uncertified(name, key):
    """The detail of access or edge ``key`` the certificate lacks."""
    if name == "accesses":
        kind = ("load", "store")[key >> 16 & 1]
        how = "the %s at $%04X of $%04X" % (kind, key >> 17, key & 0xFFFF)
    else:
        how = "the edge from $%04X to $%04X" % (key >> 16, key & 0xFFFF)
    return how + " is not certified"


def verify(state, certificate, *, remap=None, remap_id=None, taint="calls"):
    """The :class:`Verdict` on ``certificate`` for ``state``, by replay of its calls.

    Checked in the order :attr:`Verdict.failed` lists; the taint is by data alone, of every
    location some replayed call writes.

    Args:
        remap, remap_id: the remap certified under and its :func:`remap_digest`, if any.
        taint: ``"calls"`` taints every call, ``"classes"`` the first call of each class of a
            census of them.
    """
    cert = certificate
    why = _mismatch(state, cert, _remap_id(remap, remap_id))
    if why:
        return Verdict(False, "input", why)
    failure, acc, edges = _replay(state, cert, remap)
    if failure:
        return Verdict(False, *failure, calls=cert.calls)
    mine, theirs = _access_keys(cert.accesses), _edge_keys(cert.edges)
    found = {
        "calls": cert.calls,
        "accesses": _accesses(np.setdiff1d(mine, acc)),
        "edges": _edges(np.setdiff1d(theirs, edges)),
    }
    for name, new in (
        ("accesses", np.setdiff1d(acc, mine)),
        ("edges", np.setdiff1d(edges, theirs)),
    ):
        if new.size:
            return Verdict(False, name, _uncertified(name, int(new[0])), **found)
    stores = acc[acc >> 16 & 1 == 1] & 0xFFFF
    written = np.union1d(stores, np.arange(C.STACK.start, C.STACK.stop))
    got = _TAINT[taint](state, cert, written.tolist(), remap).learned()
    wrote = np.frombuffer(got["written"], np.uint16).astype(np.int64)
    if np.setdiff1d(wrote, written).size or not np.array_equal(
        np.frombuffer(got["edges"], np.uint64).astype(np.int64), edges
    ):
        return Verdict(False, "replay", "the tainted calls are not the calls replayed", **found)
    sites, deps, assumed = _outside(got, cert, wrote)
    found["assumed"] = assumed
    if sites:
        key, loc = min(sites, key=lambda s: (s[0] & 3 != 2, s))
        how = "%s at $%04X is tainted by %s, outside the feed set"
        how %= (_WHAT[key & 3], key >> 2, C.location(loc))
        return Verdict(False, _SITE[key & 3], how, **found)
    if deps:
        dst, src = deps[0]
        how = "%s is tainted by %s, outside the feed set" % (C.location(dst), C.location(src))
        return Verdict(False, "autonomy", how, **found)
    return Verdict(True, **found)


def _orbit(m, cert, limit, **state):
    """:meth:`~jennings.devices.mpu6502.NativeMPU.orbit` of ``cert``'s calls on ``m``."""
    return memo.orbit(m, cert.pc, cert.acc, cert.max_cycles, limit, **state)[:4]


def closure(state, certificate, limit, *, remap=None, remap_id=None):
    """The :class:`Closure` of ``certificate``'s calls from ``state`` within ``limit`` calls.

    The state is every register a call hands the next, the stack page and the memory the calls
    store to. The calls run in stages, each from ``state`` over more calls than the last; a
    stage that stores to an address not yet in the state is run again with it. Each stage is
    :func:`jennings.memo.orbit`, which gives what
    :meth:`~jennings.devices.mpu6502.NativeMPU.orbit` gives."""
    cert = certificate
    why = _mismatch(state, cert, _remap_id(remap, remap_id))
    if why:
        raise ValueError(why)
    mine, theirs = _access_keys(cert.accesses), _edge_keys(cert.edges)
    stores = mine[mine >> 16 & 1 == 1] & 0xFFFF
    n = min(limit, max(_STAGE * cert.calls, C.CHUNK))
    with tqdm(
        total=limit, desc="closure", unit="call", disable=limit <= C.CHUNK, mininterval=1
    ) as progress:
        while True:
            cells = np.union1d(stores, np.arange(C.STACK.start, C.STACK.stop)).tolist()
            end, states, acc, edges = _orbit(
                C.machine(state), cert, n, cells=cells + list(REGISTERS), remap=remap
            )
            acc, edges = np.asarray(acc, np.int64), np.asarray(edges, np.int64)
            new = np.setdiff1d(acc[acc >> 16 & 1 == 1] & 0xFFFF, stores)
            if new.size:
                stores = np.union1d(stores, new)
                continue
            calls = states - (end == "open")
            progress.update(calls - progress.n)
            if end != "open" or n >= limit:
                break
            n = min(limit, _STAGE * n)
    return Closure(
        end, calls, _accesses(np.setdiff1d(acc, mine)), _edges(np.setdiff1d(edges, theirs))
    )
