"""State a repeatedly called routine carries that nothing follows, and its counters.

:func:`loose` gives the memory that carries state from call to call and taints no address,
control transfer or location that does: what to set apart as a memoising orbit's fibre.
:func:`recognise` gives the cells every store steps by a constant, and how each is used.
"""

import math
from dataclasses import dataclass

import numpy as np

from jennings.coverage import components as C
from jennings.coverage import segments as S
from jennings.devices.mpu6502 import REG

__all__ = ["Counter", "UNREAD", "READ", "COMPARED", "INDEX", "loose", "recognise", "progression"]

#: How a counter's value is used: by nothing; in values alone; in where control goes; in an
#: address.
UNREAD, READ, COMPARED, INDEX = "unread", "read", "compared", "index"
_DECIMAL = REG + 11
#: Opcodes that step the byte at their zero-page or absolute operand by one.
_STEPS = {0xE6: (1, 1), 0xEE: (1, 2), 0xC6: (-1, 1), 0xCE: (-1, 2)}
#: Opcodes loading A from, and storing A to, a zero-page or absolute operand, by its bytes.
_LOADS, _STORES = {1: 0xA5, 2: 0xAD}, {0x85: 1, 0x8D: 2}
_CLC, _SEC, _ADC, _SBC = 0x18, 0x38, 0x69, 0xE9


@dataclass(frozen=True)
class Counter:
    """A cell whose every store adds a constant to it.

    Attributes:
        cell: its address.
        steps: ``(pc, step)`` of each store site: what a store there adds, as a signed byte.
        use: :data:`UNREAD`, :data:`READ`, :data:`COMPARED` or :data:`INDEX`.
        sites: ``(pc, kind)`` of the instructions its value taints, other than its stores:
            an address (:data:`~jennings.coverage.components.LOAD`, ``STORE``), where control
            goes (``CONTROL``) or a byte stored (``VALUE``).
        period: the calls after which it holds what it held, where every store site runs a
            fixed number of times in every call; None where the calls differ in that.
        indexed: ``(pc, first, step, count)`` per load or store it indexes whose addresses
            over the calls are that arithmetic progression.
    """

    cell: int
    steps: tuple
    use: str
    sites: tuple
    period: int | None
    indexed: tuple


def loose(state, pc, calls=C.CHUNK, **kw):
    """The memory cells, ascending, that carry state over ``calls`` calls of ``pc`` from
    ``state`` and are outside the feed set: they taint no address and no control transfer
    (``kw`` as :func:`~jennings.coverage.components.learn` takes it)."""
    comps = C.learn(state, pc, calls, **kw)
    return tuple(c for c in comps.persistent if c < REG and c not in comps.feed)


def progression(addresses):
    """``(first, step, count)`` of ascending ``addresses`` that are an arithmetic progression,
    None of others; one address steps by 0."""
    at = np.asarray(addresses, np.int64)
    gaps = np.diff(at)
    if at.size == 0 or (gaps.size and (gaps != gaps[0]).any()):
        return None
    return int(at[0]), int(gaps[0]) if gaps.size else 0, int(at.size)


def _operand(ram, pc, size):
    return int.from_bytes(ram[pc + 1 : pc + 1 + size], "little")


def _added(ram, site, cell):
    """What the store at ``site`` adds to ``cell``, None where it is not seen to add a
    constant: an ``INC`` or ``DEC`` of it, or a ``STA`` of it after an ``LDA`` of that mode,
    ``CLC`` or ``SEC`` (either first) and ``ADC`` or ``SBC`` of an immediate."""
    op = ram[site]
    if op in _STEPS:
        step, size = _STEPS[op]
        return step if _operand(ram, site, size) == cell else None
    size = _STORES.get(op)
    if size is None or _operand(ram, site, size) != cell or ram[site - 2] not in (_ADC, _SBC):
        return None
    load, flag = site - 4 - size, site - 3
    if ram[flag] not in (_CLC, _SEC):
        load, flag = site - 3 - size, site - 4 - size
    if ram[flag] not in (_CLC, _SEC) or ram[load] != _LOADS[size]:
        return None
    carry, operand = int(ram[flag] == _SEC), int(ram[site - 1])
    return operand + carry if ram[site - 2] == _ADC else carry - 1 - operand


def _signed(step):
    return (step + 128) % 256 - 128


def _counter(cell, steps, flows, trace):
    """The :class:`Counter` of ``cell``, stored at the sites of ``steps`` (``{pc: step}``)."""
    tainted = sorted(k for k, taint in flows.sites.items() if cell in taint)
    sites = tuple(k for k in tainted if not (k[0] in steps and k[1] == C.VALUE))
    kinds = {kind for _, kind in sites}
    read = any(cell in taint for loc, taint in flows.deps.items() if loc != cell)
    use = UNREAD
    for kind, name in ((C.VALUE, READ), (C.CONTROL, COMPARED), (C.LOAD, INDEX), (C.STORE, INDEX)):
        use = name if kind in kinds else use
    use = READ if use == UNREAD and read else use
    period = None
    if all(pc in trace.arrivals for pc in steps):
        each = sum(step * trace.arrivals[pc] for pc, step in steps.items())
        period = 256 // math.gcd(each % 256, 256)
    indexed = []
    for site, kind in sites:
        if kind in (C.LOAD, C.STORE):
            keys = trace.accesses[trace.accesses >> 16 == site << 1 | (kind == C.STORE)]
            found = progression(keys & 0xFFFF)
            indexed += [(site, *found)] if found else []
    by_site = tuple((pc, _signed(step)) for pc, step in sorted(steps.items()))
    return Counter(cell, by_site, use, sites, period, tuple(indexed))


def recognise(state, pc, calls=C.CHUNK, *, acc=0, max_cycles=C.MAX_CYCLES, remap=None):
    """The :class:`Counter` cells of ``calls`` calls of ``pc`` from ``state``, ascending.

    A counter's stored bytes depend by data on the cell alone (so the carry into an ``ADC``
    is a constant; the decimal flag may be read where it is clear and no call sets it), and
    each of its store sites is seen to add a constant to it and is not itself stored to."""
    kw = {"acc": acc, "max_cycles": max_cycles, "remap": remap}
    flows = C.explicit(state, pc, calls, **kw)
    trace = S.cut(state, pc, calls, (), **kw)
    start = C.machine(state)
    ram = bytes(start.memory)
    kept = flows.deps.get(_DECIMAL, {_DECIMAL}) == {_DECIMAL}
    binary = {_DECIMAL} if kept and not start.p & 8 else set()
    stores = trace.accesses[trace.accesses >> 16 & 1 == 1]
    by_cell = {}
    for key in stores.tolist():
        by_cell.setdefault(key & 0xFFFF, set()).add(key >> 17)
    out = []
    for cell, taint in sorted(flows.stored.items()):
        sites = by_cell.get(cell, ())
        code = {a for site in sites for a in range(site - 6, site + 3)}
        if taint - binary != {cell} or flows.addressed.get(cell) or code & set(by_cell):
            continue
        steps = {site: _added(ram, site, cell) for site in sites}
        if None not in steps.values():
            out.append(_counter(cell, steps, flows, trace))
    return tuple(out)
