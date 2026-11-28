"""An entry's orbit without emulating what is already known.

:func:`orbit` gives what :meth:`~jennings.devices.mpu6502.NativeMPU.orbit` gives, from
determinism alone: calls are cut into units, each memoised on exactly the registers and written
memory it read, so a unit whose reads match a stored one's is replayed rather than run, and a
repeated state is confirmed by comparing the states (see ``docs/memo.md``).
"""

import importlib

#: Where calls are cut into units: a whole call; at subroutine entries and returns; at those
#: and backward branch targets; or at those and inside the units of a start PC that come to be
#: many.
GRAINS = ("call", "sub", "loop", "split")
#: The ``fibre`` that is chosen from the routine: the state it carries that nothing follows.
AUTO = "auto"


def _loose(mpu, pc, **kw):
    """:func:`jennings.coverage.counters.loose`, which needs the ``coverage`` extra, of ``mpu``."""
    return importlib.import_module("jennings.coverage.counters").loose(mpu, pc, **kw)


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def orbit(mpu, pc, acc, max_cycles, limit, grain="split", fibre=None, **state):
    """``(end, states, accesses, edges, stats)`` of calls of ``pc`` on the native ``mpu``.

    All but ``stats``, and the memory and registers left, are those of ``mpu.orbit`` with the
    same ``state`` (its ``keep``, ``volatile``, ``remap`` and ``cells``). ``grain`` is one of
    :data:`GRAINS`, ``fibre`` addresses of tracked cells to set apart (``docs/memo.md``), or
    :data:`AUTO` for those :func:`jennings.coverage.counters.loose` finds from ``mpu``'s state.
    """
    core = mpu._core
    state["grain"] = GRAINS.index(grain)
    if fibre == AUTO:
        fibre = list(_loose(mpu, pc, acc=acc, max_cycles=max_cycles, remap=state.get("remap")))
    *found, stats = core.memo_orbit(pc, acc, max_cycles, limit, fibre=fibre, **state)
    return found[0], found[1], memoryview(found[2]).cast("Q"), memoryview(found[3]).cast("Q"), stats
