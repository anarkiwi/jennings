"""An entry's orbit without emulating what is already known.

:func:`orbit` gives what :meth:`~jennings.devices.mpu6502.NativeMPU.orbit` gives, from
determinism alone: calls are cut into units, each memoised on exactly the registers and written
memory it read, so a unit whose reads match a stored one's is replayed rather than run, and a
repeated state is confirmed by comparing the states (see ``docs/memo.md``).
"""

#: Where calls are cut into units: a whole call; at subroutine entries and returns; at those
#: and backward branch targets; or at those and inside the units of a start PC that come to be
#: many.
GRAINS = ("call", "sub", "loop", "split")


# pylint: disable-next=too-many-arguments,too-many-positional-arguments
def orbit(mpu, pc, acc, max_cycles, limit, grain="split", **state):
    """``(end, states, accesses, edges, stats)`` of calls of ``pc`` on the native ``mpu``.

    The first four are those of :meth:`~jennings.devices.mpu6502.NativeMPU.orbit` with the
    same arguments (``state``: its ``keep``, ``volatile``, ``remap`` and ``cells``); ``stats`` counts the work: ``calls`` run (those run again to confirm a
    repeat included), units ``hits`` replayed and ``misses`` emulated, ``steps`` instructions
    emulated, ``entries`` units stored, ``live`` units held, ``flushes`` of the memo, ``splits``
    made, ``raw`` start PCs no longer memoised, ``dropped`` units, ``collisions`` of the state
    hash, ``bytes`` held, and the ``roots`` and ``fanout`` that tell where the held units are
    (``docs/memo.md``). ``grain`` is one of :data:`GRAINS`. The memory must have nothing
    mapped, watched or observed.
    """
    core = mpu._core
    *found, stats = core.memo_orbit(pc, acc, max_cycles, limit, grain=GRAINS.index(grain), **state)
    return found[0], found[1], memoryview(found[2]).cast("Q"), memoryview(found[3]).cast("Q"), stats
