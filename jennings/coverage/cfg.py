"""Static control flow of single 6502 instructions, shared by the listing and DWARF mappers."""

import numpy as np

from jennings.coverage import EXEC, FALLTHRU, TAKEN
from jennings.devices.mpu6502 import _JAM
from jennings.isa import is_branch, length, s8

__all__ = ["LEN", "successors", "is_jump"]

LEN = np.array([length(op) for op in range(256)], np.int64)
EXIT = frozenset((0x00, 0x40, 0x60, 0x6C)) | _JAM
JMP, JSR = 0x4C, 0x20


def successors(m, a):
    """``([(flag, target)], jump_target, callee)`` static flow of the instruction at ``a``:
    branches go to ``FALLTHRU`` and ``TAKEN`` targets, ``JMP abs`` to its target, ``JSR`` and
    other instructions to the next one (``JSR`` naming its callee); exits have none."""
    op = int(m[a])
    nxt = (a + int(LEN[op])) & 0xFFFF
    lo = int(m[(a + 1) & 0xFFFF])
    word = lo | int(m[(a + 2) & 0xFFFF]) << 8
    if is_branch(op):
        t = (nxt + s8(lo)) & 0xFFFF
        return [(FALLTHRU, nxt), (TAKEN, t)], t, None
    if op == JMP:
        return [(EXEC, word)], word, None
    if op in EXIT:
        return [], None, None
    return [(EXEC, nxt)], None, word if op == JSR else None


def is_jump(flag, jump):
    """True if successor ``flag`` of an instruction with ``jump`` target is a transfer of
    control (taken branch or ``JMP``) rather than falling through."""
    return flag == TAKEN or (flag == EXEC and jump is not None)
