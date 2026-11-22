"""Static control flow of 6502 instructions, shared by the listing and DWARF mappers: single
instructions' successors, the bytes of executed instructions, return edges, control
dependence and the branches a coverage map saw go one way."""

import collections

import numpy as np

from jennings.coverage._core import EXEC, FALLTHRU, TAKEN
from jennings.devices.mpu6502 import _JAM
from jennings.isa import is_branch, length, s8

__all__ = [
    "LEN",
    "control_dependence",
    "instruction_bytes",
    "is_jump",
    "one_way",
    "return_edges",
    "successors",
]

LEN = np.array([length(op) for op in range(256)], np.int64)
EXIT = frozenset((0x00, 0x40, 0x60, 0x6C)) | _JAM
JMP, JSR, _RTS = 0x4C, 0x20, 0x60


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


def instruction_bytes(mem, starts):
    """bool [65536]: every byte of the instructions starting where the bool mask ``starts``
    is set, wrapping at ``$FFFF``; ``mem`` an array, sequence or buffer of 65536 bytes."""
    pcs = np.flatnonzero(starts)
    m = (
        np.asarray(mem)
        if isinstance(mem, (np.ndarray, list, tuple))
        else np.frombuffer(mem, np.uint8)
    )
    n = LEN[m[pcs]]
    out = np.zeros(0x10000, bool)
    for i in range(3):
        out[(pcs + i)[n > i] & 0xFFFF] = True
    return out


def return_edges(mem, succ, calls):
    """``{rts: {return sites}}``: each ``RTS`` reached from a callee entry of ``calls``
    (``{entry: {return sites}}``) along ``succ`` (``{pc: [successor pcs]}``) without
    entering a nested ``JSR`` (which continues at its own return site) returns to the
    entry's sites."""
    rets = collections.defaultdict(set)
    for entry, sites in calls.items():
        stack, own = [entry], set()
        while stack:
            p = stack.pop()
            if p in own or p not in succ:
                continue
            own.add(p)
            op = int(mem[p])
            if op == _RTS:
                rets[p] |= sites
            elif op == JSR:
                stack.append((p + 3) & 0xFFFF)
            else:
                stack.extend(succ[p])
    return dict(rets)


def control_dependence(edges, exit_=-1):
    """``{node: {branch}}``: the nodes with two or more successors in the graph of ``edges``
    (``(src, dst)`` pairs) each node is directly control dependent on, by the postdominator
    tree with the virtual exit ``exit_`` after every node without successors."""
    import networkx as nx  # pylint: disable=import-outside-toplevel

    g = nx.DiGraph()
    g.add_edges_from(edges)
    rg = g.reverse(copy=True)
    rg.add_edges_from((exit_, n) for n in g if g.out_degree(n) == 0)
    ipdom = nx.immediate_dominators(rg, exit_)
    cd = collections.defaultdict(set)
    for b in g:
        if g.out_degree(b) < 2:
            continue
        for s in g.successors(b):
            p = s
            while p not in (ipdom.get(b), exit_, None):
                cd[p].add(b)
                p = ipdom.get(p)
    return dict(cd)


def _one_way_pcs(cov):
    """``(branch flags, pcs)``: the ``TAKEN | FALLTHRU`` flags of ``cov`` and the branches
    seen going exactly one way."""
    bt = cov.flags & (TAKEN | FALLTHRU)
    return bt, np.flatnonzero((bt == TAKEN) | (bt == FALLTHRU))


def one_way(cov, mem):
    """``{(branch pc, target never taken)}`` of the branches ``cov`` saw go one way only,
    their targets decoded from ``mem``."""
    bt, pcs = _one_way_pcs(cov)
    return {
        (int(pc), t) for pc in pcs for flag, t in successors(mem, int(pc))[0] if not bt[pc] & flag
    }
