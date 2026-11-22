"""Static flow helpers of jennings.coverage.cfg and the coverage driver's raster counter."""

import itertools

import networkx as nx
import numpy as np
import pytest

import jennings.coverage
from jennings.coverage import FALLTHRU, RETURN, TAKEN, CoverageMPU, call
from jennings.coverage.cfg import (
    LEN,
    STOPS,
    control_dependence,
    control_operands,
    instruction_bytes,
    one_way,
    return_edges,
    successors,
)
from jennings.opcodes import OPCODES


def test_instruction_bytes_marks_each_instruction_and_wraps():
    rng = np.random.default_rng(3)
    mem = rng.integers(0, 256, 0x10000, dtype=np.uint8)
    starts = rng.random(0x10000) < 0.01
    starts[[0xFFFE, 0xFFFF]] = True
    mem[0xFFFE], mem[0xFFFF] = 0xAD, 0x20
    want = np.zeros(0x10000, bool)
    for pc in np.flatnonzero(starts):
        for k in range(LEN[mem[pc]]):
            want[(pc + k) & 0xFFFF] = True
    assert want[[0, 1]].all()
    for m in (mem, bytes(mem), list(mem.tolist())):
        assert (instruction_bytes(m, starts) == want).all()


def test_control_operands_are_transfer_operands_and_wrap():
    """The operand bytes of branches, JMP (both modes) and JSR only, by mnemonic and mode."""
    rng = np.random.default_rng(4)
    mem = rng.integers(0, 256, 0x10000, dtype=np.uint8)
    starts = rng.random(0x10000) < 0.05
    starts[0xFFFE] = True
    mem[0xFFFE] = 0x4C
    want = np.zeros(0x10000, bool)
    for pc in np.flatnonzero(starts):
        name, md = OPCODES[mem[pc]]
        if md == "rel" or name in ("JMP", "JSR"):
            want[(pc + np.arange(1, LEN[mem[pc]])) & 0xFFFF] = True
    assert want[[0xFFFF, 0]].all() and want.sum() < instruction_bytes(mem, starts).sum()
    for m in (mem, bytes(mem)):
        assert (control_operands(m, starts) == want).all()


def test_stops_never_fall_through():
    """STOPS are the opcodes whose static successors omit the next instruction."""
    mem = np.zeros(0x10000, np.uint8)
    falls = set()
    for op in range(256):
        mem[0x1000], mem[0x1001], mem[0x1002] = op, 0x10, 0x20
        nxt = 0x1000 + LEN[op]
        if any(t == nxt for _, t in successors(mem, 0x1000)[0]):
            falls.add(op)
    assert STOPS == frozenset(range(256)) - falls
    assert {OPCODES[op][0] for op in STOPS} == {"BRK", "JMP", "RTI", "RTS", "JAM"}


def _flow(mem, entry):
    """``(succ, calls)`` of the code reachable from ``entry``: a JSR's successor is its
    callee, and its return site is explored too."""
    succ, calls, work = {}, {}, [entry]
    while work:
        pc = work.pop()
        if pc in succ:
            continue
        out, _, callee = successors(mem, pc)
        succ[pc] = [callee] if callee is not None else [t for _, t in out]
        if callee is not None:
            calls.setdefault(callee, set()).add((pc + 3) & 0xFFFF)
            work.append((pc + 3) & 0xFFFF)
        work.extend(succ[pc])
    return succ, calls


def test_return_edges_route_each_rts_to_its_callers():
    mem = bytearray(0x10000)
    mem[0x1000:0x1007] = [0x20, 0x00, 0x20, 0x20, 0x00, 0x20, 0x60]
    # $2000: BEQ $2003 ; RTS ; JSR $3000 ; RTS. $3000: RTS.
    mem[0x2000:0x2007] = [0xF0, 0x01, 0x60, 0x20, 0x00, 0x30, 0x60]
    mem[0x3000] = 0x60
    succ, calls = _flow(mem, 0x1000)
    assert calls == {0x2000: {0x1003, 0x1006}, 0x3000: {0x2006}}
    assert return_edges(mem, succ, calls) == {
        0x2002: {0x1003, 0x1006},
        0x2006: {0x1003, 0x1006},
        0x3000: {0x2006},
    }


def _postdominates(g, n, s, exit_):
    """Brute force: every path from ``s`` to ``exit_`` passes ``n``."""
    return n == s or not nx.has_path(g.subgraph(set(g) - {n}), s, exit_)


@pytest.mark.parametrize("seed", range(40))
def test_control_dependence_is_ferrante_ottenstein_warren(seed):
    """A node depends on branch ``b`` when it postdominates a successor of ``b`` but does not
    strictly postdominate ``b`` (random graphs whose every node reaches the sink)."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 12))
    edges = {(i, int(rng.integers(i + 1, n))) for i in range(n - 1)}
    edges |= {tuple(int(v) for v in rng.integers(0, n - 1, 2)) for _ in range(n // 2)}
    g = nx.DiGraph(edges)
    sinks = [v for v in g if g.out_degree(v) == 0]
    full = g.copy()
    full.add_edges_from((v, -1) for v in sinks)
    want = {}
    for b in g:
        if g.out_degree(b) < 2:
            continue
        for s, v in itertools.product(g.successors(b), g):
            if _postdominates(full, v, s, -1) and not (v != b and _postdominates(full, v, b, -1)):
                want.setdefault(v, set()).add(b)
    assert control_dependence(edges) == want


def test_control_dependence_of_a_diamond_and_a_loop():
    assert control_dependence([(1, 2), (1, 3), (2, 4), (3, 4)]) == {2: {1}, 3: {1}}
    assert control_dependence([(1, 2), (2, 1), (2, 3)]) == {1: {2}, 2: {2}}
    assert control_dependence([("a", "b"), ("a", "c")], exit_="X") == {"b": {"a"}, "c": {"a"}}


# LDA $10 ; BEQ +2 ; LDX #1 ; LDY $11 ; BNE -2 (to itself) ; RTS
BRANCHES = [0xA5, 0x10, 0xF0, 0x02, 0xA2, 0x01, 0xA4, 0x11, 0xD0, 0xFE, 0x60]


def _covered(m10, m11=0):
    mpu = CoverageMPU()
    mpu.memory[0x1000 : 0x1000 + len(BRANCHES)] = BRANCHES
    mpu.memory[0x10], mpu.memory[0x11] = m10, m11
    call(mpu, 0x1000)
    return mpu


def test_one_way_names_the_untaken_target():
    mem = _covered(0).memory
    assert jennings.coverage.one_way is one_way
    assert one_way(_covered(0).cov, mem) == {(0x1002, 0x1004), (0x1008, 0x1008)}
    assert one_way(_covered(1).cov, mem) == {(0x1002, 0x1006), (0x1008, 0x1008)}
    both = _covered(0).cov
    both |= _covered(1).cov
    assert one_way(both, mem) == {(0x1008, 0x1008)}
    bt = both.flags & (TAKEN | FALLTHRU)
    assert bt[0x1002] == TAKEN | FALLTHRU


# $D012 poll: LDA $D012 ; CMP #$03 ; BNE -7 ; INC $10 ; LDA $D012 ; RTS
POLL = [0xAD, 0x12, 0xD0, 0xC9, 0x03, 0xD0, 0xF9, 0xE6, 0x10, 0xAD, 0x12, 0xD0, 0x60]


def _init_cover(mpu, init, song, steps=1_000_000):
    """gravity-inversion's disasm init driver: a raster counter at $D012 advancing every 64
    steps for polls."""
    sp = mpu.sp
    mpu._push_word((RETURN - 1) & 0xFFFF)
    mpu.a, mpu.pc = song, init
    for n in range(steps):
        mpu.memory[0xD012] = n >> 6 & 0xFF
        mpu.step()
        if mpu.pc == RETURN and mpu.sp == sp:
            return
    raise RuntimeError(f"init ${init:04X} did not return")


def _poller():
    mpu = CoverageMPU()
    mpu.memory[0x1000 : 0x1000 + len(POLL)] = POLL
    return mpu


def _state(mpu):
    c = mpu.cov
    return (
        bytes(mpu.memory),
        (mpu.a, mpu.x, mpu.y, mpu.sp, mpu.pc, mpu.p, mpu.processorCycles),
        c.flags.tobytes(),
        c.hits.tobytes(),
        c.edges,
        c.entries,
    )


def test_call_raster_is_the_init_cover_counter():
    ref, got = _poller(), _poller()
    _init_cover(ref, 0x1000, 7)
    call(got, 0x1000, a=7, raster=64)
    assert _state(got) == _state(ref)
    assert got.a == 3 and got.memory[0x10] == 1 and got.cov.hits[0x1000] > 64
    plain = _poller()
    with pytest.raises(RuntimeError):
        call(plain, 0x1000, max_steps=1000)
    assert plain.memory[0xD012] == 0
    fast = _poller()
    call(fast, 0x1000, raster=1)
    assert fast.cov.hits[0x1000] < 4
