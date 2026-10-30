"""Why a branch only went one way: the dynamic backward slice of its condition.

:func:`explain` replays a deterministic driver on a tracing :class:`CoverageMPU`, then walks
the trace backward from the last instance of each half-covered branch through
:func:`jennings.dataflow.flow`, down to the input bytes and constants the condition came from.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from jennings.coverage._core import EXEC, TAKEN, FALLTHRU, CoverageMPU, CoverageVM
from jennings.dataflow import REGISTERS, flow

__all__ = [
    "Constant",
    "Explanation",
    "Explanations",
    "TraceMPU",
    "explain",
    "format_explanations",
    "slice_of",
]

_FLAGS = ("C", "Z", "I", "D", "V", "N")
_NAMES = tuple(REGISTERS.get(i, "r%d" % i) for i in range(16))
_JSR, _RTS, _RTI, _BRK = 0x20, 0x60, 0x40, 0x00
_CONTROL = frozenset((_JSR, _RTS, _RTI, _BRK))
_SP = frozenset(("SP",))


def _stack(sp, k):
    return 0x100 | ((sp + k) & 0xFF)


def _frame(sp):
    """Flow of a BRK or interrupt entry at stack pointer ``sp``: constant return address,
    status pushed from the flags, I set, SP moved."""
    return {
        _stack(sp, 0): frozenset(),
        _stack(sp, -1): frozenset(),
        _stack(sp, -2): frozenset(_FLAGS),
        "I": frozenset(),
        "SP": _SP,
    }


def _control(op, sp):
    """Flow the engine applies outside the pcode of JSR, RTS, RTI and BRK."""
    if op == _JSR:
        return {_stack(sp, 0): frozenset(), _stack(sp, -1): frozenset(), "SP": _SP}
    if op == _RTS:
        return {"SP": _SP}
    if op == _RTI:
        return dict.fromkeys(_FLAGS, frozenset((_stack(sp, 1), "SP"))) | {"SP": _SP}
    return _frame(sp)


@lru_cache(maxsize=None)
def _template(pc, code, nloads, nstores):
    """``(flow, register outputs)`` of instruction bytes ``code`` at ``pc`` with placeholder
    ``("L", i)``/``("S", i)`` load and store addresses and ``("O", k)`` operand bytes."""
    mem = bytearray(0x10000)
    for i, b in enumerate(code):
        mem[(pc + i) & 0xFFFF] = b
    loads = [("L", i) for i in range(nloads)]
    out = flow(mem, pc, loads, [("S", i) for i in range(nstores)], {1: ("O", 1), 2: ("O", 2)})
    return out, frozenset(k for k in out if isinstance(k, str))


class _TraceVM(CoverageVM):
    """``CoverageVM`` appending ``(pc, code, loads, values, stores, sp, written operand
    offsets)`` per instruction and ``(None, flow)`` for interrupt entries and registers set
    between steps; ``initial`` is the register file at the first step."""

    __slots__ = ("trace", "initial", "_ld", "_val", "_st", "_after")

    def __init__(self, mem_bytes, cov=None):
        super().__init__(mem_bytes, cov)
        self.trace, self.initial, self._after = [], None, None
        self._ld, self._val, self._st = [], [], []

    def _rd(self, addr, sz):
        v = CoverageVM._rd(self, addr, sz)
        for i in range(sz):
            self._ld.append((addr + i) & 0xFFFF)
            self._val.append((v >> (8 * i)) & 0xFF)
        return v

    def _wr(self, addr, val, sz):
        self._st.extend((addr + i) & 0xFFFF for i in range(sz))
        CoverageVM._wr(self, addr, val, sz)

    def sync(self):
        """Record registers set from outside the engine since the last step as constants."""
        reg, after = self.reg, self._after
        if after is None:
            self.initial = tuple(reg)
        elif reg != after:
            names = [_NAMES[i] for i in range(16) if reg[i] != after[i]]
            self.trace.append((None, dict.fromkeys(names, frozenset())))
        self._after = list(reg)

    def step(self, pc, cache, lifter):
        self.sync()
        mem, sp = self.mem, self.reg[3]
        code = bytes(mem[(pc + i) & 0xFFFF] for i in range(3))
        smc = tuple(k for k in (1, 2) if self._lw[(pc + k) & 0xFFFF] >= 0)
        self._ld, self._val, self._st = [], [], []
        if code[0] == _RTI:
            self._ld.append(_stack(sp, 1))
            self._val.append(mem[_stack(sp, 1)])
        nxt = CoverageVM.step(self, pc, cache, lifter)
        self.trace.append((pc, code, tuple(self._ld), tuple(self._val), tuple(self._st), sp, smc))
        self._after = list(self.reg)
        return nxt


class TraceMPU(CoverageMPU):
    """``CoverageMPU`` recording a ``trace`` of every instruction's loads and stores."""

    VM = _TraceVM

    @property
    def trace(self):
        return self._vm.trace

    def _enter(self, take):
        vm = self._vm
        vm.sync()
        sp = vm.reg[3]
        take()
        vm.trace.append((None, _frame(sp)))
        vm._after = list(vm.reg)

    def irq(self):
        if not self.p & self.INTERRUPT:
            self._enter(super().irq)

    def nmi(self):
        self._enter(super().nmi)


def _outputs(rec):
    """``({output: inputs}, {loaded address: value})`` of a trace record."""
    if rec[0] is None:
        return rec[1], {}
    pc, code, loads, vals, stores, sp, smc = rec
    tmpl, _ = _template(pc, code, len(loads), len(stores))
    sub = {("L", i): a for i, a in enumerate(loads)}
    sub.update((("S", i), a) for i, a in enumerate(stores))
    sub.update((("O", k), (pc + k) & 0xFFFF) for k in smc)
    out = {
        sub.get(k, k): frozenset(sub.get(d, d) for d in v if isinstance(d, str) or d in sub)
        for k, v in tmpl.items()
    }
    if code[0] in _CONTROL:
        out.update(_control(code[0], sp))
    return out, dict(zip(loads, vals))


def _touches(rec, need):
    """Whether trace record ``rec`` may set a location in ``need``."""
    if rec[0] is None:
        return not need.isdisjoint(rec[1])
    pc, code, loads, _, stores, _, _ = rec
    return (
        code[0] in _CONTROL
        or not need.isdisjoint(stores)
        or not need.isdisjoint(_template(pc, code, len(loads), len(stores))[1])
    )


@dataclass(frozen=True)
class Constant:
    """``loc`` set by the instruction at ``pc`` (``None``: the driver) from no data input."""

    pc: object
    loc: object


@dataclass(frozen=True)
class Explanation:
    """Slice of the condition of the branch at ``branch`` seen only going ``direction``
    (TAKEN or FALLTHRU) at trace record ``index``: ``inputs`` ``{location: value}``,
    ``constants``, instruction ``pcs``, ``writers`` ``{address: pcs}`` of the sliced bytes and
    ``alternatives`` ``{address: never-executed static writers}``."""

    branch: int
    direction: int
    index: int
    inputs: dict
    constants: frozenset
    pcs: frozenset
    writers: dict
    alternatives: dict


class Explanations(list):
    """:class:`Explanation` list whose ``str`` is :func:`format_explanations` with
    ``symbols``."""

    def __init__(self, items, symbols=None):
        super().__init__(items)
        self.symbols = symbols

    def __str__(self):
        return format_explanations(self, self.symbols)


def slice_of(trace, index, roots=("PC",), initial=None):
    """``(inputs, constants, pcs, writers)`` of the dynamic backward slice of outputs ``roots``
    of trace record ``index``: locations never set before it with their observed values
    (registers from ``initial``), constants, instruction pcs and ``{address: writer pcs}``."""
    outs, vals = _outputs(trace[index])
    need = {}
    for r in roots:
        for d in outs.get(r, ()):
            need.setdefault(d, vals.get(d))
    consts, pcs, writers = set(), {trace[index][0]}, {}
    for i in range(index - 1, -1, -1):
        if not need:
            break
        rec = trace[i]
        if not _touches(rec, need.keys()):
            continue
        outs, vals = _outputs(rec)
        hit = need.keys() & outs.keys()
        if hit:
            pcs.add(rec[0])
        for loc in hit:
            del need[loc]
            if isinstance(loc, int):
                writers.setdefault(loc, set()).add(rec[0])
            if not outs[loc]:
                consts.add(Constant(rec[0], loc))
            for d in outs[loc]:
                need.setdefault(d, vals.get(d))
    if initial is not None:
        index_of = {n: i for i, n in enumerate(_NAMES)}
        need.update((n, initial[index_of[n]]) for n in need if n in index_of)
    pcs.discard(None)
    return need, frozenset(consts), frozenset(pcs), {a: frozenset(w) for a, w in writers.items()}


def explain(run, cov, smcmap=None, symbols=None, progress=False):
    """:class:`Explanations` of every branch of ``cov`` that went exactly one way, from a replay
    of the deterministic driver ``run(mpu)`` on a fresh :class:`TraceMPU`; ``smcmap`` adds
    each sliced address's static writers that ``cov`` never executed."""
    from tqdm import tqdm  # pylint: disable=import-outside-toplevel

    mpu = TraceMPU()
    run(mpu)
    trace = mpu.trace
    bt = cov.flags & (TAKEN | FALLTHRU)
    half = set(np.flatnonzero((bt == TAKEN) | (bt == FALLTHRU)).tolist())
    last = {rec[0]: i for i, rec in enumerate(trace) if rec[0] in half}
    taken = {v for v, _ in cov.entries}
    out = []
    for pc in tqdm(sorted(last), desc="explain", unit="branch", disable=not progress):
        inputs, consts, pcs, writers = slice_of(trace, last[pc], initial=mpu._vm.initial)
        alts = {}
        if smcmap is not None:
            mem = writers.keys() | {a for a in inputs if isinstance(a, int)}
            for w, wr in smcmap.writers.items():
                if not cov.flags[w] & EXEC and w not in taken:
                    for a in mem & wr.targets:
                        alts.setdefault(a, set()).add(w)
        alts = {a: frozenset(w) for a, w in alts.items()}
        out.append(Explanation(pc, int(bt[pc]), last[pc], inputs, consts, pcs, writers, alts))
    return Explanations(out, symbols)


def format_explanations(explanations, symbols=None):
    """Compact text of ``explanations``, naming addresses ``label+offset`` from ``symbols``
    ``{name: address}`` (e.g. from :func:`jennings.coverage.elf.load_elf`)."""
    names = {}
    for n, a in sorted((symbols or {}).items(), key=lambda kv: (kv[1], kv[0])):
        names.setdefault(a, n)
    keys = np.array(sorted(names), np.int64)

    def lab(a):
        if a is None:
            return "driver"
        if not isinstance(a, int):
            return a
        k = int(np.searchsorted(keys, a, "right")) - 1
        if k < 0:
            return "$%04X" % a
        base = int(keys[k])
        return names[base] + ("+%d" % (a - base) if a > base else "")

    lines = []
    for e in explanations:
        way = "taken" if e.direction == TAKEN else "fallthrough"
        lines.append("%s: only %s (record %d)" % (lab(e.branch), way, e.index))
        for loc, v in sorted(e.inputs.items(), key=lambda kv: str(kv[0])):
            lines.append("  input    %s = %s" % (lab(loc), "?" if v is None else "$%02X" % v))
        for c in sorted(e.constants, key=lambda c: (str(c.pc), str(c.loc))):
            lines.append("  constant %s at %s" % (lab(c.loc), lab(c.pc)))
        for a, w in sorted(e.writers.items()):
            lines.append("  %s set by %s" % (lab(a), " ".join(map(lab, sorted(w)))))
        for a, w in sorted(e.alternatives.items()):
            lines.append("  %s also written by %s" % (lab(a), " ".join(map(lab, sorted(w)))))
    return "\n".join(lines) + ("\n" if lines else "")
