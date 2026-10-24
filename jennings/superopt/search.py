"""Minimal straight-line sequences: enumerate candidates by increasing length over an
:class:`~.alphabet.Alphabet`, keep those matching a reference on random test vectors (numpy, or
CUDA where available), and take the first z3 proves equivalent, fewest bytes then cycles."""

import random
import time
from dataclasses import dataclass, field

import numpy as np

from jennings.devices.mpu6502 import MPU
from jennings.opcodes import MODE_LEN, OPCODES
from jennings.superopt.alphabet import REGS, Alphabet, apply
from jennings.superopt.symbolic import FLAGS, REGISTERS, Spec, execute, prove

CHUNK = 1 << 16
FILTER = 4
STEPS = 4096


@dataclass
class Problem:
    """Code to improve: ``reference`` bytes at ``origin`` under ``spec`` (whose outputs are
    the live-out locations), ``immediates`` ``{pc: (parameter, high)}`` its parameter
    immediates."""

    spec: Spec
    reference: bytes
    origin: int = 0x1000
    immediates: dict = field(default_factory=dict)

    @property
    def end(self) -> int:
        """The address after the reference."""
        return self.origin + len(self.reference)


@dataclass
class Found:
    """A search's result: the sequence (None if nothing smaller was proved), its bytes and
    cycles, the candidates tried, the seconds taken, the backend, and its code and parameter
    immediates as :class:`Problem` takes them."""

    instructions: tuple
    size: int
    cycles: int
    candidates: int
    seconds: float
    backend: str
    proofs: int = 0
    code: bytes = b""
    immediates: dict = field(default_factory=dict)

    @property
    def rate(self) -> float:
        """Candidates per second."""
        return self.candidates / self.seconds if self.seconds else 0.0


def _disassemble(problem: Problem):
    """``[(pc, mnemonic, mode, operand)]`` of the reference."""
    out, pc, code = [], 0, problem.reference
    while pc < len(code):
        mnemonic, mode = OPCODES[code[pc]]
        size = MODE_LEN[mode]
        operand = int.from_bytes(code[pc + 1 : pc + size], "little") if size > 1 else None
        out.append((problem.origin + pc, mnemonic, mode, operand))
        pc += size
    return out


class Vectors:
    """``count`` random machine states of a problem and the reference's live-outs on each,
    run on the jennings MPU."""

    def __init__(self, problem: Problem, alphabet: Alphabet, immediates, count: int, seed: int):
        # pylint: disable=too-many-locals
        spec, rng = problem.spec, random.Random(seed)
        self.count = count
        reach = max([o for _, offsets in alphabet.tables for o in offsets] + [0]) + 256
        self.regs = np.zeros((count, len(REGS)), dtype=np.int32)
        self.slots = np.zeros((count, len(alphabet.slots)), dtype=np.int32)
        self.imms = np.zeros((count, max(len(immediates), 1)), dtype=np.int32)
        self.tables = np.zeros((count, max(len(alphabet.tables), 1), reach), dtype=np.int32)
        outputs = locations(problem.spec, alphabet)
        self.expect = np.zeros((count, len(outputs)), dtype=np.int32)
        for k in range(count):
            memory = bytearray(0x10000)
            for t in spec.tables:
                memory[t.address : t.address + t.length] = rng.randbytes(t.length)
            for v in spec.vars:
                memory[v.address : v.address + v.size] = rng.randbytes(v.size)
            params = {p: rng.getrandbits(16) for p in spec.params}
            memory[problem.origin : problem.end] = problem.reference
            for pc, (name, high) in problem.immediates.items():
                memory[pc + 1] = params[name] >> 8 if high else params[name] & 0xFF
            regs = [spec.entry.get(r, rng.getrandbits(8)) for r in "AXY"]
            regs += [spec.entry.get(f, rng.getrandbits(1)) for f in "CZNV"]
            self.regs[k] = regs
            self.slots[k] = [memory[a] for a in alphabet.slots]
            for j, value in enumerate(immediates):
                self.imms[k, j] = (
                    value if isinstance(value, int) else params[value[0]] >> 8 * value[1] & 0xFF
                )
            for j, (address, _) in enumerate(alphabet.tables):
                self.tables[k, j] = list(memory[address : address + reach])
            after, regs_after = _run(problem, memory, regs, spec)
            self.expect[k] = [
                regs_after[loc] if isinstance(loc, int) and loc < len(REGS) else after[loc[1]]
                for loc in outputs
            ]


def locations(spec: Spec, alphabet: Alphabet) -> list:
    """The live-out locations as register indices into ``REGS`` and ``("m", address)``."""
    out = []
    for name in spec.outputs:
        if name in REGISTERS or name in FLAGS:
            out.append(REGS.index(name))
        else:
            var = next(v for v in spec.vars if v.name == name)
            out += [("m", var.address + k) for k in range(var.size)]
    del alphabet
    return out


def _run(problem: Problem, memory: bytearray, regs, spec: Spec):
    """``(memory, registers)`` after the reference on the MPU."""
    del spec
    mpu = MPU(memory=memory, pc=problem.origin)
    mpu.a, mpu.x, mpu.y = regs[0], regs[1], regs[2]
    mpu.sp = 0xFF
    flags = dict(zip("CZNV", regs[3:]))
    mpu.p = (
        flags["C"] * mpu.CARRY
        | flags["Z"] * mpu.ZERO
        | flags["N"] * mpu.NEGATIVE
        | flags["V"] * mpu.OVERFLOW
        | mpu.INTERRUPT
    )
    for _ in range(STEPS):
        if mpu.pc == problem.end or (memory[mpu.pc] == 0x60 and mpu.sp == 0xFF):
            break
        mpu.step()
    p = mpu.p
    out = [mpu.a, mpu.x, mpu.y]
    out += [int(bool(p & bit)) for bit in (mpu.CARRY, mpu.ZERO, mpu.NEGATIVE, mpu.OVERFLOW)]
    return memory, out


class Search:
    """One problem's search state: alphabet, pruning, vectors and the candidates' checks."""

    def __init__(self, problem: Problem, *, constants=(), illegal=False, vectors=64, seed=0):
        # pylint: disable=too-many-locals
        self.problem = problem
        spec = problem.spec
        listing = _disassemble(problem)
        values = set(constants)
        offsets = {t.name: set() for t in spec.tables}
        for pc, _, mode, operand in listing:
            if mode == "imm" and pc not in problem.immediates:
                values.add(operand)
            for t in spec.tables:
                if mode in ("absx", "absy") and t.address <= operand < t.address + t.length:
                    offsets[t.name].add(operand - t.address)
        params = sorted(set(problem.immediates.values()))
        self.immediates = sorted(values) + [(p, int(high)) for p, high in params]
        slots = [v.address + k for v in spec.vars for k in range(v.size)]
        tables = [(t.address, tuple(sorted(offsets[t.name]))) for t in spec.tables]
        self.alphabet = Alphabet(slots, len(self.immediates), tables, illegal)
        self.outputs = locations(spec, self.alphabet)
        tokens = [REGS[o] if isinstance(o, int) else ("m", slots.index(o[1])) for o in self.outputs]
        self.allowed, self.last = self.alphabet.pruning(tokens)
        self.vectors = Vectors(problem, self.alphabet, self.immediates, vectors, seed)
        self.out_index = [
            o if isinstance(o, int) else len(REGS) + slots.index(o[1]) for o in self.outputs
        ]

    def run(self, digits: np.ndarray, vectors: slice) -> np.ndarray:
        """Which candidate rows of ``digits`` (``[candidate, position]`` alphabet indices)
        reproduce the reference's live-outs on ``vectors``."""
        vec = self.vectors
        regs, slots = vec.regs[vectors], vec.slots[vectors]
        imms, tables, expect = vec.imms[vectors], vec.tables[vectors], vec.expect[vectors]
        n_cand, n_vec = digits.shape[0], regs.shape[0]
        which = np.repeat(np.arange(n_cand), n_vec)
        vector = np.tile(np.arange(n_vec), n_cand)
        state = {r: regs[vector, k].copy() for k, r in enumerate(REGS)}
        state["M"] = slots[vector].copy()
        codes = self.alphabet.codes
        for position in range(digits.shape[1]):
            column = digits[which, position]
            for entry in np.unique(column):
                rows = np.nonzero(column == entry)[0]
                op, kind, a, b = codes[entry]
                v = vector[rows]
                if kind == 1:
                    operand = imms[v, a]
                elif kind == 2:
                    operand = state["M"][rows, a]
                elif kind in (3, 4):
                    index = state["X" if kind == 3 else "Y"][rows]
                    operand = tables[v, a, b + index]
                else:
                    operand = None
                apply(state, (op, kind, a, b), rows, operand)
        final = np.stack([state[r] for r in REGS], axis=1)
        final = np.concatenate([final, state["M"]], axis=1)[:, self.out_index]
        good = (final == expect[vector]).all(axis=1).reshape(n_cand, n_vec).all(axis=1)
        return good

    def valid(self, digits: np.ndarray, budget: int) -> np.ndarray:
        """Which rows of ``digits`` pass the pruning and fit in ``budget`` bytes."""
        ok = self.last[digits[:, -1]].astype(bool)
        ok &= self.alphabet.sizes[digits].sum(axis=1) <= budget
        for k in range(digits.shape[1] - 1):
            ok &= self.allowed[digits[:, k], digits[:, k + 1]].astype(bool)
        return ok

    def candidates(self, n: int, budget: int):
        """``(survivors, tried)``: the length-``n`` sequences within ``budget`` bytes matching
        the reference on every vector, by numpy over chunks of the index space."""
        k = len(self.alphabet)
        total, survivors, tried = k**n, [], 0
        for start in range(0, total, CHUNK):
            index = np.arange(start, min(start + CHUNK, total), dtype=np.int64)
            digits = np.stack([(index // k**p) % k for p in range(n - 1, -1, -1)], axis=1)
            tried += len(digits)
            digits = digits[self.valid(digits, budget)]
            if digits.size == 0:
                continue
            digits = digits[self.run(digits, slice(0, FILTER))]
            if len(digits):
                digits = digits[self.run(digits, slice(None))]
            survivors += [tuple(int(d) for d in row) for row in digits]
        return survivors, tried

    def assemble(self, sequence) -> tuple:
        """``(code, immediates)`` of ``sequence`` at the problem's origin, ``immediates``
        ``{pc: (parameter, high)}``."""
        alphabet, pc = self.alphabet, self.problem.origin
        code, immediates = bytearray(), {}
        placeholder = [v if isinstance(v, int) else 0 for v in self.immediates]
        for d in sequence:
            ins = alphabet.instructions[d]
            if ins.form == "imm" and not isinstance(self.immediates[ins.operand], int):
                name, high = self.immediates[ins.operand]
                immediates[pc] = (name, bool(high))
            data = alphabet.encode(ins, placeholder)
            code += data
            pc += len(data)
        return bytes(code), immediates

    def prove(self, sequence) -> bool:
        """Whether z3 proves ``sequence`` equal to the reference on the live-outs."""
        problem = self.problem
        code, immediates = self.assemble(sequence)
        memory = bytearray(0x10000)
        memory[problem.origin : problem.end] = problem.reference
        old = execute(memory, problem.origin, problem.end, problem.spec, problem.immediates)
        memory = bytearray(0x10000)
        memory[problem.origin : problem.origin + len(code)] = code
        new = execute(memory, problem.origin, problem.origin + len(code), problem.spec, immediates)
        return prove(old.outputs, new.outputs) is None


def search(problem: Problem, *, backend="auto", max_bytes=None, max_length=8, **options):
    """The :class:`Found` minimal sequence for ``problem``: the fewest bytes (then cycles,
    then enumeration order) below ``max_bytes`` (default the reference's size) that z3 proves
    equal. ``backend`` ``numpy``, ``cuda`` or ``auto``; ``options`` as :class:`Search`."""
    started = time.perf_counter()
    state = Search(problem, **options)
    if backend == "auto":
        backend = "cuda" if cuda_available() else "numpy"
    if backend == "cuda":
        from jennings.superopt.cuda import (  # pylint: disable=import-outside-toplevel
            candidates as gpu_candidates,
        )

        def find(n, budget):
            return gpu_candidates(state, n, budget)

    else:
        find = state.candidates
    budget = (max_bytes or len(problem.reference)) - 1
    sizes, cycles = state.alphabet.sizes, state.alphabet.cycles
    best, best_size, tried, proofs = None, budget + 1, 0, 0
    for n in range(1, max_length + 1):
        if n > budget:
            break
        found, count = find(n, budget)
        tried += count
        ranked = sorted(found, key=lambda s: (sizes[list(s)].sum(), cycles[list(s)].sum(), s))
        for sequence in ranked:
            size = int(sizes[list(sequence)].sum())
            if size >= best_size:
                break
            proofs += 1
            if state.prove(sequence):
                best, best_size, budget = sequence, size, size - 1
                break
    seconds = time.perf_counter() - started
    if best is None:
        return Found(None, 0, 0, tried, seconds, backend, proofs)
    instructions = tuple(state.alphabet.instructions[d] for d in best)
    total = int(cycles[list(best)].sum())
    code, immediates = state.assemble(best)
    return Found(instructions, best_size, total, tried, seconds, backend, proofs, code, immediates)


def cuda_available() -> bool:
    """Whether CuPy and a CUDA device are present."""
    try:
        import cupy  # pylint: disable=import-outside-toplevel

        return cupy.cuda.runtime.getDeviceCount() > 0  # pylint: disable=c-extension-no-member
    except Exception:  # pylint: disable=broad-exception-caught
        return False
