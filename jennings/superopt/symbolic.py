"""Symbolic 6502 semantics over z3: deity-informant's p-code (what the MPU executes) interpreted
over bitvectors, one term per live-out location. Branches fork and merge as if-then-else (dead
arms pruned, path-forced values made constant); a counted shift loop becomes a variable shift.
"""

from dataclasses import dataclass, field

import z3
from deity_informant import lift

from jennings.opcodes import OPCODES

REGISTERS = {"A": 0, "X": 1, "Y": 2}
FLAGS = {"C": 8, "Z": 9, "I": 10, "D": 11, "V": 13, "N": 14}
_SP = 3
_STACK = 0x100
BUDGET = 4096
_ONE, _ZERO = z3.BitVecVal(1, 8), z3.BitVecVal(0, 8)
_BIN = {
    "INT_ADD": lambda a, b: a + b,
    "INT_SUB": lambda a, b: a - b,
    "INT_AND": lambda a, b: a & b,
    "INT_OR": lambda a, b: a | b,
    "INT_XOR": lambda a, b: a ^ b,
    "INT_LEFT": lambda a, b: a << b,
    "INT_RIGHT": z3.LShR,
}
_CMP = {
    "INT_EQUAL": lambda a, b: a == b,
    "INT_NOTEQUAL": lambda a, b: a != b,
    "INT_LESS": z3.ULT,
    "INT_LESSEQUAL": z3.ULE,
    "INT_CARRY": lambda a, b: z3.ULT(a + b, a),
}


class SymbolicError(ValueError):
    """Code outside what the symbolic executor models."""


@dataclass(frozen=True)
class Var:
    """A memory variable: ``size`` bytes, little-endian, from ``address``."""

    name: str
    address: int
    size: int = 1


@dataclass(frozen=True)
class Table:
    """A read-only region of ``length`` bytes from ``address``; an indexed read past it reads
    a byte the model leaves unconstrained."""

    name: str
    address: int
    length: int = 256


@dataclass(frozen=True)
class Spec:
    """What code reads and writes: its variables, tables and 16-bit parameters (immediates
    bind their low or high byte), the registers and flags it starts with (``entry``
    ``{name: value}``; the others symbolic) and its live-out locations (variable, register
    and flag names)."""

    vars: tuple = ()
    tables: tuple = ()
    params: tuple = ()
    entry: dict = field(default_factory=dict)
    outputs: tuple = ()

    def inputs(self) -> dict:
        """``{name: z3 term}`` of every input: variables (8 bits per byte), parameters (16),
        tables (arrays of 16-bit offset to byte), registers (8) and flags (1) not in
        ``entry``."""
        out = {v.name: z3.BitVec(v.name, 8 * v.size) for v in self.vars}
        out.update((p, z3.BitVec(p, 16)) for p in self.params)
        out.update(
            (t.name, z3.Array(t.name, z3.BitVecSort(16), z3.BitVecSort(8))) for t in self.tables
        )
        out.update((r, z3.BitVec(r, 8)) for r in REGISTERS if r not in self.entry)
        out.update((f, z3.BitVec(f, 1)) for f in FLAGS if f not in self.entry)
        return out

    def width(self, name: str) -> int:
        """The bits of location ``name``."""
        if name in REGISTERS:
            return 8
        if name in FLAGS:
            return 1
        return 8 * next(v.size for v in self.vars if v.name == name)


@dataclass
class Semantics:
    """Code's live-out locations as z3 terms over its spec's inputs, and the loops
    summarised as ``(kind, variable, bits)``."""

    spec: Spec
    inputs: dict
    outputs: dict
    loops: list = field(default_factory=list)


@dataclass
class _State:
    regs: dict
    mem: dict
    stack: dict
    pc: int

    def copy(self, pc=None) -> "_State":
        return _State(dict(self.regs), dict(self.mem), dict(self.stack), pc or self.pc)


def _part(word, k: int):
    """Byte ``k`` of a z3 bitvector."""
    return z3.Extract(8 * k + 7, 8 * k, word) if word.size() > 8 else word


def _word(parts):
    return z3.Concat(*reversed(parts)) if len(parts) > 1 else parts[0]


def _split(address):
    """``(base, index)`` of a z3 address: a constant, or a constant plus a 16-bit term."""
    if (
        z3.is_app_of(address, z3.Z3_OP_BADD)
        and address.num_args() == 2
        and z3.is_bv_value(address.arg(0))
    ):
        return address.arg(0).as_long(), address.arg(1)
    address = z3.simplify(address)
    if z3.is_bv_value(address):
        return address.as_long(), None
    raise SymbolicError(f"address {address}")


def _const(term):
    term = z3.simplify(term)
    return term.as_long() if z3.is_bv_value(term) else None


class _Program:
    """Code in memory between ``start`` and ``end``, its instructions lifted."""

    def __init__(self, memory, start: int, end: int, spec: Spec, immediates: dict):
        self.spec, self.start, self.end = spec, start, end
        self.inputs = spec.inputs()
        self.immediates = immediates
        self.vars = {}
        for v in spec.vars:
            self.vars.update((v.address + k, (v.name, k)) for k in range(v.size))
        self.ops, pc = {}, start
        while pc < end:
            self.ops[pc] = lift(memory, pc)
            pc += self.ops[pc]["len"]

    def immediate(self, pc: int):
        """The z3 byte of a parameter immediate at ``pc``, else None."""
        if pc not in self.immediates:
            return None
        name, high = self.immediates[pc]
        word = self.inputs[name]
        return z3.Extract(15, 8, word) if high else z3.Extract(7, 0, word)

    def load(self, state: _State, address):
        """The byte at z3 16-bit ``address`` in ``state``."""
        base, index = _split(address)
        if index is None and base in self.vars:
            name, k = self.vars[base]
            return state.mem.get((name, k), _part(self.inputs[name], k))
        if index is None and _STACK <= base < _STACK + 0x100:
            return state.stack[base]
        for t in self.spec.tables:
            if t.address <= base < t.address + t.length:
                at = z3.BitVecVal(base - t.address, 16)
                return z3.Select(self.inputs[t.name], at if index is None else at + index)
        raise SymbolicError(f"reads ${base:04x}{'' if index is None else ',index'}")

    def store(self, state: _State, address, value) -> None:
        """Stores byte ``value`` at z3 ``address``."""
        base, index = _split(address)
        if index is None and base in self.vars:
            state.mem[self.vars[base]] = value
        elif index is None and _STACK <= base < _STACK + 0x100:
            state.stack[base] = value
        else:
            raise SymbolicError(f"writes ${base:04x}{'' if index is None else ',index'}")

    def step(self, state: _State):
        """Runs the instruction at ``state.pc`` in place; None or ``(condition, taken,
        fall-through)``."""
        rec = self.ops[state.pc]
        imm = self.immediate(state.pc)
        from_operand = {at for at, (srcs, _) in rec["prov"]["ops"].items() if srcs == [1]}
        uniq = {}

        def get(at, vn):
            space, off, size = vn
            if space == "c":
                if imm is not None and at in from_operand:
                    return imm
                return z3.BitVecVal(off, 8 * size)
            return (state.regs if space == "r" else uniq)[off]

        for i, (name, dst, ins) in enumerate(rec["ops"]):
            args = [get((i, j), vn) for j, vn in enumerate(ins)]
            if name == "STORE":
                self.store(state, *args)
                continue
            if name == "LOAD":
                value = self.load(state, args[0])
            elif name == "COPY":
                value = args[0]
            elif name == "INT_ZEXT":
                value = z3.ZeroExt(8 * dst[2] - args[0].size(), args[0])
            elif name in _BIN:
                a, b = args
                if b.size() < a.size():
                    b = z3.ZeroExt(a.size() - b.size(), b)
                value = _BIN[name](a, b)
            elif name in _CMP:
                value = z3.If(_CMP[name](*args), _ONE, _ZERO)
            else:
                raise SymbolicError(f"p-code {name}")
            (state.regs if dst[0] == "r" else uniq)[dst[1]] = value
        ctrl = rec["ctrl"]
        if ctrl[0] == "br":
            flag, pol, taken, fall = ctrl[1:]
            return state.regs[flag[1]] == pol, taken, fall
        if ctrl[0] not in ("next", "jmp", "rts"):
            raise SymbolicError(f"control {ctrl[0]}")
        state.pc = ctrl[1] if ctrl[0] == "jmp" else state.pc + rec["len"]
        return None


class _Executor:
    """Forking symbolic execution of one :class:`_Program`."""

    def __init__(self, program: _Program):
        self.program = program
        self.solver = z3.Solver()
        self.loops = {}

    def feasible(self, condition) -> bool:
        """Whether ``condition`` may hold on the current path."""
        return self.solver.check(condition) != z3.unsat

    def run(self, state: _State, budget: int = BUDGET) -> dict:
        """The merged outputs of every path from ``state``."""
        program = self.program
        while budget > 0:
            budget -= 1
            if state.pc == program.end:
                return self.outputs(state)
            if state.pc not in program.ops:
                raise SymbolicError(f"runs outside its code at ${state.pc:04x}")
            if program.ops[state.pc]["ctrl"][0] == "rts" and _const(state.regs[_SP]) == 0xFF:
                return self.outputs(state)
            if (loop := self.loop(state.pc)) is not None:
                state = self.summary(state, *loop)
                continue
            branch = program.step(state)
            if branch is None:
                continue
            condition, taken, fall = branch
            live = [
                (c, pc)
                for c, pc in ((condition, taken), (z3.Not(condition), fall))
                if self.feasible(c)
            ]
            out = []
            for c, pc in live:
                self.solver.push()
                self.solver.add(c)
                out.append(self.run(self.forced(state.copy(pc)), budget))
                self.solver.pop()
            if len(out) == 1:
                return out[0]
            return {k: z3.If(condition, out[0][k], out[1][k]) for k in out[0]}
        raise SymbolicError(f"runs past {BUDGET} instructions")

    def forced(self, state: _State) -> _State:
        """``state`` with each register and written byte the path forces constant replaced by
        that constant."""
        solver = self.solver
        if solver.check() != z3.sat:
            return state
        model = solver.model()
        for where in (state.regs, state.mem):
            for key, term in where.items():
                if z3.is_bv_value(term) or (where is state.regs and key == _SP):
                    continue
                value = model.eval(term, model_completion=True)
                if solver.check(term != value) == z3.unsat:
                    where[key] = value
        return state

    def outputs(self, state: _State) -> dict:
        """The live-out terms of a final state."""
        spec, inputs, out = self.program.spec, self.program.inputs, {}
        for name in spec.outputs:
            if name in REGISTERS:
                out[name] = state.regs[REGISTERS[name]]
            elif name in FLAGS:
                out[name] = z3.Extract(0, 0, state.regs[FLAGS[name]])
            else:
                size = inputs[name].size() // 8
                parts = [state.mem.get((name, k), _part(inputs[name], k)) for k in range(size)]
                out[name] = _word(parts)
        return out

    def loop(self, head: int):
        """``(kind, end, counter, name, offsets)`` of a counted shift loop at ``head``: its
        body shifts one variable (or one byte of it) by one bit, ``kind`` ``lsr`` or ``asl``,
        then decrements X or Y and branches back on not zero; else None."""
        if head not in self.loops:
            self.loops[head] = self._loop(head)
        return self.loops[head]

    def _loop(self, head: int):
        ops = self.program.ops
        pcs = sorted(ops)
        end = next((pc for pc in pcs if pc > head and ops[pc]["ctrl"][0] != "next"), None)
        if end is None or OPCODES[ops[end]["prov"]["op0"]][0] != "BNE":
            return None
        if ops[end]["ctrl"][3] != head:
            return None
        body = [pc for pc in pcs if head <= pc < end]
        last = OPCODES[ops[body[-1]]["prov"]["op0"]][0]
        if last not in ("DEX", "DEY"):
            return None
        counter = REGISTERS["X" if last == "DEX" else "Y"]
        cells = set()
        for pc in body[:-1]:
            if OPCODES[ops[pc]["prov"]["op0"]][1] not in ("zp", "abs"):
                return None
            cells.add(self.program.vars.get(self.address(pc)))
        if None in cells or len({name for name, _ in cells}) != 1:
            return None
        name, offsets = next(iter(cells))[0], sorted(k for _, k in cells)
        if offsets not in ([0], [0, 1]):
            return None
        regs = {r: z3.BitVec(f"r{r}", 8) for r in range(16)}
        mem = {(name, o): z3.BitVec(f"m{o}", 8) for o in offsets}
        state = _State(dict(regs), dict(mem), {}, head)
        for pc in body:
            state.pc = pc
            self.program.step(state)
        before = _word([mem[(name, o)] for o in offsets])
        after = _word([state.mem[(name, o)] for o in offsets])
        top = before.size() - 1
        kept = z3.And(
            [state.regs[r] == regs[r] for r in (0, 1, 2, FLAGS["V"]) if r != counter]
            + [state.regs[counter] == regs[counter] - 1]
        )
        for kind, shifted, bit in (
            ("lsr", z3.LShR(before, 1), z3.Extract(0, 0, before)),
            ("asl", before << 1, z3.Extract(top, top, before)),
        ):
            claim = z3.And(kept, after == shifted, state.regs[FLAGS["C"]] == z3.ZeroExt(7, bit))
            if z3.Solver().check(z3.Not(claim)) == z3.unsat:
                return kind, end, counter, name, offsets
        return None

    def address(self, pc: int) -> int:
        """The absolute or zero-page operand address of the instruction at ``pc``."""
        rec = self.program.ops[pc]
        consts = [vn[1] for _, _, ins in rec["ops"] for vn in ins if vn[0] == "c" and vn[2] == 2]
        return consts[0] if consts else -1

    def summary(self, state: _State, kind, end, counter, name, offsets) -> _State:
        """``state`` after the loop at its pc: the variable shifted by the counter (0 meaning
        256), the carry the last bit out, the counter 0, Z set and N clear."""
        # pylint: disable=too-many-arguments,too-many-positional-arguments
        inputs = self.program.inputs
        before = _word([state.mem.get((name, o), _part(inputs[name], o)) for o in offsets])
        width, n = before.size(), state.regs[counter]
        count = z3.ZeroExt(width - 8, n)

        def shifted(by):
            return z3.LShR(before, by) if kind == "lsr" else before << by

        last = shifted(count - 1)
        if kind == "lsr":
            carry = _part(last, 0) & 1
        else:
            carry = z3.LShR(_part(last, width // 8 - 1), 7)
        word = shifted(count)
        if self.feasible(n == 0):
            word = z3.If(n == 0, z3.BitVecVal(0, width), word)
            carry = z3.If(n == 0, _ZERO, carry)
        out = state.copy(end + 2)
        for k, o in enumerate(offsets):
            out.mem[(name, o)] = _part(word, k)
        out.regs.update({FLAGS["C"]: carry, counter: _ZERO, FLAGS["Z"]: _ONE, FLAGS["N"]: _ZERO})
        return out


def execute(memory, start: int, end: int, spec: Spec, immediates=None) -> Semantics:
    """The :class:`Semantics` of the code in ``memory`` from ``start``, ending at ``end`` or
    at an ``rts`` on its entry stack (SP $ff); ``immediates`` ``{pc: (parameter, high)}``
    binds the immediate of the instruction at ``pc`` to a parameter's byte."""
    program = _Program(memory, start, end, spec, dict(immediates or {}))
    regs = dict.fromkeys(range(16), _ZERO)
    regs[_SP] = z3.BitVecVal(0xFF, 8)
    inputs = program.inputs
    for name, reg in REGISTERS.items():
        regs[reg] = z3.BitVecVal(spec.entry[name], 8) if name in spec.entry else inputs[name]
    for name, reg in FLAGS.items():
        value = spec.entry.get(name)
        regs[reg] = z3.BitVecVal(value, 8) if value is not None else z3.ZeroExt(7, inputs[name])
    executor = _Executor(program)
    outputs = executor.run(_State(regs, {}, {}, start))
    loops = [(k, n, 8 * len(o)) for k, _, _, n, o in filter(None, executor.loops.values())]
    return Semantics(spec, inputs, outputs, loops)


def prove(left: dict, right: dict):
    """None where z3 proves the terms ``left`` and ``right`` (same names) equal for all
    inputs, else a counterexample model."""
    solver = z3.Solver()
    solver.add(z3.Or([left[k] != right[k] for k in left]))
    return None if solver.check() == z3.unsat else solver.model()
