"""jennings.superopt: symbolic semantics against the MPU, the search's alphabet semantics
against the MPU, minimal-sequence searches, and the CUDA filter against numpy."""

import random

import numpy as np
import pytest
import z3

from jennings.devices.mpu6502 import MPU
from jennings.superopt.alphabet import REGS, Alphabet, Instruction, apply
from jennings.superopt.search import Problem, Search, cuda_available, search
from jennings.superopt.symbolic import FLAGS, Spec, SymbolicError, Table, Var, execute, prove

ORIGIN = 0x1000
SPEC = Spec(
    vars=(Var("v", 0x10, 2), Var("t", 0x12), Var("k", 0x13)),
    tables=(Table("T", 0x2000, 512),),
    params=("p",),
    outputs=("v", "t", "A", "C"),
)

# (code, immediates): a parameter add, branches, a shift loop, a table read and the stack.
PROGRAMS = {
    "add": (
        [0x18, 0xA5, 0x10, 0x69, 0, 0x85, 0x10, 0xA5, 0x11, 0x69, 0, 0x85, 0x11, 0x60],
        {0x1003: ("p", False), 0x1009: ("p", True)},
    ),
    "branch": ([0xA5, 0x12, 0x30, 0x04, 0xE6, 0x12, 0xD0, 0x02, 0xC6, 0x12, 0x60], {}),
    "shift": ([0xA6, 0x13, 0xF0, 0x07, 0x46, 0x11, 0x66, 0x10, 0xCA, 0xD0, 0xF9, 0x60], {}),
    "table": ([0xA4, 0x12, 0xB9, 0x01, 0x20, 0x85, 0x10, 0xB9, 0x00, 0x21, 0x85, 0x11, 0x60], {}),
    "stack": ([0xA5, 0x12, 0x48, 0xA9, 0x07, 0x68, 0x0A, 0x85, 0x12, 0x60], {}),
}


def _machine(code, env):
    memory = bytearray(0x10000)
    memory[ORIGIN : ORIGIN + len(code)] = bytes(code)
    memory[0x10], memory[0x11] = env["v"] & 0xFF, env["v"] >> 8
    memory[0x12], memory[0x13] = env["t"], env["k"]
    memory[0x2000:0x2200] = bytes(env["T"])
    return memory


def _env(rng):
    return {
        "v": rng.getrandbits(16),
        "t": rng.choice([0, 1, 0x7F, 0x80, 0xFF, rng.getrandbits(8)]),
        "k": rng.choice([0, 1, 3, rng.getrandbits(8)]),
        "p": rng.getrandbits(16),
        "T": [rng.getrandbits(8) for _ in range(512)],
        "A": rng.getrandbits(8),
        "X": rng.getrandbits(8),
        "Y": rng.getrandbits(8),
        **{f: rng.getrandbits(1) for f in "CZNV"},
    }


def _mpu(code, immediates, env):
    memory = _machine(code, env)
    for pc, (name, high) in immediates.items():
        memory[pc + 1] = env[name] >> 8 if high else env[name] & 0xFF
    mpu = MPU(memory=memory, pc=ORIGIN)
    mpu.a, mpu.x, mpu.y, mpu.sp = env["A"], env["X"], env["Y"], 0xFF
    mpu.p = env["C"] | env["Z"] << 1 | env["V"] << 6 | env["N"] << 7 | mpu.INTERRUPT
    while not (memory[mpu.pc] == 0x60 and mpu.sp == 0xFF):
        mpu.step()
    return {"v": memory[0x10] | memory[0x11] << 8, "t": memory[0x12], "A": mpu.a, "C": mpu.p & 1}


def _evaluate(sem, env):
    inputs = sem.inputs
    pairs = [(inputs[n], z3.BitVecVal(env[n], inputs[n].size())) for n in ("v", "t", "k", "p")]
    pairs += [(inputs[r], z3.BitVecVal(env[r], 8)) for r in "AXY"]
    pairs += [(inputs[f], z3.BitVecVal(env[f], 1)) for f in "CZNV" if f in inputs]
    table = z3.K(z3.BitVecSort(16), z3.BitVecVal(0, 8))
    for k, value in enumerate(env["T"]):
        table = z3.Store(table, k, value)
    pairs.append((inputs["T"], table))
    return {k: z3.simplify(z3.substitute(t, *pairs)).as_long() for k, t in sem.outputs.items()}


@pytest.mark.parametrize("name", sorted(PROGRAMS))
def test_symbolic_matches_mpu(name):
    code, immediates = PROGRAMS[name]
    memory = _machine(code, _env(random.Random(0)))
    sem = execute(memory, ORIGIN, ORIGIN + len(code), SPEC, immediates)
    rng = random.Random(len(name))
    for _ in range(60):
        env = _env(rng)
        assert _evaluate(sem, env) == _mpu(code, immediates, env)
    if name == "shift":
        assert sem.loops == [("lsr", "v", 16)]


def test_symbolic_refusals_and_proof():
    memory = _machine([0xB5, 0x10, 0x60], _env(random.Random(1)))
    with pytest.raises(SymbolicError):
        execute(memory, ORIGIN, ORIGIN + 3, SPEC)
    memory = _machine([0x8D, 0x00, 0x30, 0x60], _env(random.Random(1)))
    with pytest.raises(SymbolicError):
        execute(memory, ORIGIN, ORIGIN + 4, SPEC)
    a, b = z3.BitVecs("a b", 8)
    assert prove({"x": a + b}, {"x": b + a}) is None
    assert prove({"x": a}, {"x": b}) is not None
    assert SPEC.width("v") == 16 and SPEC.width("A") == 8 and SPEC.width("C") == 1
    assert set(FLAGS) >= {"C", "Z", "N", "V"}


def _alphabet():
    return Alphabet([0x10, 0x11, 0x12], 2, [(0x2000, (0, 1, 256))], illegal=True)


def test_alphabet_semantics_match_mpu():
    alphabet = _alphabet()
    rng = random.Random(2)
    imms = [0x80, 0x01]
    for ins in alphabet.instructions:
        for _ in range(8):
            env = _env(rng)
            memory = _machine(alphabet.encode(ins, imms) + bytes([0x60]), env)
            mpu = MPU(memory=memory, pc=ORIGIN)
            mpu.a, mpu.x, mpu.y = env["A"], env["X"], env["Y"]
            mpu.p = env["C"] | env["Z"] << 1 | env["V"] << 6 | env["N"] << 7
            state = {r: np.array([env[r]], dtype=np.int32) for r in REGS}
            state["M"] = np.array([[memory[a] for a in alphabet.slots]], dtype=np.int32)
            op, kind, a, b = alphabet.code(ins)
            operand = None
            if kind == 1:
                operand = np.array([imms[a]])
            elif kind == 2:
                operand = state["M"][:, a].copy()
            elif kind in (3, 4):
                operand = np.array([memory[0x2000 + b + env["X" if kind == 3 else "Y"]]])
            apply(state, (op, kind, a, b), np.array([0]), operand)
            mpu.step()
            want = [mpu.a, mpu.x, mpu.y, mpu.p & 1, mpu.p >> 1 & 1, mpu.p >> 7, mpu.p >> 6 & 1]
            assert [int(state[r][0]) for r in REGS] == want, ins
            assert [int(m) for m in state["M"][0]] == [memory[a] for a in alphabet.slots], ins


def test_alphabet_pruning():
    alphabet = _alphabet()
    allowed, last = alphabet.pruning(["A"])
    index = {i: k for k, i in enumerate(alphabet.instructions)}
    lda = index[Instruction("LDA", "imm", 0)]
    clc, sec = index[Instruction("CLC", "impl")], index[Instruction("SEC", "impl")]
    assert not allowed[lda, lda] and not allowed[clc, sec]
    sta0, stx1 = index[Instruction("STA", "var", 0)], index[Instruction("STX", "var", 1)]
    assert allowed[sta0, stx1] != allowed[stx1, sta0]
    assert last[lda] and not last[clc]
    assert alphabet.size(Instruction("LDA", "tabx", (0, 1))) == 3
    assert alphabet.cycle(Instruction("LDA", "tabx", (0, 1))) == 5


def _increment():
    code = bytes([0xA5, 0x10, 0x18, 0x69, 0x01, 0x85, 0x10])
    return Problem(Spec(vars=(Var("v", 0x10),), outputs=("v",)), code, ORIGIN)


def test_search_finds_minimal():
    found = search(_increment(), backend="numpy")
    assert [(i.mnemonic, i.form) for i in found.instructions] == [("INC", "var")]
    assert (found.size, found.cycles, found.proofs) == (2, 5, 1)
    assert found.rate > 0


def test_search_finds_nothing_smaller():
    code = bytes([0xA5, 0x12, 0x85, 0x10])
    spec = Spec(vars=(Var("v", 0x10), Var("t", 0x12)), outputs=("v",))
    found = search(Problem(spec, code, ORIGIN), backend="numpy")
    assert found.instructions is None and found.candidates > 0


def test_search_parameters_and_tables():
    code = bytes([0xA4, 0x12, 0xB9, 0x00, 0x20, 0x18, 0x69, 0x00, 0x85, 0x12])
    spec = Spec(
        vars=(Var("t", 0x12),),
        tables=(Table("T", 0x2000, 512),),
        params=("p",),
        entry={"C": 0},
        outputs=("t",),
    )
    problem = Problem(spec, code, ORIGIN, {ORIGIN + 6: ("p", False)})
    assert ("p", 0) in Search(problem, vectors=16).immediates
    found = search(problem, backend="numpy", max_length=4, vectors=16)
    assert [i.mnemonic for i in found.instructions] == ["LDA", "LDX", "ADC", "STA"]
    assert found.size == len(code) - 1


@pytest.mark.skipif(not cuda_available(), reason="no CUDA device")
def test_cuda_matches_numpy():
    from jennings.superopt import cuda  # pylint: disable=import-outside-toplevel

    state = Search(_increment(), illegal=True)
    for n in (1, 2, 3):
        gpu, gpu_tried = cuda.candidates(state, n, 6)
        cpu, cpu_tried = state.candidates(n, 6)
        assert gpu == sorted(cpu) and gpu_tried == cpu_tried
    found = search(_increment(), backend="cuda")
    assert found.size == 2 and found.backend == "cuda"


def test_cli(capsys):
    from jennings.superopt.__main__ import main  # pylint: disable=import-outside-toplevel

    args = ["LDA $10; CLC; ADC #$01; STA $10", "--var", "v=0x10", "--out", "v"]
    assert main(args + ["--backend", "numpy"]) == 0
    out = capsys.readouterr().out
    assert "INC $10" in out and "minimal: 2 bytes" in out
    assert (
        main(
            [
                "LDA $12; STA $10",
                "--var",
                "v=0x10",
                "--var",
                "t=0x12:1",
                "--out",
                "v",
                "--table",
                "T=0x2000",
                "--entry",
                "C=0",
                "--backend",
                "numpy",
            ]
        )
        == 0
    )
    assert "no shorter sequence" in capsys.readouterr().out
