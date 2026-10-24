# Superoptimiser

`jennings.superopt` finds the shortest straight-line 6502 sequence equivalent to a reference
on a stated interface, and proves it with z3.

```bash
pip install jennings[superopt]   # z3 + numpy
pip install jennings[cuda]       # + CuPy, for an NVIDIA GPU
python -m jennings.superopt 'LDA $10; CLC; ADC #$01; STA $10' --var v=0x10 --out v
```

```python
from jennings.superopt import Problem, Spec, Table, Var, execute, search

spec = Spec(vars=(Var("v", 0x10, 2), Var("t", 0x12)), tables=(Table("F", 0x2000, 512),),
            params=("s",), entry={"C": 0}, outputs=("v",))
found = search(Problem(spec, code, origin=0x1000, immediates={0x1004: ("s", False)}))
found.instructions, found.size, found.cycles, found.code, found.rate
```

## Spec

| field | meaning |
|---|---|
| `vars` | memory variables (`Var(name, address, size)`, little-endian); live-in symbolic, writable |
| `tables` | read-only byte regions (`Table(name, address, length)`), indexed reads symbolic |
| `params` | 16-bit parameters; `Problem.immediates` binds an immediate's byte to one (`(name, high)`) |
| `entry` | registers and flags with known entry values (`{"A": 0, "C": 0}`); the rest symbolic |
| `outputs` | live-out locations: variable names, `A` `X` `Y`, `C` `Z` `N` `V` |

Everything not live-out is dead at exit. Reads outside `vars`, `tables` and the stack are
refused. Decimal mode is not modelled (as in deity-informant).

## Semantics

`execute(memory, start, end, spec, immediates)` runs code symbolically: each instruction is
lifted by deity-informant (the lifter the MPU executes) and its p-code interpreted over z3
bitvectors. A conditional branch forks; the arms merge as if-then-else, an infeasible arm is
pruned and a value the path forces constant becomes that constant. A counted loop (`DEX` or
`DEY`, `BNE` back) whose body shifts one variable by one bit is summarised as a variable shift
by the counter (0 meaning 256); other loops unroll and need a bounded counter. Code ends at
`end` or at an `RTS` on its entry stack. `prove(left, right)` returns None or a counterexample.

## Search

1. **Alphabet.** Loads, stores, transfers, ALU, compares, shifts and rotates (accumulator and
   memory), increments, `CLC`/`SEC`, over the spec's variable bytes, the reference's constants
   and parameter bytes as immediates, and its table offsets indexed by X or Y; `illegal=True`
   adds the stable NMOS illegals (`LAX SAX DCP ISC SLO RLA SRE RRA ANC ALR SBX`).
2. **Pruning.** A pair is never enumerated where the second instruction overwrites everything
   the first wrote without reading it, or where the two are independent and out of alphabet
   order; the last instruction must write a live-out location.
3. **Filter.** Each candidate runs on test vectors (random variables, tables, parameters and
   symbolic entry registers; the reference's live-outs computed on the jennings MPU). numpy
   evaluates chunks of the index space; the CUDA kernel runs one thread per candidate index.
   The alphabet semantics are tested against the MPU instruction by instruction, and the
   kernel against numpy; a semantic slip could only drop a candidate, never admit one.
4. **Proof.** Survivors, fewest bytes then cycles then enumeration order, are proved equal to
   the reference with z3 on the symbolic semantics; the first proved is the result. Lengths
   grow until no shorter sequence can exist, so the result is minimal in bytes within
   `max_length` instructions and the alphabet, and the same spec always yields the same code.

## GPU

The CUDA path needs CuPy and a device (`cuda_available()`); tests skip it otherwise. The
`gpu` workflow (manual, on a self-hosted runner labelled `gpu`) and
`docker/cuda.Dockerfile` run the tests with a GPU:

```bash
docker build -f docker/cuda.Dockerfile -t jennings-cuda .
docker run --rm --gpus all jennings-cuda
```
