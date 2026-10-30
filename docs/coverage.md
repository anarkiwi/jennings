# Coverage

`jennings.coverage` records which bytes a 6502 program executed, which way each branch
went, and which data bytes it read or wrote. Recording lives in the engine VM
(`CoverageVM`), so deity-informant's drivers (`run_sub`, `run_irq`, `run_irq_driven`)
are covered as well as `CoverageMPU.step`.

```bash
pip install jennings[coverage]   # numpy, coverage.py
```

```python
from jennings.coverage import CoverageMPU, CoverageMap, EXEC, TAKEN, FALLTHRU

mpu = CoverageMPU(pc=0x0800)          # drop-in MPU; mpu.cov survives reset()
...                                   # load code, mpu.step() / mpu.irq() / mpu.nmi()
cov = mpu.cov
executed = (cov.flags & EXEC).nonzero()[0]
bt = cov.flags & (TAKEN | FALLTHRU)
one_way = ((bt == TAKEN) | (bt == FALLTHRU)).nonzero()[0]   # branches seen going one way
cov.save("run.npz")
total = CoverageMap.load("a.npz")
total |= CoverageMap.load("b.npz")    # OR flags, add hits, union edges/entries

from deity_informant import lift
from deity_informant.vm import run_sub
from jennings.coverage import CoverageVM
vm = CoverageVM(mem)                  # any PcodeVM driver
run_sub(vm, 0x1000, {}, lift)
vm.cov
```

## CoverageMap

| field | meaning |
|---|---|
| `flags` | `uint8[65536]`, OR of the flags below per address |
| `hits` | `uint32[65536]`, executions of the instruction starting at each address |
| `edges` | `{(src, dst)}` dynamic control transfers: `JMP (ind)`, `RTS`, `RTI`, `BRK` |
| `entries` | `{(vector, target)}` interrupts taken through `CoverageMPU.irq`/`nmi` |
| `variants` | `{(pc, bytes)}` instructions run with a written byte (bytes little-endian) |
| `defuse` | `{(writer, cell, pc)}` code bytes run at `pc` whose last store was at `writer` |
| `image` | `uint8[65536]`, 1 where an instruction ran with none of its bytes written |

| flag | value | set on |
|---|---|---|
| `EXEC` | 1 | an executed opcode byte (JAM included) |
| `OPERAND` | 2 | the operand bytes of an executed instruction |
| `TAKEN` | 4 | a conditional branch that branched |
| `FALLTHRU` | 8 | a conditional branch that did not branch |
| `READ` | 16 | a data byte loaded by an instruction |
| `WRITE` | 32 | a data byte stored by an instruction |

Branch direction comes from the opcode's tested flag, so a zero-offset branch is still
classified. `READ`/`WRITE` cover the instruction's own data accesses (including `PHA`/`PHP`
pushes and `PLA`/`PLP` pulls) and the pointer bytes read by `JMP (ind)` (with the NMOS
page wrap), `BRK`, and taken IRQ/NMI entries; opcode fetch and the return-address/status
stack traffic of `JSR`/`RTS`/`RTI`/`BRK` and interrupt entry are not flagged.
`CoverageVM.last_writer` (`int32[65536]`, -1 for the image) tracks the last instruction to
store each byte, which `variants`, `defuse` and `image` are built from. `save` writes one
`.npz`; `CoverageMap.load` reads it back (files without the newer keys load with them empty).

## Reporting

`coverage report/html/xml/lcov` (with `--fail-under`) work on 6502 code through the
`jennings.coverage.plugin` coverage.py plugin. A reported source `prog.s` needs a sidecar
`prog.s.jcov`; `Listing` produces both from memory, entry points and a `CoverageMap`, and
`write_data` turns a `CoverageMap` into coverage.py branch data.

```python
from jennings.coverage.listing import Listing
from jennings.coverage.report import write_data

listing = Listing.disassemble(mpu.memory, [0x0800], mpu.cov, ranges=[(0x0800, 0x0FFF)])
listing.write("prog.s")                       # prog.s + prog.s.jcov
write_data(mpu.cov, "prog.s")                 # .coverage-6502
write_data(mpu.cov, "prog.s", suffix=True, context="test_a")   # per-process, for combine
```

```ini
# .coveragerc
[run]
plugins = jennings.coverage.plugin
data_file = .coverage-6502
branch = True
```

```bash
coverage combine                  # merge suffixed .coverage-6502.* files
coverage report -m --fail-under=90
coverage html; coverage lcov
```

`Listing.disassemble` decodes by recursive descent from the entry points, every executed
address, and the targets of `cov.edges` and `cov.entries`, so code reached only through
`JMP (ind)` or RTS dispatch is listed. Branches continue to both successors, `JMP abs` to its
target, `JSR` to the next instruction with the callee as a new entry; `RTS`, `RTI`,
`JMP (ind)`, `BRK` and `JAM` exit. Executed instruction starts are always listed, even when
they overlap; statically found starts that overlap listed bytes are dropped. Bytes inside
`ranges` that no instruction covers become non-executable `.byte` rows.

The sidecar is JSON and independent of how the source was produced:

| key | rows | meaning |
|---|---|---|
| `lines` | `[line, address]` | instruction start addresses of each executable line |
| `arcs` | `[from, to]` | static arcs; `(-L, L)` enters at line `L`, `(L, -L)` leaves from it (a branch to its own line also uses `(L, -L)`) |
| `transitions` | `[address, mask, from, to]` | arc `(from, to)` ran when `flags[address] & mask` and line `to` executed |

`write_data` also adds each dynamic edge whose ends both map to lines. A branch line has two
exits, so a branch seen going one way is a partial branch, and the missing arc message names
both lines with their addresses.

## Sources from ELF/DWARF

`jennings.coverage.elf` reports the original assembly and C of a program built with debug
line tables (e.g. llvm-mos `-g`, whose `.prg.elf` sits beside the `.prg`).

```python
from jennings.coverage import CoverageMPU, call
from jennings.coverage.elf import load_elf, write_sidecars
from jennings.coverage.report import write_data

mem, syms = load_elf("tune.prg.elf")          # 64K image of PT_LOAD segments, {symbol: address}
mpu = CoverageMPU(memory=mem.tolist())
call(mpu, syms["init"], a=0)                  # run a subroutine to its RTS
for _ in range(300):
    call(mpu, syms["play"])
sources = write_sidecars("tune.prg.elf", ["init", "play"])   # <source>.jcov per source file
write_data(mpu.cov, sources)
```

Every instruction start inside a DWARF row's range `[address, next row address)` maps to the
row's line (line-0 rows continue the previous line), so a C line usually owns several
addresses. Arcs come from each instruction's static successors (`jennings.coverage.cfg`):
flow within a line adds no arc, a branch or jump back into its own line is `(L, -L)`, flow
with no successor in the same source exits `(L, -L)`, and the given entries, `JSR` callees
and starts with no static predecessor (e.g. RTS-dispatch targets) enter `(-L, L)`. DWARF
paths under the compilation directory (`/work` in the llvm-mos SDK container) resolve against
`root` (default: the ELF's directory); `write_sidecars` skips sources that do not exist.

`call` pushes the return address `RETURN - 1` and steps until the subroutine returns to
`RETURN` at the caller's stack depth, raising `RuntimeError` after `max_steps`.

### Fixture

`tests/fixtures/sidplayer` is an original three-voice SID playroutine (`player.s`, with the
note-to-frequency helper in `freq.c`): `init` (A = song) at `$1000`, `play` at `$1003`.
It uses self-modifying pattern/sequence pointers and voice register offsets, a PHA/PHA/RTS
effect dispatch, vibrato only in song 0 and portamento only in song 1. The committed
`sidplayer.prg`/`sidplayer.prg.elf` are rebuilt reproducibly with the llvm-mos SDK image,
and CI fails if they differ:

```bash
docker build --target artifacts --output type=local,dest=tests/fixtures/sidplayer \
    tests/fixtures/sidplayer
```

## Static SMC analysis

`jennings.coverage.smc.analyze` finds, without running the program, every instruction that can
write each code byte and the values the byte can hold.

```python
from jennings.coverage.smc import analyze

smc = analyze(mem, {syms["init"]: {"a": {0, 1}}, syms["play"]: {}}, cov=mpu.cov)
smc.cells[syms["seq_rd"] + 1]     # Cell(writers={pc: exact}, values, classes, insns, data_dependent)
smc.unresolved_transfers          # {pc: candidate targets} not followed (e.g. PHA/PHA/RTS tables)
```

Entries are addresses or `{addr: {"a": values, ..., "irq": bool}}`; unlisted registers and
flags may hold anything. Each entry runs with SP `sp` (default `$FD`, as `call` leaves a fresh
MPU) under a `call` frame (or an interrupt frame), and its return to `RETURN` is an exit.
`cov` adds a `CoverageMap`'s executed addresses and edge/interrupt targets as entries in an
unknown context (any registers, SP and stack), as `Listing.disassemble` seeds its walk.

**Domain** (`jennings.coverage.vsa`). A value of `w` bytes is the exact set of integers it may
hold (a `256**w`-bit mask); each pcode op is evaluated over the product of its input sets with
the semantics of `PcodeVM`, and `step` adds the JSR/RTS/RTI/BRK stack traffic, so on singleton
sets it is the concrete machine. Operand bytes enter through the lifter's `prov` tags, so a
self-modified operand contributes every value its bytes may hold.

**Fixpoint.** One joined state per instruction address: register sets, the bytes written along
the paths since entry (others read the global set `VS(m)` = image byte ∪ every value any store
may put at `m`), and the return addresses pushed by JSR/BRK/entry, tracked by offset above SP
through pushes and pulls. A key absent on either side of a join is dropped. Growth of `VS`
re-evaluates the instructions that read it; flags are sets, so a branch on a singleton flag
has one feasible direction (`branches`).

**Control flow** never follows computed data. Successors are static targets, RTS/RTI returns to
addresses pushed by JSR/BRK/entry (to every call site), and the variants of code bytes that the
image or constant stores put there. Any other target or opcode is recorded in
`unresolved_transfers`/`unresolved_opcodes` with its candidates; a store whose address set is
all of 64K is in `unresolved_stores` and changes no value set.

**Cells.** For every written code byte: its writers (`exact` when the byte is the store's only
target), `values` = `VS`, `classes` (`var` opcode, `ctrl` operand of JMP/JSR/branch/JMP (ind),
`addr` operand of a memory-mode instruction, `imm` immediate) and the instructions it belongs
to. `values` is exact unless `data_dependent` (a writer stores a computed value), where it is a
sound superset.

### SMC obligations

`obligations(smc)` turns the cells into test obligations: for every cell and writer, the
writer's value was executed (a `defuse` record); for cells that are not `data_dependent`,
every executable variant was executed (`variants`, or for the image byte an `image` run of an
instruction covering the cell). `write_obligations(smc, "smc.txt", labels)` writes one line per
obligation and a sidecar whose `obligations` table (`[line, kind, cell, a, b]`, kinds as in
`report.held`) `write_data` evaluates against a `CoverageMap`, so `coverage report
--fail-under` gates SMC obligations alongside code. `check(smc, cov)` lists dynamic code writes
the analysis did not predict (a writer outside W(cell), or an executed byte outside Var(cell)).

```python
from jennings.coverage.smc import analyze, check, write_obligations
from jennings.coverage.report import write_data

smc = analyze(mem, entries, cov=mpu.cov)
assert not check(smc, mpu.cov)
write_data(mpu.cov, [write_obligations(smc, "smc.txt", {0x1064: "seq_rd+1"}), "player.s"])
```
