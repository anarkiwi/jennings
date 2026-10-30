# Coverage

`jennings.coverage` records which bytes a 6502 program executed, which way each branch
went, and which data bytes it read or wrote. Recording lives in the engine VM
(`CoverageVM`), so deity-informant's drivers (`run_sub`, `run_irq`, `run_irq_driven`)
are covered as well as `CoverageMPU.step`. On top of the recording sit coverage.py
reporting (listings or ELF/DWARF sources), a static self-modifying-code and interrupt-vector
analysis with test obligations, `explain` for branches that only went one way, and a pytest
plugin.

```bash
pip install jennings[coverage]   # numpy, coverage.py, pyelftools, tqdm
```

## Walkthrough: the SID fixture

[`tests/fixtures/sidplayer`](#fixture) is a three-voice playroutine built with llvm-mos.
Build it (or use the committed `sidplayer.prg.elf`), run song 0, and report:

```bash
docker build --target artifacts --output type=local,dest=tests/fixtures/sidplayer \
    tests/fixtures/sidplayer
```

```python
from jennings.coverage import CoverageMPU, call
from jennings.coverage.elf import load_elf, write_sidecars
from jennings.coverage.explain import explain
from jennings.coverage.report import write_data
from jennings.coverage.smc import analyze, check, write_obligations

mem, syms = load_elf("sidplayer.prg.elf")

def song0(mpu):                                  # deterministic driver, reused by explain
    mpu.memory = mem.tolist()
    call(mpu, syms["init"], a=0)
    for _ in range(300):
        call(mpu, syms["play"])

mpu = CoverageMPU()
song0(mpu)
cov = mpu.cov

fx = ("fx_off", "fx_vibrato", "fx_slide", "fx_pulse")   # reached only by RTS dispatch
entries = {syms["init"]: {"a": {0, 1}}, syms["play"]: {}} | {syms[h]: {} for h in fx}
smc = analyze(mem, entries, cov=cov)             # code-byte writers, values, vectors
assert not check(smc, cov)                       # every dynamic code write was predicted

sources = write_sidecars("sidplayer.prg.elf", ["init", "play"])   # player.s, freq.c
write_data(cov, sources + [write_obligations(smc, "smc.txt")])
print(explain(song0, cov, smcmap=smc, symbols=syms))
```

```bash
coverage report -m      # with the .coveragerc below: player.s, freq.c and smc.txt
```

`player.s` shows the portamento code as missing and `voice+9` (`lda slide,x; beq vibrato`)
as a partial branch; `smc.txt` lists the self-modified operand obligations. `explain`
says why the branch never fell through:

```
voice+9: only taken (record 50432)
  constant A at init_voice+12
  constant X at voices
  slide set by init_voice+20
  slide also written by fx_slide ...
```

`slide` only ever held init's `lda #0`, and the never-executed `fx_slide` handler is the
other writer, so a test that plays song 1 (whose patterns use effect `$92`) covers the
branch. Interrupt-driven code adds vector obligations ([Interrupts](#interrupts)); the
[pytest plugin](#pytest-plugin) records each test's coverage under its node id.

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
total |= CoverageMap.load("b.npz")    # OR flags, add hits, union the sets

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
| `defuse` | `{(writer, cell, pc)}` code or pointer bytes used at `pc` whose last store was at `writer` |
| `image` | `uint8[65536]`, 1 where an instruction ran with none of its bytes written |
| `chains` | `{(pointer, previous target or -1, target)}` every vector consumption (IRQ, NMI, BRK, `JMP (ind)`) |

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
store each byte (the `JSR` or `BRK` for the return address and status it pushes, and the
vector address, e.g. `$FFFE`, for an IRQ/NMI frame), which `variants`, `defuse` and `image` are built from; a consumed pointer
byte that was written adds a `defuse` record whose `pc` is the `JMP (ind)` or `BRK`, or the
handler for IRQ/NMI. `CoverageVM.vector_target` holds the last target taken through each
pointer, the previous target of the next `chains` record. `save` writes one `.npz`;
`CoverageMap.load` reads it back (files without the newer keys load with them empty).

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

`write_data` erases a new data file on its first write; to add several contexts to one file
pass the same open `coverage.CoverageData` as `data=`.

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
smc.handlers, smc.chain, smc.hazards   # interrupt vectors, see Interrupts
```

Entries are addresses or `{addr: {"a": values, ..., "irq": bool}}`; unlisted registers and
flags may hold anything. Each entry runs with SP `sp` (default `$FD`, as `call` leaves a fresh
MPU) under a `call` frame (or an interrupt frame), and its return to `RETURN` is an exit.
`cov` resolves transfers from a `CoverageMap`: each observed edge `(src, dst)` in `cov.edges`
whose `dst` is a candidate of the unresolved transfer at `src` is followed with the state
propagated from `src` (for RTS the state at `src` with SP+2 and its return addresses popped,
likewise for RTI and `JMP (ind)`), and each interrupt in `cov.entries` enters its target in
interrupt context with the frame attributed to the vector. This runs inside the fixpoint, so
transfers in newly reached code resolve too. Only executed addresses and edge/interrupt
targets that remain unreached afterwards are entered in an unknown context (any registers, SP
and stack); their stores and loads are correspondingly imprecise.

**Domain** (`jennings.coverage.vsa`). A value of `w` bytes is the exact set of integers it may
hold (a `256**w`-bit mask); each pcode op is evaluated over the product of its input sets with
the semantics of `PcodeVM`, and `step` adds the JSR/RTS/RTI/BRK stack traffic, so on singleton
sets it is the concrete machine. Operand bytes enter through the lifter's `prov` tags, so a
self-modified operand contributes every value its bytes may hold.

**Fixpoint.** One joined state per instruction address and innermost pending return address,
so a subroutine's states stay apart per call site: register sets, the bytes written along the
paths since entry (others read the global set `VS(m)` = image byte ∪ every value any store may
put at `m`), the return addresses pushed by JSR/BRK/entry, tracked by offset above SP through
pushes and pulls, and flag origins. A flag origin records, for a branch flag set as a function
of one A/X/Y register value with scalar operands (`CPX #3`, `DEX`, `LDA mem`, ...), the
register values that give 0 and 1; a branch on the flag narrows the register on each side, so
`X` stays `{0, 1, 2}` in a `DEX`/`BPL` voice loop. A key absent on either side of a join is
dropped. Growth of `VS` re-evaluates the instructions that read it; flags are sets, so a branch
on a singleton flag has one feasible direction (`branches`).

**Control flow** never follows computed data. Successors are static targets, RTS/RTI returns to
addresses pushed by JSR/BRK/entry (the calling site of the state), the variants of code bytes
that the image or constant stores put there, and observed edges from `cov`. Any other target or
opcode is recorded in `unresolved_transfers`/`unresolved_opcodes` with its candidates (observed
targets stay listed there); a store whose address set is all of 64K is in `unresolved_stores`
and changes no value set.

**Stack pushes** are store sites: JSR and BRK push their return address (and status) at the SP
set, and an interrupt handler's frame is a store site of its vector address (any byte), the
writers `CoverageVM.last_writer` records, so code in the stack page stays consistent under
`check`.

**Cells.** For every written code byte: its writers (`exact` when the byte is the store's only
target), `values` = `VS`, `classes` (`var` opcode, `ctrl` operand of JMP/JSR/branch/JMP (ind),
`addr` operand of a memory-mode instruction, `imm` immediate) and the instructions it belongs
to. `values` is exact unless `data_dependent` (a writer stores a computed value), where it is a
sound superset.

### SMC obligations

`obligations(smc)` turns the cells into test obligations: for every cell and writer, the
writer's value was executed or, for a vector byte, consumed (`def`, a `defuse` record); for
code cells that are not `data_dependent`, every executable variant was executed (`value`:
`variants`, or for the image byte an `image` run of an instruction covering the cell); every
static interrupt handler was entered (`entry`) and every handler chain edge taken (`chain`,
see [Interrupts](#interrupts)). `write_obligations(smc, "smc.txt", labels)` writes one line per
obligation and a sidecar whose `obligations` table (`[line, kind, cell, a, b]`) `write_data`
evaluates against a `CoverageMap`, so `coverage report --fail-under` gates these obligations
alongside code. `check(smc, cov)` lists dynamic code and vector writes the analysis did not
predict (a writer outside W(cell), or an executed byte outside Var(cell)).

| row kind | holds when (`report.held`) |
|---|---|
| 0 | `(a, cell, *)` in `defuse`: writer `a`'s value at `cell` was used |
| 1 | `cell` was executed holding byte `a` after a code write (`variants`) |
| 2 | the instruction at `b` ran with none of its bytes written (`image`) |
| 3 | `(cell, *, a)` in `chains`: target `a` was taken through pointer `cell` |
| 4 | `(cell, a, b)` in `chains`: target `b` followed target `a` through `cell` |

```python
from jennings.coverage.smc import analyze, check, write_obligations
from jennings.coverage.report import write_data

smc = analyze(mem, entries, cov=mpu.cov)
assert not check(smc, mpu.cov)
write_data(mpu.cov, [write_obligations(smc, "smc.txt", {0x1064: "seq_rd+1"}), "player.s"])
```

## Interrupts

`analyze(..., vectors=VECTORS)` treats the vector bytes (default IRQ/BRK `$FFFE`, NMI
`$FFFA`, KERNAL CINV `$0314`, CBINV `$0316`, NMINV `$0318`) like code bytes: they get cells
with writers, values and class `ctrl` (no `value` obligations). A store to a vector byte is
read against the flow-sensitive state: when its partner byte was written on the same path
the overlay holds the full new vector; a first byte stays pending until the partner write,
a point where an interrupt may be taken (I may be 0), or the entry's exit, and the vector as
then seen is installed. A constant pair (image or constant stores only) becomes an analysed
entry in interrupt context: I=1 and an interrupt frame whose RTI return exits. A
data-dependent vector lands in `unresolved_transfers` at its writers. A vector's image value
becomes a handler once it is analysed code, and a `JMP (ind)` through a vector (a KERNAL
stub) enters its targets as their own handlers.

| `SmcMap` field | meaning |
|---|---|
| `vectors` | the analysed vector addresses |
| `handlers` | `{entry: vectors}` handlers installed by constant pairs or the image |
| `chain` | `{(source, vector, handler)}`: a vector writer reachable from `source` (a handler before its RTI, including callees, or a non-interrupt entry) installs `handler` |
| `hazards` | `[(first_pc, vector, partner_pc or None)]` torn vectors |

A torn vector is a byte written while its partner is not yet written on that path, with an
interrupt possible before the partner write (the first write or a later point has I
possibly 0): the interrupt could take a half-new vector, which is installed as a handler
too. `partner_pc` is None when no path writes the partner. Hazards are reported, not
obligations.

The `entry` obligation of a handler holds once it was taken through its vector; a `chain`
edge from a handler on the same vector holds when that handler's target was followed by the
new one (`chains` record `(vector, source, handler)`), and from any other source when the
handler followed no target or one of that vector's handlers. The writers of vector bytes
carry `def` obligations, held when an interrupt or `JMP (ind)` consumed their value.

```python
from jennings.coverage import CoverageMPU
from jennings.coverage.report import held
from jennings.coverage.smc import analyze, check, obligation_rows

smc = analyze(mem, [init])         # init installs h1 with SEI; h1 -> h2 -> h3 -> h1
mpu = CoverageMPU(memory=mem.tolist(), pc=init)
for _ in range(5):                 # h3's vector write is only consumed by the fifth IRQ
    while mpu.pc != idle:
        mpu.step()
    mpu.irq()
missing = [ob for ob, rows in obligation_rows(smc) if not held(rows, mpu.cov).any()]
assert not missing and not check(smc, mpu.cov) and not smc.hazards
```

`CoverageMPU.irq()`/`nmi()` always enter through the current vector. To make
deity-informant's `run_irq`/`run_irq_driven` follow a rewritten vector, pass
`handler=vector(0xFFFE)` (`from deity_informant import vector`, deity-informant >= 0.6.1),
which reads the vector from `vm.mem` at each entry.

## Explaining partial branches

`jennings.coverage.explain.explain(run, cov, smcmap=None, symbols=None, progress=False)`
answers why a branch only went one way. `run(mpu)` is the deterministic driver that produced
`cov`; it is called again on a fresh `TraceMPU` (a `CoverageMPU` recording each
instruction's loads and stores from `_rd`/`_wr`, its stack pointer and which operand bytes
were written during the run). For every branch with exactly one of `TAKEN`/`FALLTHRU`, the
dynamic backward slice of its condition (`"PC"` in `jennings.dataflow.flow`) is taken at its
last executed instance: walking the trace backward, each record that sets a needed location
replaces it with that output's inputs. Flow templates are cached per instruction bytes and
records that cannot set a needed location are skipped.

The slice follows registers, flags, memory, address dependencies (an indexed load depends on
its index; a pull on SP), self-modified operands (`flow(..., operands=)` ties an operand-built
constant to its bytes, kept only when the run wrote them), the stack traffic of
JSR/RTS/RTI/BRK and interrupt entries (RTI restores the flags from the status byte the
entry pushed), and registers a driver sets between steps (constants at `None`, "driver").

| `Explanation` field | meaning |
|---|---|
| `branch`, `direction`, `index` | branch address, the direction seen, trace record sliced |
| `inputs` | `{location: value}` never set before the branch: bytes read but not written by the run (tune data, initial image) and registers at the first step |
| `constants` | `Constant(pc, loc)`: `loc` set by `pc` from no data input, e.g. `lda #0` |
| `pcs`, `writers` | instructions in the slice; `{address: pcs}` that set its bytes |
| `alternatives` | with `smcmap`: `{address: static writers never executed}`, from `SmcMap.writers` target sets |

`explain` returns an `Explanations` list whose `str` is the compact text
(`format_explanations`, addresses as `label+offset` from `symbols`, e.g. `load_elf`'s).
`slice_of(trace, index, roots=("PC",), initial=None)` slices any record of a `TraceMPU.trace`.
Bytes the driver writes directly into memory between steps count as inputs with the value
read. Alternatives are as precise as the `SmcMap`: code reached through observed transfers
keeps its register values (on the fixture only `fx_slide` is listed for `slide`), while
executed code that `analyze` still had to seed runs in an unknown context, so indexed stores
reachable only from such seeds list every address their index can
reach.

## pytest plugin

`jennings.coverage.pytest_plugin` is registered as a `pytest11` entry point and is inert
unless `pytest --cov6502[=DATAFILE]` (default `.coverage-6502`) is given; importing it does
not load numpy or the engine. The `cov6502` fixture creates or adopts the MPUs a test runs
and names the sources to report:

```python
def test_song1(cov6502):
    mpu = cov6502.mpu(memory=mem.tolist())     # or cov6502.adopt(mpu | vm | cov)
    call(mpu, syms["init"], a=1)
    for _ in range(300):
        call(mpu, syms["play"])
    cov6502.sources(*sources, "smc.txt")        # sidecar-backed sources
```

At teardown the merged maps of the test are written with `write_data` under
`context=<nodeid>` to `DATAFILE.<worker>` (`gwN` under xdist, `main` otherwise); the
controller removes earlier `DATAFILE.*` files at session start. Combine and report with the
plugin's `.coveragerc`:

```bash
pytest -n auto --cov6502
coverage combine && coverage report -m && coverage html --show-contexts
```
