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
from jennings.coverage import CoverageMPU, CoverageMap, EXEC, call, one_way

mpu = CoverageMPU(pc=0x0800)          # drop-in MPU; mpu.cov survives reset()
...                                   # load code, mpu.step() / mpu.irq() / mpu.nmi()
call(mpu, 0x1000, a=0, raster=64)     # run to RTS; $D012 = instructions // 64 & $FF before each
cov = mpu.cov
executed = (cov.flags & EXEC).nonzero()[0]
one_way(cov, mpu.memory)              # {(branch pc, target never taken)}
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

`call(mpu, addr, a=0, x=0, y=0, max_steps=1_000_000, raster=None)` pushes the sentinel
`RETURN` and steps until it returns there; with `raster`, `$D012` holds
`n // raster & $FF` before the `n`-th instruction, so raster polls in init code terminate.

`jennings.coverage.cfg` holds the static flow helpers: `successors(mem, pc)`,
`instruction_bytes(mem, starts)` (`bool[65536]` of every byte of the instructions starting
in the mask `starts`, wrapping at `$FFFF`), `control_operands(mem, starts)` (the operand
bytes of the branches, `JMP`s and `JSR`s among them, wrapping likewise), `STOPS` (the opcodes
after which control never reaches the next instruction: `BRK`, `RTI`, `RTS`, the JAMs and
both `JMP`s), `return_edges(mem, succ, calls)` (`{rts: {return
sites}}` from `succ` `{pc: [pcs]}` and `calls` `{callee: {return sites}}`, nested `JSR`s
skipped to their return site), `control_dependence(edges, exit_=-1)` (`{node: {branch}}` by
postdominators with a virtual exit after every sink) and `one_way(cov, mem)`.

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

### Computed transfers

`Listing.disassemble(..., resolve=True)` also decodes code reached only through table entries
a run never took (`jennings.coverage.resolve`). It is opt-in because even when a run executed
every table target, an unbounded index can find a valid-looking target in data the run never
touched.

```python
listing = Listing.disassemble(mem, entries, cov, ranges=[(0x1000, 0x1FFF)], resolve=True)
listing.resolved                  # [[transfer, target]], statically resolved edges
```

| step | rule |
|---|---|
| transfers | `JMP (ind)`, `RTS`, `RTI`, and `JMP`/`JSR`/branches with an operand byte that `cov` saw written or a decoded unexecuted store may write (`abs,X`/`abs,Y` stores: base..base+255) |
| paths | every backward path of at most 16 instructions over the static predecessors, stopping at entries, callees, dynamic-edge targets and `JSR` returns; RTS/RTI paths must push |
| constants | image bytes inside `ranges` (all memory without `ranges`) not written, outside the stack page |
| forward run | `vsa.step` from unknown A/X/Y/flags with a fixed SP; stores go to a path overlay, loads read the overlay, else constants, else any value; a branch narrows its flag-origin register to the path's direction (`CMP #n`/`BCS`, `BEQ`, ...) |
| splits | an indexed load whose value reaches the transfer (backward slice over registers and the addresses each step accessed; the index registers of such loads are followed separately, so a load that only computes an index stays a set) splits the state per index value unless its least byte is written data, so a target's low and high bytes come from one entry |
| table extent | splits in index order: leading splits reading code or non-constant bytes are skipped; the first other one starts the table (none if it reads another load's base or first byte), the next one not overlapping it sets the stride (2 for an interleaved word table); the table ends at the first split on the stride reading code, a non-constant byte, a byte the table took, or another load's base or first byte (a split table's low column ends at its high column) |
| targets | each split ending in a single target; it is decoded only if its whole descent stays inside the image with documented opcodes other than `BRK`, overlaps no listed instruction at another alignment, and covers no table byte, byte the run read but never wrote or executed, written opcode byte or referenced object start |

Resolution repeats on the transfers of newly decoded code. A target that is not otherwise
listed gets a label and the trailing comment `; resolved from $XXXX` naming its transfers;
lines it reaches carry no mark, and all resolved edges, including those to listed code, are
in `resolved`. Bounds come only from the path (masks, shifts, compares, the image values of
constant bytes it reads); an index read from a variable is unbounded, so a dispatch indexed
by a command byte lists every entry up to the table's end.

Tables the resolver cannot bound or find are given by hand: `Listing.disassemble(...,
tables=[(lo, hi, n, stride, high)])` decodes from the words of each split table and keeps its
bytes as data. `resolve.split_table(mem, lo, hi, n, stride=1, high=None)` gives a table's
`(words, cells)`: word `k` has its low byte at `lo + stride * k` and its high byte at
`hi + stride * k`, or `high` for every word.

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
sets it is the concrete machine. Operand bytes enter through the `prov` bindings of
`deity_informant.lifter.template`, so a self-modified operand contributes every value its bytes
may hold.

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

## Components and cover calls

`jennings.coverage.components` splits a repeatedly called, input-free routine (a
playroutine) into autonomous components and says how many calls cover everything it can do:
every data access `(pc, read/write, address)` and every control edge. It runs on the native
core.

```python
from jennings.coverage import components as C

comps = C.learn(mpu, play, 12_000)          # NativeMPU, or 64 KiB of RAM
cyc = C.cycles(mpu, play, comps, 1_000_000)  # (preperiod, period) per closure
calls, comps = C.cover_calls(mpu, play, comps, 1_000_000)
```

Each function runs from a flat copy of the state (`machine`: RAM, plus the registers of an
MPU), calling `pc` as `run_to_rts` does (`acc`, `max_cycles`, `remap` as there); a call
that does not return raises.

**census** runs the calls once natively and sorts them into classes by what a `Learner` sees
of each (see [Learning once per class](#learning-once-per-class)), recording the edges, the
instructions that ended a call and the memory written, each with the call it was first seen
in, and optionally `cells` at each call's entry. **learn** takes `cfg.control_dependence` of
the edges over its calls with each call's last instruction leading to the exit (`control`),
then has a `Learner` taint each class's first call, which taints every location (64 KiB of
memory and the registers, `REG + index`) with the entry locations its value came from:

- data, per instruction, from `tools/gen_flow.py`'s table: `dataflow.flow` read once per
  opcode on symbolic registers, operand bytes and accesses, so it follows the same lifted
  P-Code the core is generated from. A load depends on its address and the byte it reads,
  a store's byte on the value stored; `PHA`/`PLA` flow through the stack page, `JSR`/`RTS`/
  `RTI`/`BRK` through their frames;
- control: every output of an instruction also takes the conditions of the branches it is
  directly control dependent on (and its own opcode byte), and a location's store sites
  pass their controlling conditions on at the end of a call, so a store that did not happen
  still makes the location depend on why;
- at entry, a location written by no call is a constant and taints nothing; operand bytes
  are read at fetch, so a self-modified operand or opcode is state. `Remap` applies: a
  redirected read is of the location it reads from, a `ZERO` read is a constant and a
  redirected write writes nothing.

`Components` holds `persistent` (locations some call reads before writing and some call
writes, the stack page and I/O excluded by `exclude`; of the registers only X, Y and the
flags but B), `deps` (each location's end-of-call dependence), `closures` (each sink SCC of
`deps` over the persistent locations with its ancestors: autonomous by construction),
`sites` (`(pc, kind)` to the locations tainting a load's (`LOAD`) or store's (`STORE`)
address, where control goes from `pc` (`CONTROL`): a branch's condition and offset, a
jump's target, an `RTS`/`RTI` frame, an opcode byte, or the byte a data store at `pc` stores
(`VALUE`), as the stored location gets it), `feed` (the persistent locations in any load,
store or control site's taint with their ancestors) and `feeding` (the closures containing
one). A location's dependence on itself is left out of `deps`.

**explicit** gives the explicit flows alone, as `Flows(deps, sites, stored, addressed, gated,
calls)`: it runs `control` for the control dependence and the memory the calls write, then an
explicit `Learner`, in which

- each output takes only its data inputs, from the table's `vals` (`dataflow.flow(...,
  addressed=False)`): a load's value is the byte it reads, not its address. No opcode byte, no
  branch conditions, no store-site conditions at the end of a call: control enters `gated`
  alone;
- a data load (not a stack pull or an operand fetch) of a location no call writes at `pc` is
  the constant `CONST + pc` (`CONST = 0x20000`; `location` names it `#$hhhh`), so a song-table
  read shows where it happened. A constant operand byte still taints nothing, and a `Remap`
  `ZERO` read is the constant 0, tainting nothing.

`Flows.deps` maps each location written to the entry locations and constants its end-of-call
value depends on by data, itself included (`INC c` gives `c` in `deps[c]`); `Flows.sites` is
keyed as `Components.sites`. The `LOAD`/`STORE` taints are the addresses' (an index, not the
table it reads), `CONTROL` keeps the opcode byte, condition, operands and frames, and `VALUE`
is the explicit taint of the stored byte. Pointer bytes or other data loads of constants
feeding an address or a condition appear there as `CONST + pc` too.

Sites are per `pc`, so one store serving three voices (`STA $D400,X`, X = 0, 7, 14) unions
their taints. `stored`, `addressed` and `gated` are per memory location a data store (not a
push, not a `Remap`-suppressed write) reaches, each unioned over every such store of every
call:

- `stored[loc]`: the explicit taint of each byte stored to `loc` (every store, not only the
  call's last, as `deps` has);
- `addressed[loc]`: the taint of each such store's address, what its `STORE` site gets;
- `gated[loc]`: the conditions each store was made under: the union of `cond[br]` over the
  branches `br` it is directly control dependent on that ran in the call, `cond[br]` being
  `br`'s `CONTROL` taint with its own conditions, so nesting is transitive; and, at the end of
  each call, `cond[br]` of each branch `br` a store site of `loc` is control dependent on that
  ran in the call, the implicit flow `learn` puts in `deps` (the branch's last condition in
  the call, so per `pc`).

```python
f = C.explicit(mpu, play, 2_000)
f.sites[(sta_pc, C.VALUE)]   # e.g. {C.CONST + lda_pc}: a table byte, read at lda_pc
f.sites[(lda_pc, C.LOAD)]    # {pos}: the index
f.stored[0xD400]             # {pos}: voice 0's, where f.sites[(sta_pc, C.VALUE)] has all three
f.gated[0xD404]              # {flag}: the condition voice 0's gate register was stored under
```

**trajectory** records the persistent locations at each call's entry natively, one uint8
row per call (`NativeMPU.trajectory`). **cycles** finds each closure's and the feed set's
first repeat `(preperiod, period)` on it (`orbit` is the max preperiod and lcm period of all
the closures) and checks autonomy: equal projections must have equal successors. A conflict
`(t1, t2, cells)` means a dependence learning did not see; `widen` adds the cells that told
the two entries apart, with their ancestors.

**cycles** runs natively on the columns of the rows (`periods_of`): per column set, a 64-bit
hash per row finds the first repeat (rows equal on the set, checked byte by byte, so a hash
collision is never a repeat), then each column is compared with itself `period` rows on, a
block at a time. Column sets measured once are not measured again while widening.

**cover_calls** widens until there is no conflict and returns the feed set's preperiod plus
period: the state the addresses and control of a call are a function of has then taken every
value it ever takes, so every reachable access and edge has been seen. The taints vouch only
for the paths learning saw, so while that bound exceeds the calls learned from, it learns
again over the bound and repeats. One census of the `limit + 1` calls gives the trajectory
and every relearning; a relearning takes the last learner on where the calls between saw no
new edge, exit or written address (so its control dependence and written memory are those of
the longer run), and starts a learner afresh where they did. None means the feed set does not repeat within `limit`. A feed set's period can exceed the calls after which coverage
stops growing when a branch condition reads a long-period component whose both outcomes come
early (a sweep reversing at its bounds): the bound is the one the components prove.

### Learning once per class

A call's class is its sequence of events as the learner's bus sees them: each instruction's PC
and opcode, each followed by its accesses in order, each with its kind (load, store, stack
pull or push, other read or write) and the location it reaches after `remap` (an address, a
`ZERO` read or a suppressed write). Two calls of one class are learned alike:

- everything a `Learner` computes for an instruction is a function of its opcode (the flow
  table), the locations its accesses reach, the taints those locations hold, the control
  dependence of its PC and the memory any call writes. A location's taint at a call's entry is
  its own (or nothing, for memory no call writes), and within the call it is set only by the
  call's own instructions, so by induction over the events every taint, site, condition and
  end-of-call value is the same;
- what is added to the learned state is unioned in (`deps`, sites, `stored`, `addressed`,
  `gated`, edges, exits, locations read first and written), so the second call adds nothing,
  except one thing that is not a function of the call: the implicit flows. A store site's
  controlling branches enter `deps` at the end of every call that ran the branch, from the
  first call that reached the store site on. A repeat therefore adds, for each implicit pair
  `(loc, branch)` learned since its class last ran and each branch the class ran, that
  branch's condition as the class's first call left it. The learner keeps those conditions per
  class and adds exactly these.

So a `Learner` fed a census (`NativeMPU.learn(..., census=...)`) taints each class's first
call, from its entry state (the registers and the memory any earlier call modified, kept by
the census with an undo log of first writes), and for each later call adds only the implicit
flows it would. Its result equals running every call (`test_learning_each_class_once_is_learning_every_call`).
Class identity is exact: calls are hashed, and equal hashes compared event by event.

A read of memory no call writes taints nothing whatever its address (and in explicit mode is
the constant of its PC), and that memory is never persistent, so the learner coarsens the
census classes: it makes each such read alike (one read of a constant, by its kind), the
written memory being its own `written` and the census' own, and taints the first call of each
coarse class. A song read at an advancing pointer then learns as one class
(`Learner.classes` counts them).

The control dependence and written memory are of all the calls learned from, so a learner
over `n` calls is taken on to `m` only when calls `n` to `m` see no new edge, exit or written
address. Edges and accesses alone do not decide what a call teaches: a call can take only
edges and accesses seen before and still combine them into a path no call took, making a
new dependence (`COMBO` in the tests: `z = y` happens only when bits 7 and 6 of a counter are
both set, after every edge and access has been seen). The class of that call is new, so it is
learned.

`NativeMPU.orbit(..., cells=locations)` hashes only those locations (addresses and
`REG + index`) for the state, e.g. `comps.feed`, where RAM holding stale scratch would not
repeat.
