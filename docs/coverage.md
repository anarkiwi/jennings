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
`save` writes one `.npz`; `CoverageMap.load` reads it back.

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
