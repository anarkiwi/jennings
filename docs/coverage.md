# Coverage

`jennings.coverage` records which bytes a 6502 program executed, which way each branch
went, and which data bytes it read or wrote. Recording lives in the engine VM
(`CoverageVM`), so deity-informant's drivers (`run_sub`, `run_irq`, `run_irq_driven`)
are covered as well as `CoverageMPU.step`.

```bash
pip install jennings[coverage]   # numpy
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
pushes and `PLA`/`PLP` pulls); opcode fetch, `JMP (ind)` pointer reads, and the stack
traffic of `JSR`/`RTS`/`RTI`/`BRK` and interrupt entry are not flagged.
`save` writes one `.npz`; `CoverageMap.load` reads it back.
