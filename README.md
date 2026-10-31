# jennings

A py65-compatible 6502/6510 MPU, assembler, and disassembler with first-class
support for all 105 documented NMOS 6510 illegal opcodes. Drop-in for py65:
change `import py65…` to `import jennings…`.

Where py65 stubs the illegals (`inst_not_implemented` — a 0-cycle skip),
jennings executes them byte- and cycle-exact via deity-informant's
hardware-validated `lift` + `PcodeVM` engine, and its assembler/disassembler
decode them as real mnemonics instead of `???`.

## Install

```bash
pip install jennings          # pulls deity-informant (pure-Python engine)
pip install jennings[dev]     # + test/lint tooling (pytest, black, pylint, py65)
```

## Use

```python
from jennings.devices.mpu6502 import MPU
from jennings.memory import ObservableMemory
from jennings.disassembler import Disassembler
from jennings.assembler import Assembler

mpu = MPU(memory=ObservableMemory())     # same API as py65.devices.mpu6502.MPU
mpu.memory[0:2] = [0xAF, 0x10]           # LAX $10 — an illegal py65 cannot run
mpu.step()                               # a, x <- mem[$10]; cycle-exact

Assembler(MPU()).assemble("SLO $20")     # [0x07, 0x20]
Disassembler(MPU()).instruction_at(pc)   # (len, "SLO $20")

from jennings import dataflow
dataflow.flow(mem, pc, loads=[0x1236])   # ADC $1234,X: {"A": {"A", "C", "X", 0x1236}, ...}
```

`jennings.superopt` finds the minimal equivalent of straight-line code for a given spec
(live-in variables, tables and parameters, live-out locations): z3 semantics from the same
p-code, an enumerative search on test vectors (numpy, or CUDA with `pip install
jennings[cuda]`), and a z3 proof of the result. See [docs/superopt.md](docs/superopt.md).

`jennings.dataflow.flow` reads each instruction's output-to-input dependencies
(registers, flags, memory bytes, a branch's condition) off the same lifted pcode the
MPU executes, given the addresses an observed run accessed.

`jennings.coverage` records 6502 code coverage (executed bytes, branch directions, data
accesses, dynamic edges), reports it through coverage.py, derives self-modifying-code and
interrupt-vector obligations, explains one-way branches, and records per-test coverage
from pytest. See [docs/coverage.md](docs/coverage.md).

## Native core

Wheels carry a Rust 6510 core (`jennings._core`) whose opcode semantics are generated
from deity-informant's lifter. An `MPU` over a `NativeMemory` runs it; `run_until`,
`run_to_rts` and `run_to_rts_until` release the GIL, so independent tunes run in parallel
threads. An sdist install keeps the pure-Python engine. See [docs/native.md](docs/native.md).

```python
from jennings.memory import NativeMemory
mem = NativeMemory(image)                 # 64 KiB, banked 4 KiB pages, watch sets
mem.io = True; mem.raster_model(False)    # $D011/$D012 follow the PAL beam natively
mpu = MPU(memory=mem)                     # mpu.native is True
mpu.run_to_rts(play, 0, 200_000)
```

## Compatibility

Mirrors the py65 API surface used in practice: `devices.mpu6502.MPU`
(`.step/.reset/.irq/.nmi`, `a/x/y/sp/p/pc/memory/processorCycles`,
`ByteAt/WordAt`, `disassemble/instruct/cycletime/extracycles`),
`memory.ObservableMemory`, `disassembler.Disassembler`, `assembler.Assembler`,
`utils.addressing.AddressParser`.

Every legal opcode is validated byte-, flag-, and cycle-exact against py65 in CI;
the illegals are validated by NMS decomposition and (via the engine) byte-exact
against sidplayfp. See [docs/design.md](docs/design.md),
[docs/illegal-opcodes.md](docs/illegal-opcodes.md), and
[docs/migrating-from-py65.md](docs/migrating-from-py65.md).

## Limitations

No BCD (decimal-mode ADC/SBC), matching the deity-informant engine and the C64
target domain (playroutines run `CLD`). `py65.monitor` is not reimplemented.
