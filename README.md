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
