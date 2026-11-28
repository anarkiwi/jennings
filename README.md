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
Assembler(MPU()).assemble_program("loop: DEX; BNE loop", 0x1000)  # (b"\xca\xd0\xfd", {"loop": 0x1000})
Assembler(MPU()).assemble_program("JMP (v); v: .word v", 0x1000)  # (b"\x6c\x03\x10\x03\x10", {"v": 0x1003})
Disassembler(MPU()).instruction_at(pc)   # (len, "SLO $20")

from jennings import dataflow
dataflow.flow(mem, pc, loads=[0x1236])   # ADC $1234,X: {"A": {"A", "C", "X", 0x1236}, ...}
```

`jennings.superopt` finds the minimal equivalent of straight-line code for a given spec
(live-in variables, tables and parameters, live-out locations): z3 semantics from the same
p-code, an enumerative search on test vectors (numpy, or CUDA with `pip install
jennings[cuda]`), and a z3 proof of the result. See [docs/superopt.md](docs/superopt.md).

`jennings.acme` assembles ACME 0.97 sources (the 6502/6510 subset) byte-identically to
`acme`, with its `-o`/`-l`/`!to` command line as `jennings-acme`. See [docs/acme.md](docs/acme.md).

`jennings.isa` classifies opcodes, lists the addresses an instruction touches
(`addresses(mem, pc, x, y, sp)`: data, pointer and stack bytes with the NMOS wraps) and
computes NMOS ADC/SBC including decimal mode; `jennings.opcodes` holds the opcode, cycle
(`CYCLES`) and penalty (`EXTRA`) tables; `jennings.testing` is the decimal-mode reference
model for tests of other 6502 implementations.

`jennings.decoding` decodes instructions from a 64 KiB image (`decode`, and `decode_many` as
numpy records), holds per-opcode role tables read off the lifted pcode (values read, registers
written, memory read/written, index register, kind, stored-value sources), and gives the data
addresses (`accesses`) and destination value sets (`transfer`) of an instruction over value
sets of its inputs. Its stable names, with `MPU`, `Assembler`, `Disassembler` and
`AddressParser`, import lazily from `jennings`. See [docs/api.md](docs/api.md).

`jennings.dataflow.flow` reads each instruction's output-to-input dependencies
(registers, flags, memory bytes, a branch's condition) off the same lifted pcode the
MPU executes, given the addresses an observed run accessed; `addressed=False` gives the
explicit flow, a load's value apart from its address.

`jennings.coverage` records 6502 code coverage (executed bytes, branch directions, data
accesses, dynamic edges), reports it through coverage.py on listings that can follow computed
jumps through constant tables, derives self-modifying-code and interrupt-vector obligations,
explains one-way branches, and records per-test coverage from pytest. `jennings.coverage.components`
splits a repeatedly called routine into autonomous components by native data and control taint,
measures each one's cycle and gives the calls that cover every access and edge it can make;
`jennings.coverage.certificate` records that claim and verifies it by replay, and where voices
repeat each with its own period certifies them by segments of a call instead;
`jennings.coverage.counters` finds the counters and the state nothing follows. See
[docs/coverage.md](docs/coverage.md).

## Command line

```bash
jennings dis tune.sid -f --resolve                 # recursive descent from INIT/PLAY
jennings dis drive.bin -f -e 600 --symbols 1541    # raw dump at $0000, ROM calls external
jennings dis game.prg 0810:0900 --source > g.a     # ACME source that reassembles
jennings xref kernal.bin@e000 0314 --symbols c64   # who reads/writes an address
jennings find game.prg "8d ?? d4" --dis            # byte pattern search
jennings run tune.sid --init --play 1000           # cycles, registers, changed memory
```

See [docs/cli.md](docs/cli.md).

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

`jennings.memo.orbit` gives the orbit of a repeatedly called, input-free routine (calls to
the first repeated state, every data access and control edge) without emulating what it has
already seen. See [docs/memo.md](docs/memo.md).

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

Decimal mode follows the NMOS 6502 (ADC, SBC, ARR, and RRA/ISC through them; no cycle
penalty), including the flags and invalid BCD digits (Bruce Clark, "Decimal Mode").
`py65.monitor` is not reimplemented.
