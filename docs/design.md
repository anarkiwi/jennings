# Design

jennings is a py65-compatible façade over the deity-informant execution engine.

## Layers

- **`jennings.devices.mpu6502.MPU`** — presents py65's MPU attribute/method
  surface (`a/x/y/sp/p/pc/memory/processorCycles`, `step/reset/irq/nmi`,
  `ByteAt/WordAt`, the `disassemble/instruct/cycletime/extracycles` tables). It
  holds a `deity_informant.PcodeVM` and calls `PcodeVM.step(pc, cache, lift)` for
  each instruction. Registers are properties over `PcodeVM.reg`
  (`0-3` = A/X/Y/SP; `8-14` = C/Z/I/D/B/V/N); `p` packs/unpacks those flags.
- **`jennings.memory.ObservableMemory`** — original reimplementation of py65's
  observer memory. The MPU points the engine's memory (`vm.mem`) directly at the
  caller's object, so read/write observers (e.g. SID `$D400` traps) fire on
  engine accesses. `vm.volatile` is forced off, so reads are plain RAM.
- **`jennings.disassembler` / `jennings.assembler`** — original reimplementations
  that consume the MPU `disassemble` table. Because that table is populated from
  deity-informant's `OPS` (all 256 opcodes), illegal mnemonics decode and
  assemble with no special-casing.

## Why a façade

deity-informant's lifter + VM already execute all 256 opcodes byte- and
cycle-exact (legal opcodes cross-checked against py65; illegals validated by NMS
decomposition and against sidplayfp). Reusing it — rather than porting 6502
semantics a third time — keeps jennings small and inherits that validation.

## JAM

The 12 NMOS JAM/KIL opcodes lock the CPU. `MPU.step()` detects them, sets
`mpu.jammed`, and leaves `pc` unchanged (a re-fetch loop), rather than raising.

## Non-goals

No BCD (decimal-mode ADC/SBC) — the engine and the C64 target domain assume
`CLD`. The interactive `py65.monitor` is not reimplemented (no observed use).
