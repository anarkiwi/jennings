# Illegal opcodes

jennings executes and decodes all 105 documented NMOS 6510 illegal opcodes:

| Group | Mnemonics |
|---|---|
| RMW combos | SLO, RLA, SRE, RRA, DCP, ISC |
| store/load combos | SAX, LAX |
| immediate ALU | ANC, ALR, ARR, SBX, ANE, LXA, SBC ($EB alias) |
| stack/SP | LAS, TAS |
| unstable "store high" | SHA, SHX, SHY |
| NOP variants (DOP/TOP) | NOP #imm / zp / zp,X / abs / abs,X, and implied |
| lock-up | JAM (KIL/HLT) |

Semantics, cycle counts, and the ANE/LXA magic constant (`$EE`) come from the
deity-informant engine, which follows "No More Secrets — NMOS 6510 Unintended
Opcodes" and is validated byte-exact against sidplayfp. The full per-opcode table
lives in deity-informant's
[docs/illegal-opcodes.md](https://github.com/anarkiwi/deity-informant/blob/main/docs/illegal-opcodes.md).

`mpu.disassemble[op]` carries the `(mnemonic, mode)` for every opcode, so the
assembler and disassembler handle illegals with no special cases. The only
documented behavioural gap versus real hardware is decimal mode (see
[design.md](design.md)).
