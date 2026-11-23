# Import surface and instruction API

## Top-level names

`import jennings` is cheap: the names below load their module on first access
(`jennings.EXPORTS` maps each name to its module).

| Name | Defined in |
| --- | --- |
| `MPU` | `jennings.devices.mpu6502` |
| `Assembler` | `jennings.assembler` |
| `Disassembler` | `jennings.disassembler` |
| `AddressParser` | `jennings.utils.addressing` |
| `ObservableMemory`, `NativeMemory` | `jennings.memory` |
| `OPCODES` | `jennings.opcodes` |
| `decode`, `decode_many`, `roles`, `accesses`, `transfer`, `taken`, `Instruction`, `Roles`, `UNKNOWN`, `as_mask`, `as_values` | `jennings.decoding` |

The py65 module paths (`jennings.devices.mpu6502`, `jennings.utils.addressing`, ...) remain the
canonical homes; there are no alias modules.

## `jennings.decoding`

Mode names are jennings' own (`jennings.opcodes.MODE_LEN`): `imm zp zpx zpy abs absx absy
indx indy acc impl rel ind` (py65's disassembler tokens `zpg abx aby inx iny imp` are
`opcodes.MODE_TOKEN`'s values).

### Decoding

```python
decode(mem, pc) -> Instruction(pc, opcode, mnemonic, mode, operand, length, text)
```

`mem` is any indexable 64 KiB image (numpy `uint8`, `bytes`, `bytearray`, list). `operand` is
effective: the byte for `imm` and zero-page modes, the word for absolute modes and `ind`, the
branch target for `rel`, `None` for `impl`/`acc`. Operand bytes wrap at `$FFFF`. `text` equals
`Disassembler(MPU(memory=mem)).instruction_at(pc)[1]`.

```python
decode_many(mem, starts) -> numpy structured array of DTYPE
```

`starts` is an index array/list or a 65536-entry bool mask. Fields: `pc` (u2), `opcode` (u1),
`mode` (u1, an index into `MODES`; `MODE_CODE` maps names to codes), `operand` (i4, -1 for
none), `length` (u1). Combined with the per-opcode tables:

```python
from jennings.decoding import KIND, KIND_CODE, MODE_CODE, decode_many
d = decode_many(mem, starts)
indexed = np.isin(d["mode"], [MODE_CODE[m] for m in ("absx", "absy", "indy")])
loads = d[(KIND[d["opcode"]] == KIND_CODE["load"]) & indexed]
```

### Roles

Per-opcode numpy tables, indexed by opcode, read off the lifted pcode the MPU executes:

| Table | dtype | Meaning |
| --- | --- | --- |
| `MODE`, `LENGTH` | u1 | mode code, byte length |
| `KIND` | u1 | index into `KINDS` |
| `READS` | u2 mask | locations read as values (not as addresses) |
| `WRITES` | u2 mask | registers and flags written |
| `STORED` | u2 mask | locations the byte written to the memory operand comes from |
| `ADDRESSED` | u2 mask | locations the data/pointer addresses come from |
| `INDEX` | u2 | `X`, `Y` or 0: the mode's index register |
| `MEM_READ`, `MEM_WRITE` | bool | the data memory operand is read / written on the bus |

Masks are bit `i` = `LOCATIONS[i]`: `A X Y SP` (bits 0-3), `M` (the memory operand's value),
`OPERAND` (the instruction's operand bytes: an immediate, or `SHx`'s `H + 1`), and the flags
`C Z I D B V N` at their PcodeVM register index (bits 8-14); constants `A`, `X`, `Y`, `SP`,
`M`, `OPERAND`, `C`, `FLAGS` and `names(mask)` help. Pointer and stack bytes are not
locations: a stack access reads `SP`.

`KINDS`: `load` (value from memory/immediate only: LDA LDX LDY LAX), `store` (writes memory
without reading it: STA STX STY SAX SHA SHX SHY TAS), `rmw`, `transfer` (TAX TAY TXA TYA TSX
TXS), `incdec` (INX INY DEX DEY), `alu` (other register results, e.g. ADC, ASL A, ANC, SBX,
LXA, LAS), `compare` (flags only: CMP CPX CPY BIT), `branch`, `jump`, `call`, `return`,
`stack` (PHA PHP PLA PLP), `flag` (CLC ... SED), `nop`, `other` (BRK, JAM).

`roles(op)` gives one row as names:

```python
>>> roles(0x87)
Roles(mnemonic='SAX', mode='zp', kind='store', reads=('A', 'X'), writes=(), mem_read=False,
      mem_write=True, index=None, stored=('A', 'X'))
```

### Value sets

A value set is `None` (any value), a Python `set`/`frozenset`/list of integers, a numpy
integer array, a numpy bool array of length 256, or an int bitmask as
`jennings.coverage.vsa` uses (bit `v` set when `v` is possible; `{5}` is `1 << 5`, not `5`).
`as_values(v, n=256)` gives the sorted values, `as_mask(v, width=1)` the bitmask.

```python
accesses(op, operand, x=None, y=None, words=None, mem=None, a=None) -> sorted int64 array
```

Data addresses over the value sets: `zp`/`abs` the operand, `zpx`/`zpy` wrapping in page
zero, `absx`/`absy` at `$FFFF`; `indy` each pointer word plus each `Y` (16-bit wrap); `indx`
the pointers at `(operand + X) & $FF` and its successor in page zero. Pointer words come from
`words`, else from `mem`, else are any. `SHA`/`SHX`/`SHY`/`TAS` crossing a page use the
NMOS high byte `reg & (H + 1)` (`a` needed for `SHA`/`TAS`). On singletons it equals
`isa.addresses(...)["data"]`.

```python
transfer(op, a=None, x=None, y=None, m=None, c=None, sp=None, operand=None,
         n=None, z=None, v=None, pulled=None)
    -> {destination: mask} | UNKNOWN
```

The value sets `op` leaves in its destinations, evaluated over the opcode's pcode with `D`
clear (binary mode). Destinations are exactly the locations the pcode writes:

| Key | Value set |
| --- | --- |
| `"A"`, `"X"`, `"Y"`, `"SP"` | the register |
| `"C"`, `"Z"`, `"V"`, `"N"` (`OUTPUT_FLAGS`) | the flag, over `{0, 1}`: mask `1` clear, `2` set, `3` either |
| `"M"` | the byte written to the memory operand |
| `"PUSHED"` | the byte `PHA`/`PHP` writes to the stack |

A key is absent when the opcode leaves that location unchanged (`STA` gives only `"M"`), so a
caller carries its own value forward. `m` is the memory operand's value, or the immediate in
`imm` mode (taken from `operand` when `m` is None); `operand` also supplies `SHx`'s `H + 1`;
`pulled` is the byte `PLA`/`PLP` read from the stack. `c`, `n`, `z`, `v` are the input flags
as value sets over `{0, 1}` (a mask of `1`, `2` or `3`, or a set such as `{0}`; ValueError
otherwise); `I` is any and `D` clear (after `SED`/`PLP` setting `D` the results do not hold).
Unset inputs are any value.

Every destination is exact. Sets propagate as masks op by op while the pcode combines
independent sets; a destination whose computation combines two sets derived from the same
input (`ADC`/`SBC`/`ISC`'s `C` and `V`, `ARR`'s `C` and `V`, `RRA`'s result) is instead
evaluated on numpy arrays over the cartesian product of the input sets it depends on (at most
`A × M × C`). Each flag's set is exact on its own: the relation between flags of one result
(or between a flag and the register it describes) is not kept, so `Z` and `N` of `{0, 1}` admit
the combination `Z = N = 1` that no single value gives. Branch, jump, call, return and
`BRK`/`JAM` opcodes return `UNKNOWN`; `CLI`/`SEI`/`CLD`/`SED` and `NOP` `{}`.

```python
>>> from jennings.coverage.vsa import bits
>>> {k: bits(v).tolist() for k, v in transfer(0x69, a={1, 2}, m={10}, c={0, 1}).items()}
{'V': [0], 'C': [0], 'A': [11, 12, 13], 'Z': [0], 'N': [0]}
>>> transfer(0xC9, a={5}, m={5})
{'C': 2, 'Z': 2, 'N': 1}
```

```python
taken(op, flags=None) -> frozenset of {True, False}
```

The outcomes conditional branch `op` may have (`True`: taken), its flag (`N` for BPL/BMI, `V`
for BVC/BVS, `C` for BCC/BCS, `Z` for BNE/BEQ, as the branch template tests it) read from
`flags`, a `{name: value set over {0, 1}}` dict such as `transfer`'s result; a missing flag
is either. ValueError for any other opcode.

A forward interpreter with concrete inputs follows the one edge the flags decide:

```python
from jennings.coverage.vsa import bits, is_single
from jennings.decoding import KIND, KIND_CODE, accesses, decode, taken, transfer

INPUTS = ("A", "X", "Y", "SP", "C", "Z", "V", "N")
STOP = {KIND_CODE[k] for k in ("stack", "jump", "call", "return", "other")}


def run(mem, pc, state, steps):
    """Step from ``pc`` over the numpy image ``mem`` while every result is a single value;
    ``state`` maps INPUTS names to masks (missing: any)."""
    for _ in range(steps):
        ins = decode(mem, pc)
        op = ins.opcode
        if KIND[op] == KIND_CODE["branch"]:
            edges = taken(op, state)
            if len(edges) != 1:
                break
            pc = ins.operand if True in edges else pc + ins.length
            continue
        ea = accesses(op, ins.operand or 0, x=state.get("X"), y=state.get("Y"), mem=mem,
                      a=state.get("A"))
        if KIND[op] in STOP or len(ea) > 1:
            break
        regs = {k.lower(): v for k, v in state.items() if k in INPUTS}
        out = transfer(op, operand=ins.operand, m=set(mem[ea].tolist()) or None, **regs)
        if not all(map(is_single, out.values())):
            break
        if "M" in out:
            mem[ea[0]] = bits(out.pop("M"))[0]
        state.update(out)
        pc += ins.length
    return pc, state
```
