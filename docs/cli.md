# Command line

`jennings` (or `python -m jennings`) disassembles, dumps, searches, cross-references,
assembles and runs 6502 images without writing a script. Full ACME sources go to
`jennings-acme` ([acme.md](acme.md)).

```
jennings {dis,hex,find,xref,run,info} IMAGE [options]
jennings asm "STATEMENTS" [--org ADDR] [-o OUT]
```

## Images

`IMAGE` is `PATH[@ADDR]`. The format is detected: `PSID`/`RSID` magic is a SID tune
(v1-v4 header; a zero load address takes the first two data bytes, a zero init the load
address), a `.prg` has a 2-byte little-endian load address, anything else is raw at
address 0. `--format {auto,prg,sid,raw}` forces one; `@ADDR` or `--base ADDR` sets the
load address. `--overlay PATH[@ADDR]` (repeatable) loads more files on top, e.g. a RAM
dump over a ROM. Commands default to the loaded spans.

## Addresses, ranges and labels

An address is hex (`c000`, `$c000`, `0xc000`), a label, or `label+hex`/`label-hex`.
A `RANGE` is `LO:HI` or `LO-HI` (half-open), `LO+N` (N instructions for `dis`, N bytes
otherwise, N decimal) or a bare `LO` (`-n` units). Use `:` when a bound needs arithmetic
(`INIT+4:INIT+20`).

Labels come from `--labels FILE` (repeatable; each line an ACME symbol list entry, a
VICE/ld65 `-Ln` entry `al C:1234 .name` or `al 1234 name`, or `name = $1234` /
`name=0x1234`), `-D NAME=ADDR`, built-in `--symbols {c64,1541,1571,1581}` (I/O
registers, KERNAL entries and vectors; drive job queue, track/sector table and buffers)
and a tune's `INIT`/`PLAY`. Later sources win.

## dis

```
$ jennings dis tune.sid -f --resolve
1000  4C 06 10  INIT:  JMP L1006
1003  4C 72 11  PLAY:  JMP L1172
1006  AA        L1006: TAX
1007  BD E5 12         LDA $12e5,X
```

Linear by default, one block per range. `--no-bytes` drops the byte column,
`--auto-labels` names branch/`JMP`/`JSR` targets inside the ranges `L%04X`. `-f/--follow`
decodes by recursive descent (`jennings[coverage]`) from `-e/--entry ADDR`, else
`INIT`/`PLAY`, else the range starts; unreached bytes in the ranges are `.byte` rows,
printable runs commented with their text. `--resolve` follows computed jumps through
constant tables. Targets outside the image are not descended into: they are named
(`X%04X` unless known) and marked `; external`.

```
$ jennings dis drive.bin -f -e 600 --symbols 1541
0552  20 00 C1         JSR XC100  ; external
05B8                   .byte $50,$52,$4F,$54,$45,$43,$54,$45,$44  ; "PROTECTED"
```

`--source` emits ACME source (`!cpu 6510`, `* = $XXXX` per range, labels on their own
lines, `!byte` data and opcodes ACME would encode differently) that `jennings-acme`
assembles back to the same bytes.

## hex

```
$ jennings hex sidplayer.prg 12e5+20
12E5  06 04 F3 12 F7 12 FA 12 FC 12 FF 12 01 13 00 00  ................
12F5  01 FF 02 03                                      ....
```

`-w` bytes per row; `--petscii` (mixed-case set) or `--screen` (screen codes) for the
text column.

## find

```
$ jennings find sidplayer.prg "8d ?? d4" --dis
1050  STA $d418
```

Hex bytes with `??` wildcards (spaces optional), or `--text STRING`.

## xref

```
$ jennings xref drive.bin 1c00 --symbols 1541
0500  read    LDA VIA2_PB
054C  write   STA VIA2_PB
054F  rmw     INC VIA2_PB
```

Instructions whose operand (indexed modes by their base) or branch target is an `ADDR`
or within `--range LO:HI`, with the kind: `read`, `write`, `rmw`, `jump`, `call`,
`branch` or `ref`. A linear sweep of the image, or only followed code with `-f`/`-e`.

## run

```
$ jennings run tune.sid --init --play 1000
INIT $1000  525 cycles  A=0F X=03 Y=06 SP=FF P=31
PLAY $1003 x1000  678870 cycles (min 570 max 2619)  A=00 X=FF Y=06 SP=FF P=B1
changed 00E2:00E6  59 04 0C 9C
...
```

Calls `--init [SUBTUNE]` (A = subtune, default the start song), each
`--call ADDR[:A[:X[:Y]]]` in order, then `PLAY` `--play N` times (with a progress bar),
each to its `RTS` within `--max-cycles` (`(budget)` if it ran out). Then lists memory that
changed as `LO:HI` ranges; `--dump RANGE` hex dumps after the run, `--trace` prints every
instruction with the registers after it. `--engine {native,python}` defaults to the native
core when it is installed.

## info

```
$ jennings info tests/fixtures/sidplayer/sidplayer.prg
format  prg
span    1000:138A  (906 bytes)
```

Format, spans, SID header fields and, when loaded, the `$FFFA`-`$FFFF` vectors.

## asm

```
$ jennings asm "loop: INC $D020; JMP loop" --org c000 -o border.prg
C000  EE 20 D0 4C 00 C0
loop = $C000
```

`;`-separated statements (`Assembler.assemble_program`); `-o` writes a `.prg` with its
load address, raw bytes for other names.
