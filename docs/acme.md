# ACME-compatible assembler

`jennings.acme` assembles [ACME](https://sourceforge.net/projects/acme-crossass/) 0.97
sources for the 6502 and NMOS 6510 and writes the same output file and `-l` symbol
list as `acme`, byte for byte. It needs no subprocess and no ACME install.

## Use

```python
from jennings.acme import Acme, assemble

data, symbols = assemble(open("tune.asm").read())   # output file bytes, {name: value}

acme = Acme([("tune.asm", text)], outname="tune.sid").run()
acme.output()        # file bytes in the selected format
acme.image()         # (start address, bytes) without a format header
acme.symbol_list()   # the -l / --symbollist text
acme.outname         # -o, else the !to file name, else None
```

`Acme(sources, cpu="6502", fmt=None, outname=None)` takes `[(name, text)]` sources,
assembled in order every pass, as ACME does with several files. Errors raise
`AcmeError("name:line: message")` with ACME's message text.

```
jennings-acme [-f plain|cbm|apple] [-l LABELS] [-o OUT] [--cpu CPU] SOURCE...
python -m jennings.acme ...
```

The command line follows `acme`: `-o` overrides `!to` (which then only warns, so its
format is ignored too), `-l`/`--symbollist` writes the symbol list, and with no output
file named it says so and writes nothing.

## Semantics carried over from ACME

- **Passes.** Every pass reparses the whole source with the symbols kept; passes repeat
  while the number of undefined expression results falls, then a final pass names the
  first undefined symbol.
- **Operand sizes** come from ACME's number flags, not from a layout search: a value
  read before its definition is "ever undefined" and takes the 16-bit mode in every
  pass (unless `<`/`>` made it fit a byte); hex, binary and octal literals with leading
  zeros force their width (`$0012` is absolute); `+1`/`+2` postfixes on mnemonics and
  symbols force it explicitly.
- **Symbol list.** Global numeric symbols in the order of ACME's 256 hash-keyed binary
  trees, with `+2`/`+3` for forced sizes, `; ?` for ever-undefined symbols and
  `; unused` for symbols never read.
- **Output.** The bytes between the lowest and highest address written; `cbm` prefixes
  the load address, `apple` the load address and length; `!to` without a format means
  `cbm`, no `!to` and no `-f` means plain.
- **Expressions** use ACME's priorities (`<`/`>` bind looser than `+`, `^` is a
  right-associative power), C integer division and remainder, and 64-bit wrap-around.

## Supported subset

Statements: global, `.local` and anonymous (`-`, `--`, `+`, `++`) labels, `name = expr`,
`*= expr` (with `overlay`/`invisible`), `:` separators, `;` and `//` comments.
Pseudo opcodes: `!to`, `!cpu` (`6502`, `6510`, `nmos6502`, optionally with a block),
`!pseudopc expr { }`, `!macro name .param, ... { }` with `+name args` calls,
`!8`/`!08`/`!by`/`!byte`, `!16`/`!wo`/`!word`/`!le16`/`!be16`, `!24`/`!le24`/`!be24`,
`!32`/`!le32`/`!be32`, `!h`/`!hex`, `!fi`/`!fill`, `!tx`/`!text`.

Not implemented, and rejected with `AcmeError("... Unsupported: ...")` rather than
assembled differently: other CPUs, `dop`/`top`, cheap `@locals`, zones, conditionals and
loops (`!if`, `!for`, `!do`, `!while`), `!set`, includes (`!src`, `!binary`), encodings
other than raw (`!ct`, `!pet`, `!scr`, `!scrxor`), `!skip`/`!align`/`!initmem`/`!xor`,
`!pseudopc` without a block, call-by-reference macro parameters, string symbols and
string operators, lists, floats and functions. Warnings are not printed.

## Differential testing against acme

`tools/acme_oracle.py` assembles sources with both `acme` and `jennings.acme`, with `-o`
and again with only `!to`, and compares every file written and every rejection:

```bash
python tools/acme_oracle.py SOURCE... [--fuzz 1000 --seed 0] [--keep DIR]
```

`--fuzz` adds random programs (labels, anonymous labels, expressions, every 6510
addressing mode, data, a macro, `!pseudopc`). `tests/test_acme_oracle.py` runs the same
comparison on the test sources and fuzz seeds when `acme` is installed; the CI `acme`
job installs it.
