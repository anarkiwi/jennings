"""``python -m jennings.superopt "LDA $10; CLC; ADC #$01; STA $10" --var v=0x10 --out v``:
the minimal sequence for straight-line code, its size, cycles and the search's rate."""

import argparse
import sys

from jennings.assembler import Assembler
from jennings.devices.mpu6502 import MPU
from jennings.disassembler import Disassembler
from jennings.superopt.search import Problem, search
from jennings.superopt.symbolic import Spec, Table, Var

ORIGIN = 0x1000


def _located(text: str, kind, default: int):
    """A :class:`Var` or :class:`Table` of ``name=address[:size]``."""
    name, where = text.split("=")
    address, _, size = where.partition(":")
    return kind(name, int(address, 0), int(size, 0) if size else default)


def assemble(text: str, origin: int = ORIGIN) -> bytes:
    """The bytes of ``;``-separated statements."""
    assembler, out = Assembler(MPU()), bytearray()
    for statement in filter(None, (s.strip() for s in text.split(";"))):
        out += bytes(assembler.assemble(statement, origin + len(out)))
    return bytes(out)


def listing(code: bytes, origin: int = ORIGIN) -> list:
    """The statements of ``code``."""
    mpu = MPU()
    mpu.memory[origin : origin + len(code)] = list(code)
    out, pc, disassembler = [], origin, Disassembler(mpu)
    while pc < origin + len(code):
        size, text = disassembler.instruction_at(pc)
        out.append(text)
        pc += size
    return out


def main(argv=None) -> int:
    """Searches for the minimal equivalent of the code given."""
    parser = argparse.ArgumentParser(prog="python -m jennings.superopt", description=__doc__)
    parser.add_argument("code", help="';'-separated statements")
    parser.add_argument("--var", action="append", default=[], help="name=address[:size]")
    parser.add_argument("--table", action="append", default=[], help="name=address[:length]")
    parser.add_argument("--out", action="append", default=[], help="a live-out location")
    parser.add_argument("--entry", action="append", default=[], help="register=value")
    parser.add_argument("--illegal", action="store_true", help="stable illegal opcodes too")
    parser.add_argument("--backend", default="auto", choices=("auto", "numpy", "cuda"))
    parser.add_argument("--max-length", type=int, default=6)
    args = parser.parse_args(argv)
    spec = Spec(
        vars=tuple(_located(v, Var, 1) for v in args.var),
        tables=tuple(_located(t, Table, 256) for t in args.table),
        entry={k: int(v, 0) for k, v in (e.split("=") for e in args.entry)},
        outputs=tuple(args.out),
    )
    code = assemble(args.code)
    found = search(
        Problem(spec, code, ORIGIN),
        backend=args.backend,
        max_length=args.max_length,
        illegal=args.illegal,
    )
    print(f"reference: {len(code)} bytes")
    if found.instructions is None:
        print("no shorter sequence")
    else:
        for text in listing(found.code):
            print(f"  {text}")
        print(f"minimal: {found.size} bytes, {found.cycles} cycles")
    rate = f"{found.rate:.3g}/s, {found.backend}"
    print(f"{found.candidates} candidates in {found.seconds:.2f}s ({rate})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
