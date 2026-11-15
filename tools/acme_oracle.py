"""Differential test of ``jennings.acme`` against the real ``acme`` binary.

Each source (files, or ``--fuzz N`` random programs) is assembled with ``-l``/``-o`` and
again without ``-o``; written files must match, and acme's rejections must be matched.
"""

import argparse
import os
import pathlib
import random
import shutil
import subprocess
import sys
import tempfile

from tqdm import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# pylint: disable=wrong-import-position
from jennings.acme import Acme, AcmeError, mnemonics  # noqa: E402

ACME = shutil.which("acme")
OPERANDS = {
    "imm": "#{v8}",
    "abs": "{v}",
    "x": "{v},x",
    "y": "{v},Y",
    "ind": "({v})",
    "xind": "({v8},X)",
    "indy": "({v8}),y",
    "rel": "{rel}",
}
DATA = ("by", "byte", "wo", "word", "be16", "le16", "24", "be24", "32", "be32")
OPS = ("+", "-", "|", "&", "^", "*", "/", " xor ", "<<", ">>", ">>>", "%", " or ", " div ", "=")


def reference(text, outfile=True, acme=ACME):
    """``{file name: bytes}`` acme writes for ``text``, or ``None`` if it rejects it."""
    with tempfile.TemporaryDirectory() as tmp:
        (pathlib.Path(tmp) / "src.asm").write_bytes(text.encode("latin-1"))
        args = [acme, "-l", "labels"] + (["-o", "out"] if outfile else []) + ["src.asm"]
        if subprocess.run(args, cwd=tmp, capture_output=True, check=False).returncode:
            return None
        return {p.name: p.read_bytes() for p in pathlib.Path(tmp).iterdir() if p.name != "src.asm"}


def candidate(text, outfile=True):
    """The same ``{file name: bytes}`` from :class:`jennings.acme.Acme`, or ``None``."""
    try:
        acme = Acme([("src.asm", text)], outname="out" if outfile else None).run()
    except AcmeError:
        return None
    files = {"labels": acme.symbol_list().encode("latin-1")}
    if acme.outname is not None:
        files[acme.outname] = acme.output()
    return files


def differences(text, acme=ACME):
    """Descriptions of every way ``jennings.acme`` differs from acme on ``text``."""
    found = []
    for outfile in (True, False):
        want, got = reference(text, outfile, acme), candidate(text, outfile)
        mode = "-o" if outfile else "!to"
        if (want is None) != (got is None):
            found.append("%s: acme %s" % (mode, "error" if want is None else "ok"))
        elif want is not None:
            found += [
                "%s: %s differs" % (mode, n)
                for n in sorted(set(want) | set(got))
                if want.get(n) != got.get(n)
            ]
    return found


def program(rng):
    """A random, mostly valid ACME source over labels (backward, forward, forward-defined
    constants), anonymous labels, expressions, every 6510 addressing mode, data, a macro
    and ``!pseudopc``."""
    table, labels = mnemonics("6510"), ["l%d" % i for i in range(6)]
    late = ["k%d" % i for i in range(3)]
    lines = ["!cpu 6510", "zp = $%02x" % rng.randrange(256), "-", "--"]
    lines += ["*= $%04x" % rng.randrange(0x800, 0x9000)]
    if rng.random() < 0.5:
        lines.insert(0, '!to "prog.bin"%s' % rng.choice(["", ", plain", ", cbm", ", apple"]))
    lines += ["!macro m .a, .b {", "  !byte .a, <.b, >.b : lda #.a", "}"]

    def atom():
        return rng.choice(
            [
                "$%0*x"
                % (rng.choice([1, 2, 4]), rng.randrange(256 if rng.random() < 0.5 else 0x10000)),
                str(rng.randrange(300)),
                "%" + format(rng.randrange(256), "08b"),
                rng.choice(labels + late + ["zp", "*", "-", "--", "+", "++"]),
                "'%s'" % rng.choice("az09"),
            ]
        )

    def value(width):
        expr = atom()
        for _ in range(rng.randrange(3)):
            op = rng.choice(OPS)
            right = (
                str(rng.randrange(1, 9))
                if op.strip() in ("/", "%", "div", "<<", ">>", ">>>")
                else atom()
            )
            expr = rng.choice(["(%s)%s%s", "%s%s%s"]) % (expr, op, right)
        if width == 8:
            return rng.choice("<>") + expr
        return rng.choice(["(%s) & $ffff", "%s", "<%s", "(%s) & $ff"]) % expr

    for _ in range(rng.randrange(10, 60)):
        kind = rng.randrange(20)
        if kind == 0 and labels:
            lines.append(labels.pop(rng.randrange(len(labels))))
        elif kind == 1:
            lines.append(rng.choice(["-", "+", "--", "++"]))
        elif kind == 2:
            directive = rng.choice(DATA)
            width = 8 if directive in ("by", "byte") else 16
            items = ", ".join(value(width) for _ in range(rng.randrange(1, 4)))
            lines.append("  !%s %s" % (directive, items))
        elif kind == 3:
            lines.append("  !fill %d, %s" % (rng.randrange(5), value(8)))
        elif kind == 4:
            lines.append('  !text "%s", %s' % (rng.choice(["abc", "Q:;x}", ""]), value(8)))
        elif kind == 5:
            lines.append("  !h " + " ".join("%02x" % rng.randrange(256) for _ in range(3)))
        elif kind == 6:
            lines.append("  +m %s, %s" % (value(8), value(16)))
        elif kind == 7:
            lines += [
                "  !pseudopc $%04x {" % rng.randrange(0x10000),
                "-  nop : bne - : jmp +",
                "+ }",
            ]
        else:
            mnemonic = rng.choice(sorted(table))
            syntax = rng.choice(sorted(table[mnemonic]))
            rel = "*%+d" % rng.randrange(-100, 100)
            operand = OPERANDS.get(syntax, "").format(v=value(16), v8=value(8), rel=rel)
            lines.append("  %s %s" % (rng.choice([mnemonic, mnemonic.upper()]), operand))
    lines += ["%s = *" % l for l in labels] + ["%s = $%02x" % (k, rng.randrange(256)) for k in late]
    return "\n".join(lines + ["+", "++"]) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    ap.add_argument("sources", nargs="*", type=pathlib.Path)
    ap.add_argument("--fuzz", type=int, default=0, help="random programs to compare")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep", type=pathlib.Path, help="directory to save differing sources in")
    ap.add_argument("--acme", default=ACME)
    args = ap.parse_args(argv)
    cases = [(str(p), p.read_bytes().decode("latin-1")) for p in args.sources]
    rng = random.Random(args.seed)
    cases += [("fuzz%d" % i, program(rng)) for i in range(args.fuzz)]
    bad = 0
    for name, text in tqdm(cases, desc="acme oracle", unit="src"):
        found = differences(text, args.acme)
        if found:
            bad += 1
            tqdm.write("%s: %s" % (name, "; ".join(found)))
            if args.keep:
                args.keep.mkdir(parents=True, exist_ok=True)
                (args.keep / (os.path.basename(name) + ".asm")).write_bytes(text.encode("latin-1"))
    print("%d/%d sources identical" % (len(cases) - bad, len(cases)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
