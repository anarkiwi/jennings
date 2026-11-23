"""jennings: disassemble, dump, search, cross-reference, assemble and run 6502 images."""

import argparse
import re
import sys

from jennings.acme import canonical
from jennings.assembler import Assembler
from jennings.devices.mpu6502 import MPU
from jennings.disassembler import Disassembler
from jennings.image import FORMATS, Image, parse_address, read_labels
from jennings.isa import is_read, is_rmw, is_store, length, s8
from jennings.memory import NATIVE, NativeMemory
from jennings.symbols import SETS, symbols
from jennings.utils.addressing import AddressParser

__all__ = ["main"]

_TABLE = MPU.disassemble
_LEN = [length(op) for op in range(256)]
_WORD = frozenset(("abs", "abx", "aby", "ind", "iax"))
_BYTE = frozenset(("zpg", "zpx", "zpy", "inx", "iny", "zpi"))
_JSR, _JMP = 0x20, 0x4C
_IDENT = re.compile(r"(?![AaXxYy]\Z)[A-Za-z_]\w*\Z")
_DATA_ROW, _TEXT_ROW = 8, 32
_PRINTABLE = "".join(chr(b) if 0x20 <= b < 0x7F else "." for b in range(256))
_PETSCII = "".join(
    (
        chr(b + 0x20)
        if 0x41 <= b <= 0x5A
        else chr(b - 0x20) if 0x61 <= b <= 0x7A else chr(b - 0x80) if 0xC1 <= b <= 0xDA else c
    )
    for b, c in enumerate(_PRINTABLE[:0x60] + "." * 0xA0)
)
_SCREEN = "".join(
    _PETSCII[(s + 0x40, s, s + 0x80, 0)[s >> 5]] if s < 0x60 else "." for s in range(128)
)
_CHARSETS = {"ascii": _PRINTABLE, "petscii": _PETSCII, "screen": _SCREEN * 2}


class CliError(Exception):
    pass


class _Names(AddressParser):
    """Dict-backed :class:`AddressParser` that records the addresses it names."""

    def __init__(self, names):
        super().__init__()
        self.names, self.used = names, set()

    def label_for(self, address, default=None):
        name = self.names.get(address)
        if name is None:
            return default
        self.used.add(address)
        return name


def _num(text, labels, top=0xFFFF):
    m = re.fullmatch(r"\s*([^+\-\s]+)\s*(?:([+-])\s*(\S+))?\s*", text)
    if m is None:
        raise CliError("bad address: %r" % text)
    try:
        v = labels[m[1]] if m[1] in labels else parse_address(m[1])
        if m[2]:
            v += parse_address(m[3]) * (1 if m[2] == "+" else -1)
    except ValueError:
        raise CliError("bad address or unknown label: %r" % text)
    if not 0 <= v <= top:
        raise CliError("address out of range: %r" % text)
    return v


def _range(text, labels, count):
    """``(lo, hi, count)``: ``LO:HI``/``LO-HI`` half-open, ``LO+N`` N units, bare ``LO``."""
    if ":" in text or "-" in text:
        lo, _, hi = text.partition(":") if ":" in text else text.rpartition("-")
        return _num(lo, labels), _num(hi, labels, 0x10000), None
    if "+" in text:
        lo, _, n = text.rpartition("+")
        if not n.isdigit():
            raise CliError("LO+N wants a decimal N: %r" % text)
        return _num(lo, labels), 0x10000, int(n)
    return _num(text, labels), 0x10000, count


def _span(text, labels, count):
    """``(lo, hi)`` of a byte ``RANGE``: ``LO+N`` and a bare ``LO`` span ``N``/``count`` bytes."""
    lo, hi, n = _range(text, labels, count)
    return lo, hi if n is None else min(hi, lo + n)


def _ranges(args, img, labels, count):
    if args.ranges:
        return [_range(r, labels, count) for r in args.ranges]
    return [(lo, hi, None) for lo, hi in img.merged()]


def _load(args):
    img = Image()
    try:
        img.load(args.image, args.format, None if args.base is None else parse_address(args.base))
        for spec in args.overlay:
            img.load(spec, "auto")
        sources = [img.labels] + [symbols(s) for s in args.symbols]
        for path in args.labels:
            with open(path, encoding="latin-1") as f:
                sources.append(read_labels(f.read()))
    except (OSError, ValueError, IndexError) as e:
        raise CliError(str(e))
    labels = {k: v for src in sources for k, v in src.items()}
    for d in args.define:
        name, eq, value = d.partition("=")
        if not eq:
            raise CliError("-D wants NAME=ADDR: %r" % d)
        labels[name] = _num(value, labels)
    return img, labels, {a: name for name, a in labels.items()}


def _operand(mem, a):
    """``(kind, address)`` of the instruction at ``a``'s operand, ``(None, None)`` if none."""
    op = mem[a]
    mn, md = _TABLE[op]
    if md == "rel":
        return "branch", (a + 2 + s8(mem[(a + 1) & 0xFFFF])) & 0xFFFF
    if md in _BYTE:
        target = mem[(a + 1) & 0xFFFF]
    elif md in _WORD:
        target = mem[(a + 1) & 0xFFFF] | mem[(a + 2) & 0xFFFF] << 8
    else:
        return None, None
    kind = (
        "call"
        if op == _JSR
        else (
            "jump"
            if mn == "JMP"
            else next(
                (k for k, f in (("rmw", is_rmw), ("write", is_store), ("read", is_read)) if f(op)),
                "ref",
            )
        )
    )
    return kind, target


def _target(mem, a):
    """The static transfer target (branch, ``JMP abs``, ``JSR``) of the instruction at ``a``."""
    kind, t = _operand(mem, a)
    return t if kind == "branch" or mem[a] in (_JSR, _JMP) else None


def _text(b):
    return 0x20 <= b < 0x7F


def _rows(mem, lo, hi, count=None, starts=None, breaks=()):
    """``(address, length, is_code)`` rows from ``lo`` to ``hi``: instructions at every row
    (or only at ``starts``) that end by ``hi``, the rest data rows, which end at code,
    ``breaks`` and changes between printable and other bytes."""
    a, k = lo, 0
    while a < hi and (count is None or k < count):
        n = _LEN[mem[a]]
        if (starts is None or a in starts) and a + n <= hi:
            yield a, n, True
        else:
            n, text = 1, _text(mem[a])
            while (
                a + n < hi
                and n < (_TEXT_ROW if text else _DATA_ROW)
                and _text(mem[a + n]) == text
                and a + n not in breaks
                and (starts is None or a + n not in starts)
            ):
                n += 1
            yield a, n, False
        a, k = a + n, k + 1


def _ascii(data):
    return '  ; "%s"' % data.decode("latin-1") if all(map(_text, data)) else ""


def _strip(line, keep):
    return line if keep else line[:6] + line[16:]


def _dis_lines(mem, rows, dis, names):
    for a, n, code in rows:
        label = names.get(a, "")
        hexb = " ".join("%02X" % mem[a + i] for i in range(n)) if code else ""
        text = dis.instruction_at(a)[1] if code else ".byte " + _hexlist(mem[a : a + n])
        tail = "" if code else _ascii(mem[a : a + n])
        yield ("%04X  %-8s  %-7s%s" % (a, hexb, label + ":" if label else "", text)).rstrip() + tail


def _hexlist(data):
    return ",".join("$%02X" % b for b in data)


def _source(img, spans, starts, names):
    """ACME source reproducing the bytes of ``spans`` (``[(lo, hi)]``)."""
    mem, ok = img.mem, canonical("6510", _TABLE)
    names = {a: n for a, n in names.items() if _IDENT.match(n)}
    parser = _Names(names)
    dis = Disassembler(MPU(memory=mem), parser)
    body, placed = [], set()
    for lo, hi in spans:
        body.append("\n* = $%04X" % lo)
        for a, n, code in _rows(mem, lo, hi, starts=starts, breaks=names):
            if a in names:
                body.append(names[a])
                placed.add(a)
            if code and mem[a] in ok:
                mn, md = _TABLE[mem[a]]
                text = mn if md == "acc" else dis.instruction_at(a)[1]
                word = mem[a + 1] | mem[a + 2] << 8 if n == 3 else None
                if word is not None and word < 0x100 and word in names and md != "ind":
                    text = text.replace(" ", "+2 ", 1)
                body.append("        " + text)
            else:
                body.append("        !byte " + _hexlist(mem[a : a + n]) + _ascii(mem[a : a + n]))
    defs = ["%s = $%0*X" % (names[a], 2 + 2 * (a > 0xFF), a) for a in sorted(parser.used - placed)]
    return "\n".join(["!cpu 6510"] + defs + body) + "\n"


def _follow(img, args, ranges, labels, names):
    """``(listing, code starts, external sources, names)`` of a recursive descent."""
    try:
        from jennings.coverage.listing import Listing  # pylint: disable=import-outside-toplevel
    except ImportError as e:
        raise CliError("--follow needs numpy: pip install jennings[coverage] (%s)" % e)
    entries = [_num(e, labels) for e in args.entry] or [
        img.labels[k] for k in ("INIT", "PLAY") if k in img.labels
    ]
    entries = entries or [lo for lo, _, _ in ranges]
    bounded = [(lo, hi - 1) for lo, hi, n in ranges if n is None and hi > lo]
    inside = [(lo, hi - 1) for lo, hi in img.merged()]
    names = dict(names)
    while True:
        listing = Listing.disassemble(
            img.mem, entries, ranges=bounded, resolve=args.resolve, names=names, inside=inside
        )
        starts = set(listing.lines[:, 1].tolist())
        external = {}
        for a in starts:
            t = _target(img.mem, a)
            if t is not None and not img.inside(t):
                external[a] = t
        new = {t: "X%04X" % t for t in external.values() if t not in names}
        if not new:
            return listing, starts, external, names
        names.update(new)


def _labelled(img, ranges, names, starts=None):
    names = dict(names)
    span = [(lo, hi) for lo, hi, _ in ranges]
    for lo, hi, n in ranges:
        for a, _, code in _rows(img.mem, lo, hi, n, starts):
            t = _target(img.mem, a) if code else None
            if t is not None and t not in names and any(x <= t < y for x, y in span):
                names[t] = "L%04X" % t
    return names


def cmd_dis(img, args, labels, names):
    ranges = _ranges(args, img, labels, args.count)
    if args.follow:
        listing, starts, external, names = _follow(img, args, ranges, labels, names)
    else:
        starts, external = None, {}
    if args.auto_labels or args.follow and args.source:
        names = _labelled(img, ranges, names, starts)
    if args.source:
        spans = [(lo, hi) for lo, hi, n in ranges if n is None] or img.merged()
        print(_source(img, spans, starts, names), end="")
        return
    if args.follow:
        data = []
        for line in list(filter(None, listing.source.splitlines())) + ["FFFF+"]:
            a = int(line[:4], 16)
            if ".byte" in line:
                if data and data[-1][1] == a:
                    data[-1][1] = a + line.count("$")
                else:
                    data.append([a, a + line.count("$")])
                continue
            for lo, hi in data:
                for row in _dis_lines(
                    img.mem, _rows(img.mem, lo, hi, None, (), names), None, names
                ):
                    print(_strip(row, not args.no_bytes))
            data = []
            if line != "FFFF+":
                print(_strip(line + ("  ; external" if a in external else ""), not args.no_bytes))
        return
    dis = Disassembler(MPU(memory=img.mem), _Names(names))
    for i, (lo, hi, n) in enumerate(ranges):
        if i:
            print()
        for line in _dis_lines(img.mem, _rows(img.mem, lo, hi, n, breaks=names), dis, names):
            print(_strip(line, not args.no_bytes))


def cmd_hex(img, args, labels, names):
    chars = _CHARSETS[args.charset]
    w = args.width
    for i, (lo, hi, n) in enumerate(_ranges(args, img, labels, args.count)):
        if i:
            print()
        hi = min(hi, lo + n) if n is not None else hi
        for a in range(lo, hi, w):
            row = img.mem[a : min(a + w, hi)]
            text = "".join(chars[b] for b in row)
            print("%04X  %-*s  %s" % (a, 3 * w - 1, " ".join("%02X" % b for b in row), text))


def cmd_find(img, args, labels, names):
    if args.text is not None:
        pattern = re.escape(args.text.encode("latin-1"))
    else:
        if not args.pattern:
            raise CliError("give a PATTERN or --text")
        digits = re.sub(r"\s+", "", " ".join(args.pattern))
        if len(digits) % 2 or not re.fullmatch(r"(?:[0-9a-fA-F]{2}|\?\?)*", digits):
            raise CliError("bad pattern: %r" % digits)
        pattern = b"".join(
            b"." if digits[i] == "?" else re.escape(bytes([int(digits[i : i + 2], 16)]))
            for i in range(0, len(digits), 2)
        )
    rx = re.compile(b"(?=" + pattern + b")", re.DOTALL)
    dis = Disassembler(MPU(memory=img.mem), _Names(names))
    for lo, hi in img.merged():
        for m in rx.finditer(bytes(img.mem[lo:hi])):
            a = lo + m.start()
            print("%04X" % a + ("  " + dis.instruction_at(a)[1] if args.dis else ""))


def cmd_xref(img, args, labels, names):
    want = [(a, a + 1) for a in (_num(t, labels) for t in args.addresses)]
    want += [_span(r, labels, 1) for r in args.range]
    if not want:
        raise CliError("give ADDR or --range")
    spans = [(lo, hi, None) for lo, hi in img.merged()]
    starts = None
    if args.follow or args.entry:
        _, starts, _, names = _follow(img, args, spans, labels, names)
    dis = Disassembler(MPU(memory=img.mem), _Names(names))
    for lo, hi, _ in spans:
        for a, _, code in _rows(img.mem, lo, hi, starts=starts):
            kind, t = _operand(img.mem, a) if code else (None, None)
            if kind and any(x <= t < y for x, y in want):
                print("%04X  %-6s  %s" % (a, kind, dis.instruction_at(a)[1]))


def cmd_asm(args):
    try:
        org = parse_address(args.org)
        code, found = Assembler(MPU()).assemble_program(args.source, org)
    except (SyntaxError, KeyError, ValueError, OverflowError) as e:
        raise CliError("assembly failed: %s" % e)
    for i in range(0, len(code), 16):
        print("%04X  %s" % (org + i, " ".join("%02X" % b for b in code[i : i + 16])))
    for name, a in found.items():
        print("%s = $%04X" % (name, a))
    if args.output:
        head = org.to_bytes(2, "little") if args.output.lower().endswith(".prg") else b""
        with open(args.output, "wb") as f:
            f.write(head + code)


def _call(mpu, pc, regs, budget, dis):
    """Call ``pc`` with ``regs`` (A, X, Y) to its RTS: ``(cycles, returned)``."""
    mpu.x, mpu.y = regs[1], regs[2]
    c0, sp0 = mpu.processorCycles, mpu.sp
    if mpu.native and dis is None:
        mpu.run_to_rts(pc, regs[0], budget)
    else:
        for b in (0x00, 0x01):
            mpu.memory[0x100 | mpu.sp] = b
            mpu.sp -= 1
        mpu.a, mpu.pc, end = regs[0], pc, c0 + budget + 1
        while mpu.sp < sp0 and mpu.processorCycles < end:
            at = mpu.pc
            text = dis.instruction_at(at)[1] if dis else ""
            mpu.step()
            if dis:
                print("  %04X  %-16s %s" % (at, text, _regs(mpu)))
    return mpu.processorCycles - c0, mpu.sp >= sp0


def _regs(mpu):
    return "A=%02X X=%02X Y=%02X SP=%02X P=%02X" % (mpu.a, mpu.x, mpu.y, mpu.sp, mpu.p)


def cmd_run(img, args, labels, names):
    from tqdm import tqdm  # pylint: disable=import-outside-toplevel

    calls = []
    if args.init is not None:
        song = (img.sid["startSong"] - 1) & 0xFF if img.sid else 0
        a = _num(args.init, labels) & 0xFF if args.init else song
        calls.append(("INIT", _num("INIT", labels), (a, 0, 0)))
    for spec in args.call:
        addr, *regs = spec.split(":")
        regs = [_num(r, labels) & 0xFF for r in regs] + [0] * (3 - len(regs))
        calls.append((addr, _num(addr, labels), tuple(regs[:3])))
    if args.play:
        _num("PLAY", labels)
    if not calls and not args.play:
        raise CliError("nothing to run: give --call, --init or --play")
    native = args.engine == "native"
    mem = NativeMemory(bytes(img.mem)) if native else bytearray(img.mem)
    mpu = MPU(memory=mem)
    dis = Disassembler(mpu, _Names(names)) if args.trace else None
    for name, pc, regs in calls:
        cycles, ok = _call(mpu, pc, regs, args.max_cycles, dis)
        print(
            "%s $%04X  %d cycles%s  %s" % (name, pc, cycles, "" if ok else " (budget)", _regs(mpu))
        )
    if args.play:
        pc, spent, short = labels["PLAY"], [], 0
        for _ in tqdm(range(args.play), desc="PLAY", file=sys.stderr, disable=None):
            cycles, ok = _call(mpu, pc, (0, 0, 0), args.max_cycles, dis)
            spent.append(cycles)
            short += not ok
        print(
            "PLAY $%04X x%d  %d cycles (min %d max %d)%s  %s"
            % (
                pc,
                args.play,
                sum(spent),
                min(spent),
                max(spent),
                " (%d over budget)" % short if short else "",
                _regs(mpu),
            )
        )
    final = bytes(mem[0:0x10000])
    changed = [a for a in range(0x10000) if final[a] != img.mem[a]]
    runs = []
    for a in changed:
        if runs and runs[-1][1] == a:
            runs[-1][1] = a + 1
        else:
            runs.append([a, a + 1])
    for lo, hi in runs:
        data = final[lo : min(hi, lo + 16)]
        print(
            "changed %04X:%04X  %s%s"
            % (lo, hi, data.hex(" ").upper(), " ..." if hi - lo > 16 else "")
        )
    for r in args.dump:
        lo, hi = _span(r, labels, 256)
        for a in range(lo, hi, 16):
            print("%04X  %s" % (a, final[a : min(a + 16, hi)].hex(" ").upper()))


def cmd_info(img, args, labels, names):
    print("format  %s" % img.format)
    for lo, hi in img.spans:
        print("span    %04X:%04X  (%d bytes)" % (lo, hi, hi - lo))
    if img.sid:
        for k, v in img.sid.items():
            if k != "data":
                print("%-8s%s" % (k, "$%04X" % v if k in ("load", "init", "play") else v))
    if any(lo <= 0xFFFA and hi >= 0x10000 for lo, hi in img.spans):
        for name, at in (("NMI", 0xFFFA), ("RESET", 0xFFFC), ("IRQ", 0xFFFE)):
            print("%-8s$%04X" % (name, img.mem[at] | img.mem[at + 1] << 8))


_ENGINE = "native" if NATIVE else "python"


def _arg(*flags, **kw):
    return flags, kw


_ENTRY = _arg("-e", "--entry", action="append", default=[], metavar="ADDR")
_FOLLOW = _arg("-f", "--follow", action="store_true", help="recursive descent from entries")
_RANGES = _arg("ranges", nargs="*", metavar="RANGE", help="LO:HI, LO-HI, LO+N or LO")
_COMMANDS = {
    "dis": (
        cmd_dis,
        "disassemble ranges (or the loaded span)",
        [
            _RANGES,
            _arg("-n", "--count", type=int, default=32, help="instructions for a bare LO"),
            _arg("--no-bytes", action="store_true"),
            _arg("--auto-labels", action="store_true", help="name branch/JMP/JSR targets"),
            _FOLLOW,
            _ENTRY,
            _arg("--resolve", action="store_true", help="follow constant jump tables"),
            _arg("--source", action="store_true", help="emit ACME source"),
        ],
    ),
    "hex": (
        cmd_hex,
        "hex dump",
        [
            _RANGES,
            _arg("-n", "--count", type=int, default=256, help="bytes for a bare LO"),
            _arg("-w", "--width", type=int, default=16),
            _arg("--petscii", dest="charset", action="store_const", const="petscii"),
            _arg("--screen", dest="charset", action="store_const", const="screen"),
        ],
    ),
    "find": (
        cmd_find,
        "search for a byte pattern (?? wildcards) or text",
        [
            _arg("pattern", nargs="*", metavar="PATTERN", help='e.g. "8d ?? d4"'),
            _arg("--text"),
            _arg("--dis", action="store_true", help="disassemble at each match"),
        ],
    ),
    "xref": (
        cmd_xref,
        "instructions referencing addresses",
        [
            _arg("addresses", nargs="*", metavar="ADDR"),
            _arg("--range", action="append", default=[], metavar="LO:HI"),
            _FOLLOW,
            _ENTRY,
        ],
    ),
    "run": (
        cmd_run,
        "call routines to their RTS and report registers and changed memory",
        [
            _arg("--call", action="append", default=[], metavar="ADDR[:A[:X[:Y]]]"),
            _arg("--init", nargs="?", const="", metavar="SUBTUNE", help="call INIT"),
            _arg("--play", type=int, default=0, metavar="N", help="call PLAY N times"),
            _arg("--max-cycles", type=int, default=1_000_000, help="per call"),
            _arg("--engine", choices=("native", "python"), default=_ENGINE),
            _arg("--trace", action="store_true", help="print every instruction"),
            _arg("--dump", action="append", default=[], metavar="RANGE"),
        ],
    ),
    "info": (cmd_info, "image format, spans, SID header and vectors", []),
}


def _parser():
    image = argparse.ArgumentParser(add_help=False)
    image.add_argument("image", metavar="IMAGE", help="PATH[@ADDR]: .prg, .sid or raw")
    image.add_argument("--format", choices=FORMATS, default="auto")
    image.add_argument("--base", metavar="ADDR", help="load address of a raw IMAGE")
    image.add_argument("--overlay", action="append", default=[], metavar="PATH[@ADDR]")
    image.add_argument("--labels", action="append", default=[], metavar="FILE")
    image.add_argument("-D", dest="define", action="append", default=[], metavar="NAME=ADDR")
    image.add_argument("--symbols", action="append", default=[], choices=sorted(SETS))
    ap = argparse.ArgumentParser(prog="jennings", description=__doc__.split("\n", maxsplit=1)[0])
    sub = ap.add_subparsers(dest="command", required=True)
    for name, (fn, text, spec) in _COMMANDS.items():
        p = sub.add_parser(name, parents=[image], help=text, description=text)
        for flags, kw in spec:
            p.add_argument(*flags, **kw)
        p.set_defaults(fn=fn, charset="ascii", resolve=False)
    p = sub.add_parser("asm", help="assemble statements (full sources: jennings-acme)")
    p.add_argument("source", help='e.g. "LDA #$01; STA $D020"')
    p.add_argument("--org", default="0")
    p.add_argument("-o", "--output", help="OUT.prg (with load address) or raw")
    p.set_defaults(fn=None)
    return ap


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        if args.fn is None:
            cmd_asm(args)
        else:
            img, labels, names = _load(args)
            args.fn(img, args, labels, names)
    except CliError as e:
        print("jennings: error: %s" % e, file=sys.stderr)
        return 1
    return 0
