"""Generate ``crates/jennings-core/src/cpu_gen.rs`` from deity-informant's lifter.

Each opcode is lifted under distinct sentinel operands; operand-derived constants
bind via the lifter's provenance map (or a lift-time pointer read), each binding
is checked against every lift, and the P-Code is emitted as straight-line Rust.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from deity_informant.lifter import CYCLETIME, OPS, lift

OUT = Path(__file__).resolve().parent.parent / "crates/jennings-core/src/cpu_gen.rs"
PC = 0x1234
# (b1, b2, mem[b1], mem[b1 + 1]) per lift; every byte differs between sets.
SENTINELS = ((0x5A, 0xC3, 0x71, 0x9E), (0xA7, 0x3C, 0x18, 0x2B))
REG = {0: "a", 1: "x", 2: "y", 3: "sp", 8: "c", 9: "z", 10: "i", 11: "d", 12: "b", 13: "v", 14: "n"}
BINOP = {"INT_ADD": "+", "INT_AND": "&", "INT_OR": "|", "INT_XOR": "^", "INT_LEFT": "<<"}
CMPOP = {"INT_EQUAL": "==", "INT_NOTEQUAL": "!=", "INT_LESS": "<", "INT_LESSEQUAL": "<="}
BINDINGS = {
    ((1,), "id"): ("b1", lambda b1, b2: b1),
    ((1, 2), "word"): ("w", lambda b1, b2: b1 | b2 << 8),
    ((1,), "hi1"): ("((b1 + 1) & 0xff)", lambda b1, b2: (b1 + 1) & 0xFF),
    ((2,), "hi1"): ("((b2 + 1) & 0xff)", lambda b1, b2: (b2 + 1) & 0xFF),
}
CTRL = {
    "jmp": "s.pc = w as u16;",
    "jsr": "s.jsr(bus, pc.wrapping_add(2), w as u16);",
    "jmpind": "s.jmp_ind(bus, w as u16);",
    "rts": "s.rts(bus);",
    "rti": "s.rti(bus);",
    "brk": "s.brk(bus, pc.wrapping_add(2));",
}


def _lift(op, b1, b2, p1, p2):
    mem = bytearray(0x10000)
    mem[PC], mem[PC + 1], mem[PC + 2] = op, b1, b2
    mem[b1], mem[(b1 + 1) & 0xFF] = p1, p2
    return lift(mem, PC), mem


def _mask(sz):
    return (1 << (8 * sz)) - 1


class _Op:
    """One opcode's lifts and the bindings that turn them into Rust."""

    def __init__(self, op):
        self.op = op
        self.mn, self.mode = OPS[op]
        self.lifts = [_lift(op, *s) for s in SENTINELS]
        self.rec = self.lifts[0][0]
        for rec, _m in self.lifts[1:]:
            assert len(rec["ops"]) == len(self.rec["ops"]), "op-list shape depends on operands"
        self.defs = {
            out[1]: i
            for i, (_mn, out, _ins) in enumerate(self.rec["ops"])
            if out is not None and out[0] == "u"
        }
        self.consts = self._bind_consts()
        self.stack = self._stack_temps()

    def _bind_consts(self):
        """``{(op index, arg index): rust expr}`` for every operand-dependent constant."""
        prov = self.rec["prov"]["ops"]
        out = {}
        for i, (_mn, _out, ins) in enumerate(self.rec["ops"]):
            for j, vn in enumerate(ins):
                if vn is None or vn[0] != "c":
                    continue
                vals = [rec["ops"][i][2][j][1] for rec, _m in self.lifts]
                key = prov.get((i, j))
                if key is not None:
                    expr, fn = BINDINGS[(tuple(key[0]), key[1])]
                    got = [fn(b1, b2) & _mask(vn[2]) for b1, b2, _p1, _p2 in SENTINELS]
                elif len(set(vals)) > 1:
                    expr, got = self._lift_time_read(vals)
                else:
                    continue
                assert got == vals, (hex(self.op), i, j, got, vals)
                out[(i, j)] = expr
        return out

    def _lift_time_read(self, vals):
        """Bind a constant computed from memory the lifter read: a pointer byte the P-Code LOADs."""
        prov = self.rec["prov"]["ops"]
        for i, (mn, out, _ins) in enumerate(self.rec["ops"]):
            if mn != "LOAD" or (i, 0) not in prov:
                continue
            afn = BINDINGS[(tuple(prov[(i, 0)][0]), prov[(i, 0)][1])][1]
            for delta in (0, 1):
                got = [
                    (m[afn(b1, b2)] + delta) & 0xFF
                    for (b1, b2, _p1, _p2), (_r, m) in zip(SENTINELS, self.lifts)
                ]
                if got == vals:
                    return ("((u%d + %d) & 0xff)" % (out[1], delta)), got
        raise NotImplementedError("unexplained operand-dependent constant in %02X" % self.op)

    def _stack_temps(self):
        """Unique offsets whose value derives from SP (stack-page addresses)."""
        sp = set()
        for _mn, out, ins in self.rec["ops"]:
            if out is not None and out[0] == "u":
                if any(v[:2] == ["r", 3] or (v[0] == "u" and v[1] in sp) for v in ins):
                    sp.add(out[1])
        return sp

    def rd(self, i, j, vn):
        sp, off, _sz = vn
        if sp == "c":
            return self.consts.get((i, j), "0x%xu32" % off)
        if sp == "r":
            return "(s.%s as u32)" % REG[off]
        return "u%d" % off

    def body(self):
        lines = []
        for i, (mn, out, ins) in enumerate(self.rec["ops"]):
            a = [self.rd(i, j, v) for j, v in enumerate(ins)]
            stack = ins[0][0] == "u" and ins[0][1] in self.stack
            if mn == "STORE":
                assert ins[1][2] == 1
                fn = "stack_write" if stack else "store"
                lines.append("bus.%s(%s as u16, %s as u8);" % (fn, a[0], a[1]))
                continue
            if mn == "LOAD":
                assert out[2] == 1
                expr = "bus.%s(%s as u16) as u32" % ("stack_read" if stack else "load", a[0])
            elif mn in ("COPY", "INT_ZEXT"):
                expr = a[0]
            elif mn == "INT_RIGHT":
                expr = "%s >> %s" % (a[0], a[1])
            elif mn == "INT_SUB":
                expr = "%s.wrapping_sub(%s) & 0x%x" % (a[0], a[1], _mask(out[2]))
            elif mn in BINOP:
                expr = "(%s %s %s) & 0x%x" % (a[0], BINOP[mn], a[1], _mask(out[2]))
            elif mn in CMPOP:
                expr = "(%s %s %s) as u32" % (a[0], CMPOP[mn], a[1])
            elif mn == "INT_CARRY":
                expr = "(%s + %s > 0x%x) as u32" % (a[0], a[1], _mask(ins[0][2]))
            else:
                raise NotImplementedError(mn)
            if out[0] == "r":
                lines.append("s.%s = (%s) as u8;" % (REG[out[1]], expr))
            else:
                lines.append("let u%d: u32 = %s;" % (out[1], expr))
        return lines

    def iy_base(self):
        """The temp holding the ``(zp)`` pointer: the base the ``,Y`` index is added to."""
        ops = self.rec["ops"]
        for mn, _out, ins in ops:
            if mn == "INT_ADD" and all(v[0] == "u" for v in ins):
                zx, base = ops[self.defs[ins[1][1]]], ops[self.defs[ins[0][1]]]
                if zx[0] == "INT_ZEXT" and zx[2][0][:2] == ["r", 2] and base[0] == "INT_OR":
                    return "u%d" % ins[0][1]
        raise AssertionError("no (zp),Y base in %02X" % self.op)

    def emit(self):
        rec = self.rec
        length, ctrl = rec["len"], rec["ctrl"]
        body = []
        if length > 1:
            body.append("let b1 = bus.fetch(pc.wrapping_add(1)) as u32;")
        if length > 2:
            body.append("let b2 = bus.fetch(pc.wrapping_add(2)) as u32;")
            body.append("let w = b1 | (b2 << 8);")
        body += self.body()
        cyc = str(CYCLETIME[self.op])
        pen = rec["pen"]
        if pen is not None and pen[0] != "branch":
            base = self.iy_base() if pen[0] == "iy" else "w"
            cyc += " + cross(%s, s.%s)" % (base, "x" if pen[0] == "ax" else "y")
        for (b1, b2, _p1, _p2), (r, _m) in zip(SENTINELS, self.lifts):
            if ctrl[0] == "br":
                assert r["ctrl"][3:] == (
                    (PC + 2 + (b1 - 256 if b1 & 0x80 else b1)) & 0xFFFF,
                    (PC + 2) & 0xFFFF,
                )
            elif ctrl[0] in ("jmp", "jsr", "jmpind"):
                assert r["ctrl"][1] == b1 | b2 << 8 and length == 3
            elif ctrl[0] in ("next", "jam", "brk"):
                assert len(r["ctrl"]) == 1
        if ctrl[0] == "next":
            body.append("s.pc = pc.wrapping_add(%d);" % length)
        elif ctrl[0] == "br":
            cyc += " + s.branch(s.%s == %d, pc, b1)" % (REG[ctrl[1][1]], ctrl[2])
        elif ctrl[0] == "jam":
            body.append("s.jam(pc);")
            cyc = "2"
        else:
            body.append(CTRL[ctrl[0]])
        return (
            [
                "// %s %s" % (self.mn, self.mode),
                "#[inline(always)]",
                "fn op_%02x<B: Bus>(s: &mut Cpu, bus: &mut B, pc: u16) -> u32 {" % self.op,
            ]
            + ["    " + ln for ln in body + [cyc]]
            + ["}", ""]
        )


HEADER = """\
// Generated by tools/gen_cpu.py from deity_informant.lifter; do not edit.

use crate::bus::Bus;
use crate::cpu::{cross, Cpu};
"""


def generate() -> str:
    ops = [_Op(op) for op in range(256)]
    out = [HEADER]
    for o in ops:
        out += o.emit()
    out += [
        "/// Execute the instruction at `s.pc`; cycles taken.",
        "#[inline(always)]",
        "pub(crate) fn execute<B: Bus>(s: &mut Cpu, bus: &mut B) -> u32 {",
        "    let pc = s.pc;",
        "    match bus.fetch(pc) {",
    ]
    out += ["        0x%02x => op_%02x(s, bus, pc)," % (o.op, o.op) for o in ops]
    out += ["    }", "}"]
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="fail if the checked-in file is stale")
    args = ap.parse_args(argv)
    src = generate()
    if args.check:
        if OUT.read_text() != src:
            print("%s is stale; run tools/gen_cpu.py" % OUT, file=sys.stderr)
            return 1
        return 0
    OUT.write_text(src)
    return 0


if __name__ == "__main__":
    sys.exit(main())
