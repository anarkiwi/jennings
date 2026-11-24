"""Generate ``crates/jennings-core/src/cpu_gen.rs`` from deity-informant's lifter.

Each opcode's ``deity_informant.lifter.template`` (P-Code with operand-bound constants,
control class and page-cross penalty) is emitted as straight-line Rust.
"""

from __future__ import annotations

import sys
from pathlib import Path

import gen_common
from deity_informant.lifter import OPS, bus_only, template

OUT = Path(__file__).resolve().parent.parent / "crates/jennings-core/src/cpu_gen.rs"
REG = {0: "a", 1: "x", 2: "y", 3: "sp", 8: "c", 9: "z", 10: "i", 11: "d", 12: "b", 13: "v", 14: "n"}
BINOP = {"INT_ADD": "+", "INT_AND": "&", "INT_OR": "|", "INT_XOR": "^", "INT_LEFT": "<<"}
CMPOP = {"INT_EQUAL": "==", "INT_NOTEQUAL": "!=", "INT_LESS": "<", "INT_LESSEQUAL": "<="}
BINDINGS = {
    ((1,), "id"): "b1",
    ((1, 2), "word"): "w",
    ((1,), "hi1"): "((b1 + 1) & 0xff)",
    ((2,), "hi1"): "((b2 + 1) & 0xff)",
}
CTRL = {
    "jmp": "s.pc = w as u16;",
    "jsr": "(s.sp, s.pc) = s.jsr(bus, pc.wrapping_add(2), w as u16);",
    "jmpind": "s.pc = Cpu::jmp_ind(bus, w as u16);",
    "rts": "(s.sp, s.pc) = s.rts(bus);",
    "rti": "let p: u8;\n(s.sp, s.pc, p) = s.rti(bus);\ns.set_p(p);",
    "brk": "(s.sp, s.pc) = s.brk(bus, pc.wrapping_add(2));\ns.i = 1;",
}


def _mask(sz):
    return (1 << (8 * sz)) - 1


class _Op:
    """One opcode's template and the bindings that turn it into Rust."""

    def __init__(self, op):
        self.op = op
        self.mn, self.mode = OPS[op]
        self.t = template(op)
        self.consts = {k: BINDINGS[v] for k, v in self.t.prov.items()}
        self.stack = self._stack_temps()
        self.dummy = bus_only(self.t.ops)
        self.live = {}

    def _stack_temps(self):
        """Unique offsets of stack-page addresses: ``SP | 0x100`` or a value derived from it."""
        sp, page1 = set(), set()
        for mn, out, ins in self.t.ops:
            if out is None or out[0] != "u":
                continue
            derived = any(v[:2] == ["r", 3] or (v[0] == "u" and v[1] in sp) for v in ins)
            if derived:
                sp.add(out[1])
                if mn == "INT_OR" and any(v[0] == "c" and v[1] == 0x100 for v in ins):
                    page1.add(out[1])
        return page1

    def rd(self, i, j, vn):
        sp, off, _sz = vn
        if sp == "c":
            return self.consts.get((i, j), "0x%xu32" % off)
        if sp == "r":
            return self.live.get(REG[off], "(s.%s as u32)" % REG[off])
        return "u%d" % off

    def body(self):
        """Straight-line P-Code; register results stay in locals (``self.live``) until the
        commit, so ``*s`` is the instruction-start state across every bus access."""
        lines = []
        self.live = {}
        for i, (mn, out, ins) in enumerate(self.t.ops):
            a = [self.rd(i, j, v) for j, v in enumerate(ins)]
            stack = ins[0][0] == "u" and ins[0][1] in self.stack
            if i in self.dummy:
                continue
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
                name = "r%d_%s" % (i, REG[out[1]])
                lines.append("let %s: u32 = ((%s) as u8) as u32;" % (name, expr))
                self.live[REG[out[1]]] = name
            else:
                lines.append("let u%d: u32 = %s;" % (out[1], expr))
        return lines

    def emit(self):
        length, ctrl, pen = self.t.len, self.t.ctrl, self.t.pen
        body = []
        if length > 1:
            body.append("let b1 = bus.fetch(pc.wrapping_add(1)) as u32;")
        if length > 2:
            body.append("let b2 = bus.fetch(pc.wrapping_add(2)) as u32;")
            body.append("let w = b1 | (b2 << 8);")
        body += self.body()
        assert not (self.live and ctrl[0] in CTRL), "%02X writes registers before %s" % (
            self.op,
            ctrl[0],
        )
        body += ["s.%s = %s as u8;" % (r, n) for r, n in self.live.items()]
        cyc = str(self.t.cyc)
        if pen is not None and pen[0] != "branch":
            base = "u%d" % pen[1] if pen[0] == "iy" else "w"
            cyc += " + cross(%s, s.%s)" % (base, "x" if pen[0] == "ax" else "y")
        if ctrl[0] == "next":
            body.append("s.pc = pc.wrapping_add(%d);" % length)
        elif ctrl[0] == "br":
            cyc += " + s.branch(s.%s == %d, pc, b1)" % (REG[ctrl[1]], ctrl[2])
        elif ctrl[0] == "jam":
            body.append("s.jam(pc);")
            cyc = "2"
        else:
            body += CTRL[ctrl[0]].split("\n")
        return (
            [
                "// %s %s" % (self.mn, self.mode),
                "#[inline(always)]",
                "fn op_%02x<B: Bus>(s: &mut Cpu, bus: &mut B, pc: u16) -> u32 {" % self.op,
                "    bus.sync(s);",
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
    return gen_common.main(argv, __doc__, OUT, generate, "gen_cpu.py")


if __name__ == "__main__":
    sys.exit(main())
