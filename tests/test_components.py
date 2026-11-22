"""jennings.coverage.components: native dependence learning, closures, cycles and cover calls."""

import dataclasses
import random
from collections import defaultdict

import numpy as np
import pytest

from deity_informant.lifter import template
from jennings.acme import assemble
from jennings.coverage import components as C
from jennings.coverage.cfg import LEN
from jennings.dataflow import REGISTERS, flow
from jennings.devices.mpu6502 import REG, Learner, Remap
from jennings.isa import addresses
from jennings.memory import NATIVE

pytestmark = pytest.mark.skipif(not NATIVE, reason="native core not built")

PLAY = 0x1000
STATE = {REG + i for i, n in REGISTERS.items() if n in ("X", "Y", "C", "Z", "I", "D", "V", "N")}
NAME = {n: REG + i for i, n in REGISTERS.items()}
FLAGS = [NAME[f] for f in "CZIDVN"]


def image(src, ram=None):
    """64 KiB RAM with ``src`` (ACME, ``* = $1000`` prepended) assembled at ``PLAY``."""
    out, syms = assemble("* = $1000\n" + src, cpu="6510")
    ram = bytearray(0x10000) if ram is None else bytearray(ram)
    ram[PLAY : PLAY + len(out)] = out
    return ram, syms


def seen(ram, calls):
    """Distinct data accesses and edges of ``calls`` calls."""
    _end, _states, acc, edges = C.machine(ram).orbit(PLAY, 0, C.MAX_CYCLES, calls)
    return set(acc.tolist()), set(edges.tolist())


COUNTERS = """
    inc c3
    lda c3
    cmp #3
    bne +
    lda #0
    sta c3
+   dec c5
    bpl +
    lda #4
    sta c5
+   rts
c3 !byte 0
c5 !byte 4
"""


def test_two_independent_counters_have_coprime_closures():
    ram, s = image(COUNTERS)
    comps = C.learn(ram, PLAY, 40)
    assert comps.persistent == (s["c3"], s["c5"])
    assert comps.closures == (frozenset({s["c3"]}), frozenset({s["c5"]}))
    cyc = C.cycles(ram, PLAY, comps, 100)
    assert cyc.closures == {0: (0, 3), 1: (0, 5)} and not cyc.conflicts
    assert cyc.orbit == (0, 15)
    assert C.cover_calls(ram, PLAY, comps, 100)[0] == 15
    assert C.location(s["c3"]) == "$%04X" % s["c3"] and C.location(NAME["X"]) == "X"


GATED = """
    lda a
    bne +
    inc b
+   inc a
    lda a
    and #3
    sta a
    lda b
    sta $0400
    rts
a !byte 0
b !byte 0
"""


def test_a_counter_gating_another_is_a_control_dependence():
    ram, s = image(GATED)
    comps = C.learn(ram, PLAY, 20)
    a, b = s["a"], s["b"]
    assert comps.persistent == (a, b)
    assert a in comps.deps[b], "b advances only when a is 0: control, not data"
    assert comps.closures == (frozenset({a, b}),)
    cyc = C.cycles(ram, PLAY, comps, 2000)
    assert cyc.closures == {0: (0, 1024)}
    assert comps.feed == frozenset({a}) and C.cover_calls(ram, PLAY, comps, 2000)[0] == 4


SHARED = """
    jsr sub
    lda a
br  bne +
ib  inc b
+   inc a
    lda a
    and #3
    sta a
    jmp done
sub lda #0
done rts
a !byte 0
b !byte 0
"""


def test_a_final_rts_shared_with_a_subroutine_still_exits_the_call():
    ram, s = image(SHARED)
    comps = C.learn(ram, PLAY, 20)
    assert s["a"] in comps.deps[s["b"]]
    assert comps.closures == (frozenset({s["a"], s["b"]}),)
    cd, _ = C.control(ram, PLAY, 4)
    assert cd[s["ib"]] == [s["br"]]


SMC = """
op  lda #0
    clc
    adc #1
    and #7
    sta op + 1
    ldx op + 1
rd  lda $0400,x
    sta $0500
    rts
"""


def test_a_self_modified_operand_is_state():
    ram, s = image(SMC)
    comps = C.learn(ram, PLAY, 20)
    cell = s["op"] + 1
    assert comps.persistent == (cell,)
    assert comps.closures == (frozenset({cell}),) and comps.feeding == (0,)
    assert comps.sites[(s["rd"], C.LOAD)] == frozenset({cell, NAME["D"]}), "ADC reads D"
    assert C.cycles(ram, PLAY, comps, 30).closures == {0: (0, 8)}
    assert C.cover_calls(ram, PLAY, comps, 30)[0] == 8


SCRATCH = """
    lda n
    clc
    adc #1
    and #3
    sta n
    sta tmp
    lda tmp
    beq +
    sta last
+   rts
n    !byte 0
tmp  !byte 0
last !byte $ff
"""


def test_scratch_written_before_it_is_read_is_not_persistent():
    ram, s = image(SCRATCH)
    comps = C.learn(ram, PLAY, 20)
    assert comps.persistent == (s["n"],)
    m = C.machine(ram)
    full = m.orbit(PLAY, 0, C.MAX_CYCLES, 100)
    cells = C.machine(ram).orbit(PLAY, 0, C.MAX_CYCLES, 100, cells=list(comps.persistent))
    assert (full[0], full[1]) == ("closed", 5), "last's power-on $FF is never seen again"
    assert (cells[0], cells[1]) == ("closed", 4)
    assert C.machine(ram).orbit(PLAY, 0, C.MAX_CYCLES, 100, cells=[NAME["X"]])[:2] == (
        "closed",
        1,
    )
    with pytest.raises(ValueError):
        C.machine(ram).orbit(PLAY, 0, C.MAX_CYCLES, 100, cells=[REG + 16])


TABLE = """
    ldx idx
    lda tab,x
    sta $d400
    lda vib
    sta $d401
    inc vib
    inx
    txa
    and #3
    sta idx
    rts
idx !byte 0
vib !byte 0
tab !byte 1, 2, 3, 4
"""


def test_an_index_feeding_a_table_read_is_a_feeding_closure():
    ram, s = image(TABLE)
    comps = C.learn(ram, PLAY, 20)
    idx, vib = s["idx"], s["vib"]
    assert comps.persistent == (idx, vib)
    assert comps.feed == frozenset({idx})
    assert comps.feeding == (comps.closures.index(frozenset({idx})),)
    cyc = C.cycles(ram, PLAY, comps, 600)
    assert cyc.feed == (0, 4) and cyc.orbit == (0, 256)
    calls, _ = C.cover_calls(ram, PLAY, comps, 600)
    assert calls == 4
    assert seen(ram, calls) == seen(ram, 2000)
    assert seen(ram, calls - 1)[0] < seen(ram, calls)[0]


SWEEP = """
    lda pos
    clc
    adc #1
    cmp #3
    bne +
    lda #0
+   sta pos
    tax
    lda tab,x
    sta $d400
    ldy sweep
    lda $0400,y
    sta $d401
    lda sweep
    clc
    adc #5
    sta sweep
    rts
pos   !byte 0
sweep !byte 0
tab   !byte 1, 2, 3
"""


def test_cover_calls_coverage_equals_a_much_longer_run():
    ram, s = image(SWEEP)
    comps = C.learn(ram, PLAY, 600)
    assert comps.feed == frozenset({s["pos"], s["sweep"]})
    calls, _ = C.cover_calls(ram, PLAY, comps, 2000)
    assert calls == 768
    assert seen(ram, calls) == seen(ram, 20 * calls)
    assert C.cover_calls(ram, PLAY, comps, 500)[0] is None


CARRY = """
    lda $0400
    adc #0
    sta $0400
    lda #$ff
    adc #0
    rts
"""


def test_a_flag_read_before_written_is_state():
    ram, _ = image(CARRY)
    comps = C.learn(ram, PLAY, 20)
    assert NAME["C"] in comps.persistent and 0x0400 in comps.persistent
    assert NAME["C"] in comps.deps[0x0400]
    assert C.trajectory(ram, PLAY, [0x0400, NAME["C"]], 3).tolist() == [
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
    ]
    m = C.machine(ram)
    m.p |= 1
    assert C.trajectory(m, PLAY, [0x0400, NAME["C"]], 3).tolist() == [
        [0, 1],
        [1, 0],
        [1, 0],
        [1, 0],
    ]


GATE = """
    lda m
    cmp #2
    bcc +
    inc n
+   inc m
    lda m
    and #3
    sta m
    rts
n !byte 0
m !byte 0
"""


def test_a_missed_dependence_is_a_conflict_that_widening_closes():
    ram, s = image(GATE)
    n, m = s["n"], s["m"]
    learned = C.learn(ram, PLAY, 8)
    assert learned.closures == (frozenset({n, m}),) and m in learned.deps[n]
    missed = dataclasses.replace(learned, closures=(frozenset({n}), frozenset({m})))
    cyc = C.cycles(ram, PLAY, missed, 100)
    assert set(cyc.conflicts) == {0} and cyc.conflicts[0][2] == frozenset({m})
    assert cyc.closures[1] == (0, 4)
    wide = C.widen(missed, cyc)
    assert wide.closures == learned.closures and wide.feeding == (0,)
    calls, got = C.cover_calls(ram, PLAY, missed, 2000)
    assert got.closures == learned.closures and calls == 4
    blind = dataclasses.replace(missed, feed=frozenset({n}))
    assert "feed" in C.cycles(ram, PLAY, blind, 100).conflicts
    assert C.widen(blind, C.cycles(ram, PLAY, blind, 100)).feed == frozenset({n, m})


LATE = """
    inc m
    lda m
    cmp #200
    bcc +
    ldx k
    lda tab,x
    sta $d400
    inx
    cpx #3
    bne ++
    ldx #0
++  stx k
+   rts
m   !byte 0
k   !byte 0
tab !byte 1, 2, 3
"""


def test_cover_calls_relearns_over_the_calls_it_vouches_for():
    ram, s = image(LATE)
    early = C.learn(ram, PLAY, 50)
    assert early.persistent == (s["m"],) and early.feed == frozenset({s["m"]})
    calls, comps = C.cover_calls(ram, PLAY, early, 4000)
    assert comps.calls >= calls and s["k"] in comps.feed
    assert calls == 768, "k advances 56 of every 256 calls: mod 3 it returns after 768"
    assert seen(ram, calls) == seen(ram, 8 * calls)


def test_remap_redirects_what_a_call_reads_and_suppresses_its_writes():
    ram, s = image("""
ld  lda $0400
    sta $0401
st  inc $0402
    inc $0403
    rts
""")
    table = np.full(0x10000, -1, np.int32)
    table[0x0400] = 0x0403
    zero = np.full(0x10000, -1, np.int32)
    zero[0x0402] = -2
    remap = Remap({s["ld"]: table, s["st"]: zero})
    comps = C.learn(ram, PLAY, 4, remap=remap)
    assert comps.persistent == (0x0403,)
    assert comps.deps[0x0401] == frozenset({0x0403})
    rows = C.trajectory(ram, PLAY, [0x0401, 0x0402, 0x0403], 3, remap=remap)
    assert rows.tolist() == [[0, 0, 0], [0, 0, 1], [1, 0, 2], [2, 0, 3]]


def test_a_call_that_never_returns_or_bad_state_raises():
    ram, _ = image("loop jmp loop\n")
    with pytest.raises(RuntimeError, match="'cycles'"):
        C.learn(ram, PLAY, 1, max_cycles=1000)
    with pytest.raises(RuntimeError, match="'jam'"):
        C.learn(image("!byte $02\n")[0], PLAY, 1)
    with pytest.raises(ValueError):
        C.machine(b"\0" * 10)


def test_a_learner_runs_in_chunks_and_long_runs_show_progress(monkeypatch):
    ram, s = image(COUNTERS)
    monkeypatch.setattr(C, "CHUNK", 7)
    comps = C.learn(ram, PLAY, 40)
    assert comps.persistent == (s["c3"], s["c5"]) and comps.calls == 40
    assert len(C.trajectory(ram, PLAY, comps.persistent, 20)) == 21
    learner = Learner({}, [])
    m = C.machine(ram)
    assert m.learn(learner, PLAY, 0, 1000, 3) == "sp" and learner.calls == 3


def test_a_counter_depends_on_itself_explicitly_but_components_drop_it():
    ram, s = image("""
    inc c
    rts
c   !byte 0
""")
    c = s["c"]
    flows = C.explicit(ram, PLAY, 4)
    assert flows.deps[c] == frozenset({c}) and flows.calls == 4
    assert c not in C.learn(ram, PLAY, 4).deps


INDEXED = """
    ldx pos
ld  lda table,x
st  sta $d400
    inx
    txa
    and #3
    sta pos
    rts
pos   !byte 0
table !byte 1, 2, 3, 4
"""


def test_a_table_read_is_a_constant_by_value_and_its_index_by_address():
    ram, s = image(INDEXED)
    pos, ld, st = s["pos"], s["ld"], s["st"]
    flows = C.explicit(ram, PLAY, 8)
    assert flows.sites[(st, C.VALUE)] == frozenset({C.CONST + ld})
    assert flows.sites[(ld, C.LOAD)] == frozenset({pos})
    assert flows.deps[0xD400] == frozenset({C.CONST + ld})
    assert flows.deps[pos] == frozenset({pos})
    assert C.location(C.CONST + ld) == "#$%04X" % ld
    comps = C.learn(ram, PLAY, 8)
    assert comps.sites[(st, C.VALUE)] == frozenset({pos})
    assert comps.feed == frozenset({pos})


FLAGGED = """
    lda flag
br  beq +
    lda #1
    sta out
+   lda flag
    eor #1
    sta flag
    rts
flag !byte 0
out  !byte 0
"""


def test_a_store_under_a_branch_is_control_not_an_explicit_flow():
    ram, s = image(FLAGGED)
    flag, out, br = s["flag"], s["out"], s["br"]
    assert flag in C.learn(ram, PLAY, 4).deps[out]
    flows = C.explicit(ram, PLAY, 4)
    assert flag not in flows.deps.get(out, ())
    assert flows.sites[(br, C.CONTROL)] == frozenset({flag})


def test_explicit_flows_follow_a_remap_and_zero_reads_are_empty():
    ram, s = image("""
ld  lda $0400
    sta $0401
st  inc $0402
    inc $0403
    rts
""")
    table = np.full(0x10000, -1, np.int32)
    table[0x0400] = 0x0403
    zero = np.full(0x10000, -1, np.int32)
    zero[0x0402] = -2
    flows = C.explicit(ram, PLAY, 4, remap=Remap({s["ld"]: table, s["st"]: zero}))
    assert flows.deps[0x0401] == frozenset({0x0403})
    assert flows.deps[0x0403] == frozenset({0x0403})
    assert (s["st"], C.VALUE) not in flows.sites


def test_closures_are_sink_sccs_with_their_ancestors():
    deps = {1: {2}, 2: {1, 3}, 3: {3}, 4: {3}, 5: set()}
    cls, g = C.closures(deps, [1, 2, 3, 4, 5])
    assert cls == (frozenset({5}), frozenset({3, 4}), frozenset({1, 2, 3}))
    assert set(g.edges) == {(2, 1), (1, 2), (3, 2), (3, 4)}


# A Python reference of the native taint run, from dataflow.flow per executed instruction.

DATA = [0xF0 + k for k in range(16)]
LOADS = ["LDA", "LDX", "LDY", "ADC", "SBC", "AND", "ORA", "EOR", "CMP", "CPX", "CPY", "LAX"]
RMW = ["INC", "DEC", "ASL", "LSR", "ROL", "ROR", "SLO", "RLA", "SRE", "RRA", "DCP", "ISC"]
IMPLIED = ["TAX", "TAY", "TXA", "TYA", "INX", "INY", "DEX", "DEY", "CLC", "SEC", "CLV", "NOP"]
IMM = ["ANC", "ALR", "ARR", "SBX"]
BRANCHES = ["BCC", "BCS", "BEQ", "BNE", "BMI", "BPL", "BVC", "BVS"]
Y_SAFE = {"LDY", "TAY", "INY", "DEY"}


class _Gen:
    """Random terminating programs: forward branches, a bounded loop, JSR, BRK/RTI, JMP (ind),
    PHA/PLA pairs, self-modified operands and opcodes."""

    def __init__(self, rng):
        self.rng, self.n, self.smc = rng, 0, []

    def label(self, p):
        self.n += 1
        return "%s%d" % (p, self.n)

    def operand(self):
        r = self.rng
        z, a = r.choice(DATA), 0x0400 + r.randrange(16)
        return r.choice(
            ["$%02x" % z, "$%02x,x" % z, "$%04x" % a, "$%04x,x" % a, "$%04x,y" % a, "($f0),y"]
        )

    def instr(self, avoid=()):
        r = self.rng
        k = r.randrange(9)
        if k < 3:
            mn = r.choice([m for m in LOADS if m not in avoid])
            if r.random() < 0.3 and mn != "LAX":
                lab = self.label("s")
                self.smc.append(lab)
                return ["%s %s #$%02x" % (lab, mn, r.randrange(256))]
            if mn in ("LDX", "CPX", "CPY", "LDY") or (mn == "LAX" and r.random() < 0.5):
                return ["%s $%02x" % (mn, r.choice(DATA))]
            return ["%s %s" % (mn, self.operand() if mn != "LAX" else "$%02x" % r.choice(DATA))]
        if k == 3:
            mn = r.choice(RMW)
            return [
                (
                    "%s $%02x" % (mn, r.choice(DATA))
                    if r.random() < 0.5
                    else "%s $04%02x" % (mn, r.randrange(16))
                )
            ]
        if k == 4:
            return [
                r.choice(["STA $%02x,x", "STA $04%02x,y", "STX $%02x", "SAX $%02x"])
                % r.choice(DATA)
            ]
        if k == 5:
            return ["%s #$%02x" % (r.choice(IMM), r.randrange(256))]
        if k == 6 and self.smc:
            return ["STA %s + 1" % r.choice(self.smc)]
        if k == 7:
            return [r.choice(["SED", "CLD", "BIT $f0"])]
        return [r.choice([m for m in IMPLIED if m not in avoid])]

    def block(self, depth, size, avoid=(), stack=True):
        r, out = self.rng, []
        for _ in range(size):
            k = r.randrange(12) if depth > 0 else 0
            if k == 1:
                lab = self.label("f")
                out += ["%s %s" % (r.choice(BRANCHES), lab)]
                out += self.block(depth - 1, r.randrange(1, 4), avoid, stack) + [lab]
            elif k == 2 and stack:
                pair = r.choice([("PHA", "PLA"), ("PHP", "PLP")])
                out += [pair[0]] + self.block(depth - 1, 2, avoid, False) + [pair[1]]
            elif k == 3 and "LDY" not in avoid:
                lab = self.label("l")
                out += ["LDY #2", lab] + self.block(0, 2, Y_SAFE) + ["DEY", "BPL %s" % lab]
            elif k == 4 and stack and not avoid:
                out += ["JSR sub"]
            elif k == 5 and stack and not avoid:
                out += ["BRK", "!byte 0"]
            elif k == 6 and not avoid:
                j, e = self.label("j"), self.label("e")
                out += ["LDA $f1", "AND #1", "TAX", "LDA %s_lo,x" % j, "STA $0500"]
                out += ["LDA %s_hi,x" % j, "STA $0501", "JMP ($0500)"]
                out += [
                    "%s_lo !byte <%s_0, <%s_1" % (j, j, j),
                    "%s_hi !byte >%s_0, >%s_1" % (j, j, j),
                ]
                out += ["%s_0 INC $f2" % j, "JMP %s" % e, "%s_1 DEC $f3" % j, e]
            elif k == 7 and not avoid:
                m, ops = self.label("m"), self.label("o")
                out += ["LDA $f4", "AND #1", "TAY", "LDA %s,y" % ops, "STA %s" % m]
                out += ["JMP %s_go" % m, "%s !byte $ee, $ce" % ops, "%s_go" % m]
                out += ["%s INC $040e" % m]
            else:
                out += self.instr(avoid)
        return out

    def program(self):
        play = self.block(3, 24)
        sub = self.block(1, 4, stack=False)
        irq = self.block(1, 3, stack=False)
        return (
            "\n".join(
                ["    " + x if " " in x or x.isupper() else x for x in play + ["RTS"]]
                + ["sub"]
                + ["    " + x for x in sub + ["RTS"]]
                + ["irq"]
                + ["    " + x for x in irq + ["RTI"]]
            )
            + "\n"
        )


class _Code:
    """The instruction bytes at ``pc``, indexed as memory."""

    def __init__(self, code, pc):
        self.code, self.pc = code, pc

    def __getitem__(self, a):
        k = (a - self.pc) & 0xFFFF
        return self.code[k] if k < len(self.code) else 0


def _loc(x):
    return NAME[x] if isinstance(x, str) else x


def reference(ram, calls, control, written, explicit=False):
    """``(persistent, deps, edges, sites)`` of ``calls`` calls of ``PLAY``, by
    :func:`jennings.dataflow.flow` on each executed instruction; ``explicit`` as an explicit
    ``Learner``."""
    m = C.machine(ram)
    mem = m.memory
    loads, stores = [], []
    mem.observe_data_access(loads.append, lambda a, _v: stores.append(a), stack=True)
    deps, first, wrote = defaultdict(set), set(), set()
    sites, edges, store_sites, implicit = defaultdict(set), set(), set(), set()
    for _ in range(calls):
        taint, cond, touched, prev = {}, {}, set(), None

        def get(loc):
            if loc in taint:
                return taint[loc]
            own = loc in written if loc < REG else loc in STATE
            return {loc} if own else set()

        def touch(loc, read):
            if loc not in touched:
                touched.add(loc)
                if read:
                    first.add(loc)

        start = m.sp
        for byte in (0, 1):
            mem[0x100 + m.sp] = byte
            m.sp = (m.sp - 1) & 0xFF
        m.a, m.pc = 0, PLAY
        while m.sp < start:
            pc = m.pc
            code = bytes(mem[pc + k] for k in range(3))
            op, n = code[0], int(LEN[code[0]])
            kind = template(op).ctrl[0]
            other = addresses(mem, pc, m.x, m.y, m.sp)
            if prev is not None:
                edges.add((prev, pc))
            prev = pc
            loads.clear()
            stores.clear()
            m.step()
            assert not m.jammed
            addrs = []
            got = flow(_Code(code, pc), pc, loads, stores, {1: pc + 1, 2: pc + 2}, addrs)
            vals = (
                flow(_Code(code, pc), pc, loads, stores, {1: pc + 1, 2: pc + 2}, addressed=False)
                if explicit
                else got
            )
            data = set() if op in (0x28, 0x68) else set(loads)
            reads = other["stack"] if kind in ("rts", "rti") else other["ptr"]
            reads = reads if kind in ("rts", "rti", "jmpind", "brk") else []
            writes = other["stack"] if kind in ("jsr", "brk") else []
            for k in range(n):
                touch(pc + k, True)
            for a in [*loads, *reads]:
                touch(a, True)
            regs = set()
            for ins in [*got.values(), *(i for _s, _a, i in addrs)]:
                regs |= {NAME[x] for x in ins if isinstance(x, str)}
            regs |= {NAME["SP"]} if kind in ("jsr", "rts", "rti", "brk") else set()
            regs |= set(FLAGS) if kind == "brk" else set()
            for r in regs:
                touch(r, True)

            def t(loc, pc=pc, data=data):
                if explicit and loc in data and loc not in written:
                    return {C.CONST + pc}
                return get(loc)

            def u(ins):
                return set().union(*(t(_loc(x)) for x in ins))

            code_t = get(pc)
            ctl = set().union(*(cond.get(b, set()) for b in control.get(pc, ())))
            base = set() if explicit else code_t | ctl
            ctrl, out = set(code_t), []
            for dst, ins in vals.items():
                if dst == "PC":
                    ctrl |= u(ins)
                else:
                    out.append((_loc(dst), u(ins) | base))
                    if isinstance(dst, int) and op not in (0x08, 0x48):
                        sites[(pc, C.VALUE)] |= out[-1][1]
            if op not in (0x08, 0x28, 0x48, 0x68):
                for store, _a, ins in addrs:
                    if u(ins):
                        sites[(pc, C.STORE if store else C.LOAD)] |= u(ins)
            if kind == "br":
                ctrl |= get(pc + 1)
            if kind in ("jmp", "jsr", "jmpind"):
                ctrl |= get(pc + 1) | get(pc + 2)
            for a in reads[1:] if kind == "rti" else reads:
                ctrl |= get(a)
            if kind in ("jsr", "rts", "rti", "brk"):
                out.append((NAME["SP"], get(NAME["SP"]) | base))
            if kind == "rti":
                out += [(f, get(reads[0]) | base) for f in FLAGS]
            for k, a in enumerate(writes):
                out.append((a, (u(FLAGS) | base) if kind == "brk" and k == 2 else base))
            if kind == "brk":
                out.append((NAME["I"], base))
            if ctrl:
                sites[(pc, C.CONTROL)] |= ctrl
            cond[pc] = ctrl | ctl
            for loc, s in out:
                touch(loc, False)
                wrote.add(loc)
                taint[loc] = s
                if not explicit and control.get(pc) and (loc, pc) not in store_sites:
                    store_sites.add((loc, pc))
                    implicit |= {(loc, b) for b in control[pc]}
        for loc, s in taint.items():
            deps[loc] |= s
        for loc, b in implicit:
            if b in cond:
                deps[loc] |= cond[b]
    persistent = sorted(loc for loc in first & wrote if loc < REG or loc in STATE)
    return (
        persistent,
        {k: v for k, v in deps.items() if v},
        edges,
        {k: v for k, v in sites.items() if v},
    )


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("seed", range(24))
def test_native_taint_matches_the_flow_reference(seed, explicit):
    rng = random.Random(seed)
    src = _Gen(rng).program()
    ram = bytearray(0x10000)
    ram[0:0x100] = rng.randbytes(0x100)
    ram[0x400:0x500] = rng.randbytes(0x100)
    ram, syms = image(src, ram)
    ram[0xFFFE], ram[0xFFFF] = syms["irq"] & 0xFF, syms["irq"] >> 8
    calls = 10
    cd, written = C.control(ram, PLAY, calls)
    learner = Learner(cd, written, explicit)
    assert C.machine(ram).learn(learner, PLAY, 0, C.MAX_CYCLES, calls) == "sp"
    got = learner.learned()
    d = np.frombuffer(got["deps"], np.uint64)
    got_deps = defaultdict(set)
    for loc, src_ in zip((d >> 32).tolist(), (d & 0xFFFFFFFF).tolist()):
        got_deps[loc].add(src_)
    s = np.frombuffer(got["sites"], np.uint64)
    got_sites = defaultdict(set)
    for key, loc in zip((s >> 18).tolist(), (s & 0x3FFFF).tolist()):
        got_sites[(key >> 2, "rwbv"[key & 3])].add(loc)
    e = np.frombuffer(got["edges"], np.uint64)
    written = set(written)
    want = reference(ram, calls, cd, written, explicit)
    assert np.frombuffer(got["persistent"], np.uint32).tolist() == want[0]
    assert dict(got_deps) == want[1]
    assert set(zip((e >> 16).tolist(), (e & 0xFFFF).tolist())) == want[2]
    assert dict(got_sites) == want[3]


def test_first_repeats_are_found_on_growing_prefixes():
    rng = np.random.default_rng(1)
    for mu, lam, n in ((0, 1, 10), (3, 7, 50), (5000, 30000, 80000), (2, 9000, 9001)):
        states = rng.permutation(1 << 20)[: mu + lam].astype(np.uint32)
        seq = np.concatenate([states[:mu], np.resize(states[mu:], n - mu)])
        sub = np.ascontiguousarray(seq.view(np.uint8).reshape(-1, 4))
        want = None if n < mu + lam + 1 else (mu, lam)
        assert C._first_repeat(sub, start=16) == want
