"""Decoding, roles, accesses and transfer: checked against the Disassembler, the lifted
data flow, ``isa.addresses`` and concrete execution."""

import itertools
import random

import numpy as np
import pytest
from deity_informant import PcodeVM

import jennings
from jennings import dataflow, decoding, isa
from jennings.coverage import vsa
from jennings.decoding import (
    DTYPE,
    INDEX,
    KIND,
    KINDS,
    MEM_READ,
    MEM_WRITE,
    MODE,
    MODES,
    READS,
    STORED,
    UNKNOWN,
    WRITES,
    accesses,
    as_mask,
    as_values,
    decode,
    decode_many,
    names,
    roles,
    transfer,
)
from jennings.devices.mpu6502 import MPU
from jennings.disassembler import Disassembler
from jennings.opcodes import MEM_MODES, OPCODES

PC = 0x0400
OPERANDS = [(0x00, 0x00), (0x7F, 0x12), (0x80, 0xFF), (0xFF, 0x7F), (0x34, 0xC0)]


def _image(op, b1, b2, pc=PC):
    mem = bytearray(0x10000)
    for k, v in enumerate((op, b1, b2)):
        mem[(pc + k) & 0xFFFF] = v
    return mem


@pytest.mark.parametrize("op", range(256))
def test_decode_matches_disassembler_and_decode_many(op):
    for pc in (PC, 0xFFFE, 0xFFFF):
        for b1, b2 in OPERANDS:
            mem = _image(op, b1, b2, pc)
            ins = decode(mem, pc)
            n, text = Disassembler(MPU(memory=bytearray(mem))).instruction_at(pc)
            assert (ins.length, ins.text) == (n, text)
            assert (ins.pc, ins.opcode, ins.mnemonic, ins.mode) == (pc, op, *OPCODES[op])
            word = mem[(pc + 1) & 0xFFFF] | mem[(pc + 2) & 0xFFFF] << 8
            want = {1: None, 2: word & 0xFF, 3: word}[n]
            if ins.mode == "rel":
                want = (pc + 2 + isa.s8(b1)) & 0xFFFF
            assert ins.operand == want
            rec = decode_many(np.frombuffer(bytes(mem), np.uint8), [pc])[0]
            assert MODES[rec["mode"]] == ins.mode
            assert (rec["pc"], rec["opcode"], rec["length"]) == (pc, op, n)
            assert rec["operand"] == (-1 if want is None else want)


def test_decode_many_mask_and_containers():
    rng = np.random.default_rng(1)
    mem = rng.integers(0, 256, 0x10000, dtype=np.uint8)
    mask = rng.random(0x10000) < 0.3
    recs = decode_many(mem, mask)
    assert recs.dtype == DTYPE and (recs["pc"] == np.flatnonzero(mask)).all()
    for container in (bytes(mem), bytearray(mem), mem.tolist()):
        assert (decode_many(container, mask) == recs).all()
    for rec in recs[:: max(1, len(recs) // 500)]:
        ins = decode(mem.tolist(), int(rec["pc"]))
        assert rec["operand"] == (-1 if ins.operand is None else ins.operand)
    loads = recs[
        (KIND[recs["opcode"]] == KINDS.index("load"))
        & np.isin(MODE[recs["opcode"]], [MODES.index(m) for m in ("absx", "absy", "indy")])
    ]
    assert all(
        roles(int(o)).kind == "load" and roles(int(o)).mode in ("absx", "absy", "indy")
        for o in loads["opcode"]
    )


class _Observed(PcodeVM):
    __slots__ = ("log",)

    def _rd(self, addr, sz):
        self.log[0].extend((addr + i) & 0xFFFF for i in range(sz))
        return PcodeVM._rd(self, addr, sz)

    def _wr(self, addr, val, sz):
        self.log[1].extend((addr + i) & 0xFFFF for i in range(sz))
        PcodeVM._wr(self, addr, val, sz)


def _observe(mem, state):
    mpu = MPU(memory=bytearray(mem), pc=PC)
    vm = _Observed(b"")
    for attr in ("mem", "reg", "uniq", "cycles", "volatile", "vicirq", "ciaicr", "wlog"):
        setattr(vm, attr, getattr(mpu._vm, attr))
    vm.volatile, vm.dummies, vm.log = False, False, ([], [])
    mpu._vm = vm
    mpu.a, mpu.x, mpu.y, mpu.p, mpu.sp = state
    mpu.step()
    return vm.log


def _loc(n, data):
    """Role location of a flow location: registers by name, data bytes as ``M``."""
    if isinstance(n, str):
        return {"PC": set()}.get(n, {n})
    return {"M"} if n in data else set()


_CTRL = {"BRK", "JSR", "RTS", "RTI", "JMP", "JAM"}


@pytest.mark.parametrize("op", [op for op, (mn, _) in OPCODES.items() if mn not in _CTRL])
def test_roles_match_dataflow(op):
    rng = random.Random(op)
    reads, writes, stored, addressed = set(), set(), set(), set()
    mem_read = mem_write = False
    for _ in range(3):
        mem = bytearray(rng.randbytes(0x10000))
        mem[PC] = op
        state = [rng.randrange(256) for _ in range(3)] + [rng.randrange(256) | 0x30, 0xF0]
        loads, stores = _observe(mem, state)
        where = isa.addresses(mem, PC, state[1], state[2], state[4], state[0])
        data, stack = set(where["data"]), set(where["stack"])
        own = {(PC + 1) & 0xFFFF, (PC + 2) & 0xFFFF}

        got = []
        out = dataflow.flow(
            mem, PC, loads, stores, {1: "OPERAND", 2: "OPERAND"}, got, addressed=False
        )
        for k, deps in out.items():
            if isinstance(k, str):
                writes |= {k} - {"PC"}
            for d in deps:
                reads |= _loc(d, data)
            if k in data:
                stored.update(*(_loc(d, data) for d in deps))
        reads |= {"SP"} if stack & (set(loads) | set(stores)) else set()
        mem_read |= bool(data & set(loads))
        mem_write |= bool(data & set(stores))
        for _, a, deps in got:
            if a not in stack and a not in own:
                addressed.update(*(_loc(d, data) - {"M"} for d in deps))
    r = roles(op)
    assert set(r.reads) == reads
    assert set(r.writes) == writes
    assert set(r.stored) == stored
    assert (r.mem_read, r.mem_write) == (mem_read, mem_write)
    assert set(names(decoding.ADDRESSED[op])) == addressed
    if r.index:
        assert r.index in addressed


def test_role_tables_against_isa_and_vsa():
    for op in range(256):
        r = roles(op)
        assert names(WRITES[op]) == r.writes and names(READS[op]) == r.reads
        assert names(STORED[op]) == r.stored and KINDS[KIND[op]] == r.kind
        assert {vsa.writes(op)} == {
            frozenset(i for i, n in enumerate(decoding.LOCATIONS) if n in r.writes)
        }
        assert isa.is_store(op) == (r.kind == "store")
        assert isa.is_rmw(op) == (r.kind == "rmw")
        assert isa.is_read(op) == (r.mem_read and not r.mem_write and r.kind != "nop")
        assert isa.is_branch(op) == (r.kind == "branch")
        assert bool(INDEX[op]) == (r.mode[-1] in "xy" and r.mode in MEM_MODES)
        assert (MEM_READ[op] or MEM_WRITE[op]) == bool(
            r.mode in MEM_MODES and r.mnemonic not in ("JMP", "JSR")
        )
    by_kind = {k: {roles(op).mnemonic for op in range(256) if roles(op).kind == k} for k in KINDS}
    assert by_kind["transfer"] == set("TAX TAY TXA TYA TSX TXS".split())
    assert by_kind["incdec"] == set("INX INY DEX DEY".split())
    assert by_kind["flag"] == set("CLC SEC CLI SEI CLV CLD SED".split())
    assert by_kind["stack"] == set("PHA PHP PLA PLP".split())
    assert by_kind["load"] == set("LDA LDX LDY LAX".split())
    assert by_kind["compare"] == set("CMP CPX CPY BIT".split())
    assert by_kind["store"] == isa.STORE_MNEMONICS
    assert by_kind["rmw"] == isa.RMW_MNEMONICS
    assert by_kind["other"] == {"BRK", "JAM"} and by_kind["nop"] == {"NOP"}
    assert roles(0x87).stored == ("A", "X") and roles(0x8E).stored == ("X",)


@pytest.mark.parametrize("op", [op for op in range(256) if MEM_READ[op] or MEM_WRITE[op]])
def test_accesses_match_isa_addresses(op):
    rng = random.Random(op)
    for _ in range(40):
        mem = bytearray(rng.randbytes(0x10000))
        a, x, y = (rng.randrange(256) for _ in range(3))
        operand = rng.randrange(0x10000)
        mem[PC], mem[PC + 1], mem[PC + 2] = op, operand & 0xFF, operand >> 8
        want = isa.addresses(mem, PC, x, y, 0xF0, a)["data"]
        got = accesses(op, operand, x={x}, y={y}, mem=mem, a={a})
        assert got.tolist() == want
    xs, ys, as_ = ({rng.randrange(256) for _ in range(5)} for _ in range(3))
    union = set()
    for a, x, y in itertools.product(as_, xs, ys):
        mem[PC + 1], mem[PC + 2] = 0xF7, 0x30
        union |= set(isa.addresses(mem, PC, x, y, 0xF0, a)["data"])
    assert set(accesses(op, 0x30F7, x=xs, y=ys, mem=mem, a=as_).tolist()) == union


def test_accesses_value_set_forms_and_defaults():
    lda_absx, lda_indy, lda_indx, sta_zpx, lda_zp = 0xBD, 0xB1, 0xA1, 0x95, 0xA5
    assert accesses(lda_absx, 0xFFF0, x=np.array([0x20, 0x01])).tolist() == [0x10, 0xFFF1]
    mask = np.zeros(256, bool)
    mask[[1, 2]] = True
    assert accesses(lda_absx, 0x1000, x=mask).tolist() == [0x1001, 0x1002]
    assert accesses(lda_absx, 0x1000, x=0b110).tolist() == [0x1001, 0x1002]
    assert accesses(sta_zpx, 0xF0, x=None).tolist() == list(range(256))
    assert accesses(lda_zp, 0x1234).tolist() == [0x34]
    assert accesses(lda_indy, 0x10, y={0, 2}, words={0xFFFF, 0x2000}).tolist() == [
        1,
        0x2000,
        0x2002,
        0xFFFF,
    ]
    assert accesses(lda_indx, 0x10, words=[0x1234]).tolist() == [0x1234]
    assert len(accesses(lda_indy, 0x10, y={0})) == 0x10000
    assert len(accesses(0xEA, 0)) == 0 and len(accesses(0x4C, 0x1234)) == 0


def test_value_set_helpers():
    assert as_values({3, 1}).tolist() == [1, 3] and as_values(0b1010).tolist() == [1, 3]
    assert as_values(None, 4).tolist() == [0, 1, 2, 3]
    assert as_values([3, 3, 1]).tolist() == [1, 3]
    assert as_mask({1, 3}) == 0b1010 == as_mask(0b1010) == as_mask(np.array([0, 1, 0, 1], bool))
    assert as_mask(None) == vsa.TOP8 and as_mask(None, 2) == vsa.TOP16


def _flagged(r):
    return {k for k, v in r.items() if v is UNKNOWN}


_MODE_OPERAND = {"zp": 0x80, "zpx": 0x80, "zpy": 0x80, "indx": 0x80, "indy": 0x80}


def _concrete(op, regs, m, operand):
    a, x, y, c, sp = regs
    md = OPCODES[op][1]
    mem = bytearray(0x10000)
    if md in ("indx", "indy"):
        mem[:0x100] = b"\x20" * 0x100
    if md == "imm":
        operand = m
    mem[PC], mem[PC + 1], mem[PC + 2] = op, operand & 0xFF, operand >> 8 & 0xFF
    data = isa.addresses(mem, PC, x, y, sp, a)["data"]
    for e in data:
        mem[e] = m
    mpu = MPU(memory=mem, pc=PC)
    mpu.a, mpu.x, mpu.y, mpu.p, mpu.sp = a, x, y, 0x30 | c, sp
    mpu.step()
    out = {"A": mpu.a, "X": mpu.x, "Y": mpu.y, "SP": mpu.sp}
    if data:
        out["M"] = mpu.memory[data[0]]
    return out


#: Destinations transfer gives exactly for every value set.
EXACT = set(
    "LDA LDX LDY LAX TAX TXA TAY TYA TSX TXS INX INY DEX DEY INC DEC ADC SBC AND ORA EOR"
    " ASL LSR ROL ROR STA STX STY SAX ANC ALR LXA ANE SBX LAS SHA SHX SHY".split()
)
SHA_INDY = 0x93
ILLEGAL_M = set("DCP ISC SLO RLA SRE RRA".split())


@pytest.mark.parametrize("op", [op for op in range(256) if transfer(op) is not UNKNOWN])
def test_transfer_matches_concrete_execution(op):
    rng = random.Random(op)
    r = roles(op)
    md = r.mode
    operand = _MODE_OPERAND.get(md, 0x3000)
    for trial in range(4):
        live = set(r.reads) | ({"M"} if {"M", "OPERAND"} & set(r.reads) else set())
        sets = {
            n: (
                set(
                    rng.sample(range(2 if n == "C" else 256), rng.choice((1, 2)) if n == "C" else 3)
                )
                if n in live
                else {rng.randrange(2 if n == "C" else 256)}
            )
            for n in "A X Y C SP M".split()
        }
        if trial == 3:
            sets["M"] = {0x00, 0xFF, 0x80}
        got = transfer(
            op,
            a=sets["A"],
            x=sets["X"],
            y=sets["Y"],
            m=sets["M"],
            c=sets["C"],
            sp=sets["SP"],
            operand=None if md == "imm" else operand,
        )
        want = {k: set() for k in got}
        for regs in itertools.product(*(sorted(sets[n]) for n in "A X Y C SP".split())):
            for m in sets["M"]:
                out = _concrete(op, regs, m, operand)
                for k in want:
                    want[k].add(out[k])
        dests = {n for n in r.writes if n in ("A", "X", "Y", "SP")}
        assert set(got) == dests | ({"M"} if r.mem_write else set())
        for k, v in got.items():
            if v is not UNKNOWN and op == SHA_INDY:
                assert want[k] <= set(vsa.bits(v).tolist())
            elif v is not UNKNOWN:
                assert vsa.bits(v).tolist() == sorted(want[k]), (k, sets)
        if r.mnemonic in EXACT:
            assert not _flagged(got)
        if r.mnemonic in ILLEGAL_M:
            assert got["M"] is not UNKNOWN


def test_transfer_unknowns_and_defaults():
    assert transfer(0x60) is UNKNOWN and transfer(0x48) is UNKNOWN and transfer(0xD0) is UNKNOWN
    assert transfer(0x18) == {} and transfer(0xC9) == {} and transfer(0xEA) == {}
    assert transfer(0xE8) == {"X": vsa.TOP8}
    assert transfer(0xA9, operand=0x42) == {"A": 1 << 0x42}
    assert transfer(0x9E, x={0xFF}, operand=0x3000) == {"M": 1 << 0x31}
    assert transfer(0x9E, x={0xFF})["M"] == vsa.TOP8
    assert transfer(0x7F, a={1}, m={3, 4}, c={0, 1})["A"] is UNKNOWN
    assert repr(UNKNOWN) == "UNKNOWN"
    with pytest.raises(ValueError):
        transfer(0xAA, a=set())


def test_lazy_top_level_exports():
    for name, module in jennings.EXPORTS.items():
        assert getattr(jennings, name) is getattr(__import__(module, fromlist=[name]), name)
    assert set(jennings.EXPORTS) <= set(dir(jennings))
    assert jennings.decode is decode
    with pytest.raises(AttributeError):
        jennings.no_such_name  # pylint: disable=pointless-statement
