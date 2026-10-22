"""Per-instruction data flow: checked against execution, not restated."""

import random

import pytest
from deity_informant import PcodeVM

from jennings import dataflow
from jennings.devices.mpu6502 import MPU
from jennings.opcodes import MODE_LEN, OPCODES

REGS = ("A", "X", "Y")
FLAGS = {"C": 0x01, "Z": 0x02, "I": 0x04, "D": 0x08, "V": 0x40, "N": 0x80}
PC = 0x0400
SKIP = {"JAM", "BRK", "RTI", "RTS", "JSR", "JMP"}


class _Observed(PcodeVM):
    __slots__ = ("log",)

    def _rd(self, addr, sz):
        self.log[0].extend((addr + i) & 0xFFFF for i in range(sz))
        return PcodeVM._rd(self, addr, sz)

    def _wr(self, addr, val, sz):
        self.log[1].extend((addr + i) & 0xFFFF for i in range(sz))
        PcodeVM._wr(self, addr, val, sz)


def _run(mem, state):
    """Step the instruction at :data:`PC`: ``(outputs, loads, stores)``."""
    mpu = MPU(memory=bytearray(mem), pc=PC)
    vm = _Observed(b"")
    for attr in ("mem", "reg", "uniq", "cycles", "volatile", "vicirq", "ciaicr", "wlog"):
        setattr(vm, attr, getattr(mpu._vm, attr))
    vm.volatile, vm.log = False, ([], [])
    mpu._vm = vm
    mpu.a, mpu.x, mpu.y, mpu.p, mpu.sp = state
    mpu.step()
    out = {"A": mpu.a, "X": mpu.x, "Y": mpu.y, "SP": mpu.sp, "PC": mpu.pc}
    out.update({f: int(bool(mpu.p & bit)) for f, bit in FLAGS.items()})
    out.update({a: mpu.memory[a] for a in vm.log[1]})
    return out, vm.log[0], vm.log[1]


def _case(op, rng):
    mem = bytearray(rng.randrange(256) for _ in range(0x10000))
    mem[PC] = op
    state = [rng.randrange(256) for _ in REGS] + [rng.randrange(256) | 0x30, 0xF0]
    return mem, state


@pytest.mark.parametrize("op", [op for op, (mn, _md) in OPCODES.items() if mn not in SKIP])
def test_every_output_that_moves_lists_the_input_that_moved_it(op):
    rng = random.Random(op)
    for _ in range(6):
        mem, state = _case(op, rng)
        base, loads, stores = _run(mem, state)
        deps = dataflow.flow(mem, PC, loads, stores)
        for i, name in enumerate(REGS):
            moved = list(state)
            moved[i] ^= 0x5A
            _check(deps, base, _run(mem, moved), name, loads, stores)
        for name, bit in FLAGS.items():
            moved = list(state)
            moved[3] ^= bit
            _check(deps, base, _run(mem, moved), name, loads, stores)
        for addr in set(loads):
            poked = bytearray(mem)
            poked[addr] ^= 0x5A
            if addr not in range(PC, PC + MODE_LEN[OPCODES[op][1]]):
                _check(deps, base, _run(poked, state), addr, loads, stores)


def _check(deps, base, moved, source, loads, stores):
    after, moved_loads, moved_stores = moved
    if (moved_loads, moved_stores) != (loads, stores):
        return  # the input chose the cells accessed: an address dependence
    for loc, value in base.items():
        if after.get(loc) != value:
            assert source in deps.get(loc, {loc}), (loc, source, deps)


def test_flow_is_exact_on_a_few_instructions():
    mem = bytearray(0x10000)
    mem[PC : PC + 3] = bytes([0x7D, 0x34, 0x12])  # ADC $1234,X
    flow = dataflow.flow(mem, PC, [0x1236])
    assert flow["A"] == flow["C"] == {"A", "C", "X", 0x1236}
    mem[PC : PC + 2] = bytes([0xC9, 0x02])  # CMP #$02
    assert dataflow.flow(mem, PC) == {r: {"A"} for r in ("C", "Z", "N")}
    mem[PC : PC + 2] = bytes([0xB1, 0x34])  # LDA ($34),Y
    assert dataflow.flow(mem, PC, [0x34, 0x35, 0x2005])["A"] == {"Y", 0x34, 0x35, 0x2005}
    mem[PC] = 0x48  # PHA: the pushed byte is A, the address is SP's
    assert dataflow.flow(mem, PC, (), [0x1FF]) == {0x1FF: {"A"}, "SP": {"SP"}}
    mem[PC : PC + 2] = bytes([0xF0, 0x02])  # BEQ
    assert dataflow.flow(mem, PC) == {"PC": {"Z"}}


def test_register_names_come_from_the_lifter():
    assert dataflow.REGISTERS == {
        0: "A",
        1: "X",
        2: "Y",
        3: "SP",
        8: "C",
        9: "Z",
        10: "I",
        11: "D",
        12: "B",
        13: "V",
        14: "N",
    }
