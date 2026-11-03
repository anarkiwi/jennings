"""ADC, SBC, ARR, RRA and ISC on the Python and native MPU, against Bruce Clark's reference.

Every accumulator, operand and carry, with the decimal flag set (invalid BCD included) and clear.
"""

import pytest

import decimal_ref as ref
from jennings.devices.mpu6502 import MPU
from jennings.memory import NATIVE, NativeMemory

CASES = {
    "ADC": (0x65, ref.adc),
    "SBC": (0xE5, ref.sbc),
    "ARR": (0x6B, ref.arr),
    "RRA": (0x67, ref.rra),
    "ISC": (0xE7, ref.isc),
}
KINDS = ["python"] + (["native"] if NATIVE else [])
N, V, Z, C, D = MPU.NEGATIVE, MPU.OVERFLOW, MPU.ZERO, MPU.CARRY, MPU.DECIMAL


def machine(kind):
    mem = NativeMemory() if kind == "native" else bytearray(0x10000)
    return MPU(memory=mem), mem


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("d", (0, 1))
@pytest.mark.parametrize("name", CASES)
@pytest.mark.parametrize("eighth", range(8))
def test_matches_the_reference(kind, d, name, eighth):
    op, fn = CASES[name]
    mpu, mem = machine(kind)
    mem[0x1000], mem[0x1001] = op, 0x10 if op != 0x6B else 0
    for m in range(256):
        if op == 0x6B:
            mem[0x1001] = m
        for a in range(eighth * 32, eighth * 32 + 32):
            for c in (0, 1):
                mem[0x10] = m
                mpu.pc, mpu.a, mpu.p = 0x1000, a, MPU.UNUSED | c | d * D
                mpu.step()
                want = fn(a, m, c, d)
                flags = mpu.p
                got = (mpu.a, flags >> 7 & 1, flags >> 6 & 1, flags >> 1 & 1, flags & 1)
                assert got == tuple(want[:5]), (name, a, m, c, d)
                assert flags & D == d * D and mpu.pc == 0x1000 + 2
                if name in ("RRA", "ISC"):
                    assert mem[0x10] == want[5], (name, a, m, c, d)
