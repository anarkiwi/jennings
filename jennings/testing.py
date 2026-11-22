"""Reference models for tests of 6502 implementations.

NMOS ADC, SBC, ARR, RRA and ISC as Bruce Clark's "Decimal Mode" tutorial states them, each
over ``(a, operand, carry, decimal)`` returning ``(result, N, V, Z, C)`` (RRA and ISC also
the memory byte they write). The binary and decimal forms agree with every SingleStepTests
6502 case of $69, $E9 and $6B.
"""

from jennings import isa

__all__ = ["adc", "arr", "isc", "rra", "sbc"]


def adc(a, b, c, d):
    return isa.adc(a, b, c, decimal=bool(d), flags=True)


def sbc(a, b, c, d):
    return isa.sbc(a, b, c, decimal=bool(d), flags=True)


def arr(a, imm, c, d):
    t = a & imm
    r = (t >> 1) | (c << 7)
    n, z, v = r >> 7, int(r == 0), ((t ^ r) >> 6) & 1
    if not d:
        return r, n, v, z, (r >> 6) & 1
    if (t & 0xF) + (t & 1) > 5:
        r = (r & 0xF0) | ((r + 6) & 0xF)
    carry = int((t & 0xF0) + (t & 0x10) > 0x50)
    if carry:
        r = (r & 0xF) | ((r + 0x60) & 0xF0)
    return r, n, v, z, carry


def rra(a, m, c, d):
    """``(A, N, V, Z, C, memory)`` of ROR memory then ADC."""
    rotated = (m >> 1) | (c << 7)
    return (*adc(a, rotated, m & 1, d), rotated)


def isc(a, m, c, d):
    """``(A, N, V, Z, C, memory)`` of INC memory then SBC."""
    inc = (m + 1) & 0xFF
    return (*sbc(a, inc, c, d), inc)
