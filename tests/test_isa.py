"""ISA-level helpers: classification, lengths, ALU, addresses, and the instruction walker."""

import numpy as np
import pytest

from jennings import isa
from jennings.devices.mpu6502 import MPU
from jennings.memory import NATIVE, NativeMemory
from jennings.opcodes import OPCODES

LEGAL_STORES = {"STA", "STX", "STY"}
LEGAL_RMW = {"ASL", "LSR", "ROL", "ROR", "INC", "DEC"}
LEGAL_READS = {"LDA", "LDX", "LDY", "CMP", "CPX", "CPY", "BIT", "ADC", "AND", "EOR", "ORA", "SBC"}
CONTROL = (0x20, 0x4C, 0x6C, 0x00, 0x40, 0x60, 0xEA)  # JSR/JMP/BRK/RTI/RTS/NOP
REG_ONLY = (0x0A, 0x4A, 0x2A, 0x6A, 0x18, 0x38, 0xAA)  # acc shifts + flag/transfer ops


def test_lengths_cover_all_opcodes_and_are_legal():
    assert {isa.length(op) for op in range(256)} <= {1, 2, 3}
    assert isa.length(0xA9) == 2  # LDA #imm
    assert isa.length(0x8D) == 3  # STA abs
    assert isa.length(0x60) == 1  # RTS
    assert isa.length(0x4C) == 3  # JMP abs
    assert isa.length(0xEA) == 1  # NOP


def test_branches_are_exactly_the_rel_opcodes():
    branch = {op for op in range(256) if isa.is_branch(op)}
    assert branch == {op for op, (_mn, mode) in OPCODES.items() if mode == "rel"}
    assert branch == {0x10, 0x30, 0x50, 0x70, 0x90, 0xB0, 0xD0, 0xF0}


def test_data_ops_are_memory_and_control_ops_are_not():
    for op in (0xA5, 0xAD, 0x85, 0x8D, 0xC6, 0x06, 0x24):
        assert isa.is_memory(op), hex(op)
    for op in CONTROL:
        assert not isa.is_read(op)
        assert not isa.is_store(op)
        assert not isa.is_rmw(op)
    for op in REG_ONLY:
        assert not isa.is_memory(op), hex(op)  # acc shifts/flag ops are register-only


def test_families_are_mutually_exclusive():
    for op in range(256):
        assert (isa.is_read(op) + isa.is_store(op) + isa.is_rmw(op)) <= 1, hex(op)


def test_store_family_includes_illegal_stores_and_only_writers():
    store_mn = {isa.mnemonic(op) for op in range(256) if isa.is_store(op)}
    assert LEGAL_STORES <= store_mn
    for writer in ("SAX", "SHA", "SHX", "SHY", "TAS"):
        assert writer in store_mn
    assert not isa.is_store(0xA9)  # LDA #imm
    assert not isa.is_store(0x0A)  # ASL A


def test_rmw_requires_a_memory_mode():
    assert {isa.mnemonic(op) for op in range(256) if isa.is_rmw(op)} >= LEGAL_RMW
    for mem in (0x06, 0xC6, 0x0E, 0xEE):  # ASL/DEC zp, ASL abs, INC abs
        assert isa.is_rmw(mem), hex(mem)
    for acc in (0x0A, 0x4A, 0x2A, 0x6A):  # acc shifts
        assert not isa.is_rmw(acc), hex(acc)


def test_read_omits_register_only_and_control_reads():
    assert {isa.mnemonic(op) for op in range(256) if isa.is_read(op)} >= LEGAL_READS
    for mem in (0xA5, 0xAD, 0xC5, 0x24):  # LDA/CMP zp, BIT zp
        assert isa.is_read(mem), hex(mem)
    for op in (0xE9, 0xA9, 0xCA, 0x00):  # SBC #, LDA #, DEX, BRK
        assert not isa.is_read(op), hex(op)


def test_s8_rounds_a_byte_to_signed():
    assert isa.s8(0x00) == 0
    assert isa.s8(0x7F) == 127
    assert isa.s8(0x80) == -128
    assert isa.s8(0xFF) == -1


def test_adc_and_sbc_carry_semantics():
    assert isa.adc(0xFF, 0x01, 0) == (0x00, 1)
    assert isa.adc(0x10, 0x20, 1) == (0x31, 0)
    assert isa.sbc(0x00, 0x01, 1) == (0xFF, 0)
    assert isa.sbc(0x05, 0x03, 1) == (0x02, 1)
    assert isa.sbc(0x05, 0x08, 1) == (0xFD, 0)  # borrow: 5-8 negative clears C
    assert isa.sbc(0x05, 0x03, 0) == (0x01, 1)  # C=0 subtracts 1, 5-3-1=1 stays set


def test_walk_stops_at_opcode_and_respects_budget():
    code = bytes([0xA9, 0x01, 0x8D, 0x00, 0xD4, 0x60])
    assert list(isa.walk(code, 0, {0x60})) == [0, 2, 5]
    assert list(isa.walk(bytes([0xEA] * 10), 0, set(), budget=3)) == [0, 1, 2]
    assert list(isa.walk(code, 2, set())) == [2, 5]  # mid-program start


def test_walk_stops_at_end_of_memory():
    assert list(isa.walk(bytes([0xEA] * 3), 0, set())) == [0, 1, 2]
    assert list(isa.walk(bytes([0xEA] * 3), 1, set())) == [1, 2]


def test_alu_carry_is_the_flagged_carry():
    for a, b, c, d in [(0x99, 0x01, 0, 1), (0x10, 0x25, 1, 1), (0x7F, 0x01, 0, 0)]:
        for fn in (isa.adc, isa.sbc):
            full = fn(a, b, c, decimal=d, flags=True)
            assert fn(a, b, c, decimal=d) == (full[0], full[4]) and len(full) == 5
    assert isa.adc(0x99, 0x01, 0, decimal=True, flags=True) == (0x00, 1, 0, 0, 1)
    assert isa.sbc(0x00, 0x01, 1, decimal=True, flags=True) == (0x99, 1, 0, 0, 0)


def _mem(pairs):
    m = bytearray(0x10000)
    for a, v in pairs:
        m[a & 0xFFFF] = v
    return m


@pytest.mark.parametrize(
    "pairs,x,y,sp,want",
    [
        # LDA ($FF,X), X=0: pointer $FF/$00 wraps in page zero.
        (
            [(0x1000, 0xA1), (0x1001, 0xFF), (0xFF, 0x34), (0x00, 0x12)],
            0,
            0,
            0xFF,
            {"data": [0x1234], "ptr": [0xFF, 0x00], "stack": []},
        ),
        # LDA ($FF),Y: pointer $FF/$00, Y added past the page.
        (
            [(0x1000, 0xB1), (0x1001, 0xFF), (0xFF, 0xF0), (0x00, 0x12)],
            0,
            0x20,
            0xFF,
            {"data": [0x1310], "ptr": [0xFF, 0x00], "stack": []},
        ),
        # LDA $F0,X wraps at $FF; LDA $FFF0,X at $FFFF.
        ([(0x1000, 0xB5), (0x1001, 0xF0)], 0x20, 0, 0xFF, {"data": [0x10], "ptr": [], "stack": []}),
        (
            [(0x1000, 0xBD), (0x1001, 0xF0), (0x1002, 0xFF)],
            0x20,
            0,
            0xFF,
            {"data": [0x10], "ptr": [], "stack": []},
        ),
        # JMP ($10FF) reads its high byte from $1000.
        (
            [(0x1000, 0x6C), (0x1001, 0xFF), (0x1002, 0x10)],
            0,
            0,
            0xFF,
            {"data": [], "ptr": [0x10FF, 0x1000], "stack": []},
        ),
        # JSR pushes two bytes, wrapping in page 1; BRK three; RTI pulls three.
        ([(0x1000, 0x20)], 0, 0, 0x00, {"data": [], "ptr": [], "stack": [0x100, 0x1FF]}),
        (
            [(0x1000, 0x00)],
            0,
            0,
            0x01,
            {"data": [], "ptr": [0xFFFE, 0xFFFF], "stack": [0x101, 0x100, 0x1FF]},
        ),
        ([(0x1000, 0x40)], 0, 0, 0xFE, {"data": [], "ptr": [], "stack": [0x1FF, 0x100, 0x101]}),
    ],
)
def test_addresses_hazards(pairs, x, y, sp, want):
    assert isa.addresses(_mem(pairs), 0x1000, x, y, sp) == want


def test_pointer_bytes():
    m = _mem([(0x1000, 0xA1), (0x1001, 0xFE), (0x2000, 0xA9)])
    assert isa.pointer_bytes(m, 0x1000) == (0xFE, 0xFF)
    assert isa.pointer_bytes(m, 0x1000, x=1) == (0xFF, 0x00)
    assert isa.pointer_bytes(m, 0x2000) is None


def test_sha_crossing_a_page_needs_a():
    m = _mem([(0x1000, 0x9F), (0x1001, 0xF0), (0x1002, 0x12)])
    with pytest.raises(ValueError):
        isa.addresses(m, 0x1000, 0xFF, 0x20, 0xFF)
    assert isa.addresses(m, 0x1000, 0xFF, 0x20, 0xFF, a=0xFF)["data"] == [0x1310]
    assert isa.addresses(m, 0x1000, 0xFF, 0x20, 0xFF, a=0x02)["data"] == [0x0210]


def _bases(img, pc, addrs):
    """The base of each access ``addrs`` of the instruction at ``pc`` by its mode in
    ``OPCODES``: the operand of an indexed absolute or zero-page mode, the pointer word of
    ``(zp),Y`` after its two pointer loads, else the address."""
    md = OPCODES[int(img[pc])][1]
    b1, b2 = int(img[(pc + 1) & 0xFFFF]), int(img[(pc + 2) & 0xFFFF])
    if md == "indy":
        word = int(img[b1]) | int(img[(b1 + 1) & 0xFF]) << 8
        return addrs[:2] + [word] * (len(addrs) - 2)
    fixed = {"absx": b1 | b2 << 8, "absy": b1 | b2 << 8, "zpx": b1, "zpy": b1}.get(md)
    return [a if fixed is None else fixed for a in addrs]


@pytest.mark.skipif(not NATIVE, reason="native core not built")
def test_addresses_are_the_native_cores_accesses():
    """Every opcode from random states, page and zero-page edges included: the addresses
    are those the native core reads or writes (watch callbacks, which see every access
    but fetches), and its logged data loads and stores are ``data`` plus page-zero
    pointers."""
    rng = np.random.default_rng(1)
    mem = NativeMemory()
    mpu = MPU(memory=mem)
    seen = set()
    mem.subscribe_to_read(range(0x10000), seen.add)
    mem.subscribe_to_write(range(0x10000), lambda a, v: seen.add(a))
    for op in range(256):
        for t in range(16):
            img = rng.integers(0, 256, 0x10000, dtype=np.uint8)
            pc = int(rng.integers(0x200, 0x10000))
            if t < 4:
                pc = (pc & 0xFF00) | 0xFE + (t & 1)
            img[pc] = op
            if t % 3 == 0:
                img[(pc + 1) & 0xFFFF] = 0xFF
            a, x, y, sp = (int(v) for v in rng.integers(0, 256, 4))
            sp = (0, 1, 0xFE, 0xFF)[t % 4] if t % 5 == 0 else sp
            img[0x100 | (sp + 2) & 0xFF], img[0x100 | (sp + 1) & 0xFF] = 0x00, 0x01
            want = isa.addresses(img, pc, x, y, sp, a)
            # run_to_rts pushes its return address above SP, so its SP must not wrap.
            for logged in (False, True)[: 2 if sp < 0xFE else 1]:
                mem.write(0, img.tobytes())
                mpu.jammed = False
                mpu.a, mpu.x, mpu.y, mpu.p = a, x, y, 0x24
                if logged:
                    mpu.sp = (sp + 2) & 0xFF
                    _, log, bases = mpu.run_to_rts_accesses(pc, a, 0, bases=True)
                    got = {k & 0xFFFF for k in log}
                    zp = want["ptr"] if OPCODES[op][1] in ("indx", "indy") else []
                    assert got == set(want["data"] + zp), (hex(op), hex(pc))
                    assert list(bases) == _bases(img, pc, [k & 0xFFFF for k in log]), hex(op)
                else:
                    mpu.pc, mpu.sp = pc, sp
                    seen.clear()
                    mpu.step()
                    assert seen == set(want["data"] + want["ptr"] + want["stack"]), (
                        hex(op),
                        hex(pc),
                    )
