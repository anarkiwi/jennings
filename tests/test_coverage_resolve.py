from pathlib import Path

import numpy as np
import pytest

from jennings.assembler import Assembler
from jennings.coverage import CoverageMPU
from jennings.coverage.elf import load_elf
from jennings.coverage.listing import Listing
from jennings.coverage.resolve import _View, split_table
from jennings.coverage.vsa import TOP8, bits, from_values
from jennings.devices.mpu6502 import MPU

ORG = 0x0800
SIDPLAYER = Path(__file__).parent / "fixtures" / "sidplayer" / "sidplayer.prg.elf"
HANDLERS = "h0: INX; RTS; h1: INY; RTS; h2: DEX; RTS"


def _prog(text):
    code, labels = Assembler(MPU()).assemble_program(text, ORG)
    mem = np.zeros(0x10000, np.uint8)
    mem[ORG : ORG + len(code)] = np.frombuffer(bytes(code), np.uint8)
    return mem, labels, (ORG, ORG + len(code) - 1)


def _listing(text, resolve=True, cov=None, extra=(), tables=lambda _: ()):
    mem, labels, span = _prog(text)
    lst = Listing.disassemble(
        mem, [ORG, *extra], cov, ranges=[span], resolve=resolve, tables=tables(labels)
    )
    return lst, labels


def _edges(lst):
    return {tuple(e) for e in lst.resolved.tolist()}


def _resolved(lst, labels, names, source):
    return {(labels[source], labels[n]) for n in names} == _edges(lst) and all(
        lst.line(labels[n]) is not None for n in names
    )


def test_jmp_indirect_through_split_table_bounded_by_compare():
    text = (
        "LDA $10; CMP #3; BCS out; TAX; LDA lo,X; STA $20; LDA hi,X; STA $21; jmp: JMP ($20);"
        f"out: RTS; {HANDLERS}; lo: .byte <h0, <h1, <h2; hi: .byte >h0, >h1, >h2"
    )
    lst, labels = _listing(text)
    assert _resolved(lst, labels, ("h0", "h1", "h2"), "jmp")
    assert "L%04X: INX  ; resolved from $%04X" % (labels["h0"], labels["jmp"]) in lst.source
    plain, _ = _listing(text, resolve=False)
    assert plain.line(labels["h0"]) is None and not plain.resolved.size
    assert ".byte" in plain.source.splitlines()[plain.line(labels["out"])]


def test_split_table_words_cells_and_wrap():
    m = np.zeros(0x10000, np.uint8)
    m[[0xFFFF, 0x0001, 0x0003]] = [0x10, 0x11, 0x12]
    m[[0x2000, 0x2002, 0x2004]] = [0xA0, 0xA1, 0xA2]
    words, cells = split_table(m, 0xFFFF, 0x2000, 3, 2)
    assert words.tolist() == [0xA010, 0xA111, 0xA212]
    assert sorted(cells.tolist()) == [1, 3, 0x2000, 0x2002, 0x2004, 0xFFFF]
    words, cells = split_table(m, 0xFFFF, None, 2, 2, high=0x40)
    assert words.tolist() == [0x4010, 0x4011] and sorted(cells.tolist()) == [1, 0xFFFF]


def test_hand_written_tables_decode_what_the_resolver_cannot():
    """A dispatch through an unknown zero-page pointer resolves nothing alone; the table
    given by hand roots its handlers and stays data."""
    text = (
        "LDY $10; LDA ($12),Y; STA $20; INY; LDA ($12),Y; STA $21; JMP ($20);"
        f"{HANDLERS}; tbl: .word h0, h1, h2"
    )
    plain, labels = _listing(text)
    assert plain.line(labels["h1"]) is None
    for table in (
        lambda lab: [(lab["tbl"], lab["tbl"] + 1, 3, 2)],
        lambda lab: [(lab["tbl"], None, 3, 2, lab["h0"] >> 8)],
    ):
        lst, labels = _listing(text, tables=table)
        assert all(lst.line(labels[h]) is not None for h in ("h0", "h1", "h2"))
        assert lst.line(labels["tbl"]) is None and ".byte" in lst.source


def test_rts_dispatch_through_word_table_to_image_end():
    text = (
        "LDA $10; ASL; TAY; LDA tbl+1,Y; PHA; LDA tbl,Y; PHA; rts: RTS;"
        f"{HANDLERS}; tbl: .word h0-1, h1-1, h2-1"
    )
    lst, labels = _listing(text)
    assert _resolved(lst, labels, ("h0", "h1", "h2"), "rts")


def test_unbounded_split_table_ends_at_its_high_column():
    text = (
        "LDX $10; LDA hi,X; PHA; LDA lo,X; PHA; rts: RTS;"
        f"{HANDLERS}; lo: .byte <h0-1, <h1-1, <h2-1; hi: .byte >h0-1, >h1-1, >h2-1"
    )
    lst, labels = _listing(text)
    assert _resolved(lst, labels, ("h0", "h1", "h2"), "rts")


def test_unbounded_table_ends_at_code():
    text = (
        "LDA $10; ASL; TAX; LDA tbl,X; STA jmp+1; LDA tbl+1,X; STA jmp+2; jmp: JMP $0000;"
        f"{HANDLERS}; tbl: .word h0, h1; tail: RTS; RTS; .word h2"
    )
    tail = _prog(text)[1]["tail"]
    lst, labels = _listing(text, extra=[tail, tail + 1])
    assert _resolved(lst, labels, ("h0", "h1"), "jmp")
    lst, labels = _listing(text)
    assert _resolved(lst, labels, ("h0", "h1", "h2"), "jmp") and lst.line(tail) is None


def test_smc_jump_bounded_by_mask_pairs_bytes_of_one_entry():
    text = (
        "LDA $10; AND #1; TAY; LDA lo,Y; STA jmp+1; LDA hi,Y; STA jmp+2; jmp: JMP $0000;"
        "h0: INX; RTS; .byte 0, 0, 0; h1: INY; RTS; lo: .byte <h0, <h1; hi: .byte >h0, >h1;"
        ".byte $FF, $FF"
    )
    lst, labels = _listing(text)
    assert _resolved(lst, labels, ("h0", "h1"), "jmp")


def test_smc_branch_offset():
    text = (
        "LDA $10; AND #1; TAY; LDA offs,Y; STA br+1; CLC; br: BCC base; base: RTS;"
        "h0: INX; RTS; h1: INY; RTS; offs: .byte 1, 3"
    )
    lst, labels = _listing(text)
    assert _resolved(lst, labels, ("h0", "h1"), "br")


def test_table_starting_inside_code_skips_leading_entries():
    text = (
        "LDA $10; AND #3; TAX; LDA tbl,X; STA jmp+1; LDA tbl2,X; STA jmp+2; jmp: JMP $0000;"
        "tbl: RTS; RTS; .byte <h0, <h1; tbl2: RTS; RTS; .byte >h0, >h1; h0: INX; RTS; h1: INY;"
        "RTS"
    )
    mem, labels, span = _prog(text)
    rts = [labels["tbl"], labels["tbl"] + 1, labels["tbl2"], labels["tbl2"] + 1]
    lst = Listing.disassemble(mem, [ORG, *rts], ranges=[span], resolve=True)
    assert _resolved(lst, labels, ("h0", "h1"), "jmp")


@pytest.mark.parametrize("entry", ["h0", "bad-1"])
def test_misaligned_or_garbage_targets_are_not_taken(entry):
    text = (
        "LDA $10; AND #1; TAX; LDA hi,X; PHA; LDA lo,X; PHA; rts: RTS;"
        f"h0: LDA #$EA; RTS; bad: .byte $02, $60; lo: .byte <h0-1, <{entry};"
        f"hi: .byte >h0-1, >{entry}"
    )
    lst, labels = _listing(text)
    assert _edges(lst) == {(labels["rts"], labels["h0"])}
    assert lst.line(labels["h0"] + 1) is None and lst.line(labels["bad"]) is None


def test_variable_or_written_tables_and_plain_returns_resolve_nothing():
    text = (
        "LDA $10; AND #1; TAX; LDA hi,X; PHA; LDA lo,X; PHA; STA lo,Y; RTS;"
        "h0: INX; RTS; h1: INY; RTS; lo: .byte <h0-1, <h1-1; hi: .byte >h0-1, >h1-1"
    )
    lst, labels = _listing(text)
    assert not lst.resolved.size and lst.line(labels["h0"]) is None
    lst, _ = _listing("LDA ($10),Y; STA $20; PHA; JSR sub; RTS; sub: PLA; RTS")
    assert not lst.resolved.size


def test_resolution_follows_into_newly_decoded_code_and_marks_executed_targets_once():
    text = (
        "LDA $10; AND #1; TAX; LDA lo,X; STA $20; LDA hi,X; STA $21; jmp: JMP ($20);"
        "h0: LDA $11; AND #1; TAY; LDA hi2,Y; PHA; LDA lo2,Y; PHA; rts: RTS; h1: INY; RTS;"
        "g0: INX; RTS; g1: DEX; RTS;"
        "lo: .byte <h0, <h1; hi: .byte >h0, >h1; lo2: .byte <g0-1, <g1-1; hi2: .byte >g0-1, >g1-1"
    )
    mem, labels, span = _prog(text)
    mpu = CoverageMPU(memory=mem.tolist(), pc=ORG)
    mpu.memory[0x10] = 1
    while mpu.pc != labels["h1"] + 1:
        mpu.step()
    lst = Listing.disassemble(mem, [ORG], mpu.cov, ranges=[span], resolve=True)
    want = {("jmp", "h0"), ("jmp", "h1"), ("rts", "g0"), ("rts", "g1")}
    assert _edges(lst) == {(labels[a], labels[b]) for a, b in want}
    rows = lst.source.splitlines()
    assert "resolved" not in rows[lst.line(labels["h1"]) - 1]
    assert "resolved from" in rows[lst.line(labels["h0"]) - 1]
    assert [-lst.line(labels["g0"]), lst.line(labels["g0"])] in lst.arcs.tolist()


def test_fully_executed_program_listing_is_unchanged():
    text = (
        "LDA $10; CMP #2; BCS out; ASL; TAY; LDA tbl+1,Y; PHA; LDA tbl,Y; PHA; RTS;"
        "out: RTS; h0: INX; RTS; h1: INY; RTS; tbl: .word h0-1, h1-1"
    )
    mem, labels, span = _prog(text)
    cov = None
    for v in (0, 1, 2):
        mpu = CoverageMPU(memory=mem.tolist(), pc=ORG)
        mpu.memory[0x10] = v
        for _ in range(12):
            mpu.step()
        cov = mpu.cov if cov is None else cov.merge(mpu.cov)
    on = Listing.disassemble(mem, [ORG], cov, ranges=[span], resolve=True)
    off = Listing.disassemble(mem, [ORG], cov, ranges=[span])
    assert on.source == off.source and (on.arcs == off.arcs).all()
    assert len(on.resolved) == 2


def test_indirect_indexed_table_through_pointer_set_on_path():
    text = (
        "LDA #<tbl; STA $FB; LDA #>tbl; STA $FC; LDA $10; AND #1; ASL; TAY;"
        "LDA ($FB),Y; STA $20; INY; LDA ($FB),Y; STA $21; jmp: JMP ($20);"
        "h0: INX; RTS; h1: INY; RTS; tbl: .word h0, h1"
    )
    lst, labels = _listing(text)
    assert _resolved(lst, labels, ("h0", "h1"), "jmp")


@pytest.mark.parametrize("where", ["before", "after"])
def test_store_through_unknown_pointer_forgets_memory(where):
    smash = "STA ($30),Y;"
    text = (
        f"LDA $10; AND #1; TAX; {smash if where == 'before' else ''}LDA lo,X; STA $20;"
        f"LDA hi,X; STA $21; {smash if where == 'after' else ''}JMP ($20);"
        "h0: INX; RTS; h1: INY; RTS; lo: .byte <h0, <h1; hi: .byte >h0, >h1"
    )
    lst, labels = _listing(text)
    assert not lst.resolved.size and lst.line(labels["h0"]) is None


def test_many_paths_fall_back_to_the_straight_line_block():
    diamonds = "".join(f"BEQ d{k}; NOP; d{k}:" for k in range(6))
    text = (
        f"LDA $11; {diamonds} LDA $10; AND #1; TAX; LDA lo,X; STA $20; LDA hi,X; STA $21;"
        "jmp: JMP ($20); h0: INX; RTS; h1: INY; RTS; lo: .byte <h0, <h1; hi: .byte >h0, >h1"
    )
    lst, labels = _listing(text)
    assert _resolved(lst, labels, ("h0", "h1"), "jmp")


def test_two_independent_indexes_are_not_paired():
    text = (
        "LDX $10; LDY $11; LDA lo,X; STA $20; LDA hi,Y; STA $21; JMP ($20);"
        "h0: INX; RTS; h1: INY; RTS; lo: .byte <h0, <h1; hi: .byte >h0, >h1"
    )
    lst, labels = _listing(text)
    assert not lst.resolved.size and lst.line(labels["h0"]) is None


def test_sidplayer_effect_handlers_resolve_without_a_run():
    mem, syms = load_elf(SIDPLAYER)
    used = np.flatnonzero(mem)
    lst = Listing.disassemble(
        mem, [syms["init"], syms["play"]], ranges=[(used[0], used[-1])], resolve=True
    )
    fx = ("fx_off", "fx_vibrato", "fx_slide", "fx_pulse")
    assert {tuple(e) for e in lst.resolved.tolist()} == {
        (syms["dispatch"] + 14, syms[h]) for h in fx
    }
    plain = Listing.disassemble(mem, [syms["init"], syms["play"]], ranges=[(used[0], used[-1])])
    assert all(plain.line(syms[h]) is None for h in fx)


def test_view_load_unions_overlay_and_constants():
    const = np.zeros(0x10000, bool)
    const[0x10:0x12] = True
    m = np.zeros(0x10000, np.uint8)
    m[0x10], m[0x11] = 5, 6
    view = _View(const, m, {0x12: 1 << 7})
    assert bits(view.load(from_values([0x10, 0x11, 0x12], 2))).tolist() == [5, 6, 7]
    assert view.load(from_values([0x10, 0x13], 2)) == TOP8


@pytest.mark.parametrize(
    "touch, taken", [("LDA #5; STA h0+1; LDA h0+1", True), ("LDA h0+1", False)]
)
def test_run_read_bytes_reject_targets_unless_also_written(touch, taken):
    text = (
        f"{touch}; LDA $10; AND #1; TAX; LDA lo,X; STA $20; LDA hi,X; STA $21; JMP ($20);"
        "h0: LDA #$00; RTS; h1: INY; RTS; lo: .byte <h0, <h1; hi: .byte >h0, >h1"
    )
    mem, labels, span = _prog(text)
    mpu = CoverageMPU(memory=mem.tolist(), pc=ORG)
    mpu.memory[0x10] = 1
    while mpu.pc != labels["h1"] + 1:
        mpu.step()
    lst = Listing.disassemble(mem, [ORG], mpu.cov, ranges=[span], resolve=True)
    assert (lst.line(labels["h0"]) is not None) == taken
