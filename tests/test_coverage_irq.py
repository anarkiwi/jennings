import numpy as np
import pytest

from jennings.assembler import Assembler
from jennings.coverage import CoverageMap, CoverageMPU, READ
from jennings.coverage.report import held
from jennings.coverage.smc import analyze, check, obligation_rows, obligations, write_obligations
from jennings.devices.mpu6502 import MPU


class _Refs(dict):
    """Label references for ``str.format_map``: ``{x}`` absolute, ``{x_lo}``/``{x_hi}`` bytes;
    unknown labels assemble with same-width placeholders."""

    def __missing__(self, k):
        return "$08" if k.endswith(("_lo", "_hi")) else "$0800"


def _asm(blocks, **image):
    """Two-pass assembly of ``{org: statements}`` (``label:`` lines define labels);
    ``image`` sets bytes such as ``_FFFE=0x00``."""
    asm, labels = Assembler(MPU()), {}
    for _ in range(2):
        refs = _Refs({k: "$%04X" % v for k, v in labels.items()})
        refs.update({k + "_lo": "$%02X" % (v & 0xFF) for k, v in labels.items()})
        refs.update({k + "_hi": "$%02X" % (v >> 8) for k, v in labels.items()})
        mem = np.zeros(0x10000, np.uint8)
        for org, statements in blocks.items():
            pc = org
            for st in statements:
                if st.endswith(":"):
                    labels[st[:-1]] = pc
                    continue
                code = asm.assemble(st.format_map(refs), pc)
                assert code is not None, st
                mem[pc : pc + len(code)] = code
                pc += len(code)
    for k, v in image.items():
        mem[int(k[1:], 16)] = v
    return mem, labels


def _install(vec, target, tail):
    lo, hi = "$%04X" % vec, "$%04X" % (vec + 1)
    return ["LDA #{%s_lo}" % target, "STA " + lo, "LDA #{%s_hi}" % target, "STA " + hi] + tail


def _run(mem, labels, interrupts):
    mpu = CoverageMPU(memory=mem.tolist(), pc=labels["init"])
    for k in range(interrupts + 1):
        if k:
            mpu.irq()
        while mpu.pc != labels["loop"]:
            mpu.step()
    return mpu.cov


def _missing(smc, cov):
    return [ob for ob, rows in obligation_rows(smc) if not held(rows, cov).any()]


@pytest.fixture(scope="module", name="raster")
def _raster():
    mem, labels = _asm(
        {
            0x0800: ["init:", "SEI"] + _install(0xFFFE, "h1", ["CLI", "loop:", "JMP {loop}"]),
            0x0900: ["h1:"] + _install(0xFFFE, "h2", ["RTI"]),
            0x0A00: ["h2:"] + _install(0xFFFE, "h3", ["RTI"]),
            0x0B00: ["h3:"] + _install(0xFFFE, "h1", ["RTI"]),
        }
    )
    return mem, labels, analyze(mem, [labels["init"]])


def test_raster_chain_graph(raster):
    _, lb, smc = raster
    h = [lb["h1"], lb["h2"], lb["h3"]]
    assert smc.handlers == dict.fromkeys(h, {0xFFFE})
    assert smc.chain == {(lb["init"], 0xFFFE, h[0])} | {
        (h[i], 0xFFFE, h[(i + 1) % 3]) for i in range(3)
    }
    assert not smc.hazards and not smc.unresolved_transfers
    assert smc.regs[h[0]]["i"] == {1} and (h[0] + 10, 0) in smc.edges
    cell = smc.cells[0xFFFE]
    assert cell.classes == {"ctrl"} and not cell.insns and not cell.variants
    assert set(cell.writers) == {lb["init"] + 3} | {x + 2 for x in h}
    kinds = [ob.kind for ob in obligations(smc)]
    assert kinds.count("entry") == 3 and kinds.count("chain") == 4 and kinds.count("def") == 8


def test_raster_obligations_need_every_interrupt(raster):
    mem, lb, smc = raster
    cov = _run(mem, lb, 3)
    assert {(0xFFFE, lb["h3"], lb["h1"])} == {
        (ob.cell, ob.writer, ob.value) for ob in _missing(smc, cov) if ob.kind == "chain"
    }
    cov = _run(mem, lb, 5)
    assert not _missing(smc, cov) and not check(smc, cov)
    assert cov.flags[0xFFFE] & READ and cov.flags[0xFFFF] & READ
    assert (lb["h3"] + 2, 0xFFFE, lb["h1"]) in cov.defuse


def test_stack_pushes_attributed(raster):
    mem, lb, smc = raster
    assert smc.writers[0xFFFE].targets == {0x01FE, 0x01FF, 0x0100}
    mem = mem.copy()
    mem[0x0C00:0x0C04] = [0x20, 0x10, 0x0C, 0x00]
    mem[0x0C10] = 0x60
    mem[0xFFFE:] = [lb["h1"] & 0xFF, lb["h1"] >> 8]
    mpu = CoverageMPU(memory=mem.tolist(), pc=0x0C00)
    lw = mpu._vm.last_writer

    def frame(n):
        return [0x100 | ((mpu.sp + k) & 0xFF) for k in range(1, n + 1)]

    mpu.step()
    assert (lw[frame(2)] == 0x0C00).all()
    mpu.step()
    mpu.step()
    assert mpu.pc == lb["h1"] and (lw[frame(3)] == 0x0C03).all()
    mpu.nmi()
    assert (lw[frame(3)] == 0xFFFA).all()
    mpu.p &= ~mpu.INTERRUPT
    mpu.irq()
    assert (lw[frame(3)] == 0xFFFE).all()
    assert not mpu.cov.flags[frame(3)].any()


def test_raster_listing(raster, tmp_path):
    _, lb, smc = raster
    text = write_obligations(smc, tmp_path / "smc.txt", {0xFFFE: "irqlo"}).read_text()
    assert "$FFFE  irqlo            entry enters $%04X" % lb["h2"] in text
    assert "chain $%04X installs $%04X" % (lb["h3"], lb["h1"]) in text
    assert "ctrl  written by $%04X" % (lb["h1"] + 2) in text


def _torn(pre, mid, post):
    body = ["LDA #{h_lo}", "w1:", "STA $FFFE"] + mid + ["LDA #{h_hi}", "w2:", "STA $FFFF"]
    mem, lb = _asm(
        {
            0x0800: ["init:"] + pre + body + post + ["loop:", "JMP {loop}"],
            0x0900: ["h:", "RTI"],
            0x0A00: ["old:", "RTI"],
        },
        _FFFF=0x0A,
    )
    return analyze(mem, [lb["init"]]), lb


@pytest.mark.parametrize(
    "pre,mid,post,torn",
    [
        (["CLI"], [], [], True),
        ([], [], [], True),
        (["SEI"], ["CLI"], [], True),
        (["SEI"], [], ["CLI"], False),
    ],
)
def test_torn_vector(pre, mid, post, torn):
    smc, lb = _torn(pre, mid, post)
    assert smc.hazards == ([(lb["w1"], 0xFFFE, lb["w2"])] if torn else [])
    half = {(lb["init"], 0xFFFE, lb["old"])} if torn else set()
    assert smc.chain == {(lb["init"], 0xFFFE, lb["h"])} | half


def test_single_byte_update_with_interrupts_enabled():
    mem, lb = _asm(
        {0x0800: ["init:", "CLI", "LDA #$10", "w1:", "STA $FFFE", "loop:", "JMP {loop}"]}
    )
    mem[0xFFFF] = 0x09
    smc = analyze(mem, [lb["init"]])
    assert smc.hazards == [(lb["w1"], 0xFFFE, None)] and smc.handlers[0x0910] == {0xFFFE}


def test_data_dependent_vector_unresolved():
    mem, lb = _asm({0x0800: ["init:", "SEI", "LDA $10,X", "w1:", "STA $FFFE", "RTS"]}, _11=1)
    smc = analyze(mem, [lb["init"]])
    assert not smc.handlers and smc.cells[0xFFFE].data_dependent
    assert smc.unresolved_transfers == {lb["w1"]: {0x0000, 0x0001}}


def test_kernal_stub_jmp_indirect():
    def handler(nxt):
        return _install(0x0314, nxt, ["PLA", "TAY", "PLA", "TAX", "PLA", "RTI"])

    mem, lb = _asm(
        {
            0x0800: ["init:", "CLI", "loop:", "JMP {loop}"],
            0x0900: ["stub:", "PHA", "TXA", "PHA", "TYA", "PHA", "jind:", "JMP ($0314)"],
            0x0A00: ["ha:"] + handler("hb"),
            0x0B00: ["hb:"] + handler("ha"),
        },
        _FFFF=0x09,
        _0315=0x0A,
    )
    cov = _run(mem, lb, 3)
    ha, hb, jind = lb["ha"], lb["hb"], lb["jind"]
    assert {(jind, ha), (jind, hb)} <= cov.edges
    assert {(ha + 2, 0x0314, jind), (ha + 7, 0x0315, jind), (hb + 2, 0x0314, jind)} <= cov.defuse
    assert {(0x0314, -1, ha), (0x0314, ha, hb), (0x0314, hb, ha)} <= cov.chains
    assert (0xFFFE, lb["stub"], lb["stub"]) in cov.chains
    smc = analyze(mem, {lb["init"]: {}, lb["stub"]: {"irq": True}})
    assert smc.handlers == {lb["stub"]: {0xFFFE}, ha: {0x0314}, hb: {0x0314}}
    assert smc.chain == {(ha, 0x0314, hb), (hb, 0x0314, ha)}
    assert not _missing(smc, cov) and not check(smc, cov)


def test_check_flags_unpredicted_vector_writer(raster):
    mem, lb, smc = raster
    cov = _run(mem, lb, 1)
    cov.defuse.add((0x0C00, 0xFFFE, lb["h1"]))
    assert check(smc, cov) == [("writer", 0x0C00, 0xFFFE)]


def test_chains_persist(tmp_path):
    cov = CoverageMap()
    cov.chains |= {(0xFFFE, -1, 0x0900), (0xFFFE, 0x0900, 0x0A00)}
    cov.save(tmp_path / "c.npz")
    back = CoverageMap.load(tmp_path / "c.npz")
    assert back == cov and back.chains == cov.chains
    with np.load(tmp_path / "c.npz") as z:
        np.savez(tmp_path / "old.npz", **{k: z[k] for k in z.files if k != "chains"})
    old = CoverageMap.load(tmp_path / "old.npz")
    assert not old.chains and old != cov
    assert CoverageMap().merge(old).merge(cov).chains == cov.chains
