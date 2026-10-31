"""The native core's Python surface: NativeMemory, drivers, observers, threads."""

import random
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from deity_informant import PcodeVM
from jennings.devices.mpu6502 import MPU, NativeMPU
from jennings.memory import NATIVE, NativeMemory

pytestmark = pytest.mark.skipif(not NATIVE, reason="native core not built")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import difftest  # pylint: disable=wrong-import-position

STACK_OPS = {0x08, 0x28, 0x48, 0x68}


def _ref_run_to_rts(mpu, mem, pc, acc, max_cycles, pcs=None):
    """vagrant_moses.cpu.run_to_rts / pysidtracker.emu.run_to_rts_until, verbatim."""
    start_sp = mpu.sp
    for byte in (0x00, 0x01):
        mem[0x0100 + mpu.sp] = byte
        mpu.sp = (mpu.sp - 1) & 0xFF
    mpu.a, mpu.pc = acc & 0xFF, pc
    start = mpu.processorCycles
    while mpu.sp < start_sp and mpu.processorCycles - start <= max_cycles:
        if pcs is not None and mpu.pc in pcs:
            return True
        mpu.step()
    return False


def _pair(data):
    ref_mem, nat_mem = bytearray(data), NativeMemory(data)
    return (MPU(memory=ref_mem), ref_mem), (MPU(memory=nat_mem), nat_mem)


def _state(mpu):
    return (mpu.processorCycles, mpu.pc, mpu.a, mpu.x, mpu.y, mpu.sp, mpu.p, mpu.jammed)


def test_backend_selection():
    assert MPU(memory=NativeMemory()).native and not MPU().native
    assert isinstance(NativeMPU(), NativeMPU)
    with pytest.raises(TypeError):
        NativeMPU(memory=bytearray(0x10000))
    mpu = MPU(memory=NativeMemory())
    mpu.memory = mpu.memory
    with pytest.raises(TypeError):
        mpu.memory = NativeMemory()


@pytest.mark.parametrize("seed", range(12))
def test_run_to_rts_matches_reference(seed):
    rng = random.Random(seed)
    data = bytearray(rng.randbytes(0x10000))
    (ref, rmem), (nat, nmem) = _pair(data)
    for pc in [rng.randrange(0x200, 0xF000) for _ in range(6)]:
        sp = rng.randrange(256)
        ref.sp = nat.sp = sp
        cap = rng.choice((50, 500, 5000))
        if seed % 2:
            pcs = {rng.randrange(0x200, 0xF000) for _ in range(200)} | {pc + 3}
            assert _ref_run_to_rts(ref, rmem, pc, sp, cap, pcs) == nat.run_to_rts_until(
                pc, sp, pcs, cap
            )
        else:
            _ref_run_to_rts(ref, rmem, pc, sp, cap)
            nat.run_to_rts(pc, sp, cap)
        assert _state(ref) == _state(nat)
        assert bytes(rmem) == bytes(nmem)


def test_run_to_rts_returns_on_balancing_rts():
    mem = NativeMemory()
    mem[0x1000:0x1006] = [0x20, 0x00, 0x20, 0xE8, 0xE8, 0x60]  # JSR $2000; INX; INX; RTS
    mem[0x2000] = 0x60
    mpu = MPU(memory=mem)
    mpu.run_to_rts(0x1000, 0x42, 1000)
    assert (mpu.pc, mpu.a, mpu.x, mpu.sp) == (0x0002, 0x42, 2, 0xFF)
    assert mpu.run_to_rts_until(0x1000, 0, {0x2000}, 1000) and mpu.pc == 0x2000


def test_run_until_reasons():
    mem = NativeMemory()
    mem[0x1000:0x1003] = [0x4C, 0x00, 0x10]  # JMP $1000
    mpu = MPU(memory=mem, pc=0x1000)
    assert mpu.run_until(max_cycles=300) == "cycles" and mpu.processorCycles == 303
    assert mpu.run_until(stop_pcs=[0x1000]) == "pc"
    assert mpu.run_until(stop_ranges=[(0x0FF0, 0x1001)]) == "pc"
    assert mpu.run_until(stop_ranges=[(0x1001, 0x2000)], max_cycles=9) == "cycles"
    with pytest.raises(ValueError):
        mpu.run_until(stop_ranges=[(0, 0x10001)])
    mem[0x1000] = 0x02
    assert mpu.run_until() == "jam" and mpu.jammed and mpu.pc == 0x1001
    mem[0x1000:0x1002] = [0x68, 0x02]  # PLA; JAM
    mpu.pc, mpu.sp = 0x1000, 0xF0
    assert mpu.run_until(sp_above=0xF0) == "sp" and mpu.sp == 0xF1
    mpu.pc = 0x1001
    assert mpu.run_until(stop_on_jam=False, max_cycles=10) == "cycles"


def test_stop_from_callback_and_exceptions():
    mem = NativeMemory()
    mem[0x1000:0x1006] = [0xEE, 0x00, 0xC0, 0x4C, 0x00, 0x10]  # INC $C000; JMP $1000
    mpu = MPU(memory=mem, pc=0x1000)
    hits = []

    def on_write(addr, value):
        hits.append(value)
        if len(hits) == 5:
            mpu.stop = True

    mem.subscribe_to_write([0xC000], on_write)
    assert mpu.run_until(max_cycles=10**6) == "stop" and hits == [1, 2, 3, 4, 5]
    assert mpu.pc == 0x1003 and not mpu.stop

    def boom(addr):
        raise KeyError(addr)

    mem.subscribe_to_read([0xC000], boom)
    with pytest.raises(KeyError):
        mpu.run_until(max_cycles=10**6)
    assert mpu.pc == 0x1003
    with pytest.raises(KeyError):
        mpu.pc = 0x1000
        mpu.step()
    with pytest.raises(KeyError):
        _ = mem[0xC000]


def test_callbacks_see_instruction_start_and_cannot_write_registers():
    mem = NativeMemory()
    mem[0x1000:0x1003] = [0xAD, 0x00, 0xC0]  # LDA $C000
    mpu = MPU(memory=mem, pc=0x1000)
    mpu.processorCycles = 100
    seen = []

    def on_read(addr):
        seen.append((mpu.pc, mpu.processorCycles))
        if len(seen) == 1:
            mpu.a = 1
        return 0x5A

    mem.subscribe_to_read([0xC000], on_read)
    with pytest.raises(RuntimeError):
        mpu.step()
    assert seen == [(0x1000, 100)] and mpu.pc == 0x1003 and mpu.a == 0

    def reenter(addr):
        mpu.run_until(max_cycles=1)

    mem.subscribe_to_read([0xC001], reenter)
    mem[0x1003:0x1006] = [0xAD, 0x01, 0xC0]
    with pytest.raises(RuntimeError):
        mpu.step()


def test_busy_core_refuses_other_threads():
    mem = NativeMemory()
    mem[0x1000:0x1006] = [0xAD, 0x00, 0xC0, 0x4C, 0x00, 0x10]
    mpu = MPU(memory=mem, pc=0x1000)
    entered, release, errors = threading.Event(), threading.Event(), []

    def on_read(addr):
        entered.set()
        release.wait(10)
        mpu.stop = True

    mem.subscribe_to_read([0xC000], on_read)
    t = threading.Thread(target=lambda: errors.append(mpu.run_until()))
    t.start()
    assert entered.wait(10)
    for touch in (lambda: mem[0], lambda: mpu.a, lambda: mpu.run_until(max_cycles=1)):
        with pytest.raises(RuntimeError):
            touch()
    release.set()
    t.join(10)
    assert errors == ["stop"] and mem[0x1000] == 0xAD


def test_threads_match_sequential():
    progs = [random.Random(s).randbytes(0x10000) for s in range(8)]

    def run(data):
        mpu = MPU(memory=NativeMemory(data), pc=0x0800)
        mpu.run_until(max_cycles=200_000, stop_on_jam=False)
        return _state(mpu), bytes(mpu.memory)

    want = [run(d) for d in progs]
    got = [None] * 8

    def work(k):
        got[k] = run(progs[k])

    threads = [threading.Thread(target=work, args=(k,)) for k in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert got == want


def _observed_python(data, steps):
    """pysidtracker.emu.observe_data_access's VM hook, minus stack ops (its documented contract)."""
    events = []
    mpu = MPU(memory=bytearray(data))

    class Observed(PcodeVM):
        def _rd(self, addr, sz):
            v = PcodeVM._rd(self, addr, sz)
            events.extend(("r", (addr + k) & 0xFFFF) for k in range(sz))
            return v

        def _wr(self, addr, val, sz):
            PcodeVM._wr(self, addr, val, sz)
            events.extend(("w", (addr + k) & 0xFFFF, (val >> 8 * k) & 0xFF) for k in range(sz))

    vm = Observed(b"")
    vm.mem, vm.volatile = mpu._vm.mem, False
    mpu._vm = vm
    mpu.pc = 0x0800
    out = []
    for _ in range(steps):
        op = mpu.memory[mpu.pc]
        mpu.step()
        if op not in STACK_OPS:
            out += events
        events.clear()
    return out, mpu


@pytest.mark.parametrize("seed", range(4))
def test_observe_data_access_matches_engine_loads_and_stores(seed):
    data = random.Random(seed).randbytes(0x10000)
    want, ref = _observed_python(data, 5000)
    mem = NativeMemory(data)
    got = []
    mem.observe_data_access(lambda a: got.append(("r", a)), lambda a, v: got.append(("w", a, v)))
    mpu = MPU(memory=mem, pc=0x0800)
    for _ in range(5000):
        mpu.step()
    assert got == want and _state(mpu) == _state(ref)


def test_observe_data_access_filters_and_last_load():
    mem = NativeMemory()
    prog = [0xB1, 0x10, 0x48, 0x8D, 0x00, 0xD4, 0x8D, 0x01, 0xD4, 0x20, 0x00, 0x20]
    mem[0x1000 : 0x1000 + len(prog)] = prog  # LDA ($10),Y; PHA; STA $D400; STA $D401; JSR
    mem[0x10], mem[0x11], mem[0x3000] = 0x00, 0x30, 0x77
    mem[0x2000] = 0x60
    got = []
    mem.observe_data_access(
        lambda a: got.append(("r", a)),
        lambda a, v: got.append(("w", a, v)),
        reads=[0x3000],
        writes=range(0xD400, 0xD401),
    )
    mpu = MPU(memory=mem, pc=0x1000)
    for _ in range(5):
        mpu.step()
    assert got == [("r", 0x3000), ("w", 0xD400, 0x77)]
    assert mem.last_load == 0x3000
    mem.observe_data_access()
    mpu.pc = 0x1000
    mpu.step()
    assert got == [("r", 0x3000), ("w", 0xD400, 0x77)]


def test_memory_protocols():
    mem = NativeMemory(bytes(range(256)))
    assert len(mem) == 0x10000 and mem[255] == 255 and mem[-1] == 0 and mem[0x10005] == 5
    assert mem[0:8:2] == [0, 2, 4, 6] and list(mem)[:3] == [0, 1, 2]
    view = np.frombuffer(mem, dtype=np.uint8)
    view[0x4000] = 0x99
    assert mem[0x4000] == 0x99 and memoryview(mem)[0x4000] == 0x99
    mem[0x10:0x13] = b"\x01\x02\x03"
    mem[0x20:0x22] = bytearray([9, 8])
    assert bytes(mem)[0x10:0x13] == b"\x01\x02\x03" and mem.count(9) >= 1
    with pytest.raises(ValueError):
        mem[0] = 256
    with pytest.raises(ValueError):
        NativeMemory(bytes(0x10001))
    seen = []
    mem.subscribe_to_write([0x50], lambda a, v: seen.append(v))
    mem.write(0x50, b"\x07")
    assert mem[0x50] == 7 and not seen


def test_page_table_rom_under_ram():
    mem = NativeMemory()
    rom = bytes([0xEA]) * 0x1000
    mem.set_page(0xE, rom)
    assert mem.page_mapped(0xE) and mem[0xE123] == 0xEA
    mem[0xE123] = 0x11
    assert mem[0xE123] == 0xEA and bytes(mem)[0xE123] == 0x11
    mem.set_page(0xE)
    assert not mem.page_mapped(0xE) and mem[0xE123] == 0x11
    for bad in ((16, rom), (1, b"short")):
        with pytest.raises(ValueError):
            mem.set_page(*bad)


@pytest.mark.parametrize("ntsc,cpl,lines", [(False, 63, 312), (True, 65, 263)])
def test_raster_model_follows_the_cycle_counter(ntsc, cpl, lines):
    mem = NativeMemory()
    mem[0xD011] = 0x1B
    mem.raster_model(ntsc)
    mpu = MPU(memory=mem)
    for cycles in (0, cpl * 5, cpl * 300 + 7, cpl * lines * 3 + cpl * 260):
        mpu.processorCycles = cycles
        line = cycles // cpl % lines
        mem.io = True
        assert mem[0xD012] == line & 0xFF
        assert mem[0xD011] == 0x1B | (line >> 8) << 7
        mem.io = False
        assert mem[0xD012] == 0 and not mem.io
    mem.io = True
    mem.raster_model(None)
    assert mem[0xD012] == 0


def test_raster_wait_loop_exits_natively():
    mem = NativeMemory()
    mem.io = True
    mem.raster_model(False)
    mem[0x1000:0x1005] = [0xAD, 0x12, 0xD0, 0xD0, 0xFB]  # LDA $D012; BNE *-3
    mpu = MPU(memory=mem, pc=0x1000)
    assert mpu.run_until(stop_pcs=[0x1005], max_cycles=10**6) == "pc"
    assert mpu.processorCycles // 63 % 312 == 0 and mpu.processorCycles > 0


def test_differential_harness_smoke():
    for seed in range(6):
        assert difftest.run_program(seed, 20_000, seed % 2 == 0) is None


def test_differential_harness_detects_a_divergence(monkeypatch):
    """Sensitivity: a reference raster one line late must show up as a divergence."""
    raster = difftest._Side._raster

    def late(self, addr):
        self.mpu.processorCycles -= difftest.PAL_CPL
        try:
            return raster(self, addr)
        finally:
            self.mpu.processorCycles += difftest.PAL_CPL

    monkeypatch.setattr(difftest._Side, "_raster", late)
    assert any(difftest.run_program(seed, 100_000, False) for seed in range(8))


def test_difftest_cli(capsys):
    assert difftest.main(["--programs", "2", "--steps", "2000", "--jobs", "1"]) == 0
    assert "0 diverged" in capsys.readouterr().err


@pytest.mark.parametrize("seed", range(6))
def test_run_until_irq_period_matches_stepped_irqs(seed):
    data = bytearray(random.Random(seed).randbytes(0x10000))
    data[0xFFFE], data[0xFFFF] = 0x00, 0x30
    data[0x3000:0x3004] = bytes((0xE6, 0x02, 0x58, 0x40))  # INC $02; CLI; RTI
    (ref, _rmem), (nat, nmem) = _pair(data)
    period, total = 997, 60_000
    ref.pc = nat.pc = 0x3002
    due = period
    while ref.processorCycles < total:
        ref.step()
        if ref.processorCycles >= due:
            ref.irq()
            due += period
    for end in (total // 2, total):
        nat.run_until(
            max_cycles=end - nat.processorCycles - 1, irq_period=period, stop_on_jam=False
        )
    assert _state(ref) == _state(nat) and nat.irq_due == due
    assert bytes(_rmem) == bytes(nmem)
    nat.irq_due = None
    assert nat.irq_due is None
    with pytest.raises(ValueError):
        nat.run_until(max_cycles=10, irq_period=0)


def _trace_program():
    mem = NativeMemory()
    mem[0x1000:0x1008] = [
        0xA2,
        0x03,
        0xCA,
        0xD0,
        0xFD,
        0x20,
        0x00,
        0x20,
    ]  # LDX #3; DEX; BNE; JSR $2000
    mem[0x2000] = 0x60
    mem[0x1008] = 0x60
    return MPU(memory=mem)


def test_run_to_rts_traced_matches_step_loop():
    mpu, ref = _trace_program(), _trace_program()
    tr = mpu.run_to_rts_traced(0x1000, 0, 1000, regs=True)
    ref.sp -= 2
    ref.pc, ref.a = 0x1000, 0
    rows = []
    while ref.sp < 0xFF:
        pc, cyc = ref.pc, ref.processorCycles
        ref.step()
        rows.append((pc, cyc, ref.a, ref.x, ref.y, ref.sp, ref.p))
    assert len(tr) == len(rows) and len(tr) > 8
    got = list(zip(tr.pc, tr.cycles, tr.a, tr.x, tr.y, tr.sp, tr.p))
    assert got == rows
    assert _trace_program().run_to_rts_traced(0x1000, 0, 1000).a is None


def test_run_until_trace():
    mpu = _trace_program()
    mpu.pc = 0x1000
    reason, tr = mpu.run_until(stop_pcs=[0x2000], trace=True)
    assert reason == "pc" and tr.pc[0] == 0x1000 and tr.pc[-1] == 0x1005
    assert tr.cycles[0] == 0 and tr.x is None
    assert mpu.run_until(max_cycles=5) in ("cycles", "pc", "jam", "sp")
    _, tr = mpu.run_until(max_cycles=0, regs=True)
    assert len(tr) == 1 and len(tr.sp) == 1


def test_steps_matches_trace_rows():
    mem = NativeMemory()
    mem[0x1000:0x1008] = [0xEE, 0x00, 0xC0, 0xEE, 0x00, 0xC0, 0x4C, 0x00]  # INC $C000 x2; JMP
    mem[0x1008] = 0x10
    mpu = MPU(memory=mem, pc=0x1000)
    assert mpu.steps == 0
    seen = []
    mem.observe_data_access(on_write=lambda a, v: seen.append((mpu.steps, v)))
    _, tr = mpu.run_until(max_cycles=40, trace=True)
    assert seen and all(tr.pc[i] in (0x1000, 0x1003) for i, _ in seen)
    assert mpu.steps == len(tr)
    mpu.reset()
    assert mpu.steps == 0


def test_observe_stack_and_last_load_none():
    mem = NativeMemory()
    mem[0x1000:0x1006] = [0x48, 0x68, 0x08, 0x28, 0x20, 0x00]  # PHA PLA PHP PLP JSR $20xx
    mem[0x1006] = 0x20
    mem[0x2000] = 0x60
    assert mem.last_load is None
    got = []
    mem.observe_data_access(lambda a: got.append(("r", a)), lambda a, v: got.append(("w", a)))
    mpu = MPU(memory=mem, pc=0x1000)
    for _ in range(6):
        mpu.step()
    assert got == [] and mem.last_load is None
    mem.observe_data_access(
        lambda a: got.append(("r", a)), lambda a, v: got.append(("w", a)), stack=True
    )
    mpu.pc = 0x1000
    for _ in range(6):
        mpu.step()  # JSR/RTS frames are not PHA/PHP/PLA/PLP accesses
    assert got == [("w", 0x1FF), ("r", 0x1FF), ("w", 0x1FF), ("r", 0x1FF)]
    mem[0x1000:0x1003] = [0xAD, 0x00, 0x00]  # LDA $0000
    mpu.pc = 0x1000
    mpu.step()
    assert mem.last_load == 0


def test_shared_memory_aliases_buffer():
    ram = bytearray(0x10000)
    ram[0x1000:0x1003] = [0xE6, 0x10, 0x60]  # INC $10; RTS
    mem = NativeMemory(ram, share=True)
    mpu = MPU(memory=mem)
    mpu.run_to_rts(0x1000, 0, 100)
    assert ram[0x10] == 1 and mem[0x10] == 1
    ram[0x10] = 7  # host writes show through
    assert mem[0x10] == 7
    for bad in (bytearray(10), bytes(0x10000)):
        with pytest.raises(ValueError):
            NativeMemory(bad, share=True)
    with pytest.raises(ValueError):
        NativeMemory(share=True)
    assert bytes(NativeMemory(ram)) == bytes(ram) and NativeMemory(ram) is not mem
