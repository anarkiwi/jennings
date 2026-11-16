"""The native core under the cycle collector: callbacks that reach back are freed."""

import gc
import weakref

import pytest

from jennings.devices.mpu6502 import NativeMPU
from jennings.memory import NATIVE, NativeMemory

pytestmark = pytest.mark.skipif(not NATIVE, reason="native core not built")

PORT = 0x01
PROG = bytes([0xA5, PORT, 0x85, PORT, 0xE6, 0x10, 0x60])  # LDA $01; STA $01; INC $10; RTS


class Bus:
    """Holds its memory and subscribes its own bound methods, as a machine's bus does."""

    def __init__(self, how, mpu=False):
        self.mem = NativeMemory()
        self.mem.write(0x1000, PROG)
        self.seen = []
        if how == "subscribe":
            self.mem.subscribe_to_read([PORT], self.read)
            self.mem.subscribe_to_write([PORT], self.write)
        else:
            self.mem.observe_data_access(self.read, self.write)
        self.mpu = NativeMPU(self.mem) if mpu else None

    def read(self, addr):
        self.seen.append(("r", addr))

    def write(self, addr, value):
        self.seen.append(("w", addr, value))


def _freed(make, n=1):
    refs = [weakref.ref(make()) for _ in range(n)]
    gc.collect()
    return sum(r() is not None for r in refs)


@pytest.mark.parametrize("how", ["subscribe", "observe"])
@pytest.mark.parametrize("mpu", [False, True])
def test_bus_cycle_is_collected(how, mpu):
    assert _freed(lambda: Bus(how, mpu)) == 0


def test_closure_over_the_mpu_is_collected():
    def make():
        mpu = NativeMPU(NativeMemory())
        mpu.memory.subscribe_to_read([PORT], lambda _: mpu.a)
        return mpu

    assert _freed(make) == 0


def test_traverse_reports_callbacks_and_memory():
    bus = Bus("subscribe", mpu=True)
    refs = gc.get_referents(bus.mem)
    assert refs.count(bus.read) == 1
    assert refs.count(bus.write) == 1
    core = bus.mpu._core  # pylint: disable=protected-access
    assert bus.mem in gc.get_referents(core)
    bus.mem.observe_data_access(bus.read)
    assert gc.get_referents(bus.mem).count(bus.read) == 2
    assert gc.is_tracked(bus.mem) and gc.is_tracked(core)


@pytest.mark.parametrize("how", ["subscribe", "observe"])
def test_running_owners_are_all_collected(how):
    def make():
        bus = Bus(how, mpu=True)
        bus.mpu.run_to_rts(0x1000, 0, 1000)
        assert [s[0] for s in bus.seen] == ["r", "w"] + (["r", "w"] if how == "observe" else [])
        return bus

    gc.collect()
    before = sum(isinstance(o, Bus) for o in gc.get_objects())
    assert _freed(make, 64) == 0
    assert sum(isinstance(o, Bus) for o in gc.get_objects()) == before


def test_subscribe_tolerates_reentrant_eq():
    mem = NativeMemory()

    class Cb:
        """A callback whose ``==`` subscribes another address."""

        def __call__(self, addr):
            return addr

        def __eq__(self, other):
            mem.subscribe_to_read(range(0x100, 0x180), lambda _: None)
            return self is other

        __hash__ = object.__hash__

    cb = Cb()
    mem.subscribe_to_read([0x20], cb)
    mem.subscribe_to_read([0x20], Cb())
    assert mem[0x20] == 0x20 and mem[0x17F] == 0
    assert sum(isinstance(r, Cb) for r in gc.get_referents(mem)) == 2
