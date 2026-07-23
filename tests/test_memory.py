"""ObservableMemory: subject-backed store with read/write observers."""

from jennings.memory import ObservableMemory


def test_read_write_and_slice():
    mem = ObservableMemory()
    mem[0x10] = 0x42
    assert mem[0x10] == 0x42
    mem[0x20:0x23] = [1, 2, 3]
    assert mem[0x20:0x23] == [1, 2, 3]


def test_write_observer_can_override():
    mem = ObservableMemory()
    seen = []
    mem.subscribe_to_write([0xD400], lambda a, v: seen.append((a, v)))
    mem[0xD400] = 0x7F
    assert seen == [(0xD400, 0x7F)] and mem[0xD400] == 0x7F
    mem.subscribe_to_write([0x50], lambda a, v: 0xAA)
    mem[0x50] = 0x01
    assert mem[0x50] == 0xAA


def test_read_observer():
    mem = ObservableMemory()
    mem.subscribe_to_read([0xD012], lambda a: 0x33)
    assert mem[0xD012] == 0x33


def test_write_helper_and_delegation():
    mem = ObservableMemory(subject=bytearray(0x10000))
    mem.write(0x100, bytes([9, 8, 7]))
    assert mem[0x100] == 9 and mem[0x102] == 7
    assert mem.count(7) >= 1  # __getattr__ delegates to the subject's methods


def test_duplicate_subscribe_is_idempotent():
    mem = ObservableMemory()
    calls = []
    cb = lambda a, v: calls.append(a)
    mem.subscribe_to_write([0x10], cb)
    mem.subscribe_to_write([0x10], cb)
    mem[0x10] = 1
    assert calls == [0x10]
