"""Test helpers shared by several test modules."""

from deity_informant import PcodeVM

from jennings.acme import assemble
from jennings.assembler import Assembler
from jennings.devices.mpu6502 import MPU


def flags(p):
    """The C, Z, I, D, V, N bits of status ``p``."""
    return (p & 1, (p >> 1) & 1, (p >> 2) & 1, (p >> 3) & 1, (p >> 6) & 1, (p >> 7) & 1)


def mk_p(rng):
    """A random status byte (B clear, U set, D clear)."""
    b = rng.integers(0, 2, 6)
    return int(b[0] | (b[1] << 1) | (b[2] << 2) | 0x20 | (b[4] << 6) | (b[5] << 7))


def regs(mpu):
    """``(a, x, y, sp)`` followed by the :func:`flags` of ``mpu``."""
    return (mpu.a, mpu.x, mpu.y, mpu.sp) + flags(mpu.p)


def load_regs(mpu, mem, a, x, y, sp, p, pc):
    """Put ``mem`` and the registers into ``mpu``."""
    mpu.memory[:] = mem
    mpu.a, mpu.x, mpu.y, mpu.sp, mpu.p, mpu.pc = a, x, y, sp, p, pc


def _addrs(addr, size):
    return [(addr + i) & 0xFFFF for i in range(size)]


class Observed(PcodeVM):
    """A VM logging the addresses it reads and writes in ``log`` as ``(reads, writes)``."""

    __slots__ = ("log",)

    def __init__(self, *args):
        super().__init__(*args)
        self.log = ([], [])

    def _rd(self, addr, sz):
        self.log[0].extend(_addrs(addr, sz))
        return PcodeVM._rd(self, addr, sz)

    def _wr(self, addr, val, sz):
        self.log[1].extend(_addrs(addr, sz))
        PcodeVM._wr(self, addr, val, sz)


def step_observed(mem, state, pc):
    """Step the instruction at ``pc`` of ``mem`` from ``state`` (a, x, y, p, sp) on an
    :class:`Observed` VM: ``(mpu, vm)``."""
    mpu = MPU(memory=bytearray(mem), pc=pc)
    vm = Observed(b"")
    for attr in ("mem", "reg", "uniq", "cycles", "volatile", "vicirq", "ciaicr", "wlog"):
        setattr(vm, attr, getattr(mpu._vm, attr))
    vm.volatile, vm.dummies = False, False
    mpu._vm = vm
    mpu.a, mpu.x, mpu.y, mpu.p, mpu.sp = state
    mpu.step()
    return mpu, vm


def load_program(mpu, statements, org):
    """Assemble ``statements`` into ``mpu``'s memory at ``org``: the address after them."""
    asm, pc = Assembler(mpu), org
    for st in statements:
        code = asm.assemble(st, pc)
        assert code is not None, st
        mpu.memory[pc : pc + len(code)] = code
        pc += len(code)
    return pc


def build_mpu(statements, org, mpu_class, **mem):
    """A ``mpu_class(pc=org)`` with ``statements`` assembled at ``org`` and ``m<addr>=value``
    bytes poked: ``(mpu, address after the code)``."""
    mpu = mpu_class(pc=org)
    end = load_program(mpu, statements, org)
    for addr, val in mem.items():
        mpu.memory[int(addr[1:], 16)] = val
    return mpu, end


def run_to(mpu, end, limit=1000):
    """Step ``mpu`` until its PC is ``end``."""
    for _ in range(limit):
        if mpu.pc == end:
            return mpu
        mpu.step()
    raise AssertionError("did not reach %04X" % end)


def check_regeneration_is_a_noop(gen):
    """The generator module ``gen``'s checked-in output is what it generates now."""
    assert gen.OUT.read_text() == gen.generate()
    assert gen.main(["--check"]) == 0


def check_flags_a_stale_file(gen, monkeypatch, name):
    """``gen.main(["--check"])`` fails on a stale ``name`` next to the real output."""
    monkeypatch.setattr(gen, "OUT", gen.OUT.with_name(name))
    gen.OUT.write_text("stale")
    try:
        assert gen.main(["--check"]) == 1
    finally:
        gen.OUT.unlink()


def image(src, ram=None):
    """64 KiB RAM with ``src`` (ACME, ``* = $1000`` prepended) assembled at ``$1000``."""
    out, syms = assemble("* = $1000\n" + src, cpu="6510")
    ram = bytearray(0x10000) if ram is None else bytearray(ram)
    ram[0x1000 : 0x1000 + len(out)] = out
    return ram, syms
