"""Coverage recording in the engine VM and MPU, re-exported by :mod:`jennings.coverage`."""

import numpy as np
from deity_informant import PcodeVM

from jennings.devices.mpu6502 import MPU, _JAM
from jennings.isa import is_branch, length

__all__ = [
    "EXEC",
    "OPERAND",
    "TAKEN",
    "FALLTHRU",
    "READ",
    "WRITE",
    "RETURN",
    "CoverageMap",
    "CoverageVM",
    "CoverageMPU",
    "call",
]

EXEC, OPERAND, TAKEN, FALLTHRU, READ, WRITE = 1, 2, 4, 8, 16, 32
RETURN = 0x0000

# Branch opcode bits 7-6 select the flag (N, V, C, Z reg index); bit 5 is the value that branches.
_BRANCH_FLAG = (14, 13, 8, 9)
_DYNAMIC = frozenset((0x00, 0x40, 0x60, 0x6C))  # BRK, RTI, RTS, JMP (ind)
_INFO = tuple(
    (
        length(op),
        (_BRANCH_FLAG[op >> 6], (op >> 5) & 1) if is_branch(op) else None,
        op in _DYNAMIC,
    )
    for op in range(256)
)


_SETS = {"edges": 2, "entries": 2, "variants": 2, "defuse": 3, "chains": 3}


def _rows(s, w):
    return np.array(sorted(s), dtype=np.int64).reshape(-1, w)


class CoverageMap:
    """Per-address ``flags`` (uint8) and ``hits`` (uint32), dynamic ``edges`` and
    interrupt ``entries`` as ``(source, target)`` / ``(vector, target)`` sets, and for
    self-modified code the executed ``variants`` ``(pc, bytes)`` (little-endian instruction
    bytes, run with a written byte), ``defuse`` ``(writer, cell, pc)`` uses of written code and
    pointer bytes, ``image`` (uint8, 1 where an instruction ran with none of its bytes written)
    and ``chains`` ``(pointer, previous target or -1, target)`` of every vector consumption."""

    __slots__ = (
        "flags",
        "hits",
        "image",
        "edges",
        "entries",
        "variants",
        "defuse",
        "chains",
        "_fv",
        "_hv",
        "_iv",
    )

    def __init__(self):
        self.flags = np.zeros(0x10000, np.uint8)
        self.hits = np.zeros(0x10000, np.uint32)
        self.image = np.zeros(0x10000, np.uint8)
        self.edges = set()
        self.entries = set()
        self.variants = set()
        self.defuse = set()
        self.chains = set()
        self._fv = memoryview(self.flags)
        self._hv = memoryview(self.hits)
        self._iv = memoryview(self.image)

    def merge(self, other):
        """Accumulate ``other`` in place (OR flags, add hits, union sets)."""
        np.bitwise_or(self.flags, other.flags, out=self.flags)
        np.add(self.hits, other.hits, out=self.hits)
        np.bitwise_or(self.image, other.image, out=self.image)
        for k in _SETS:
            getattr(self, k).update(getattr(other, k))
        return self

    __ior__ = merge

    def save(self, path):
        """Write a single ``.npz`` (numpy appends the suffix if ``path`` lacks it)."""
        np.savez_compressed(
            path,
            flags=self.flags,
            hits=self.hits,
            image=self.image,
            **{k: _rows(getattr(self, k), w) for k, w in _SETS.items()},
        )

    @classmethod
    def load(cls, path):
        cov = cls()
        with np.load(path) as z:
            cov.flags[:] = z["flags"]
            cov.hits[:] = z["hits"]
            if "image" in z:
                cov.image[:] = z["image"]
            for k in _SETS:
                if k in z:
                    setattr(cov, k, set(map(tuple, z[k].tolist())))
        return cov

    def __eq__(self, other):
        if not isinstance(other, CoverageMap):
            return NotImplemented
        return (
            np.array_equal(self.flags, other.flags)
            and np.array_equal(self.hits, other.hits)
            and np.array_equal(self.image, other.image)
            and all(getattr(self, k) == getattr(other, k) for k in _SETS)
        )

    __hash__ = None


class CoverageVM(PcodeVM):
    """``PcodeVM`` recording into ``cov``; ``cur_pc`` is the executing instruction and
    ``last_writer`` (int32 per address, -1 for the image) the last instruction to store there;
    ``vector_target`` ``{pointer: target}`` the last target consumed through each pointer."""

    __slots__ = ("cov", "cur_pc", "last_writer", "vector_target", "_lw")

    def __init__(self, mem_bytes, cov=None):
        super().__init__(mem_bytes)
        self.cov = CoverageMap() if cov is None else cov
        self.cur_pc = None
        self.last_writer = np.full(0x10000, -1, np.int32)
        self.vector_target = {}
        self._lw = memoryview(self.last_writer)

    def _rd(self, addr, sz):
        f = self.cov._fv
        for i in range(sz):
            f[(addr + i) & 0xFFFF] |= READ
        return PcodeVM._rd(self, addr, sz)

    def _wr(self, addr, val, sz):
        f, lw = self.cov._fv, self._lw
        pc = -1 if self.cur_pc is None else self.cur_pc
        for i in range(sz):
            f[(addr + i) & 0xFFFF] |= WRITE
            lw[(addr + i) & 0xFFFF] = pc
        PcodeVM._wr(self, addr, val, sz)

    def _uses(self, pc, n):
        """Record the instruction at ``pc`` (``n`` bytes) as a variant and its written bytes'
        def-use pairs."""
        mem, lw, cov = self.mem, self._lw, self.cov
        cells = [(pc + i) & 0xFFFF for i in range(n)]
        cov.variants.add((pc, sum(mem[c] << (8 * i) for i, c in enumerate(cells))))
        cov.defuse.update((lw[c], c, pc) for c in cells if lw[c] >= 0)

    def consume(self, ptr, pc, target):
        """Record the instruction or interrupt at ``pc`` taking ``target`` from the pointer at
        ``ptr`` (NMOS page wrap): READ flags, def-use of written pointer bytes, chain transition."""
        cov, lw = self.cov, self._lw
        for c in (ptr, (ptr & 0xFF00) | ((ptr + 1) & 0xFF)):
            cov._fv[c] |= READ
            if lw[c] >= 0:
                cov.defuse.add((lw[c], c, pc))
        cov.chains.add((ptr, self.vector_target.get(ptr, -1), target))
        self.vector_target[ptr] = target

    def step(self, pc, cache, lifter):
        self.cur_pc = pc
        mem = self.mem
        op = mem[pc]
        n, br, dyn = _INFO[op]
        lw = self._lw
        if (
            lw[pc] >= 0
            or n > 1
            and (lw[(pc + 1) & 0xFFFF] >= 0 or n > 2 and lw[(pc + 2) & 0xFFFF] >= 0)
        ):
            self._uses(pc, n)
        else:
            self.cov._iv[pc] = 1
        nxt = PcodeVM.step(self, pc, cache, lifter)
        cov = self.cov
        f = cov._fv
        f[pc] |= EXEC
        for i in range(1, n):
            f[(pc + i) & 0xFFFF] |= OPERAND
        cov._hv[pc] += 1
        if br:
            f[pc] |= TAKEN if self.reg[br[0]] == br[1] else FALLTHRU
        elif dyn:
            cov.edges.add((pc, nxt))
            if op == 0x6C:
                self.consume(mem[(pc + 1) & 0xFFFF] | (mem[(pc + 2) & 0xFFFF] << 8), pc, nxt)
            elif op == 0x00:
                self.consume(0xFFFE, pc, nxt)
        return nxt


class CoverageMPU(MPU):
    """``MPU`` over ``CoverageVM``; ``cov`` persists across ``reset()``."""

    VM = CoverageVM

    @property
    def cov(self):
        return self._vm.cov

    @cov.setter
    def cov(self, cov):
        self._vm.cov = cov

    def step(self):
        pc = self.pc & self.addrMask
        if self._memory[pc] in _JAM:
            cov = self._vm.cov
            cov._fv[pc] |= EXEC
            cov._hv[pc] += 1
        return super().step()

    def irq(self):
        if not self.p & self.INTERRUPT:
            super().irq()
            self._vm.consume(self.IRQ, self.pc, self.pc)
            self.cov.entries.add((self.IRQ, self.pc))

    def nmi(self):
        super().nmi()
        self._vm.consume(self.NMI, self.pc, self.pc)
        self.cov.entries.add((self.NMI, self.pc))


def call(mpu, addr, a=0, x=0, y=0, max_steps=1_000_000):
    """Run the subroutine at ``addr`` with registers ``a``/``x``/``y`` until it returns to the
    sentinel return address ``RETURN`` it pushed; ``RuntimeError`` after ``max_steps``."""
    sp = mpu.sp
    mpu._push_word((RETURN - 1) & 0xFFFF)
    mpu.a, mpu.x, mpu.y, mpu.pc = a, x, y, addr
    for _ in range(max_steps):
        mpu.step()
        if mpu.pc == RETURN and mpu.sp == sp:
            return mpu
    raise RuntimeError("$%04X did not return within %d steps" % (addr, max_steps))
