"""Code coverage for 6502 execution: per-address execute/operand/branch/data flags,
execution counts, dynamic control-transfer edges and interrupt entries.

``CoverageVM`` records inside the engine, so deity-informant's drivers
(``run_sub``/``run_irq``/``run_irq_driven``) are covered as well as ``CoverageMPU.step``.
"""

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
    "CoverageMap",
    "CoverageVM",
    "CoverageMPU",
]

EXEC, OPERAND, TAKEN, FALLTHRU, READ, WRITE = 1, 2, 4, 8, 16, 32

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


def _pairs(s):
    return np.array(sorted(s), dtype=np.uint16).reshape(-1, 2)


def _unpairs(arr):
    return set(map(tuple, arr.tolist()))


def _mark_vector(cov, ptr):
    """Flag the two pointer bytes an indirect jump or interrupt entry reads (NMOS page wrap)."""
    f = cov._fv
    f[ptr] |= READ
    f[(ptr & 0xFF00) | ((ptr + 1) & 0xFF)] |= READ


class CoverageMap:
    """Per-address ``flags`` (uint8) and ``hits`` (uint32), dynamic ``edges`` and
    interrupt ``entries`` as ``(source, target)`` / ``(vector, target)`` sets."""

    __slots__ = ("flags", "hits", "edges", "entries", "_fv", "_hv")

    def __init__(self):
        self.flags = np.zeros(0x10000, np.uint8)
        self.hits = np.zeros(0x10000, np.uint32)
        self.edges = set()
        self.entries = set()
        self._fv = memoryview(self.flags)
        self._hv = memoryview(self.hits)

    def merge(self, other):
        """Accumulate ``other`` in place (OR flags, add hits, union sets)."""
        np.bitwise_or(self.flags, other.flags, out=self.flags)
        np.add(self.hits, other.hits, out=self.hits)
        self.edges |= other.edges
        self.entries |= other.entries
        return self

    __ior__ = merge

    def save(self, path):
        """Write a single ``.npz`` (numpy appends the suffix if ``path`` lacks it)."""
        np.savez_compressed(
            path,
            flags=self.flags,
            hits=self.hits,
            edges=_pairs(self.edges),
            entries=_pairs(self.entries),
        )

    @classmethod
    def load(cls, path):
        cov = cls()
        with np.load(path) as z:
            cov.flags[:] = z["flags"]
            cov.hits[:] = z["hits"]
            cov.edges = _unpairs(z["edges"])
            cov.entries = _unpairs(z["entries"])
        return cov

    def __eq__(self, other):
        if not isinstance(other, CoverageMap):
            return NotImplemented
        return (
            np.array_equal(self.flags, other.flags)
            and np.array_equal(self.hits, other.hits)
            and self.edges == other.edges
            and self.entries == other.entries
        )

    __hash__ = None


class CoverageVM(PcodeVM):
    """``PcodeVM`` recording into ``cov``; ``cur_pc`` is the executing instruction."""

    __slots__ = ("cov", "cur_pc")

    def __init__(self, mem_bytes, cov=None):
        super().__init__(mem_bytes)
        self.cov = CoverageMap() if cov is None else cov
        self.cur_pc = None

    def _rd(self, addr, sz):
        f = self.cov._fv
        for i in range(sz):
            f[(addr + i) & 0xFFFF] |= READ
        return PcodeVM._rd(self, addr, sz)

    def _wr(self, addr, val, sz):
        f = self.cov._fv
        for i in range(sz):
            f[(addr + i) & 0xFFFF] |= WRITE
        PcodeVM._wr(self, addr, val, sz)

    def step(self, pc, cache, lifter):
        self.cur_pc = pc
        mem = self.mem
        op = mem[pc]
        n, br, dyn = _INFO[op]
        if op == 0x6C:
            ptr = mem[(pc + 1) & 0xFFFF] | (mem[(pc + 2) & 0xFFFF] << 8)
            _mark_vector(self.cov, ptr)
        elif op == 0x00:
            _mark_vector(self.cov, 0xFFFE)
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
            _mark_vector(self.cov, self.IRQ)
            self.cov.entries.add((self.IRQ, self.pc))

    def nmi(self):
        super().nmi()
        _mark_vector(self.cov, self.NMI)
        self.cov.entries.add((self.NMI, self.pc))
