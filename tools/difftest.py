"""Differential test: the native core against the Python engine, instruction by instruction.

Each program is 64 KiB of byte soup over a banked page table, run under mutating
watch callbacks, an all-address write logger and a raster model, with IRQ/NMI
injected at random boundaries; state, write log and callback sequence must match.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, field

from jennings.devices.mpu6502 import MPU
from jennings.memory import NativeMemory, ObservableMemory

PAL_CPL, PAL_LINES = 63, 312
OVERLAY_PAGES = (0xA, 0xB, 0xE, 0xF)
# 256-byte pages carrying the test's watches; never zero page, stack or vectors.
READ_PAGES = tuple(range(0xC0, 0xC8)) + (0x9F, 0xD4)
WRITE_PAGES = tuple(range(0xC4, 0xCC)) + (0x3F, 0xD4)
REENTER = 1 / 300
WATCHED = frozenset(READ_PAGES + WRITE_PAGES + (0xD0,))


class _Paged:
    """Reference subject: RAM with 4 KiB read-only overlays; writes always reach RAM."""

    def __init__(self, ram, overlays):
        self.ram = ram
        self.pages = [overlays.get(p) for p in range(16)]

    def __getitem__(self, addr):
        page = self.pages[addr >> 12]
        return self.ram[addr] if page is None else page[addr & 0xFFF]

    def __setitem__(self, addr, value):
        self.ram[addr] = value


class _NoStaleSha(dict):
    """PcodeVM cache that never keeps ``$93``: its lifted ``H+1`` comes from the zp pointer."""

    def __setitem__(self, key, rec):
        if key[1] != 0x93:
            super().__setitem__(key, rec)


@dataclass
class Program:
    seed: int
    ram: bytes
    overlays: dict
    log_writes: bool
    events: list = field(default_factory=list)

    @classmethod
    def make(cls, seed: int, log_writes: bool):
        rng = random.Random(seed)
        ram = rng.randbytes(0x10000)
        overlays = {p: rng.randbytes(0x1000) for p in OVERLAY_PAGES if rng.random() < 0.75}
        return cls(seed, ram, overlays, log_writes)


class _Side:
    """One backend: an MPU plus the callbacks whose call sequence is compared."""

    def __init__(self, prog: Program, native: bool):
        self.calls = []
        self.writes = []
        self.counts = [0, 0, 0]
        if native:
            mem = NativeMemory(prog.ram)
            for p, img in prog.overlays.items():
                mem.set_page(p, img)
            mem.io = True
            mem.raster_model(False)
            self.ram = mem
        else:
            self.ram = bytearray(prog.ram)
            mem = ObservableMemory(subject=_Paged(self.ram, prog.overlays))
            mem.subscribe_to_read((0xD011, 0xD012), self._raster)
        self.mem = mem
        self.mpu = MPU(memory=mem)
        if not native:
            self.mpu._cache = _NoStaleSha()
        for base in READ_PAGES:
            mem.subscribe_to_read(range(base << 8, (base + 1) << 8), self._read_a)
        mem.subscribe_to_read(range(0xD400, 0xD420), self._read_b)
        for base in WRITE_PAGES:
            mem.subscribe_to_write(range(base << 8, (base + 1) << 8), self._write)
        if prog.log_writes:
            mem.subscribe_to_write(range(0x10000), self._log)

    def _raster(self, addr):
        line = self.mpu.processorCycles // PAL_CPL % PAL_LINES
        if addr == 0xD012:
            return line & 0xFF
        return self.ram[0xD011] & 0x7F | (line >> 8) << 7

    def _read_a(self, addr):
        self.counts[0] += 1
        n = self.counts[0]
        self.calls.append(("ra", addr, self.mpu.pc, self.mpu.processorCycles))
        return None if n % 3 == 0 else (addr + 7 * n) & 0xFF

    def _read_b(self, addr):
        self.counts[1] += 1
        self.calls.append(("rb", addr, self.mpu.pc))
        return None if self.counts[1] % 2 else addr & 0x1F

    def _write(self, addr, value):
        self.counts[2] += 1
        self.calls.append(("w", addr, value, self.mpu.pc, self.mpu.processorCycles))
        return value ^ 0xA5 if self.counts[2] % 2 else None

    def _log(self, addr, value):
        self.writes.append((addr, value))

    def state(self):
        m = self.mpu
        return (m.processorCycles, m.pc, m.a, m.x, m.y, m.sp, m.p)


def _safe(pc):
    return not any(((pc + k) & 0xFFFF) >> 8 in WATCHED for k in range(3))


def run_program(seed: int, steps: int, log_writes: bool = True):
    """Lockstep both backends over one program; None if identical, else a divergence report."""
    prog = Program.make(seed, log_writes)
    rng = random.Random(seed ^ 0x5EED)
    ref, nat = _Side(prog, False), _Side(prog, True)
    for side in (ref, nat):
        side.mpu.pc = 0x0800
    for i in range(steps):
        r = rng.random()
        for side in (ref, nat):
            if r < 0.002:
                side.mpu.irq()
            elif r < 0.0025:
                side.mpu.nmi()
        if not _safe(ref.mpu.pc) or rng.random() < REENTER:
            pc = rng.randrange(0x0200, 0x9000)
            while not _safe(pc):
                pc = rng.randrange(0x0200, 0x9000)
            ref.mpu.pc = nat.mpu.pc = pc
        ref.mpu.step()
        nat.mpu.step()
        if ref.state() != nat.state():
            return "seed %d step %d: ref %r native %r" % (seed, i, ref.state(), nat.state())
        for name in ("calls", "writes"):
            got, want = getattr(nat, name), getattr(ref, name)
            if got != want:
                return "seed %d step %d: %s ref %r native %r" % (seed, i, name, want, got)
            got.clear()
            want.clear()
    if bytes(ref.ram) != bytes(nat.ram):
        return "seed %d: final RAM differs" % seed
    return None


def _job(args):
    seed, steps, log_writes = args
    return seed, run_program(seed, steps, log_writes)


def main(argv=None) -> int:
    from tqdm import tqdm  # pylint: disable=import-outside-toplevel

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--programs", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    args = ap.parse_args(argv)
    jobs = [(args.seed + k, args.steps, k % 4 != 3) for k in range(args.programs)]
    bad = []
    with ProcessPoolExecutor(args.jobs) if args.jobs > 1 else nullcontext() as ex:
        results = ex.map(_job, jobs) if ex else map(_job, jobs)
        progress = tqdm(results, total=len(jobs), unit="prog")
        for _seed, err in progress:
            if err:
                bad.append(err)
                progress.write(err)
            progress.set_postfix(diverged=len(bad))
    print(
        "%d programs x %d instructions: %d diverged" % (args.programs, args.steps, len(bad)),
        file=sys.stderr,
    )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
