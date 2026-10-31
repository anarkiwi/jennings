"""Benchmark the 6510 backends on a playroutine-shaped program.

A main loop waits for the next ``$D012`` raster line and calls a play routine
(indexed ``STA $D400,X``, a ``(zp),Y`` copy, illegals); an IRQ fires every
``--irq`` cycles. Prints Mcycles/s and x realtime per backend as a Markdown table.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

from tqdm import tqdm

from jennings.assembler import Assembler
from jennings.devices.mpu6502 import MPU
from jennings.memory import NativeMemory

PAL_HZ = 985248
ORG, IRQ = 0x1000, 0x2000
PROGRAM = {
    ORG: """
        main:  LDA $D012
        wait:  CMP $D012
               BEQ wait
               JSR play
               JMP main
        play:  LDX #$18
        sid:   LDA $3000,X
               EOR $FB
               STA $D400,X
               DEX
               BPL sid
               LDY #$3F
        copy:  LDA ($FC),Y
               STA ($FE),Y
               DEY
               BPL copy
               LAX $F0
               SAX $F1
               DCP $F2
               ISC $F3
               SLO $F4
               RLA $F5
               ALR #$7F
               ANC #$80
               SBX #$01
               INC $FB
               RTS
    """,
    IRQ: """
               PHA
               TXA
               PHA
               INC $D019
               LDA $DC0D
               PLA
               TAX
               PLA
               RTI
    """,
}


def assemble(source: str, org: int) -> bytes:
    """Two-pass assemble ``label: op operand`` lines; a bare-identifier operand is a label."""
    asm = Assembler(MPU())
    lines = [
        (label.strip(), stmt.strip())
        for label, _, stmt in (raw.strip().rpartition(":") for raw in source.strip().splitlines())
    ]
    labels = {}
    for _pass in range(2):
        pc, out = org, bytearray()
        for label, stmt in lines:
            if label:
                labels[label] = pc
            op, _, arg = stmt.partition(" ")
            if arg.isidentifier():
                arg = "$%04X" % labels.get(arg, pc)
            code = bytes(asm.assemble(("%s %s" % (op, arg)).strip(), pc))
            out += code
            pc += len(code)
    return bytes(out)


def image() -> bytearray:
    """64 KiB with the program, IRQ vector, SID table and zp pointers; ``$01=$35``."""
    ram = bytearray(0x10000)
    for org, src in PROGRAM.items():
        code = assemble(src, org)
        ram[org : org + len(code)] = code
    ram[0], ram[1] = 0x2F, 0x35
    ram[0xFFFE], ram[0xFFFF] = IRQ & 0xFF, IRQ >> 8
    ram[0x3000:0x3019] = bytes(range(0x19))
    ram[0xFC:0x100] = bytes((0x00, 0x40, 0x00, 0x50))
    return ram


def drive_step(mpu, total: int, period: int) -> None:
    due = period
    while mpu.processorCycles < total:
        mpu.step()
        if mpu.processorCycles >= due:
            mpu.irq()
            due += period


def drive_run_until(mpu, total: int, period: int) -> None:
    due = period
    while mpu.processorCycles < total:
        left = min(due, total) - mpu.processorCycles
        if left > 0:
            mpu.run_until(max_cycles=left - 1, stop_on_jam=False)
        if mpu.processorCycles >= due:
            mpu.irq()
            due += period


def python_list():
    mpu = MPU(memory=list(image()), pc=ORG)
    mpu._vm.volatile = True  # PcodeVM's own raster model on a flat list
    return mpu, drive_step


def python_flat_mpu():
    from vagrant_moses.cpu import flat_mpu  # pylint: disable=import-outside-toplevel

    mpu, _mem = flat_mpu(image())
    mpu.pc = ORG
    return mpu, drive_step


def native(observe: bool = False):
    mem = NativeMemory(image())
    mem.io = True
    mem.raster_model(False)
    if observe:
        log = []
        mem.observe_data_access(on_write=lambda a, v: log.append(v), writes=range(0xD400, 0xD419))
    return MPU(memory=mem, pc=ORG)


BACKENDS = {
    "(a) Python engine, list": python_list,
    "(b) Python engine, flat_mpu": python_flat_mpu,
    "(c) native step()": lambda: (native(), drive_step),
    "(d) native run_until": lambda: (native(), drive_run_until),
    "(d') native run_until + SID-write observer": lambda: (native(True), drive_run_until),
}


def _state(mpu):
    return (mpu.pc, mpu.a, mpu.x, mpu.y, mpu.sp, mpu.p)


def run(name: str, cycles: int, period: int):
    mpu, drive = BACKENDS[name]()
    t0 = time.perf_counter()
    drive(mpu, cycles, period)
    dt = time.perf_counter() - t0
    return mpu.processorCycles, dt, _state(mpu)


def threaded(n: int, cycles: int, period: int) -> float:
    mpus = [native() for _ in range(n)]
    threads = [threading.Thread(target=drive_run_until, args=(m, cycles, period)) for m in mpus]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return time.perf_counter() - t0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cycles", type=int, default=20_000_000)
    ap.add_argument("--irq", type=int, default=20_000, help="cycles between IRQs")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--thread-cycles", type=int, default=500_000_000)
    ap.add_argument("--only", nargs="*", default=None, help="backend name prefixes, e.g. (c) (d)")
    args = ap.parse_args(argv)
    names = [n for n in BACKENDS if not args.only or any(n.startswith(p) for p in args.only)]
    rows, states = [], {}
    for name in tqdm(names, unit="backend"):
        cyc, dt, state = run(name, args.cycles, args.irq)
        states[name] = state
        rows.append((name, cyc / dt / 1e6, cyc / dt / PAL_HZ, dt))
    print("| backend | Mcycles/s | x realtime | wall s |")
    print("|---|---:|---:|---:|")
    for name, mcs, rt, dt in rows:
        print("| %s | %.2f | %.1f | %.2f |" % (name, mcs, rt, dt))
    native_states = {s for n, s in states.items() if "native" in n or "flat_mpu" in n}
    if len(native_states) > 1:
        print("backends disagree on the final state: %r" % states, file=sys.stderr)
        return 1
    if args.threads:
        one = threaded(1, args.thread_cycles, args.irq)
        many = threaded(args.threads, args.thread_cycles, args.irq)
        print(
            "\n%d threads x %d cycles via run_until: wall %.2f s vs %.2f s single (%.2fx)"
            % (args.threads, args.thread_cycles, many, one, many / one)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
