"""Time :func:`jennings.memo.orbit` on 64 KiB images and check it: with ``--brute`` against
:meth:`~jennings.devices.mpu6502.NativeMPU.orbit`, each ``--fibre`` against it without one.

    python tools/memo_orbit.py --pc 0x1003 --volatile 0xd000:0xe000 [--grain split]
        [--limit N] [--brute] [--fibre 0x1b17,0x1aff]... IMAGE...

One row per image and fibre: the end and calls, the memo's counts, its bytes and the seconds
taken; the host's load average heads the rows. The state is all RAM outside ``--volatile`` and
no registers, not :func:`jennings.coverage.certificate.closure`'s (the registers a call hands
the next, the stack page and the memory stored to, volatile or not).
"""

import argparse
import os
import sys
import time

import numpy as np
from tqdm import tqdm

from jennings import memo
from jennings.devices.mpu6502 import NativeMPU
from jennings.memory import NativeMemory

MAX_CYCLES = 10_000_000


def _spans(text):
    return [range(*(int(v, 0) for v in span.split(":"))) for span in text.split(",") if span]


def _cells(text):
    return [int(v, 0) for v in text.split(",") if v]


def _keep(volatile):
    mask = np.ones(0x10000, bool)
    for span in volatile:
        mask[span.start : span.stop] = False
    edge = np.flatnonzero(np.diff(mask.astype(np.int8), prepend=0, append=0))
    return [range(int(lo), int(hi)) for lo, hi in edge.reshape(-1, 2)]


def left(mpu):
    """The memory and registers ``mpu`` holds."""
    regs = (mpu.a, mpu.x, mpu.y, mpu.sp, mpu.p, mpu.pc, mpu.processorCycles)
    return bytes(mpu.memory), regs


def _timed(image, orbit):
    """``(found, rest, seconds)`` of ``orbit`` on a machine of ``image``: what it found and
    left, what else it returned and the seconds it took."""
    t = time.perf_counter()
    mpu = NativeMPU(memory=NativeMemory(image))
    end, states, acc, edges, *rest = orbit(mpu)
    seconds = time.perf_counter() - t
    return (end, states, bytes(acc), bytes(edges), *left(mpu)), rest, seconds


def _memo(image, args, fibre=None):
    """``(found, row)`` of the memo orbit of ``image``: what it found and left, and its row."""
    state = {"keep": _keep(args.volatile), "volatile": args.volatile, "fibre": fibre}
    found, (stats,), seconds = _timed(
        image,
        lambda mpu: memo.orbit(mpu, args.pc, 0, MAX_CYCLES, args.limit, args.grain, **state),
    )
    counts = {k: v for k, v in stats.items() if k not in ("roots", "fanout")}
    if not fibre:
        counts = {k: v for k, v in counts.items() if not k.startswith("tape_")}
    return found, {"end": found[0], "states": found[1], **counts, "seconds": seconds}


def run(image, args):
    """The rows of ``image``: the memo orbit's, then one per fibre, each with whether it found
    and left what it is checked against (``same``, None unchecked)."""
    found, row = _memo(image, args)
    want, state = None, (_keep(args.volatile), args.volatile)
    if args.brute:
        want, _, row["brute seconds"] = _timed(
            image, lambda mpu: mpu.orbit(args.pc, 0, MAX_CYCLES, args.limit, *state)
        )
    rows = [{**row, "same": want and want == found}]
    for fibre in args.fibre:
        got, row = _memo(image, args, fibre)
        rows.append({"fibre": ",".join(f"{a:04x}" for a in fibre), **row, "same": got == found})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    ap.add_argument("images", nargs="+")
    ap.add_argument("--pc", type=lambda v: int(v, 0), required=True)
    ap.add_argument("--volatile", type=_spans, default=[])
    ap.add_argument("--limit", type=int, default=100_000_000)
    ap.add_argument("--grain", choices=memo.GRAINS, default="split")
    ap.add_argument("--brute", action="store_true")
    ap.add_argument("--fibre", type=_cells, action="append", default=[])
    args = ap.parse_args(argv)
    print("load", *os.getloadavg(), "cpus", os.cpu_count(), flush=True)
    total, ok = 0.0, True
    for path in tqdm(args.images, unit="image", disable=None):
        with open(path, "rb") as f:
            image = f.read()
        for row in run(image, args):
            total += row["seconds"]
            ok &= row["same"] is not False
            cells = [f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()]
            tqdm.write(" ".join([os.path.basename(path), *cells]))
    print(f"total seconds {total:.2f}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
