"""Time :func:`jennings.memo.orbit` on 64 KiB images and, with ``--brute``, check it against
:meth:`~jennings.devices.mpu6502.NativeMPU.orbit`.

    python tools/memo_orbit.py --pc 0x1003 --volatile 0xd000:0xe000 [--grain split]
        [--limit N] [--brute] IMAGE...

One row per image: the end and calls, the memo's counts, its bytes and the seconds taken; the
host's load average heads the rows. The state is all RAM outside ``--volatile`` and no
registers, not :func:`jennings.coverage.certificate.closure`'s (the registers a call hands
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


def _keep(volatile):
    mask = np.ones(0x10000, bool)
    for span in volatile:
        mask[span.start : span.stop] = False
    edge = np.flatnonzero(np.diff(mask.astype(np.int8), prepend=0, append=0))
    return [range(int(lo), int(hi)) for lo, hi in edge.reshape(-1, 2)]


def run(image, pc, volatile, limit, grain, brute):
    """``(row, same)``: the memo orbit's row for ``image`` and, with ``brute``, whether the
    brute-force orbit gives the same end, states, accesses and edges (and its seconds)."""
    keep = _keep(volatile)
    t = time.perf_counter()
    mpu = NativeMPU(memory=NativeMemory(image))
    end, states, acc, edges, stats = memo.orbit(
        mpu, pc, 0, MAX_CYCLES, limit, grain, keep=keep, volatile=volatile
    )
    counts = {k: v for k, v in stats.items() if k not in ("roots", "fanout")}
    row = {"end": end, "states": states, **counts, "seconds": time.perf_counter() - t}
    if not brute:
        return row, None
    t = time.perf_counter()
    want = NativeMPU(memory=NativeMemory(image)).orbit(pc, 0, MAX_CYCLES, limit, keep, volatile)
    row["brute seconds"] = time.perf_counter() - t
    same = want[:2] == (end, states) and bytes(want[2]) == bytes(acc)
    return row, same and bytes(want[3]) == bytes(edges)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    ap.add_argument("images", nargs="+")
    ap.add_argument("--pc", type=lambda v: int(v, 0), required=True)
    ap.add_argument("--volatile", type=_spans, default=[])
    ap.add_argument("--limit", type=int, default=100_000_000)
    ap.add_argument("--grain", choices=memo.GRAINS, default="split")
    ap.add_argument("--brute", action="store_true")
    args = ap.parse_args(argv)
    print("load", *os.getloadavg(), "cpus", os.cpu_count(), flush=True)
    total, ok = 0.0, True
    for path in tqdm(args.images, unit="image", disable=None):
        with open(path, "rb") as f:
            image = f.read()
        row, same = run(image, args.pc, args.volatile, args.limit, args.grain, args.brute)
        total += row["seconds"]
        ok &= same is not False
        cells = [f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()]
        tqdm.write(" ".join([os.path.basename(path), *cells, *([f"same {same}"] * args.brute)]))
    print(f"total seconds {total:.2f}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
