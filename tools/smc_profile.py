"""Profile :func:`jennings.coverage.smc.analyze` on the sidplayer fixture: cProfile by
cumulative time, or with ``--visits`` the instructions the fixpoint processed most often.

    python tools/smc_profile.py [--seeded] [--handlers] [--visits] [--top N]
"""

import argparse
import bisect
import cProfile
import pstats
import time
from collections import Counter
from pathlib import Path

from jennings.coverage import CoverageMap, CoverageMPU, call
from jennings.coverage import smc as smc_mod
from jennings.coverage.elf import load_elf

ELF = Path(__file__).parents[1] / "tests" / "fixtures" / "sidplayer" / "sidplayer.prg.elf"
HANDLERS = ("fx_off", "fx_vibrato", "fx_slide", "fx_pulse")


def _songs(mem, syms):
    cov = CoverageMap()
    for song in (0, 1):
        mpu = CoverageMPU(memory=mem.tolist())
        call(mpu, syms["init"], a=song)
        for _ in range(300):
            call(mpu, syms["play"])
        cov |= mpu.cov
    return cov


def _visits(top, syms, run):
    names = sorted((a, n) for n, a in syms.items())
    keys = [a for a, _ in names]
    count = Counter()
    process = smc_mod._Analysis.process  # pylint: disable=protected-access

    def counted(self, node):
        count[node[0]] += 1
        return process(self, node)

    smc_mod._Analysis.process = counted  # pylint: disable=protected-access
    try:
        run()
    finally:
        smc_mod._Analysis.process = process  # pylint: disable=protected-access
    print("%d visits" % sum(count.values()))
    for pc, n in count.most_common(top):
        a, name = names[bisect.bisect_right(keys, pc) - 1]
        print("%6d  $%04X  %s+%d" % (n, pc, name, pc - a))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeded", action="store_true", help="pass both songs' coverage as cov")
    ap.add_argument("--handlers", action="store_true", help="add the fx handlers as entries")
    ap.add_argument("--visits", action="store_true", help="count visits per instruction")
    ap.add_argument("--top", type=int, default=25)
    args = ap.parse_args()
    mem, syms = load_elf(ELF)
    entries = {syms["init"]: {"a": {0, 1}}, syms["play"]: {}}
    if args.handlers:
        entries |= {syms[h]: {} for h in HANDLERS}
    cov = _songs(mem, syms) if args.seeded else None
    out = {}

    def run():
        t = time.perf_counter()
        out["smc"] = smc_mod.analyze(mem, entries, cov=cov, progress=True)
        out["t"] = time.perf_counter() - t

    if args.visits:
        _visits(args.top, syms, run)
    else:
        prof = cProfile.Profile()
        prof.runcall(run)
        pstats.Stats(prof).sort_stats("cumulative").print_stats(args.top)
    print("%.2fs, %d starts" % (out["t"], len(out["smc"].starts)))


if __name__ == "__main__":
    main()
