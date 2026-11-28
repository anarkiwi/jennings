"""Cut the calls of an entry into segments and print what each runs on, from RAM dumps.

Each ``STATE`` is the whole RAM as the entry is first called from. Per state: the cuts
(``--cut`` or those :func:`jennings.coverage.segments.propose` finds), then a Markdown row per
segment with its start, the segments it depends on, its state and where that state repeats.
"""

from __future__ import annotations

import argparse
import sys
import time

from jennings.coverage import components as C
from jennings.coverage import segments as S


def _hex(text):
    return int(text, 16)


def _names(locs, most=8):
    names = [C.location(c) for c in sorted(locs)]
    return " ".join(names[:most]) + (" +%d" % (len(names) - most) if len(names) > most else "")


def _found(state, args):
    """``(cuts, found)`` for ``state``: the cuts given, or those proposed (None if none)."""
    volatile = C.spans(args.volatile or (C.IO,))
    if args.cut is not None:
        cuts = tuple(sorted(args.cut))
        found = S.examine(state, args.pc, args.calls, cuts, lead=args.lead, volatile=volatile)
        return cuts, found
    got = S.propose(state, args.pc, args.limit, volatile=volatile, learn=args.learn)
    return got[0], got[3]


def report(path, args):
    """Print the segments of the state at ``path``; whether every one's state repeats."""
    with open(path, "rb") as f:
        state = f.read()
    whole = S.cut(state, args.pc, args.learn, (), lead=args.learn // 2)
    print("%s: candidates %s" % (path, " ".join("$%04X" % p for p in S.candidates(whole, state))))
    start = time.perf_counter()
    try:
        cuts, found = _found(state, args)
    except S.Refused as e:
        print("refused: %s: %s" % (e.failed, e.detail))
        return False
    took = time.perf_counter() - start
    how = " ".join("$%04X" % p for p in cuts) or "none"
    lead, calls = found.trace.lead, found.trace.lead + found.trace.calls
    print(
        "cuts %s after %d calls, %d calls, closes at %s (%.1f s)"
        % (how, lead, calls, found.calls, took)
    )
    print("%d runs made again" % found.trace.forks)
    print("| segment | from | cycles | depends on | held | free | state | cells | repeat |")
    print("|---|---|---|---|---|---|---|---|---|")
    parts = found.parts
    for seg, pc in enumerate(found.trace.starts):
        row = [seg, "$%04X" % pc, found.trace.cycles[seg]]
        row += [" ".join(map(str, sorted(parts.groups[seg] - {seg}))), _names(parts.held[seg])]
        row += [_names(parts.free[seg])]
        row += [len(parts.cells[seg]), _names(parts.cells[seg]), found.cycles[seg]]
        print("| " + " | ".join(str(v) for v in row) + " |")
    sys.stdout.flush()
    return found.closed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    ap.add_argument("states", nargs="+", metavar="STATE")
    ap.add_argument("--pc", type=_hex, required=True, help="the entry, hex")
    ap.add_argument("--limit", type=int, default=2_000_000, help="calls to look for cuts in")
    ap.add_argument("--learn", type=int, default=C.CHUNK, help="calls the cuts are chosen over")
    ap.add_argument("--cut", type=_hex, action="append", help="cut here (hex), not as proposed")
    ap.add_argument("--calls", type=int, default=C.CHUNK, help="calls examined under --cut")
    ap.add_argument("--lead", type=int, default=0, help="calls left whole under --cut")
    ap.add_argument(
        "--volatile",
        type=lambda v: tuple(int(x, 16) for x in v.split(":")),
        action="append",
        metavar="LO:HI",
        help="hex, default I/O",
    )
    args = ap.parse_args(argv)
    return 0 if all([report(path, args) for path in args.states]) else 1


if __name__ == "__main__":
    sys.exit(main())
