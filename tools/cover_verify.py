"""Certify and verify the cover point of an entry from RAM dumps.

Each ``STATE`` is the whole RAM as the entry is first called from. Prints a Markdown row per
state: the cover calls, the seconds to certify and to verify, and what the verifier found;
with ``--closure`` also the calls to the whole state's first repeat and what they add.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from jennings.coverage import certificate as V
from jennings.coverage.components import IO, STACK


def _span(text):
    lo, hi = (int(v, 16) for v in text.split(":"))
    return lo, hi


def _timed(fn, *args, **kw):
    start = time.perf_counter()
    out = fn(*args, **kw)
    return out, time.perf_counter() - start


def _row(path, args):
    with open(path, "rb") as f:
        state = f.read()
    exclude = args.exclude or (STACK, IO)
    if args.segments:
        (cert, _), took = _timed(V.by_segments, state, args.pc, args.limit, exclude=exclude)
    else:
        cert, took = _timed(V.certify, state, args.pc, args.limit, exclude=exclude)
    if isinstance(cert, V.Open):
        unrepeated = ["+".join("%04X" % c for c in cells) for cells in cert.unrepeated]
        print("%s: no segments: %s" % (path, cert.segments), file=sys.stderr)
        return [path, "open at %d" % cert.calls, "%.1f" % took, "", unrepeated, cert.period]
    if args.save:
        with open(path + ".cert.json", "w", encoding="utf-8") as f:
            json.dump(cert.to_dict(), f)
    row = [path, cert.calls, "%.1f" % took]
    for taint in args.taint:
        got, took = _timed(V.verify, state, cert, taint=taint)
        unused = sum(len(v) for v in got.accesses.values()) + len(got.edges)
        row += ["%.1f" % took, "ok" if got.ok else "%s: %s" % (got.failed, got.detail)]
        row += [unused, len(got.assumed)]
    if args.closure:
        got, took = _timed(V.closure, state, cert, args.closure)
        more = sum(len(v) for v in got.accesses.values()) + len(got.edges)
        row += [got.end, got.calls, "%.1f" % took, more]
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("states", nargs="+", metavar="STATE")
    ap.add_argument("--pc", type=lambda v: int(v, 16), required=True, help="the entry, hex")
    ap.add_argument("--limit", type=int, default=2_000_000, help="calls to look for a cover in")
    ap.add_argument(
        "--exclude", type=_span, action="append", metavar="LO:HI", help="hex, default stack and I/O"
    )
    ap.add_argument("--taint", action="append", choices=V.TAINTS, help="default calls")
    ap.add_argument("--closure", type=int, metavar="LIMIT", help="also run to state closure")
    ap.add_argument("--save", action="store_true", help="write STATE.cert.json")
    ap.add_argument("--segments", action="store_true", help="certify by segments only")
    args = ap.parse_args(argv)
    args.taint = args.taint or ["calls"]
    head = ["state", "calls", "certify s"]
    for taint in args.taint:
        head += ["verify %s s" % taint, "verdict", "unused", "assumed"]
    if args.closure:
        head += ["closure", "closure calls", "closure s", "beyond certificate"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head), flush=True)
    for path in args.states:
        print("| " + " | ".join(str(v) for v in _row(path, args)) + " |", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
