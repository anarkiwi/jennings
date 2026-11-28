"""Compare what an entry's certificates claim with each other and with a brute-force orbit.

    python tools/cover_compare.py STATE --pc ENTRY [--limit N] [--orbit CALLS]

Certifies ``STATE`` (the whole RAM the entry is first called from) as ``certify`` does and by
segments alone, and prints each certificate's calls with the accesses and edges it claims.
With ``--orbit``, the calls are also run plainly to the whole state's first repeat (or that
many calls) and what they add to each certificate is counted.
"""

from __future__ import annotations

import argparse
import sys
import time

from jennings.coverage import certificate as V
from jennings.coverage import components as C
from jennings.coverage import segments as S


def claimed(cert):
    """``(accesses, edges)``: the sets of access and edge keys ``cert`` claims."""
    return set(V._access_keys(cert.accesses).tolist()), set(V._edge_keys(cert.edges).tolist())


def orbit(state, cert, calls):
    """``(end, calls, accesses, edges)`` of ``NativeMPU.orbit`` over the state ``closure``
    takes: the registers, the stack page and every address ``cert`` stores to."""
    keys = V._access_keys(cert.accesses)
    cells = sorted(set((keys[keys >> 16 & 1 == 1] & 0xFFFF).tolist()) | set(C.STACK))
    end, states, acc, edges = C.machine(state).orbit(
        cert.pc, cert.acc, cert.max_cycles, calls, cells=cells + list(V.REGISTERS)
    )
    return end, states - (end == "open"), set(acc.tolist()), set(edges.tolist())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    ap.add_argument("state")
    ap.add_argument("--pc", type=lambda v: int(v, 16), required=True, help="the entry, hex")
    ap.add_argument("--limit", type=int, default=2_000_000, help="calls to certify within")
    ap.add_argument("--orbit", type=int, metavar="CALLS", help="also run plainly this far")
    args = ap.parse_args(argv)
    with open(args.state, "rb") as f:
        state = f.read()
    certs = {"certify": V.certify(state, args.pc, args.limit)}
    try:
        certs["segments"] = V.by_segments(state, args.pc, args.limit)[0]
    except S.Refused as e:
        print("segments: %s: %s" % (e.failed, e.detail))
    certs = {k: c for k, c in certs.items() if not isinstance(c, V.Open)}
    for name, cert in certs.items():
        acc, edges = claimed(cert)
        kind = type(cert).__name__
        print(
            "%s: %s, %d calls, %d accesses, %d edges"
            % (name, kind, cert.calls, len(acc), len(edges))
        )
    alike = len({tuple(map(frozenset, claimed(c))) for c in certs.values()}) <= 1
    print("the certificates claim %s" % ("the same" if alike else "different accesses or edges"))
    if args.orbit and certs:
        start = time.perf_counter()
        end, calls, acc, edges = orbit(state, next(iter(certs.values())), args.orbit)
        print("orbit: %s after %d calls, %.1f s" % (end, calls, time.perf_counter() - start))
        for name, cert in certs.items():
            mine = claimed(cert)
            counts = (
                len(acc - mine[0]),
                len(edges - mine[1]),
                len(mine[0] - acc),
                len(mine[1] - edges),
            )
            print(
                "%s: the orbit adds %d accesses and %d edges, and lacks %d and %d" % (name, *counts)
            )
            alike &= not any(counts)
    return 0 if alike else 1


if __name__ == "__main__":
    sys.exit(main())
