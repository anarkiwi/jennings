"""What ``gen_cpu.py`` and ``gen_flow.py`` share: the ``--check`` command line."""

from __future__ import annotations

import argparse
import sys


def main(argv, doc, out, generate, script) -> int:
    """Write ``generate()`` to ``out``, or with ``--check`` fail if ``out`` is stale."""
    ap = argparse.ArgumentParser(description=doc.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="fail if the checked-in file is stale")
    args = ap.parse_args(argv)
    src = generate()
    if args.check:
        if out.read_text() != src:
            print("%s is stale; run tools/%s" % (out, script), file=sys.stderr)
            return 1
        return 0
    out.write_text(src)
    return 0
