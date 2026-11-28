"""Print the instructions of an entry's calls that access given addresses, from a RAM dump.

    python tools/cover_sites.py STATE --pc ENTRY ADDR... [--calls N] [--around BYTES]

Each ``ADDR`` (hex) is listed with the PCs that load or store it over the calls, then each such
instruction is disassembled with the bytes around it.
"""

from __future__ import annotations

import argparse
import sys

from jennings.coverage import segments as S
from jennings.disassembler import Disassembler
from jennings.devices.mpu6502 import MPU


def _hex(text):
    return int(text, 16)


def listing(state, start, stop):
    """Disassembly lines of ``state`` from ``start`` up to ``stop``."""
    mpu = MPU()
    mpu.memory[0:0x10000] = list(state)
    dis, out, pc = Disassembler(mpu), [], start
    while pc < stop:
        size, text = dis.instruction_at(pc)
        out.append("  $%04X  %s" % (pc, text))
        pc += size
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    ap.add_argument("state")
    ap.add_argument("addrs", nargs="+", type=_hex, metavar="ADDR")
    ap.add_argument("--pc", type=_hex, required=True, help="the entry, hex")
    ap.add_argument("--calls", type=int, default=4096)
    ap.add_argument("--around", type=int, default=12, help="bytes listed after each site")
    args = ap.parse_args(argv)
    with open(args.state, "rb") as f:
        state = f.read()
    keys = S.cut(state, args.pc, args.calls, ()).accesses
    for addr in args.addrs:
        mine = keys[keys & 0xFFFF == addr]
        print("$%04X:" % addr)
        for key in mine.tolist():
            pc, kind = key >> 17, ("load", "store")[key >> 16 & 1]
            print(" %s at $%04X" % (kind, pc))
            print("\n".join(listing(state, pc, pc + args.around)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
