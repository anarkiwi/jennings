"""Static 6502 listing for coverage reporting: recursive-descent disassembly from entry
points and observed execution, one instruction per source line, with a ``.jcov`` sidecar."""

import numpy as np

from jennings.coverage import EXEC, FALLTHRU, TAKEN
from jennings.coverage.plugin import save_sidecar
from jennings.devices.mpu6502 import MPU, _JAM
from jennings.disassembler import Disassembler
from jennings.isa import is_branch, length, s8
from jennings.utils.addressing import AddressParser

__all__ = ["Listing"]

_LEN = np.array([length(op) for op in range(256)], np.int64)
_EXIT = frozenset((0x00, 0x40, 0x60, 0x6C)) | _JAM
_JMP, _JSR = 0x4C, 0x20
_DATA_ROW = 8


class _Labels(AddressParser):
    def __init__(self, addresses):
        super().__init__()
        self._names = {a: "L%04X" % a for a in addresses}

    def label_for(self, address, default=None):
        return self._names.get(address, default)


def _successors(m, a):
    """``([(flag, target)], jump_target, callee)`` static flow of the instruction at ``a``."""
    op = int(m[a])
    nxt = (a + int(_LEN[op])) & 0xFFFF
    lo = int(m[(a + 1) & 0xFFFF])
    word = lo | int(m[(a + 2) & 0xFFFF]) << 8
    if is_branch(op):
        t = (nxt + s8(lo)) & 0xFFFF
        return [(FALLTHRU, nxt), (TAKEN, t)], t, None
    if op == _JMP:
        return [(EXEC, word)], word, None
    if op in _EXIT:
        return [], None, None
    return [(EXEC, nxt)], None, word if op == _JSR else None


class Listing:
    """``source`` text with ``lines`` [[line, address]], static ``arcs`` [[from, to]] and
    ``transitions`` [[address, flag, from, to]] (see ``jennings.coverage.plugin``)."""

    __slots__ = ("source", "lines", "arcs", "transitions")

    def __init__(self, source, lines, arcs, transitions):
        self.source = source
        self.lines = np.asarray(lines, np.int64).reshape(-1, 2)
        self.arcs = np.asarray(arcs, np.int64).reshape(-1, 2)
        self.transitions = np.asarray(transitions, np.int64).reshape(-1, 4)

    def line(self, address):
        """Source line of the instruction starting at ``address`` (``None`` if not code)."""
        hit = self.lines[self.lines[:, 1] == address, 0]
        return int(hit[0]) if hit.size else None

    @classmethod
    def disassemble(cls, mem, entries, cov=None, ranges=()):
        """Decode from ``entries`` and, given ``cov``, from every executed address and dynamic
        edge/interrupt target; ``ranges`` (inclusive ``(lo, hi)``) add ``.byte`` rows for
        the bytes in them that no instruction covers. Executed starts may overlap."""
        m = np.zeros(0x10000, np.uint8)
        src = np.asarray(mem, np.uint8)[:0x10000]
        m[: src.size] = src
        roots = {a & 0xFFFF for a in entries}
        execd = np.zeros(0, np.int64)
        seeds = set()
        if cov is not None:
            execd = np.flatnonzero(cov.flags & EXEC)
            roots |= {t for _, t in cov.entries}
            seeds = {t for _, t in cov.edges}
        owned = np.zeros(0x10000, bool)
        span = _LEN[m[execd]]
        for k in range(3):
            owned[(execd[span > k] + k) & 0xFFFF] = True
        forced = set(execd.tolist())
        succ, jumps, callees = {}, set(), set()
        stack = sorted(roots | seeds | forced, reverse=True)
        while stack:
            a = stack.pop()
            if a in succ:
                continue
            if a not in forced:
                cells = (a + np.arange(_LEN[m[a]])) & 0xFFFF
                if owned[cells].any():
                    continue
                owned[cells] = True
            out, jump, callee = _successors(m, a)
            succ[a] = out
            stack.extend(t for _, t in reversed(out))
            jumps.add(jump)
            if callee is not None:
                callees.add(callee)
                stack.append(callee)

        want = np.zeros(0x10000, bool)
        for lo, hi in ranges:
            want[lo : hi + 1] = True
        free = np.flatnonzero(want & ~owned)
        rows = [(a, 0) for a in succ]
        for run in np.split(free, np.flatnonzero(np.diff(free) != 1) + 1):
            rows += [(int(a), int(min(_DATA_ROW, run[-1] + 1 - a))) for a in run[::_DATA_ROW]]
        rows.sort()
        line_of = {a: i + 1 for i, (a, n) in enumerate(rows) if not n}

        targets = {t for out in succ.values() for _, t in out}
        labels = _Labels((jumps | callees | roots | seeds) & line_of.keys())
        dis = Disassembler(MPU(memory=m.tolist()), labels)
        entry = roots | callees | (succ.keys() - targets)
        text, trans = [], []
        for a, n in rows:
            if n:
                data = ",".join("$%02X" % b for b in m[a : a + n])
                text.append("%04X  %-8s  %-7s.byte %s" % (a, "", "", data))
                continue
            L = line_of[a]
            n, ins = dis.instruction_at(a)
            hexb = " ".join("%02X" % m[(a + i) & 0xFFFF] for i in range(n))
            label = labels.label_for(a, "")
            text.append(
                ("%04X  %-8s  %-7s%s" % (a, hexb, label + ":" if label else "", ins)).rstrip()
            )
            out = [(f, line_of[t]) for f, t in succ[a] if t in line_of]
            trans += [(a, f, L, -L if to == L else to) for f, to in out] or [(a, EXEC, L, -L)]
            if a in entry:
                trans.append((a, EXEC, -L, L))
        trans = np.array(trans, np.int64).reshape(-1, 4)
        lines = np.array([(line_of[a], a) for a, n in rows if not n], np.int64)
        return cls("\n".join(text) + "\n", lines, np.unique(trans[:, 2:], axis=0), trans)

    def write(self, path):
        """Write the listing to ``path`` and its sidecar to ``path + '.jcov'``."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.source)
        save_sidecar(path, lines=self.lines, arcs=self.arcs, transitions=self.transitions)
