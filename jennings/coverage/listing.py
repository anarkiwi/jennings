"""Static 6502 listing for coverage reporting: recursive-descent disassembly from entry
points and observed execution, one instruction per source line, with a ``.jcov`` sidecar."""

import numpy as np

from jennings.coverage._core import EXEC, OPERAND, READ, WRITE
from jennings.coverage.cfg import LEN, successors
from jennings.coverage.plugin import save_sidecar
from jennings.coverage.resolve import Tables, split_table
from jennings.devices.mpu6502 import MPU
from jennings.disassembler import Disassembler
from jennings.opcodes import ILLEGAL_OPCODES
from jennings.utils.addressing import AddressParser

__all__ = ["Listing"]

_DATA_ROW = 8
_GARBAGE = ILLEGAL_OPCODES | {0x00}


def _walk(m, stack, owned, known, forced=frozenset(), bad=None):
    """Recursive descent from ``stack`` past the starts in ``known``, claiming ``owned``
    bytes: ``(succ, jumps, callees)`` of the new instructions. A start overlapping owned
    bytes is dropped, or with ``bad(address, cells)`` the whole walk fails (None)."""
    succ, jumps, callees = {}, set(), set()
    while stack:
        a = stack.pop()
        if a in succ or a in known:
            continue
        if a not in forced:
            cells = (a + np.arange(LEN[m[a]])) & 0xFFFF
            if owned[cells].any() or bad is not None and bad(a, cells):
                if bad is not None:
                    return None
                continue
            owned[cells] = True
        out, jump, callee = successors(m, a)
        succ[a] = out
        stack.extend(t for _, t in reversed(out))
        jumps.add(jump)
        if callee is not None:
            callees.add(callee)
            stack.append(callee)
    return succ, jumps, callees


def _resolve(m, succ, owned, image, cov, heads, jumps, callees, tabled):
    """Decode the statically resolved targets of computed transfers into ``succ``/``owned``
    until no new transfer appears: ``(edges {(transfer, target)}, new roots)``. A target is
    taken only if its whole descent decodes documented opcodes other than BRK inside the
    image, without misaligned overlap, table bytes, read-only data or written opcodes."""
    flags = np.zeros(0x10000, np.uint8) if cov is None else cov.flags
    code = (flags & (EXEC | OPERAND)) != 0
    var = ((flags & WRITE) != 0) & ~code
    data = (((flags & READ) != 0) & ~code & ~var) | tabled
    edges, new, done, firsts = set(), set(), set(), set()
    while True:
        tables = Tables(m, succ, owned, image, cov, heads | callees, firsts)
        todo = [t for t in tables.transfers() if t not in done]
        if not todo:
            return edges, new

        def bad(a, cells, tables=tables):
            return (
                int(m[a]) in _GARBAGE
                or var[a]
                or tables.objects[a]
                or data[cells].any()
                or not image[cells].all()
            )

        plans = {t: tables.plan(t) for t in todo}
        for t in todo:
            done.add(t)
            found, cells = tables.targets(plans[t])
            data[cells] = True
            for target in found:
                if target not in succ:
                    trial = owned.copy()
                    got = _walk(m, [target], trial, succ, bad=bad)
                    if got is None:
                        continue
                    owned[:] = trial
                    succ.update(got[0])
                    jumps |= got[1]
                    callees |= got[2]
                    new.add(target)
                edges.add((t, target))


class _Labels(AddressParser):
    def __init__(self, addresses, names=None):
        super().__init__()
        self._names = {a: "L%04X" % a for a in addresses}
        self._names.update(names or {})

    def label_for(self, address, default=None):
        return self._names.get(address, default)


class Listing:
    """``source`` text with ``lines`` [[line, address]], static ``arcs`` [[from, to]] and
    ``transitions`` [[address, flag, from, to]] (see ``jennings.coverage.plugin``)."""

    __slots__ = ("source", "lines", "arcs", "transitions", "resolved")

    def __init__(self, source, lines, arcs, transitions, resolved=()):
        self.source = source
        self.lines = np.asarray(lines, np.int64).reshape(-1, 2)
        self.arcs = np.asarray(arcs, np.int64).reshape(-1, 2)
        self.transitions = np.asarray(transitions, np.int64).reshape(-1, 4)
        self.resolved = np.asarray(sorted(resolved), np.int64).reshape(-1, 2)

    def line(self, address):
        """Source line of the instruction starting at ``address`` (``None`` if not code)."""
        hit = self.lines[self.lines[:, 1] == address, 0]
        return int(hit[0]) if hit.size else None

    @classmethod
    # pylint: disable-next=too-many-arguments,too-many-positional-arguments
    def disassemble(
        cls, mem, entries, cov=None, ranges=(), resolve=False, tables=(), names=None, inside=None
    ):
        """Decode from ``entries`` and, given ``cov``, from every executed address and dynamic
        edge/interrupt target; ``ranges`` (inclusive ``(lo, hi)``) add ``.byte`` rows for
        the bytes in them that no instruction covers. Executed starts may overlap.
        ``resolve`` also decodes the targets of computed transfers read from constant tables
        (``resolved``: ``[[transfer, target]]``; new roots are marked ``; resolved from``).
        ``tables`` are hand-written split tables, :func:`~.resolve.split_table` arguments
        ``(lo, hi, n[, stride[, high]])``: their words are entries, their bytes table data.
        ``names`` ``{address: name}`` override the ``L%04X`` labels; ``inside`` (inclusive
        ``(lo, hi)`` spans) limits descent to instructions wholly within them."""
        m = np.zeros(0x10000, np.uint8)
        src = np.asarray(mem, np.uint8)[:0x10000]
        m[: src.size] = src
        roots = {a & 0xFFFF for a in entries}
        tabled = np.zeros(0x10000, bool)
        for t in tables:
            words, cells = split_table(m, *t)
            roots.update(words.tolist())
            tabled[cells] = True
        execd = np.zeros(0, np.int64)
        seeds = set()
        if cov is not None:
            execd = np.flatnonzero(cov.flags & EXEC)
            roots |= {t for _, t in cov.entries}
            seeds = {t for _, t in cov.edges}
        owned = np.zeros(0x10000, bool)
        if inside is not None:
            owned[:] = True
            for lo, hi in inside:
                owned[lo : hi + 1] = False
        span = LEN[m[execd]]
        for k in range(3):
            owned[(execd[span > k] + k) & 0xFFFF] = True
        forced = set(execd.tolist())
        stack = sorted(roots | seeds | forced, reverse=True)
        succ, jumps, callees = _walk(m, stack, owned, {}, forced)
        want = np.zeros(0x10000, bool)
        for lo, hi in ranges:
            want[lo : hi + 1] = True
        edges, new = set(), set()
        if resolve:
            image = want if ranges else np.ones(0x10000, bool)
            edges, new = _resolve(m, succ, owned, image, cov, roots | seeds, jumps, callees, tabled)
        via = {}
        for t, target in sorted(edges):
            via.setdefault(target, []).append("$%04X" % t)

        free = np.flatnonzero(want & ~owned)
        rows = [(a, 0) for a in succ]
        for run in np.split(free, np.flatnonzero(np.diff(free) != 1) + 1):
            rows += [(int(a), int(min(_DATA_ROW, run[-1] + 1 - a))) for a in run[::_DATA_ROW]]
        rows.sort()
        line_of = {a: i + 1 for i, (a, n) in enumerate(rows) if not n}

        targets = {t for out in succ.values() for _, t in out}
        labels = _Labels((jumps | callees | roots | seeds | new) & line_of.keys(), names)
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
            row = "%04X  %-8s  %-7s%s" % (a, hexb, label + ":" if label else "", ins)
            text.append(
                row.rstrip() + ("  ; resolved from " + ",".join(via[a]) if a in new else "")
            )
            out = [(f, line_of[t]) for f, t in succ[a] if t in line_of]
            trans += [(a, f, L, -L if to == L else to) for f, to in out] or [(a, EXEC, L, -L)]
            if a in entry:
                trans.append((a, EXEC, -L, L))
        trans = np.array(trans, np.int64).reshape(-1, 4)
        lines = np.array([(line_of[a], a) for a, n in rows if not n], np.int64)
        arcs = np.unique(trans[:, 2:], axis=0)
        return cls("\n".join(text) + "\n", lines, arcs, trans, edges)

    def write(self, path):
        """Write the listing to ``path`` and its sidecar to ``path + '.jcov'``."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.source)
        save_sidecar(path, lines=self.lines, arcs=self.arcs, transitions=self.transitions)
