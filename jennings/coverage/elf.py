"""Coverage sidecars for original sources from ELF/DWARF (e.g. llvm-mos ``.prg.elf``): every
instruction start in a DWARF line-table row maps to that row's line, and static flow between
instruction starts becomes coverage.py arcs (see ``jennings.coverage.plugin``)."""

import os
import posixpath
from collections import defaultdict

import numpy as np
from elftools.elf.elffile import ELFFile

from jennings.coverage._core import EXEC
from jennings.coverage.cfg import LEN, is_jump, successors
from jennings.coverage.plugin import save_sidecar

__all__ = ["load_elf", "line_starts", "sidecars", "write_sidecars"]


def _str(s):
    return s.decode() if isinstance(s, bytes) else s


def load_elf(path):
    """``(memory, symbols)``: a 64K ``uint8`` image of the ``PT_LOAD`` segments (zero-filled
    beyond their file contents) and ``{name: address}`` from the symbol table."""
    mem = np.zeros(0x10000, np.uint8)
    with open(path, "rb") as f:
        elf = ELFFile(f)
        for seg in elf.iter_segments("PT_LOAD"):
            data = np.frombuffer(seg.data(), np.uint8)
            mem[seg["p_vaddr"] : seg["p_vaddr"] + data.size] = data
        tab = elf.get_section_by_name(".symtab")
        syms = {
            s.name: s["st_value"]
            for s in (tab.iter_symbols() if tab else ())
            if s.name and s["st_info"]["type"] != "STT_FILE" and s["st_shndx"] != "SHN_UNDEF"
        }
    return mem, syms


def _files(lp, comp_dir, root):
    """Real paths of a line program's file table, indexed like ``state.file``."""
    v5 = lp["version"] >= 5
    dirs = [comp_dir] * (not v5) + [_str(d) for d in lp["include_directory"]]
    names = [posixpath.join(dirs[e.dir_index], _str(e.name)) for e in lp["file_entry"]]
    pre = comp_dir.rstrip("/") + "/"
    rel = [n[len(pre) :] if n.startswith(pre) else n for n in names]
    return [None] * (not v5) + [os.path.normpath(os.path.join(root, r)) for r in rel]


def line_starts(path, mem, root=None):
    """``{address: (source, line)}`` for every instruction start in a DWARF line row's range
    ``[address, next row address)`` (line-0 rows continue the previous line), walking
    instruction lengths over ``mem``; paths under the compilation directory resolve against
    ``root`` (default: the ELF's directory)."""
    root = os.path.dirname(os.path.abspath(path)) if root is None else root
    starts = {}
    with open(path, "rb") as f:
        dwarf = ELFFile(f).get_dwarf_info()
        for cu in dwarf.iter_CUs():
            top = cu.get_top_DIE().attributes
            comp_dir = _str(top["DW_AT_comp_dir"].value) if "DW_AT_comp_dir" in top else ""
            lp = dwarf.line_program_for_CU(cu)
            files = _files(lp, comp_dir, root)
            rows = [e.state for e in lp.get_entries() if e.state is not None]
            line = 0
            for r, nxt in zip(rows, rows[1:]):
                line = 0 if r.end_sequence else r.line or line
                a = r.address
                while line and a < nxt.address:
                    starts[a] = (files[r.file], line)
                    a += int(LEN[mem[a]])
    return starts


def sidecars(path, entries=(), root=None):
    """``{source: {"lines", "arcs", "transitions"}}`` for each DWARF source of the ELF at
    ``path``, from the static successors of each instruction start (docs/coverage.md);
    ``entries`` are addresses or symbol names entered from outside."""
    mem, syms = load_elf(path)
    starts = line_starts(path, mem, root)
    flow = {a: successors(mem, a) for a in starts}
    src_of = {a: s for a, (s, _) in starts.items()}
    preds = {t for a, (out, _, _) in flow.items() for _, t in out if src_of.get(t) == src_of[a]}
    entry = {(syms[e] if isinstance(e, str) else e) & 0xFFFF for e in entries}
    entry |= {c for _, _, c in flow.values() if c is not None}
    entry |= starts.keys() - preds
    lines, trans = defaultdict(list), defaultdict(list)
    for a in sorted(starts):
        src, L = starts[a]
        out, jump, _ = flow[a]
        lines[src].append((L, a))
        rows = trans[src]
        for f, t in out:
            ts, tl = starts.get(t, (None, 0))
            if ts != src:
                rows.append((a, f, L, -L))
            elif tl != L:
                rows.append((a, f, L, tl))
            elif is_jump(f, jump):
                rows.append((a, f, L, -L))
        if not out:
            rows.append((a, EXEC, L, -L))
        if a in entry:
            rows.append((a, EXEC, -L, L))
    tables = {}
    for src, rows in lines.items():
        t = np.array(trans[src], np.int64).reshape(-1, 4)
        tables[src] = {
            "lines": np.array(rows, np.int64),
            "arcs": np.unique(t[:, 2:], axis=0),
            "transitions": t,
        }
    return tables


def write_sidecars(path, entries=(), root=None):
    """Write ``<source>.jcov`` next to each DWARF source of ``path`` that exists and return
    those source paths (for ``jennings.coverage.report.write_data``)."""
    found = {s: t for s, t in sidecars(path, entries, root).items() if os.path.isfile(s)}
    for src, tables in found.items():
        save_sidecar(src, **tables)
    return sorted(found)
