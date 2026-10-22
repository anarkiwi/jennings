"""Per-instruction data flow, read off the lifted pcode the MPU executes.

:func:`flow` maps every register, flag and memory byte an instruction sets to the
registers, flags and bytes its value came from; a branch maps ``"PC"`` to its condition.
"""

from deity_informant import lift, lifter

__all__ = ["REGISTERS", "flow"]

# PcodeVM register index -> name, from the lifter's own register varnodes (flags lose "F").
REGISTERS = {
    vn[1]: name[1:] if len(name) == 2 and name[0] == "F" else name
    for name, vn in vars(lifter).items()
    if isinstance(vn, list) and len(vn) == 3 and vn[0] == "r"
}


def flow(mem, pc, loads=(), stores=()):
    """``{output: frozenset(inputs)}`` of the instruction at ``pc`` in ``mem``.

    Locations are :data:`REGISTERS` names and memory addresses. ``loads``/``stores`` are
    the byte addresses it read and wrote, in access order (as an observed run reports
    them); a load depends on its address, a store's byte only on the value stored.
    """
    rec = lift(mem, pc)
    loads, stores = iter(loads), iter(stores)
    regs, uniq, out = {}, {}, {}

    def src(vn):
        space, off, _size = vn
        if space == "c":
            return frozenset()
        if space == "u":
            return uniq.get(off, frozenset())
        return regs.get(off, frozenset((REGISTERS.get(off, "r%d" % off),)))

    for mn, dst, ins in rec["ops"]:
        if mn == "STORE":
            for _ in range(ins[1][2]):
                addr = next(stores, None)
                if addr is not None:
                    out[addr] = src(ins[1])
            continue
        deps = frozenset().union(*map(src, ins))
        if mn == "LOAD":
            deps |= frozenset(
                a for a in (next(loads, None) for _ in range(dst[2])) if a is not None
            )
        if dst[0] == "r":
            regs[dst[1]] = deps
            out[REGISTERS.get(dst[1], "r%d" % dst[1])] = deps
        else:
            uniq[dst[1]] = deps
    ctrl = rec.get("ctrl") or ()
    if ctrl and ctrl[0] == "br":
        out["PC"] = src(ctrl[1])
    return out
