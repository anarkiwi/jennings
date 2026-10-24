"""The instructions a search draws from, their concrete semantics over batches of machine
states (numpy), and which pairs of them never belong in a minimal sequence."""

from dataclasses import dataclass

import numpy as np
from deity_informant import CYCLETIME

from jennings.opcodes import ILLEGAL_OPCODES, MODE_LEN, OPCODES

REGS = ("A", "X", "Y", "C", "Z", "N", "V")
IMPLIED = "TAX TAY TXA TYA INX INY DEX DEY CLC SEC".split()
FORMS = {
    "LDA": ("imm", "var", "tabx", "taby"),
    "LDX": ("imm", "var", "taby"),
    "LDY": ("imm", "var", "tabx"),
    "STA": ("var",),
    "STX": ("var",),
    "STY": ("var",),
    "ADC": ("imm", "var", "tabx", "taby"),
    "SBC": ("imm", "var", "tabx", "taby"),
    "AND": ("imm", "var", "tabx", "taby"),
    "ORA": ("imm", "var", "tabx", "taby"),
    "EOR": ("imm", "var", "tabx", "taby"),
    "CMP": ("imm", "var", "tabx", "taby"),
    "CPX": ("imm", "var"),
    "CPY": ("imm", "var"),
    "ASL": ("acc", "var"),
    "LSR": ("acc", "var"),
    "ROL": ("acc", "var"),
    "ROR": ("acc", "var"),
    "INC": ("var",),
    "DEC": ("var",),
    **{m: ("impl",) for m in IMPLIED},
}
STABLE_ILLEGAL = {
    "LAX": ("var", "taby"),
    "SAX": ("var",),
    "DCP": ("var",),
    "ISC": ("var",),
    "SLO": ("var",),
    "RLA": ("var",),
    "SRE": ("var",),
    "RRA": ("var",),
    "ANC": ("imm",),
    "ALR": ("imm",),
    "SBX": ("imm",),
}
MNEMONICS = tuple(FORMS) + tuple(STABLE_ILLEGAL)
_NZ = {"N", "Z"}
_EFFECTS = {
    "LDA": ((), ("A",)),
    "LDX": ((), ("X",)),
    "LDY": ((), ("Y",)),
    "STA": (("A",), ("m",)),
    "STX": (("X",), ("m",)),
    "STY": (("Y",), ("m",)),
    "TAX": (("A",), ("X",)),
    "TAY": (("A",), ("Y",)),
    "TXA": (("X",), ("A",)),
    "TYA": (("Y",), ("A",)),
    "ADC": (("A", "C"), ("A", "C", "V")),
    "SBC": (("A", "C"), ("A", "C", "V")),
    "AND": (("A",), ("A",)),
    "ORA": (("A",), ("A",)),
    "EOR": (("A",), ("A",)),
    "CMP": (("A",), ("C",)),
    "CPX": (("X",), ("C",)),
    "CPY": (("Y",), ("C",)),
    "ASL": (("t",), ("t", "C")),
    "LSR": (("t",), ("t", "C")),
    "ROL": (("t", "C"), ("t", "C")),
    "ROR": (("t", "C"), ("t", "C")),
    "INC": (("m",), ("m",)),
    "DEC": (("m",), ("m",)),
    "INX": (("X",), ("X",)),
    "INY": (("Y",), ("Y",)),
    "DEX": (("X",), ("X",)),
    "DEY": (("Y",), ("Y",)),
    "CLC": ((), ("C",)),
    "SEC": ((), ("C",)),
    "LAX": ((), ("A", "X")),
    "SAX": (("A", "X"), ("m",)),
    "DCP": (("m", "A"), ("m", "C")),
    "ISC": (("m", "A", "C"), ("m", "A", "C", "V")),
    "SLO": (("m", "A"), ("m", "A", "C")),
    "RLA": (("m", "A", "C"), ("m", "A", "C")),
    "SRE": (("m", "A"), ("m", "A", "C")),
    "RRA": (("m", "A", "C"), ("m", "A", "C", "V")),
    "ANC": (("A",), ("A", "C")),
    "ALR": (("A",), ("A", "C")),
    "SBX": (("A", "X"), ("X", "C")),
}
_NO_NZ = {"STA", "STX", "STY", "CLC", "SEC", "SAX"}
_READS_MEMORY = set("LDA LDX LDY LAX ADC SBC AND ORA EOR CMP CPX CPY".split())
_MODES = {"imm": "imm", "acc": "acc", "impl": "impl", "tabx": "absx", "taby": "absy"}
ENCODE = {}
for _op, (_m, _mode) in sorted(OPCODES.items(), key=lambda e: (e[0] in ILLEGAL_OPCODES, e[0])):
    ENCODE.setdefault((_m, _mode), _op)
OP_IDS = {m: k for k, m in enumerate(MNEMONICS)}
KINDS = {"impl": 0, "acc": 0, "imm": 1, "var": 2, "tabx": 3, "taby": 4}


@dataclass(frozen=True)
class Instruction:
    """An instruction of a search: its mnemonic, form (``impl``, ``acc``, ``imm``, ``var``,
    ``tabx`` or ``taby``) and operand (an immediate's index, a variable byte's slot, or a
    table's ``(index, offset)``)."""

    mnemonic: str
    form: str
    operand: object = None


class Alphabet:
    """The instructions over ``slots`` (variable byte addresses), ``immediates`` (count) and
    ``tables`` ``[(address, offsets)]``, legal ones only unless ``illegal``."""

    def __init__(self, slots, immediates: int, tables, illegal: bool = False):
        self.slots, self.tables = tuple(slots), tuple(tables)
        forms = dict(FORMS, **(STABLE_ILLEGAL if illegal else {}))
        out = []
        for mnemonic, kinds in forms.items():
            for form in kinds:
                if form == "imm":
                    out += [Instruction(mnemonic, form, k) for k in range(immediates)]
                elif form == "var":
                    out += [Instruction(mnemonic, form, s) for s in range(len(self.slots))]
                elif form in ("tabx", "taby"):
                    out += [
                        Instruction(mnemonic, form, (t, offset))
                        for t, (_, offsets) in enumerate(self.tables)
                        for offset in offsets
                    ]
                else:
                    out.append(Instruction(mnemonic, form))
        self.instructions = tuple(out)
        self.codes = np.array([self.code(i) for i in out], dtype=np.int32)
        self.sizes = np.array([self.size(i) for i in out], dtype=np.int32)
        self.cycles = np.array([self.cycle(i) for i in out], dtype=np.int32)

    def __len__(self) -> int:
        return len(self.instructions)

    def mode(self, ins: Instruction) -> str:
        """The 6502 addressing mode of ``ins``."""
        if ins.form == "var":
            return "zp" if self.slots[ins.operand] < 0x100 else "abs"
        return _MODES[ins.form]

    def opcode(self, ins: Instruction) -> int:
        """The opcode of ``ins``."""
        return ENCODE[(ins.mnemonic, self.mode(ins))]

    def size(self, ins: Instruction) -> int:
        """The bytes of ``ins``."""
        return MODE_LEN[self.mode(ins)]

    def cycle(self, ins: Instruction) -> int:
        """The worst-case cycles of ``ins`` (an indexed read off a page start +1)."""
        extra = 0
        if ins.form in ("tabx", "taby"):
            table, offset = ins.operand
            extra = int((self.tables[table][0] + offset) % 256 != 0)
        return CYCLETIME[self.opcode(ins)] + extra

    def code(self, ins: Instruction) -> tuple:
        """``(op, kind, a, b)``: the op id, operand kind and its two fields."""
        kind = KINDS[ins.form]
        if kind in (3, 4):
            return OP_IDS[ins.mnemonic], kind, ins.operand[0], ins.operand[1]
        return OP_IDS[ins.mnemonic], kind, ins.operand or 0, 0

    def encode(self, ins: Instruction, immediates) -> bytes:
        """The bytes of ``ins``, ``immediates`` the immediate bytes."""
        mode = self.mode(ins)
        out = [self.opcode(ins)]
        if ins.form == "imm":
            out.append(immediates[ins.operand] & 0xFF)
        elif ins.form == "var":
            address = self.slots[ins.operand]
            out += [address & 0xFF] + ([address >> 8] if mode == "abs" else [])
        elif ins.form in ("tabx", "taby"):
            table, offset = ins.operand
            address = self.tables[table][0] + offset
            out += [address & 0xFF, address >> 8]
        return bytes(out)

    def effects(self, ins: Instruction) -> tuple:
        """``(reads, writes)``: the locations ``ins`` reads and writes (registers, flags,
        ``("m", slot)``)."""
        reads, writes = (set(x) for x in _EFFECTS[ins.mnemonic])
        cell = ("m", ins.operand)
        target = cell if ins.form == "var" else "A"
        for group in (reads, writes):
            for token, value in (("m", cell), ("t", target)):
                if token in group:
                    group.discard(token)
                    group.add(value)
        if ins.form == "var" and ins.mnemonic in _READS_MEMORY:
            reads.add(cell)
        if ins.form in ("tabx", "taby"):
            reads.add(ins.form[-1].upper())
        if ins.mnemonic not in _NO_NZ:
            writes |= _NZ
        return frozenset(reads), frozenset(writes)

    def pruning(self, outputs) -> tuple:
        """``(allowed, last)``: ``allowed[i, j]`` False where ``j`` after ``i`` makes ``i``
        dead or the pair is independent and out of order; ``last[j]`` where ``j`` may end a
        sequence (it writes one of ``outputs``)."""
        effects = [self.effects(i) for i in self.instructions]
        n = len(effects)
        allowed = np.ones((n, n), dtype=np.uint8)
        for i, (ri, wi) in enumerate(effects):
            for j, (rj, wj) in enumerate(effects):
                dead = wi <= wj and not rj & wi
                independent = not wi & (rj | wj) and not wj & ri
                if dead or (independent and j <= i):
                    allowed[i, j] = 0
        last = np.array([bool(w & set(outputs)) for _, w in effects], dtype=np.uint8)
        return allowed, last


def _nz(state, value):
    state["N"] = (value >> 7) & 1
    state["Z"] = (value == 0).astype(value.dtype)


def _adc(state, value):
    a = state["A"]
    total = a + value + state["C"]
    result = total & 0xFF
    state["V"] = ((~(a ^ value) & (a ^ result) & 0x80) != 0).astype(a.dtype)
    state["C"] = (total > 0xFF).astype(a.dtype)
    state["A"] = result
    _nz(state, result)


def _compare(state, reg, value):
    state["C"] = (reg >= value).astype(reg.dtype)
    _nz(state, (reg - value) & 0xFF)


def _shift(state, value, mnemonic):
    carry = state["C"]
    if mnemonic in ("ASL", "ROL", "SLO", "RLA"):
        result = ((value << 1) | (carry if mnemonic in ("ROL", "RLA") else 0)) & 0xFF
        state["C"] = (value >> 7) & 1
    else:
        result = (value >> 1) | ((carry << 7) if mnemonic in ("ROR", "RRA") else 0)
        state["C"] = value & 1
    return result


def apply(state: dict, code, rows, operand, slot_of=None) -> None:
    """Applies instruction ``code`` (an :meth:`Alphabet.code`) to the batch ``rows`` of
    ``state`` (arrays of registers and flags, ``M`` the variable bytes ``[row, slot]``),
    ``operand`` the operand byte of each row; a store or read-modify-write writes slot
    ``code[2]``."""
    # pylint: disable=too-many-branches,too-many-statements
    mnemonic = MNEMONICS[code[0]]
    sub = {k: v[rows] for k, v in state.items()}
    slot = code[2]
    if mnemonic in ("LDA", "LDX", "LDY", "LAX"):
        for reg in {"LDA": "A", "LDX": "X", "LDY": "Y", "LAX": "AX"}[mnemonic]:
            sub[reg] = operand
        _nz(sub, operand)
    elif mnemonic in ("STA", "STX", "STY", "SAX"):
        value = {"STA": sub["A"], "STX": sub["X"], "STY": sub["Y"]}.get(mnemonic)
        sub["M"][:, slot] = value if value is not None else sub["A"] & sub["X"]
    elif mnemonic in ("TAX", "TAY", "TXA", "TYA"):
        sub[mnemonic[2]] = sub[mnemonic[1]]
        _nz(sub, sub[mnemonic[2]])
    elif mnemonic in ("ADC", "SBC"):
        _adc(sub, operand if mnemonic == "ADC" else operand ^ 0xFF)
    elif mnemonic in ("AND", "ORA", "EOR"):
        a = sub["A"]
        sub["A"] = {"AND": a & operand, "ORA": a | operand, "EOR": a ^ operand}[mnemonic]
        _nz(sub, sub["A"])
    elif mnemonic in ("CMP", "CPX", "CPY"):
        _compare(sub, sub[{"CMP": "A", "CPX": "X", "CPY": "Y"}[mnemonic]], operand)
    elif mnemonic in ("ASL", "LSR", "ROL", "ROR"):
        value = sub["A"] if code[1] == 0 else sub["M"][:, slot]
        result = _shift(sub, value, mnemonic)
        if code[1] == 0:
            sub["A"] = result
        else:
            sub["M"][:, slot] = result
        _nz(sub, result)
    elif mnemonic in ("INC", "DEC"):
        result = (sub["M"][:, slot] + (1 if mnemonic == "INC" else 255)) & 0xFF
        sub["M"][:, slot] = result
        _nz(sub, result)
    elif mnemonic in ("INX", "INY", "DEX", "DEY"):
        reg = mnemonic[2]
        sub[reg] = (sub[reg] + (1 if mnemonic[0] == "I" else 255)) & 0xFF
        _nz(sub, sub[reg])
    elif mnemonic in ("CLC", "SEC"):
        sub["C"] = np.full_like(sub["C"], int(mnemonic == "SEC"))
    elif mnemonic in ("DCP", "ISC"):
        result = (sub["M"][:, slot] + (255 if mnemonic == "DCP" else 1)) & 0xFF
        sub["M"][:, slot] = result
        if mnemonic == "DCP":
            _compare(sub, sub["A"], result)
        else:
            _adc(sub, result ^ 0xFF)
    elif mnemonic in ("SLO", "RLA", "SRE", "RRA"):
        result = _shift(sub, sub["M"][:, slot], mnemonic)
        sub["M"][:, slot] = result
        if mnemonic == "RRA":
            _adc(sub, result)
        else:
            a = sub["A"]
            sub["A"] = {"SLO": a | result, "RLA": a & result, "SRE": a ^ result}[mnemonic]
            _nz(sub, sub["A"])
    elif mnemonic == "ANC":
        sub["A"] = sub["A"] & operand
        _nz(sub, sub["A"])
        sub["C"] = sub["N"].copy()
    elif mnemonic == "ALR":
        value = sub["A"] & operand
        sub["C"] = value & 1
        sub["A"] = value >> 1
        _nz(sub, sub["A"])
    elif mnemonic == "SBX":
        value = sub["A"] & sub["X"]
        _compare(sub, value, operand)
        sub["X"] = (value - operand) & 0xFF
    for k, v in sub.items():
        state[k][rows] = v
