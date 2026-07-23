"""Canonical 6502/6510 opcode table: the single source of truth for the ISA.

Maps every opcode byte to ``(mnemonic, mode)`` (all 151 legal + 105 NMOS illegal),
with per-mode instruction lengths and the illegal-opcode set. This module is
dependency-free so deity-informant's lifter can import it without a cycle.
"""

MODE_LEN = {
    "imm": 2,
    "zp": 2,
    "zpx": 2,
    "zpy": 2,
    "abs": 3,
    "absx": 3,
    "absy": 3,
    "indx": 2,
    "indy": 2,
    "acc": 1,
    "impl": 1,
    "rel": 2,
    "ind": 3,
}
MEM_MODES = {"zp", "zpx", "zpy", "abs", "absx", "absy", "indx", "indy"}

# deity-informant addressing-mode name -> py65 disassembler/assembler mode token.
MODE_TOKEN = {
    "imm": "imm",
    "zp": "zpg",
    "zpx": "zpx",
    "zpy": "zpy",
    "abs": "abs",
    "absx": "abx",
    "absy": "aby",
    "indx": "inx",
    "indy": "iny",
    "acc": "acc",
    "impl": "imp",
    "rel": "rel",
    "ind": "ind",
}


def _build_ops():
    t = {}
    L = [
        (0x00, "BRK", "impl"),
        (0x08, "PHP", "impl"),
        (0x28, "PLP", "impl"),
        (0x48, "PHA", "impl"),
        (0x68, "PLA", "impl"),
        (0x40, "RTI", "impl"),
        (0x60, "RTS", "impl"),
        (0x20, "JSR", "abs"),
        (0x4C, "JMP", "abs"),
        (0x6C, "JMP", "ind"),
        (0xEA, "NOP", "impl"),
        (0x18, "CLC", "impl"),
        (0x38, "SEC", "impl"),
        (0x58, "CLI", "impl"),
        (0x78, "SEI", "impl"),
        (0xB8, "CLV", "impl"),
        (0xD8, "CLD", "impl"),
        (0xF8, "SED", "impl"),
        (0xAA, "TAX", "impl"),
        (0xA8, "TAY", "impl"),
        (0x8A, "TXA", "impl"),
        (0x98, "TYA", "impl"),
        (0xBA, "TSX", "impl"),
        (0x9A, "TXS", "impl"),
        (0xCA, "DEX", "impl"),
        (0x88, "DEY", "impl"),
        (0xE8, "INX", "impl"),
        (0xC8, "INY", "impl"),
    ]
    for op, mn, md in L:
        t[op] = (mn, md)
    for op, md in (
        (0x10, "BPL"),
        (0x30, "BMI"),
        (0x50, "BVC"),
        (0x70, "BVS"),
        (0x90, "BCC"),
        (0xB0, "BCS"),
        (0xD0, "BNE"),
        (0xF0, "BEQ"),
    ):
        t[op] = (md, "rel")

    def add(mn, entries):
        for op, md in entries:
            t[op] = (mn, md)

    add(
        "ORA",
        [
            (0x09, "imm"),
            (0x05, "zp"),
            (0x15, "zpx"),
            (0x0D, "abs"),
            (0x1D, "absx"),
            (0x19, "absy"),
            (0x01, "indx"),
            (0x11, "indy"),
        ],
    )
    add(
        "AND",
        [
            (0x29, "imm"),
            (0x25, "zp"),
            (0x35, "zpx"),
            (0x2D, "abs"),
            (0x3D, "absx"),
            (0x39, "absy"),
            (0x21, "indx"),
            (0x31, "indy"),
        ],
    )
    add(
        "EOR",
        [
            (0x49, "imm"),
            (0x45, "zp"),
            (0x55, "zpx"),
            (0x4D, "abs"),
            (0x5D, "absx"),
            (0x59, "absy"),
            (0x41, "indx"),
            (0x51, "indy"),
        ],
    )
    add(
        "ADC",
        [
            (0x69, "imm"),
            (0x65, "zp"),
            (0x75, "zpx"),
            (0x6D, "abs"),
            (0x7D, "absx"),
            (0x79, "absy"),
            (0x61, "indx"),
            (0x71, "indy"),
        ],
    )
    add(
        "SBC",
        [
            (0xE9, "imm"),
            (0xE5, "zp"),
            (0xF5, "zpx"),
            (0xED, "abs"),
            (0xFD, "absx"),
            (0xF9, "absy"),
            (0xE1, "indx"),
            (0xF1, "indy"),
        ],
    )
    add(
        "CMP",
        [
            (0xC9, "imm"),
            (0xC5, "zp"),
            (0xD5, "zpx"),
            (0xCD, "abs"),
            (0xDD, "absx"),
            (0xD9, "absy"),
            (0xC1, "indx"),
            (0xD1, "indy"),
        ],
    )
    add(
        "LDA",
        [
            (0xA9, "imm"),
            (0xA5, "zp"),
            (0xB5, "zpx"),
            (0xAD, "abs"),
            (0xBD, "absx"),
            (0xB9, "absy"),
            (0xA1, "indx"),
            (0xB1, "indy"),
        ],
    )
    add(
        "STA",
        [
            (0x85, "zp"),
            (0x95, "zpx"),
            (0x8D, "abs"),
            (0x9D, "absx"),
            (0x99, "absy"),
            (0x81, "indx"),
            (0x91, "indy"),
        ],
    )
    add("LDX", [(0xA2, "imm"), (0xA6, "zp"), (0xB6, "zpy"), (0xAE, "abs"), (0xBE, "absy")])
    add("LDY", [(0xA0, "imm"), (0xA4, "zp"), (0xB4, "zpx"), (0xAC, "abs"), (0xBC, "absx")])
    add("STX", [(0x86, "zp"), (0x96, "zpy"), (0x8E, "abs")])
    add("STY", [(0x84, "zp"), (0x94, "zpx"), (0x8C, "abs")])
    add("CPX", [(0xE0, "imm"), (0xE4, "zp"), (0xEC, "abs")])
    add("CPY", [(0xC0, "imm"), (0xC4, "zp"), (0xCC, "abs")])
    add("BIT", [(0x24, "zp"), (0x2C, "abs")])
    add("ASL", [(0x0A, "acc"), (0x06, "zp"), (0x16, "zpx"), (0x0E, "abs"), (0x1E, "absx")])
    add("LSR", [(0x4A, "acc"), (0x46, "zp"), (0x56, "zpx"), (0x4E, "abs"), (0x5E, "absx")])
    add("ROL", [(0x2A, "acc"), (0x26, "zp"), (0x36, "zpx"), (0x2E, "abs"), (0x3E, "absx")])
    add("ROR", [(0x6A, "acc"), (0x66, "zp"), (0x76, "zpx"), (0x6E, "abs"), (0x7E, "absx")])
    add("INC", [(0xE6, "zp"), (0xF6, "zpx"), (0xEE, "abs"), (0xFE, "absx")])
    add("DEC", [(0xC6, "zp"), (0xD6, "zpx"), (0xCE, "abs"), (0xDE, "absx")])
    rmw = {0x00: "SLO", 0x20: "RLA", 0x40: "SRE", 0x60: "RRA", 0xC0: "DCP", 0xE0: "ISC"}
    for base, mn in rmw.items():
        for lowoff, md in (
            (0x03, "indx"),
            (0x07, "zp"),
            (0x0F, "abs"),
            (0x13, "indy"),
            (0x17, "zpx"),
            (0x1B, "absy"),
            (0x1F, "absx"),
        ):
            t[base + lowoff] = (mn, md)
    add("SAX", [(0x83, "indx"), (0x87, "zp"), (0x8F, "abs"), (0x97, "zpy")])
    add(
        "LAX",
        [
            (0xA3, "indx"),
            (0xA7, "zp"),
            (0xAF, "abs"),
            (0xB3, "indy"),
            (0xB7, "zpy"),
            (0xBF, "absy"),
        ],
    )
    add("LXA", [(0xAB, "imm")])
    add("ANC", [(0x0B, "imm"), (0x2B, "imm")])
    add("ALR", [(0x4B, "imm")])
    add("ARR", [(0x6B, "imm")])
    add("SBX", [(0xCB, "imm")])
    add("SBC", [(0xEB, "imm")])
    add("ANE", [(0x8B, "imm")])
    add("LAS", [(0xBB, "absy")])
    add("SHA", [(0x9F, "absy"), (0x93, "indy")])
    add("SHX", [(0x9E, "absy")])
    add("SHY", [(0x9C, "absx")])
    add("TAS", [(0x9B, "absy")])
    for op in (0x1A, 0x3A, 0x5A, 0x7A, 0xDA, 0xFA):
        t[op] = ("NOP", "impl")
    for op in (0x80, 0x82, 0x89, 0xC2, 0xE2):
        t[op] = ("NOP", "imm")
    for op in (0x04, 0x44, 0x64):
        t[op] = ("NOP", "zp")
    for op in (0x14, 0x34, 0x54, 0x74, 0xD4, 0xF4):
        t[op] = ("NOP", "zpx")
    t[0x0C] = ("NOP", "abs")
    for op in (0x1C, 0x3C, 0x5C, 0x7C, 0xDC, 0xFC):
        t[op] = ("NOP", "absx")
    for op in (0x02, 0x12, 0x22, 0x32, 0x42, 0x52, 0x62, 0x72, 0x92, 0xB2, 0xD2, 0xF2):
        t[op] = ("JAM", "impl")
    return t


OPCODES = _build_ops()
OPS = OPCODES  # deity-informant compatibility alias

_ILLEGAL_COMBO = {
    "SLO",
    "RLA",
    "SRE",
    "RRA",
    "DCP",
    "ISC",
    "SAX",
    "LAX",
    "LXA",
    "ANC",
    "ALR",
    "ARR",
    "SBX",
    "ANE",
    "LAS",
    "SHA",
    "SHX",
    "SHY",
    "TAS",
}


def _is_illegal(op):
    mn, _ = OPCODES[op]
    if mn in _ILLEGAL_COMBO or mn == "JAM":
        return True
    if mn == "SBC" and op == 0xEB:
        return True
    if mn == "NOP" and op != 0xEA:
        return True
    return False


ILLEGAL_OPCODES = frozenset(o for o in OPCODES if _is_illegal(o))
