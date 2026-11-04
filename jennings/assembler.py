"""Assemble a single 6502/6510 statement to bytes (py65-compatible).

Reverse-looks-up ``(mnemonic, mode)`` in the MPU ``disassemble`` table, so in
jennings every illegal mnemonic (SLO, LAX, ISC, ...) assembles like a legal one.
"""

import re

from jennings.opcodes import ILLEGAL_OPCODES
from jennings.utils.addressing import AddressParser


class Assembler:
    Statement = re.compile(r"^([A-z]{3}[0-7]?\s+\(?\s*)([^,\s\)]+)(\s*[,xXyY\s]*\)?[,xXyY\s]*)$")

    Label = re.compile(r"^\s*([A-Za-z_.][\w.]*)\s*:(.*)$")

    Addressing = (
        ("zpi", "($00FF)"),
        ("zpx", "$00FF,X"),
        ("zpy", "$00FF,Y"),
        ("zpg", "$00FF"),
        ("inx", "($00FF,X)"),
        ("iax", "($FFFF,X)"),
        ("iny", "($00FF),Y"),
        ("ind", "($FFFF)"),
        ("abx", "$FFFF,X"),
        ("aby", "$FFFF,Y"),
        ("abs", "$FFFF"),
        ("rel", "$FFFF"),
        ("imp", ""),
        ("acc", ""),
        ("acc", "A"),
        ("imm", "#$FF"),
    )

    def __init__(self, mpu, address_parser=None):
        self._mpu = mpu
        self._address_parser = address_parser or AddressParser()
        self._addressing = []
        numchars = mpu.BYTE_WIDTH // 4
        for mode, fmt in self.Addressing:
            pat = "^" + re.escape(fmt) + "$"
            pat = pat.replace("00", "0{%d}" % numchars)
            pat = pat.replace("FF", "([0-9A-F]{%d})" % numchars)
            self._addressing.append([mode, re.compile(pat)])
        # (mnemonic, mode) -> opcode, preferring the legal opcode (NOP -> 0xEA, not 0x1A).
        self._encode = {}
        for op, entry in enumerate(mpu.disassemble):
            cur = self._encode.get(entry)
            if cur is None or (cur in ILLEGAL_OPCODES and op not in ILLEGAL_OPCODES):
                self._encode[entry] = op

    def assemble(self, statement, pc=0x0000, check_range=False):
        """Assemble ``statement`` to a list of bytes; raise on failure.

        ``check_range`` makes a branch beyond -128..127 raise ``OverflowError``
        (py65 wraps it silently)."""
        opcode, operand = self.normalize_and_split(statement)
        for mode, pattern in self._addressing:
            match = pattern.match(operand)
            if not match:
                continue
            op = self._encode.get((opcode, mode))
            if op is None:
                continue
            result = [op]
            operands = match.groups()
            if mode == "rel":
                absolute = int("".join(operands), 16)
                offset = (absolute - pc) - 2
                half = (self._mpu.byteMask + 1) // 2
                if check_range and not -half <= offset < half:
                    raise OverflowError("branch out of range: %s" % statement)
                relative = offset & self._mpu.byteMask
                operands = [self._mpu.BYTE_FORMAT % relative]
            elif len(operands) == 2:
                operands = (operands[1], operands[0])
            result.extend(int(h, 16) for h in operands)
            if (pc + len(result)) > (1 << self._mpu.ADDR_WIDTH):
                raise OverflowError
            return result
        raise SyntaxError(statement)

    def assemble_program(self, text, origin=0x0000):
        """Assemble ``;`` or newline separated statements to ``(bytes, {label: address})``.

        A label is ``name:`` alone or before a statement. Operands take labels wherever
        they take numbers (``name``, ``name+1``, ``#<name``, ``#>name``). Passes repeat
        with the previous pass's addresses until they agree; the first assumes every
        label is at ``origin``, so an operand is zero page only while its label stays on
        page zero, and addresses only move up: the smallest consistent layout. Raises
        ``ValueError`` for a duplicate label, ``KeyError`` for an undefined one,
        ``OverflowError`` for a branch out of range."""
        statements = []
        for line in re.split(r"[;\n]", text):
            while match := self.Label.match(line):
                statements.append((match.group(1), None))
                line = match.group(2)
            if line.strip():
                statements.append((None, line.strip()))
        names = [name for name, _ in statements if name]
        for name in set(names):
            if names.count(name) > 1:
                raise ValueError("duplicate label: %s" % name)
        parser, saved = self._address_parser, self._address_parser.labels
        try:
            parser.labels, strict = dict.fromkeys(names, origin), False
            while True:
                code, found = bytearray(), {}
                for name, statement in statements:
                    if name:
                        found[name] = origin + len(code)
                    else:
                        code += bytes(self.assemble(statement, origin + len(code), strict))
                if strict:
                    return bytes(code), found
                strict, parser.labels = found == parser.labels, found
        finally:
            parser.labels = saved

    def normalize_and_split(self, statement):
        """Normalise ``statement`` and split into ``(opcode, operand)`` strings."""
        statement = " ".join(statement.split())
        match = self.Statement.match(statement)
        if match:
            before, target, after = match.groups()
            if target.startswith("#"):
                try:
                    if target[1] in "<>":
                        number = self._address_parser.number(target[2:])
                        number = number & self._mpu.byteMask if target[1] == "<" else number >> 8
                    elif target[1] in ("'", '"'):
                        number = ord(target[2])
                    else:
                        number = self._address_parser.number(target[1:])
                except IndexError:
                    raise SyntaxError(statement)
                if number < 0 or number > self._mpu.byteMask:
                    raise OverflowError
                statement = before + "#$" + self._mpu.BYTE_FORMAT % number
            elif target in ("a", "A"):
                pass
            else:
                address = self._address_parser.number(target)
                statement = before + "$" + self._mpu.ADDR_FORMAT % address + after
        splitted = statement.split(" ", 2)
        opcode = splitted[0].strip().upper()
        operand = splitted[1].strip().upper() if len(splitted) > 1 else ""
        return (opcode, operand)
