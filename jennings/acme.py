# pylint: disable=too-many-lines
"""ACME 0.97 compatible cross-assembler for the 6502/6510 subset of ACME sources.

Mirrors ACME's passes, operand-size flags and symbol trees, so output files and ``-l``
symbol lists are byte-identical; unsupported features raise :class:`AcmeError`.
"""

import argparse
import operator
import re
import sys

from jennings.assembler import encoding
from jennings.opcodes import ILLEGAL_OPCODES, MODE_TOKEN, OPCODES

__all__ = ["Acme", "AcmeError", "assemble", "main", "mnemonics", "read_symbol_list"]

F8, F16, F24 = 1, 2, 4
FORCE = F8 | F16 | F24
FITS = 8
EVER = 16
EOS, SOL, EOB, EOF = "\0", "\n", "}", "\x04"
MASK64 = (1 << 64) - 1

_LEX = re.compile(r"""(["'])(?:\\.|(?!\1)[^\\\r\n])*\1?|(?:;|//)[^\r\n]*|[ \t]+|\r\n?|\n|[:}]""")
_QUOTED = re.compile(r"""(["'])((?:\\[^\0]|(?!\1)[^\\\0])*)(\1?)""")
_KEYWORD = re.compile(r"[0-9A-Za-z_\x80-\xff]*")
_RUNS = {c: re.compile(re.escape(c) + "+") for c in "+-"}
_PLUSES = re.compile(r"\+*")
_DECIMAL = re.compile(r"[0-9]+")
_DIGIT_RUNS = {16: re.compile(r"[0-9a-fA-F]*"), 2: re.compile(r"[01.#]*"), 8: re.compile(r"[0-7]*")}
_HEX_PAIR = re.compile(r"[0-9a-fA-F]{0,2}")
_ESCAPES = {"\\": "\\", "'": "'", '"': '"', "0": "\0", "t": "\t", "n": "\n", "r": "\r"}
_LITERALS = {"$": (16, 2, 4), "%": (2, 8, 16), "&": (8, 3, 6)}
_PRIORITIES = (
    "37:^;34:* / div % mod;32:+ -;30:<< asl lsl >> asr >>> lsr;26:<= < >= >;24:!= <> ><;"
    "22:= ==;20:& and;18:xor eor;16:| or"
)
_DYADIC = {
    op: int(prio)
    for prio, ops in (group.split(":") for group in _PRIORITIES.split(";"))
    for op in ops.split()
}
_ALIASES = {"div": "/", "mod": "%", "asl": "<<", "lsl": "<<", "asr": ">>", "lsr": ">>>"}
_ALIASES.update({"and": "&", "or": "|", "eor": "xor", "<>": "!=", "><": "!=", "==": "="})
_COMPARE = {"<=": operator.le, "<": operator.lt, ">=": operator.ge, ">": operator.gt}
_COMPARE.update({"!=": operator.ne, "=": operator.eq})
_ARITHMETIC = {"*": operator.mul, "+": operator.add, "-": operator.sub, "&": operator.and_}
_ARITHMETIC.update({"|": operator.or_, "xor": operator.xor})
_ARITHMETIC.update({"<<": lambda a, b: a << (b & 63), ">>": lambda a, b: a >> (b & 63)})
_ARITHMETIC[">>>"] = lambda a, b: (a & MASK64) >> (b & 63)
_SYMBOLIC = re.compile(r">>>|<<|>>|<=|>=|<>|><|==|!=|[-+*/%&|^<>=!]")
_SYNTAX = {
    token: (syntax, int(size))
    for token, syntax, size in (
        entry.split(":")
        for entry in "imp:imp:0 acc:imp:0 imm:imm:1 zpg:abs:1 abs:abs:2 zpx:x:1 abx:x:2 zpy:y:1 "
        "aby:y:2 inx:xind:1 iny:indy:1 ind:ind:2 rel:rel:1".split()
    )
}
_ADDRESSING = {
    (False, None, None): "abs",
    (False, None, "x"): "x",
    (False, None, "y"): "y",
    (True, None, None): "ind",
    (True, "x", None): "xind",
    (True, None, "y"): "indy",
}
_FORMATS = ("plain", "cbm", "apple")
_DATA_SPEC = (
    "by byte 8 08:1;wo word 16 le16:2;be16:2,1;24 le24:3;be24:3,1;32 le32:4;be32:4,1;tx text:1,0,1"
)
_DATA = {
    name: tuple(int(x) for x in args.split(","))
    for names, args in (group.split(":") for group in _DATA_SPEC.split(";"))
    for name in names.split()
}
_PSEUDO = {"to": "to", "cpu": "cpu", "pseudopc": "pseudopc", "macro": "macro"}
_PSEUDO.update(h="hex", hex="hex", fi="fill", fill="fill")
_ACME_ONLY = frozenset(
    "initmem xor ct convtab cbm raw pet scr scrxor bin binary skip align realpc al as rl rs addr "
    "address set sl symbollist zn zone sz subzone src source if ifdef ifndef for do while "
    "warn error serious eof endoffile".split()
)


def _table(legal_only):
    """``{mnemonic: {syntax: {size: opcode}}}`` of ACME's 6502 or NMOS 6510 instruction set."""
    table = {}
    entries = [(m, MODE_TOKEN[md]) for m, md in (OPCODES[op] for op in range(256))]
    for (mnemonic, mode), op in encoding(entries).items():
        if not (legal_only and op in ILLEGAL_OPCODES):
            syntax, size = _SYNTAX[mode]
            table.setdefault(mnemonic.lower(), {}).setdefault(syntax, {})[size] = op
    if not legal_only:
        table["asr"] = table["alr"]
    return table


_CPUS = {"6502": _table(True), "6510": _table(False)}
_CPUS["nmos6502"] = _CPUS["6510"]
_OTHER_CPUS = frozenset("65c02 r65c02 w65c02 65816 65ce02 4502 m65 c64dtv2".split())


def mnemonics(cpu):
    """``{mnemonic: {syntax: {size: opcode}}}`` ACME assembles for ``cpu``."""
    return Acme._cpu(cpu)


_SYMBOL_LINE = re.compile(r"^\s*([^\s+=]+)(?:\+[23])?\s*=\s*\$([0-9a-fA-F]+)", re.M)


def read_symbol_list(text):
    """``{name: value}`` of an ACME ``-l``/``--symbollist`` file (:meth:`Acme.symbol_list`):
    each ``name = $hex`` line, a ``+2``/``+3`` size suffix dropped; undefined (``?``) symbols
    are left out."""
    return {m[1]: int(m[2], 16) for m in _SYMBOL_LINE.finditer(text)}


class AcmeError(Exception):
    """An ACME source error, or a feature this implementation does not support."""


class _Symbol:
    __slots__ = ("hash", "scope", "gt", "le", "name", "kind", "val", "flags", "read")

    def __init__(self, hash_value, scope, name):
        self.hash, self.scope, self.name = hash_value, scope, name
        self.gt = self.le = self.kind = self.val = None
        self.flags, self.read = 0, False


def _hash(name):
    """ACME's 32-bit rotate-xor hash of ``name`` (bytes as signed chars)."""
    value = 0
    for b in name.encode("latin-1"):
        value = (((value << 7) | (value >> 25)) & 0xFFFFFFFF) ^ (b if b < 0x80 else b | 0xFFFFFF00)
    return value


def _fold(value):
    value ^= value >> 16
    value ^= value >> 8
    return value & 0xFF


def _s64(value):
    value &= MASK64
    return value - (1 << 64) if value >> 63 else value


def _cdiv(a, b):
    q = abs(a) // abs(b)
    return -q if (a < 0) != (b < 0) else q


def _starts_keyword(c):
    return ("a" <= c.lower() <= "z") or c == "_" or c >= "\x80"


def _is_digit(c):
    return "0" <= c <= "9"


def _lex(text):
    """ACME's processed byte stream of ``text``: one blank per run, EOS for ``:``, ``;``
    and ``//`` comments, EOS+SOL per line break, EOS+EOB per ``}``, quotes kept raw."""

    def sub(match):
        token = match.group(0)
        c = token[0]
        if c in "\"'":
            return token
        if c in ";/":
            return EOS
        if c in " \t":
            return " "
        if c in "\r\n":
            return EOS + SOL
        return EOS if c == ":" else EOS + EOB

    if text.startswith("#"):
        text = ";" + text[1:]
    return _LEX.sub(sub, text) + EOS + EOF


def _unescape(raw):
    out, chars = [], iter(raw)
    for c in chars:
        if c == "\\":
            c = next(chars)
            if c not in _ESCAPES:
                raise AcmeError("Unsupported backslash sequence.")
            c = _ESCAPES[c]
        out.append(c)
    return "".join(out)


class Acme:
    """Assemble ACME ``sources`` (``[(name, text)]``, processed in order every pass).

    ``cpu`` is the default ``!cpu``; ``fmt`` the output format (``plain``, ``cbm``,
    ``apple``, or ``None``: plain unless ``!to`` picks one); ``outname`` the ``-o``
    file name, which makes ``!to`` only warn, as in ACME."""

    def __init__(self, sources, cpu="6502", fmt=None, outname=None):
        if fmt not in (None,) + _FORMATS:
            raise AcmeError("Unknown output format: %s" % fmt)
        self.default_cpu = self._cpu(cpu)
        self.fmt, self.outname = fmt, outname
        self.streams = [(name, _lex(text)) for name, text in sources]
        self.forest, self.symbol_table, self.macros = [None] * 256, {}, {}
        self.mem = bytearray(0x10000)
        self.passno, self.complain, self.depth, self.open = 0, False, 0, 0
        self.name = self.s = ""
        self.i = self.pc = self.add = self.wi = self.lo = self.hi = self.undef = 0
        self.scope = self.scope_max = 0
        self.cpu, self.pc_defined, self.out_on, self.counters = self.default_cpu, False, False, {}

    def run(self):
        """Pass until no result is undefined; if that stalls, a last pass names the culprit."""
        self._pass()
        before = self.undef + 1
        while self.undef and self.undef < before:
            before = self.undef
            self._pass()
        if self.undef:
            self.complain = True
            self._pass()
            raise AcmeError("Value not defined.")
        return self

    def _pass(self):
        self.pc, self.pc_defined, self.add, self.wi, self.out_on = 0, False, 0, 0, False
        self.lo, self.hi, self.undef, self.counters = 0xFFFF, 0, 0, {}
        self.cpu, self.scope_max, self.scope = self.default_cpu, 2, 2
        for self.name, self.s in self.streams:
            self.i = -1
            try:
                self._block()
                if self.c != EOF:
                    raise AcmeError("Found '}' instead of end-of-file.")
            except AcmeError as e:
                line = self.s.count(SOL, 0, max(self.i, 0)) + 1
                raise AcmeError("%s:%d: %s" % (self.name, line, e)) from None
        self.passno += 1

    def image(self):
        """``(start, bytes)`` of the assembled memory, lowest to highest written byte."""
        if self.hi < self.lo:
            return 0, b""
        return self.lo, bytes(self.mem[self.lo : self.hi + 1])

    def output(self):
        """The output file contents in the selected format."""
        start, data = self.image()
        if self.fmt == "cbm":
            return start.to_bytes(2, "little") + data
        if self.fmt == "apple":
            return start.to_bytes(2, "little") + (len(data) & 0xFFFF).to_bytes(2, "little") + data
        return data

    def _dump_order(self):
        for root in self.forest:
            stack = [root] if root else []
            while stack:
                node = stack.pop()
                if node.scope == 0 and node.kind == "num":
                    yield node
                stack.extend(n for n in (node.le, node.gt) if n)

    def symbols(self):
        """``{name: value}`` of the global symbols, in ACME's symbol list order."""
        return {s.name: s.val for s in self._dump_order()}

    def symbol_list(self):
        """The ``-l``/``--symbollist`` file text."""
        lines = []
        for s in self._dump_order():
            force = s.flags & FORCE
            size = "+2" if force == F16 else "+3" if force in (F24, F16 | F24) else ""
            value = " ?" if s.val is None else "$%x" % (s.val & 0xFFFFFFFF)
            ever = "\t; ?" if s.flags & EVER else ""
            unused = "" if s.read else "\t; unused"
            lines.append("\t%s%s\t= %s%s%s\n" % (s.name, size, value, ever, unused))
        return "".join(lines)

    def _find(self, scope, name):
        sym = self.symbol_table.get((scope, name))
        if sym is None:
            hash_value = _hash(name) ^ scope
            sym = self.symbol_table[scope, name] = _Symbol(hash_value, scope, name)
            slot = _fold(hash_value)
            node = self.forest[slot]
            if node is None:
                self.forest[slot] = sym
            while node:
                attr = "gt" if hash_value > node.hash else "le"
                nxt = getattr(node, attr)
                if nxt is None:
                    setattr(node, attr, sym)
                node = nxt
        return sym

    def _read(self, scope, name, shown):
        sym = self._find(scope, name)
        sym.read = True
        if sym.kind is None:
            sym.kind, sym.val, sym.flags = "num", None, EVER
        if sym.val is None and self.complain:
            raise AcmeError("Value not defined (%s)." % shown)
        return sym.val, sym.flags

    @staticmethod
    def _assign(sym, number, change=False):
        val, flags = number
        if sym.kind is None:
            sym.kind, sym.val, sym.flags = "num", val, flags
            return
        own = sym.flags
        if change:
            sym.val, own = None, own & ~FITS
        if sym.val is None:
            sym.val = val
        elif sym.val != val:
            raise AcmeError("Symbol already defined.")
        if not own & FORCE:
            own |= flags & FORCE
        if own & (EVER | FITS) == EVER:
            flags &= ~FITS
        sym.flags = own | (flags & (FITS | EVER))

    @staticmethod
    def _force_symbol(sym, force):
        if not sym.read:
            sym.flags = (sym.flags & ~FORCE) | force
        elif sym.flags & FORCE != force:
            raise AcmeError("Too late for postfix.")

    def _label(self, scope, name, force=0, change=False):
        sym = self._find(scope, name)
        self._assign(sym, (self.pc, 0), change)
        if force:
            self._force_symbol(sym, force)

    def _anon(self, name, increment):
        """Internal name of the forward anonymous label ``name`` refers to next."""
        self._find(self.scope, name)
        number = self.counters.get((self.scope, name), 0)
        if increment:
            self.counters[self.scope, name] = number + 1
        suffix = ""
        while True:
            suffix += chr(ord("a") + (number & 15))
            number >>= 4
            if not number:
                return name + suffix

    @property
    def c(self):
        return self.s[self.i]

    def _get(self):
        self.i += 1
        return self.s[self.i]

    def _skip(self):
        while self.s[self.i] == " ":
            self.i += 1

    def _run(self, pattern):
        end = pattern.match(self.s, self.i).end()
        word, self.i = self.s[self.i : end], end
        return word

    def _keyword(self, required=False):
        word = self._run(_KEYWORD)
        if required and not word:
            raise AcmeError("No string given.")
        return word

    def _scoped_keyword(self):
        self._skip()
        scope = 0
        if self.c == ".":
            self.i += 1
            scope = self.scope
        return scope, self._keyword(True)

    def _accept_comma(self):
        self._skip()
        if self.c != ",":
            return False
        self.i += 1
        self._skip()
        return True

    def _ensure_eos(self):
        self._skip()
        if self.c != EOS:
            raise AcmeError("Garbage data at end of statement.")

    def _quoted(self):
        match = _QUOTED.match(self.s, self.i)
        if not match.group(3):
            raise AcmeError("Quotes still open at end of line.")
        self.i = match.end()
        return match.group(1), _unescape(match.group(2))

    def _force_bit(self):
        force = 0
        if self.c == "+":
            force = {"1": F8, "2": F16, "3": F24}.get(self._get(), 0)
            if not force:
                raise AcmeError("Illegal postfix.")
            self.i += 1
        self._skip()
        return force

    def _skip_block(self):
        """Advance from ``{`` to its matching ``}``."""
        depth = 1
        while depth:
            c = self._get()
            if c in "\"'":
                self.i = _QUOTED.match(self.s, self.i).end() - 1
            elif c == "{":
                depth += 1
            elif c == EOB:
                depth -= 1
            elif c == EOF:
                raise AcmeError("Found end-of-file instead of '}'.")

    def _expression(self):
        """``(result, parenthesized, unclosed parentheses)``; a result is a ``str`` or
        ``(value, flags)`` with value ``None`` while undefined."""
        self.open = 0
        result, paren = self._binary(0)
        if isinstance(result, tuple):
            val, flags = result
            if flags & F24:
                flags &= ~(F16 | F8)
            elif flags & F16:
                flags &= ~F8
            result = val, flags
            if val is None:
                self.undef += 1
        return result, paren, self.open

    def _binary(self, minimum):
        left, paren = self._unary()
        while True:
            self._skip()
            op = self._peek_dyadic()
            if op is None or _DYADIC[op.lower()] < minimum:
                return left, paren
            self.i += len(op)
            op = op.lower()
            right, _ = self._binary(_DYADIC[op] + (op != "^"))
            left, paren = self._dyadic(_ALIASES.get(op, op), left, right), False

    def _peek_dyadic(self):
        c = self.c
        if _starts_keyword(c):
            word = _KEYWORD.match(self.s, self.i).group(0)
            if word.lower() not in _DYADIC:
                raise AcmeError("Unknown operator.")
            return word
        if c == "[":
            raise AcmeError("Unsupported: indexing.")
        match = _SYMBOLIC.match(self.s, self.i)
        if match and match.group(0) == "!":
            raise AcmeError("Syntax error.")
        return match and match.group(0)

    def _unary(self):
        self._skip()
        c = self.c
        if c == "+":
            name = self._run(_RUNS["+"])
            return self._read(self.scope, self._anon(name, False), name), False
        if c == "-":
            name = self._run(_RUNS["-"])
            self._skip()
            if self.c in (EOS, ")", ","):
                return self._read(self.scope, name, name), False
            if len(name) % 2 == 0:
                return self._unary()
            return self._monadic("neg", self._binary(37)[0]), False
        if c in "!<>^":
            self.i += 1
            return self._monadic(c, self._binary(39 if c == "!" else 29)[0]), False
        if c == "(":
            self.i += 1
            value, _ = self._binary(0)
            self._skip()
            if self.c == ")":
                self.i += 1
            else:
                self.open += 1
            return value, True
        if c in "\"'":
            quote, text = self._quoted()
            if quote == '"':
                return text, False
            if len(text) != 1:
                raise AcmeError("There's more than one character." if text else "No string given.")
            return (ord(text), 0), False
        if c in _LITERALS and (c != "&" or "0" <= self.s[self.i + 1] <= "7"):
            self.i += 1
            return self._literal(*_LITERALS[c]), False
        if c == "*":
            self.i += 1
            if not self.pc_defined and self.complain:
                raise AcmeError("Value not defined (*).")
            return (self.pc, 0), False
        if c == ".":
            self.i += 1
            if _is_digit(self.c):
                raise AcmeError("Unsupported: floating point.")
            word = self._keyword(True)
            return self._read(self.scope, word, "." + word), False
        if _is_digit(c):
            if c == "0" and self.s[self.i + 1] in ("b", "x"):
                self.i += 2
                return self._literal(*_LITERALS["%" if self.s[self.i - 1] == "b" else "$"]), False
            value = int(self._run(_DECIMAL))
            if self.c == ".":
                raise AcmeError("Unsupported: floating point.")
            return (value, 0), False
        if _starts_keyword(c):
            word = self._keyword()
            if word.lower() == "not":
                return self._monadic("!", self._binary(39)[0]), False
            if self.c == "(":
                raise AcmeError("Unsupported: function %s()." % word)
            return self._read(0, word, word), False
        if c in "@[&":
            raise AcmeError("Unsupported: %s." % {"@": "cheap locals", "[": "lists"}.get(c, "&"))
        raise AcmeError("No value given." if c in (EOS, ",", "{", EOB) else "Syntax error.")

    def _literal(self, base, wide, wider):
        digits = self._run(_DIGIT_RUNS[base])
        if not digits:
            raise AcmeError("Literal without any digits.")
        value = int(digits.replace(".", "0").replace("#", "1"), base)
        flags = 0
        if len(digits) > wider:
            flags = F24 if value < 65536 else 0
        elif len(digits) > wide:
            flags = F16 if value < 256 else 0
        return value, flags

    @staticmethod
    def _number(value):
        if isinstance(value, str):
            raise AcmeError("Unsupported: string operations.")
        return value

    def _monadic(self, op, value):
        val, flags = self._number(value)
        if op in "<>^":
            flags = (flags | FITS) & ~FORCE
            if val is not None:
                val = (val >> {"<": 0, ">": 8, "^": 16}[op]) & 0xFF
        else:
            flags &= ~FITS
            if val is not None:
                val = ~val if op == "!" else -val
        return val, flags

    def _dyadic(self, op, left, right):
        (a, fa), (b, fb) = self._number(left), self._number(right)
        if op in _COMPARE:
            flags = ((fa | fb) & EVER) | FITS
        else:
            flags = (fa | (fb & (EVER | FORCE))) & ~FITS
        if a is None or b is None:
            return None, flags
        if op in _COMPARE:
            return int(_COMPARE[op](a, b)), flags
        if op in ("/", "%"):
            if b == 0:
                raise AcmeError("Division by zero.")
            q = _cdiv(a, b)
            return _s64(q if op == "/" else a - b * q), flags
        if op == "^":
            if b < 0:
                raise AcmeError("Exponent is negative.")
            return _s64(pow(a, b, 1 << 64)), flags
        return _s64(_ARITHMETIC[op](a, b)), flags

    def _result(self):
        result, _, opened = self._expression()
        if opened:
            raise AcmeError("Too many '('.")
        return result

    @staticmethod
    def _int(result):
        if isinstance(result, str):
            if len(result) != 1:
                raise AcmeError("String length is not 1.")
            return ord(result), 0
        return result

    def _defined_int(self):
        complain, self.complain = self.complain, True
        try:
            val = self._int(self._result())[0]
        finally:
            self.complain = complain
        return val

    def _byte(self, value):
        if not self.out_on:
            raise AcmeError("Program counter undefined.")
        if self.wi > 0xFFFF:
            raise AcmeError("Produced too much code.")
        self.mem[self.wi] = value & 0xFF
        self.lo, self.hi = min(self.lo, self.wi), max(self.hi, self.wi)
        self.wi += 1
        self.add += 1

    def _emit(self, value, width, big=False):
        value = value or 0
        bits = 8 * width
        if width < 4 and not -(1 << (bits - 1)) <= value < (1 << bits):
            raise AcmeError("Number out of range (%d bits)." % bits)
        shifts = range(0, bits, 8)
        for shift in reversed(shifts) if big else shifts:
            self._byte(value >> shift)

    def _block(self):
        """Parse statements from the next byte to the end of the block or file."""
        self._get()
        while self.c not in (EOB, EOF):
            labelled = False
            while True:
                labelled = self._statement_part(labelled)
                if self.c == EOS:
                    break
            self.pc, self.add = (self.pc + self.add) & 0xFFFF, 0
            self._get()

    def _statement_part(self, labelled):
        """Parse one element of a statement; return whether the statement has a label."""
        c = self.c
        if c in (" ", SOL, EOS):
            self.i += c != EOS
            return labelled
        if c == "!":
            self._pseudo_op()
            return labelled
        if c == "*":
            self._set_pc_statement()
            return labelled
        if c == "+" and (self._get() == "." or _starts_keyword(self.c) or _is_digit(self.c)):
            self._macro_call()
            return labelled
        if _starts_keyword(c):
            word = self._keyword()
            table = self.cpu.get(word.lower()) if len(word) == 3 else None
            if table:
                self._instruction(table)
                return labelled
            if word.lower() in ("dop", "top") and self.cpu is _CPUS["6510"]:
                raise AcmeError("Unsupported: mnemonic %s." % word)
        if labelled or c not in "+-." and not _starts_keyword(c):
            raise AcmeError("Unsupported: cheap locals." if c == "@" else "Syntax error.")
        if c == "+":
            self._label(self.scope, self._anon("+" + self._run(_PLUSES), True))
        elif c == "-":
            self._label(self.scope, self._run(_RUNS["-"]), change=True)
        elif c == ".":
            self.i += 1
            self._definition(self.scope, self._keyword(True))
        else:
            self._definition(0, word)
        return True

    def _definition(self, scope, name):
        force = self._force_bit()
        if self.c != "=":
            self._label(scope, name, force)
            return
        self.i += 1
        sym = self._find(scope, name)
        result = self._result()
        if isinstance(result, str):
            raise AcmeError("Unsupported: string symbols.")
        self._assign(sym, result)
        if force:
            self._force_symbol(sym, force)
        self._ensure_eos()

    def _set_pc_statement(self):
        self.i += 1
        self._skip()
        if self.c != "=":
            raise AcmeError("Syntax error.")
        self.i += 1
        new = self._defined_int()
        while self._accept_comma():
            if self._keyword().lower() not in ("overlay", "invisible"):
                raise AcmeError('Unknown "*=" segment modifier.')
        self.wi = (self.wi + new - self.pc) & 0xFFFF
        self.pc, self.pc_defined, self.out_on = new, True, True
        self._ensure_eos()

    def _optional_block(self):
        self._skip()
        if self.c != "{":
            return False
        self._block()
        if self.c != EOB:
            raise AcmeError("Found end-of-file instead of '}'.")
        self.i += 1
        return True

    def _instruction(self, modes):
        if set(modes) == {"imp"}:
            self._byte(modes["imp"][0])
            self._ensure_eos()
            return
        if "rel" in modes:
            self._byte(modes["rel"][F8])
            self._branch()
            return
        force = self._force_bit()
        syntax, number = self._addressing()
        sizes = modes.get(syntax)
        if not sizes:
            raise AcmeError("CPU does not support this addressing mode for this mnemonic.")
        if syntax == "imp":
            self._byte(sizes[0])
            return
        size = self._size(force, number, sum(sizes))
        self._byte(sizes[size])
        self._emit(number[0], 1 if size == F8 else 2)

    def _addressing(self):
        self._skip()
        c = self.c
        if c == EOS:
            return "imp", None
        if c == "#":
            self.i += 1
            number = self._int(self._result())
            self._ensure_eos()
            return "imm", number
        if c == "[":
            raise AcmeError("Unsupported: long indirect addressing.")
        result, paren, opened = self._expression()
        if opened > 1:
            raise AcmeError("Too many '('.")
        pre = None
        if opened:
            pre = self._index()
            if self.c != ")":
                raise AcmeError("Syntax error.")
            self.i += 1
        syntax = _ADDRESSING.get((paren, pre, self._index()))
        self._ensure_eos()
        if syntax is None:
            raise AcmeError("CPU does not support this addressing mode for this mnemonic.")
        return syntax, self._int(result)

    def _index(self):
        if not self._accept_comma():
            return None
        c = self.c.lower()
        if c not in ("x", "y"):
            raise AcmeError("Syntax error.")
        self.i += 1
        self._skip()
        return c

    @staticmethod
    def _size(force, number, sizes):
        val, flags = number
        if force:
            if sizes & force:
                return force
            raise AcmeError("CPU does not support this postfix for this mnemonic.")
        if flags & FORCE:
            for size in (F8, F16):
                if size & sizes & flags:
                    return size
            raise AcmeError("Number out of range.")
        if sizes in (F8, F16):
            return sizes
        if flags & EVER:
            return F8 if flags & FITS else F16
        if val < 0:
            raise AcmeError("Negative value - cannot choose addressing mode.")
        if val < 256:
            return F8
        if val < 65536:
            return F16
        raise AcmeError("Number out of range.")

    def _branch(self):
        target = self._int(self._result())[0]
        offset = 0
        if self.pc_defined and target is not None:
            if target | 0xFFFF != 0xFFFF:
                raise AcmeError("Target not in bank (0x%x)." % target)
            offset = (target - (self.pc + 2)) & 0xFFFF
            offset -= 0x10000 if offset & 0x8000 else 0
            if not -128 <= offset <= 127:
                raise AcmeError("Target out of range (%d)." % offset)
        self._byte(offset)
        self._ensure_eos()

    def _pseudo_op(self):
        self.i += 1
        word = self._keyword(True).lower()
        self._skip()
        if word in _DATA:
            self._po_data(*_DATA[word])
        elif word in _PSEUDO:
            getattr(self, "_po_" + _PSEUDO[word])()
        elif word in _ACME_ONLY:
            raise AcmeError("Unsupported: !%s." % word)
        else:
            raise AcmeError("Unknown pseudo opcode.")

    def _po_data(self, width, big=False, strings=False):
        while True:
            result = self._result()
            if isinstance(result, str):
                if len(result) > 1 and not strings:
                    raise AcmeError("There's more than one character.")
                for c in result:
                    self._emit(ord(c), width, big)
            else:
                self._emit(result[0], width, big)
            if not self._accept_comma():
                break
        self._ensure_eos()

    def _po_hex(self):
        while True:
            pair = _HEX_PAIR.match(self.s, self.i).group(0)
            if len(pair) == 1:
                raise AcmeError("Hex digits are not given in pairs.")
            if pair:
                self._byte(int(pair, 16))
                self.i += 2
            elif self.c == " ":
                self.i += 1
            elif self.c == EOS:
                return
            else:
                raise AcmeError("Syntax error.")

    def _po_fill(self):
        count = self._defined_int()
        value = self._int(self._result())[0] if self._accept_comma() else 0
        if count < 0:
            raise AcmeError("Negative size argument.")
        for _ in range(count):
            self._emit(value, 1)
        self._ensure_eos()

    def _po_to(self):
        if self.c != '"':
            raise AcmeError("File name quotes not found.")
        _, filename = self._quoted()
        if not filename:
            raise AcmeError("No file name given.")
        if self.passno or self.outname is not None:
            self.i = self.s.index(EOS, self.i)
            return
        self.outname = filename
        if self._accept_comma():
            self.fmt = self._keyword().lower()
            if self.fmt not in _FORMATS:
                raise AcmeError("Unknown output format.")
        else:
            self.fmt = self.fmt or "cbm"
        self._ensure_eos()

    @staticmethod
    def _cpu(name):
        name = name.lower()
        if name in _OTHER_CPUS:
            raise AcmeError("Unsupported: !cpu %s." % name)
        if name not in _CPUS:
            raise AcmeError("Unknown processor.")
        return _CPUS[name]

    def _po_cpu(self):
        cpu, outer = self._cpu(self._keyword()), self.cpu
        self.cpu = cpu
        if self._optional_block():
            self.cpu = outer
        self._ensure_eos()

    def _po_pseudopc(self):
        new = self._defined_int()
        self._skip()
        if self.c != "{":
            raise AcmeError("Unsupported: !pseudopc without a block.")
        offset, outer = new - self.pc, self.pc_defined
        self.pc, self.pc_defined = new, True
        self._optional_block()
        self.pc, self.pc_defined = (self.pc - offset) & 0xFFFF, outer
        self._ensure_eos()

    def _po_macro(self):
        if not self.passno:
            scope, name = self._scoped_keyword()
            self._skip()
            params = []
            while self.c != "{":
                if self.c in "~@":
                    raise AcmeError("Unsupported: macro parameter %s." % self.c)
                local = self.c == "."
                self.i += local
                params.append((local, self._keyword(True)))
                if not self._accept_comma() and self.c != "{":
                    raise AcmeError("No left brace found.")
            key = scope, name, len(params)
            if key in self.macros:
                raise AcmeError("Macro already defined.")
            self.macros[key] = params, self.i
        else:
            self.i = self.s.index("{", self.i)
        self._skip_block()
        self.i += 1
        self._ensure_eos()

    def _macro_call(self):
        self.depth += 1
        if self.depth > 64:
            raise AcmeError("Too deeply nested. Recursive macro calls?")
        scope, name = self._scoped_keyword()
        self._skip()
        args = []
        while self.c != EOS:
            if self.c == "~":
                raise AcmeError("Unsupported: call by reference.")
            arg = self._result()
            if isinstance(arg, str):
                raise AcmeError("Unsupported: string macro arguments.")
            args.append(arg)
            if not self._accept_comma():
                break
        macro = self.macros.get((scope, name, len(args)))
        if macro is None:
            raise AcmeError("Macro not defined (or wrong signature).")
        params, start = macro
        resume, outer = self.i, self.scope
        self.scope_max += 2
        self.scope = self.scope_max
        for (local, param), arg in zip(params, args):
            sym = self._find(self.scope if local else 0, param)
            sym.kind, (sym.val, sym.flags) = "num", arg
        self.i = start
        self._block()
        self.i, self.scope = resume, outer
        self.depth -= 1
        self._ensure_eos()


def assemble(text, cpu="6502", fmt=None, name="<source>"):
    """``(output file bytes, {global symbol: value})`` of ACME source ``text``."""
    acme = Acme([(name, text)], cpu=cpu, fmt=fmt).run()
    return acme.output(), acme.symbols()


def main(argv=None):
    """ACME command line subset: ``[-f FORMAT] [-l LABELS] [-o OUT] SOURCE...``."""
    ap = argparse.ArgumentParser(
        prog="jennings-acme", description=__doc__.split("\n", maxsplit=1)[0]
    )
    ap.add_argument("-o", "--outfile", help="output file name (overrides !to)")
    ap.add_argument("-l", "--symbollist", "--labeldump", help="symbol list file name")
    ap.add_argument("-f", "--format", choices=_FORMATS, help="output file format")
    ap.add_argument("--cpu", default="6502", help="default target processor")
    ap.add_argument("sources", nargs="+")
    args = ap.parse_args(argv)
    try:
        sources = []
        for path in args.sources:
            with open(path, "rb") as f:
                sources.append((path, f.read().decode("latin-1")))
        acme = Acme(sources, cpu=args.cpu, fmt=args.format, outname=args.outfile).run()
    except (AcmeError, OSError) as e:
        print("Error - %s" % e, file=sys.stderr)
        return 1
    if acme.outname is None:
        print(
            'No output file specified (use the "-o" option or the "!to" pseudo opcode).',
            file=sys.stderr,
        )
    else:
        with open(acme.outname, "wb") as f:
            f.write(acme.output())
    if args.symbollist:
        with open(args.symbollist, "w", encoding="latin-1", newline="\n") as f:
            f.write(acme.symbol_list())
    return 0


if __name__ == "__main__":
    sys.exit(main())
