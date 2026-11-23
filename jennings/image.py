"""64 KiB memory images loaded from C64 program files (``.prg``), PSID/RSID tunes (``.sid``)
and raw dumps, with the spans they occupy and label files in ACME, VICE/ld65 or plain
``name = $hex`` form."""

import re
import struct

from jennings.acme import read_symbol_list

__all__ = ["FORMATS", "Image", "parse_address", "read_labels", "sid_header"]

FORMATS = ("auto", "prg", "sid", "raw")
_SID = struct.Struct(">4sHHHHHHHI32s32s32s")
_SID_FIELDS = "magic version dataOffset load init play songs startSong speed name author released"
_SID_V2 = struct.Struct(">HBBBB")
_SID_V2_FIELDS = "flags startPage pageLength secondSID thirdSID"
_HEX = re.compile(r"(?:\$|0[xX])?([0-9a-fA-F]+)")
_VICE = re.compile(r"^\s*al\s+(?:[A-Za-z]:)?([0-9A-Fa-f]+)\s+\.?(\S+)")
_PLAIN = re.compile(r"^\s*([A-Za-z_.@][\w.@]*)\s*=\s*(?:\$|0[xX])?([0-9A-Fa-f]+)\b")


def parse_address(text):
    """Integer of hex ``text`` (``c000``, ``$c000`` or ``0xc000``), ``ValueError`` if not."""
    m = _HEX.fullmatch(text.strip())
    if m is None:
        raise ValueError("not a hex address: %r" % text)
    return int(m[1], 16)


def read_labels(text):
    """``{name: address}`` of a label file; each line an ACME symbol list entry, a VICE/ld65
    ``al C:1234 .name`` (or ``al 1234 name``) or a plain ``name = $1234``/``name=0x1234``."""
    out = {}
    for line in text.splitlines():
        found = read_symbol_list(line)
        if not found:
            m = _VICE.match(line) or _PLAIN.match(line)
            if m:
                found = {m[2]: int(m[1], 16)} if m.re is _VICE else {m[1]: int(m[2], 16)}
        out.update((k, v & 0xFFFF) for k, v in found.items())
    return out


def sid_header(data):
    """The PSID/RSID header of ``data`` as a dict, ``load``/``init`` resolved (0 = the data's
    own load address / ``load``) and ``data`` the payload after any embedded load address."""
    if len(data) < _SID.size:
        raise ValueError("truncated SID header")
    h = dict(zip(_SID_FIELDS.split(), _SID.unpack_from(data)))
    for k in ("magic", "name", "author", "released"):
        h[k] = h[k].rstrip(b"\0").decode("latin-1")
    if h["version"] >= 2 and len(data) >= _SID.size + _SID_V2.size:
        h.update(zip(_SID_V2_FIELDS.split(), _SID_V2.unpack_from(data, _SID.size)))
    payload = data[h["dataOffset"] :]
    if h["load"] == 0:
        h["load"], payload = payload[0] | payload[1] << 8, payload[2:]
    h["init"] = h["init"] or h["load"]
    h["data"] = payload
    return h


class Image:
    """``mem`` (64 KiB ``bytearray``) holding the loaded files; ``spans`` the half-open
    ``[lo, hi)`` each occupies, ``format`` the main file's, ``sid`` its header (or None) and
    ``labels`` the names it defines (``INIT``/``PLAY`` of a tune)."""

    def __init__(self):
        self.mem = bytearray(0x10000)
        self.spans, self.format, self.sid, self.labels = [], None, None, {}

    @property
    def lo(self):
        return self.spans[0][0]

    @property
    def hi(self):
        return self.spans[0][1]

    def merged(self):
        """The union of ``spans`` as sorted disjoint ``[lo, hi)`` pairs."""
        out = []
        for lo, hi in sorted(self.spans):
            if out and lo <= out[-1][1]:
                out[-1][1] = max(out[-1][1], hi)
            else:
                out.append([lo, hi])
        return [tuple(s) for s in out]

    def inside(self, address):
        return any(lo <= address < hi for lo, hi in self.spans)

    def load(self, spec, fmt="auto", base=None):
        """Load ``PATH[@ADDR]`` (``base`` the default ``ADDR``) on top of what is loaded."""
        path, _, at = spec.rpartition("@")
        try:
            addr = parse_address(at) if path else base
        except ValueError:
            path, addr = spec, base
        path = path or spec
        with open(path, "rb") as f:
            data = f.read()
        if fmt == "auto":
            fmt = (
                "sid"
                if data[:4] in (b"PSID", b"RSID")
                else "prg" if path.lower().endswith(".prg") else "raw"
            )
        sid = None
        if fmt == "sid":
            sid = sid_header(data)
            load, data = sid["load"], sid["data"]
        elif fmt == "prg":
            load, data = data[0] | data[1] << 8, data[2:]
        else:
            load = 0
        lo = load if addr is None else addr
        if lo + len(data) > 0x10000:
            raise ValueError("%s at $%04X runs past $FFFF" % (path, lo))
        self.mem[lo : lo + len(data)] = data
        self.spans.append((lo, lo + len(data)))
        if self.format is None:
            self.format, self.sid = fmt, sid
            if sid:
                self.labels["INIT"] = sid["init"]
                if sid["play"] and sid["play"] != sid["init"]:
                    self.labels["PLAY"] = sid["play"]
        return self
