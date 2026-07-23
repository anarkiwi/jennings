"""Parse user input into addresses, ranges, or labels (py65-compatible)."""

import re


class AddressParser:
    def __init__(self, maxwidth=16, radix=16, labels=None):
        self.radix = radix
        self.maxwidth = maxwidth
        self.labels = {}
        for k, v in (labels or {}).items():
            self.labels[k] = self._constrain(v)

    @property
    def maxwidth(self):
        return self._maxwidth

    @maxwidth.setter
    def maxwidth(self, width):
        self._maxwidth = width
        self._maxaddr = (1 << width) - 1

    def address_for(self, label, default=None):
        return self.labels.get(label, default)

    def label_for(self, address, default=None):
        for label, label_address in self.labels.items():
            if label_address == address:
                return label
        return default

    def number(self, num):
        try:
            if num.startswith("$"):
                return self._constrain(int(num[1:], 16))
            if num.startswith("+"):
                return self._constrain(int(num[1:], 10))
            if num.startswith("%"):
                return self._constrain(int(num[1:], 2))
            if num in self.labels:
                return self.labels[num]
            matches = re.match(r"^([^\s+-]+)\s*([+\-])\s*([$+%]?\d+)$", num)
            if matches:
                label, sign, offset = matches.groups()
                if label not in self.labels:
                    raise KeyError("Label not found: %s" % label)
                base = self.labels[label]
                offset = self.number(offset)
                address = base + offset if sign == "+" else base - offset
                return self._constrain(address)
            return self._constrain(int(num, self.radix))
        except ValueError:
            raise KeyError("Label not found: %s" % num)

    def range(self, addresses):
        matches = re.match(r"^([^:,]+)\s*[:,]+\s*([^:,]+)$", addresses)
        if matches:
            start, end = map(self.number, matches.groups(0))
        else:
            start = end = self.number(addresses)
        if start > end:
            start, end = end, start
        return (start, end)

    def _constrain(self, address):
        if address < 0 or address > self._maxaddr:
            raise OverflowError(address)
        return address
