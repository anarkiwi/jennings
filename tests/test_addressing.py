"""AddressParser: numbers, labels, and ranges (py65-compatible)."""

import pytest

from jennings.utils.addressing import AddressParser


def test_number_radixes():
    p = AddressParser()
    assert p.number("$c000") == 0xC000
    assert p.number("+4096") == 4096
    assert p.number("%1111") == 15
    assert p.number("ff") == 0xFF  # default radix 16


def test_labels():
    p = AddressParser(labels={"start": 0x1000})
    assert p.number("start") == 0x1000
    assert p.number("start + $10") == 0x1010
    assert p.number("start - +16") == 0x0FF0
    assert p.address_for("start") == 0x1000
    assert p.label_for(0x1000) == "start"
    assert p.label_for(0x9999, "none") == "none"


def test_range():
    p = AddressParser()
    assert p.range("$1000:$2000") == (0x1000, 0x2000)
    assert p.range("$2000,$1000") == (0x1000, 0x2000)  # normalised
    assert p.range("$1234") == (0x1234, 0x1234)


def test_overflow_and_missing_label():
    p = AddressParser()
    with pytest.raises(OverflowError):
        p.number("$1FFFF")
    with pytest.raises(KeyError):
        p.number("nope")
