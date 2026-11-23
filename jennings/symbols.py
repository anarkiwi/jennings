"""Built-in symbol sets from the public C64 and Commodore disk drive memory maps and the
MOS 6522 VIA, MOS 6526 CIA, MOS 6581 SID, MOS 6567 VIC-II and WD1770 register maps."""

__all__ = ["SETS", "symbols"]

_VIA = "PB PA DDRB DDRA T1CL T1CH T1LL T1LH T2CL T2CH SR ACR PCR IFR IER PANH"
_CIA = "PRA PRB DDRA DDRB TALO TAHI TBLO TBHI TOD10 TODSEC TODMIN TODHR SDR ICR CRA CRB"
_WD1770 = "STATUS TRACK SECTOR DATA"
_VIC = (
    " ".join("M%dX M%dY" % (i, i) for i in range(8))
    + " MSIGX CR1 RASTER LPX LPY SPENA CR2 YXPAND MEMPTR IRR IMR MDP MMC MXE MM MD EC"
    + " B0C B1C B2C B3C MM0 MM1 "
    + " ".join("M%dC" % i for i in range(8))
)
_SID = (
    " ".join(
        "V{0}FREQLO V{0}FREQHI V{0}PWLO V{0}PWHI V{0}CTRL V{0}AD V{0}SR".format(i)
        for i in (1, 2, 3)
    )
    + " FCLO FCHI RESFILT MODEVOL POTX POTY OSC3 ENV3"
)
_KERNAL = {
    "READST": 0xFFB7,
    "SETLFS": 0xFFBA,
    "SETNAM": 0xFFBD,
    "OPEN": 0xFFC0,
    "CLOSE": 0xFFC3,
    "CHKIN": 0xFFC6,
    "CHKOUT": 0xFFC9,
    "CLRCHN": 0xFFCC,
    "CHRIN": 0xFFCF,
    "CHROUT": 0xFFD2,
    "LOAD": 0xFFD5,
    "SAVE": 0xFFD8,
    "STOP": 0xFFE1,
    "GETIN": 0xFFE4,
    "CLALL": 0xFFE7,
    "PLOT": 0xFFF0,
    "CINV": 0x0314,
    "CBINV": 0x0316,
    "NMINV": 0x0318,
    "NMI_VEC": 0xFFFA,
    "RESET_VEC": 0xFFFC,
    "IRQ_VEC": 0xFFFE,
}
_DRIVE = {
    **{"JOB%d" % i: i for i in range(6)},
    **{"%s%d" % (k, i): 6 + 2 * i + j for i in range(6) for j, k in enumerate(("TRACK", "SECTOR"))},
    **{"BUF%d" % i: 0x300 + 0x100 * i for i in range(5)},
}

#: ``{set: ([(prefix, base, register names)], {name: address})}``.
SETS = {
    "c64": (
        [
            ("VIC_", 0xD000, _VIC),
            ("SID_", 0xD400, _SID),
            ("CIA1_", 0xDC00, _CIA),
            ("CIA2_", 0xDD00, _CIA),
        ],
        _KERNAL,
    ),
    "1541": ([("VIA1_", 0x1800, _VIA), ("VIA2_", 0x1C00, _VIA)], _DRIVE),
    "1571": (
        [
            ("VIA1_", 0x1800, _VIA),
            ("VIA2_", 0x1C00, _VIA),
            ("CIA_", 0x4000, _CIA),
            ("WD_", 0x2000, _WD1770),
        ],
        _DRIVE,
    ),
    "1581": ([("CIA_", 0x4000, _CIA), ("WD_", 0x6000, _WD1770)], {}),
}


def symbols(name):
    """``{name: address}`` of built-in symbol set ``name`` (a key of :data:`SETS`)."""
    blocks, fixed = SETS[name]
    out = {p + r: base + i for p, base, regs in blocks for i, r in enumerate(regs.split())}
    out.update(fixed)
    return out
