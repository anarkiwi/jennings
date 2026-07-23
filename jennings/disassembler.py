"""Disassemble one instruction from an MPU's memory (py65-compatible).

Consumes the MPU ``disassemble`` table -- which in jennings covers all 105 NMOS
illegal opcodes -- so illegals render as real mnemonics where py65 shows ``???``.
"""

from jennings.utils.addressing import AddressParser


class Disassembler:
    def __init__(self, mpu, address_parser=None):
        self._mpu = mpu
        self._address_parser = address_parser or AddressParser()
        self.addrWidth = mpu.ADDR_WIDTH
        self.byteWidth = mpu.BYTE_WIDTH
        self.addrFmt = mpu.ADDR_FORMAT
        self.byteFmt = mpu.BYTE_FORMAT
        self.addrMask = mpu.addrMask
        self.byteMask = mpu.byteMask

    def instruction_at(self, pc):
        """Return ``(length, text)`` for the instruction at ``pc``."""
        mpu = self._mpu
        disasm, mode = mpu.disassemble[mpu.ByteAt(pc)]

        def word_label():
            address = mpu.WordAt(pc + 1)
            return address, self._address_parser.label_for(address, "$" + self.addrFmt % address)

        def byte_label():
            zp = mpu.ByteAt(pc + 1)
            return zp, self._address_parser.label_for(zp, "$" + self.byteFmt % zp)

        if mode == "acc":
            return 1, disasm + " A"
        if mode == "imp":
            return 1, disasm
        if mode == "imm":
            return 2, disasm + " #$" + self.byteFmt % mpu.ByteAt(pc + 1)
        if mode in ("abs", "abx", "aby", "ind", "iax"):
            _addr, label = word_label()
            suffix = {
                "abs": " %s",
                "abx": " %s,X",
                "aby": " %s,Y",
                "ind": " (%s)",
                "iax": " (%s,X)",
            }
            return 3, disasm + suffix[mode] % label
        if mode in ("zpg", "zpx", "zpy", "inx", "iny", "zpi"):
            _zp, label = byte_label()
            suffix = {
                "zpg": " %s",
                "zpx": " %s,X",
                "zpy": " %s,Y",
                "inx": " (%s,X)",
                "iny": " (%s),Y",
                "zpi": " (%s)",
            }
            return 2, disasm + suffix[mode] % label
        if mode == "rel":
            opv = mpu.ByteAt(pc + 1)
            targ = pc + 2 - ((opv ^ self.byteMask) + 1) if opv & 0x80 else pc + 2 + opv
            targ &= self.addrMask
            return 2, disasm + " " + self._address_parser.label_for(targ, "$" + self.addrFmt % targ)
        raise NotImplementedError("Addressing mode: %r" % mode)
