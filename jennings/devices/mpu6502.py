"""NMOS 6502 / 6510 CPU with first-class illegal-opcode support.

A py65-compatible ``MPU`` over deity-informant's hardware-validated ``lift`` +
``PcodeVM`` engine: all 256 opcodes (incl. the 105 NMOS 6510 illegals py65 stubs)
execute byte- and cycle-exact. Drop-in for ``py65.devices.mpu6502.MPU``.
"""

from deity_informant import lift, PcodeVM, OPS, CYCLETIME, EXTRACYCLES, ILLEGAL_OPCODES

# deity-informant addressing-mode name -> py65 disassembler mode token.
_MODE_TOKEN = {
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

# PcodeVM.reg indices: 0-3 A/X/Y/SP; flags C/Z/I/D/B/V/N at 8-14.
_A, _X, _Y, _SP = 0, 1, 2, 3
_FC, _FZ, _FI, _FD, _FB, _FV, _FN = 8, 9, 10, 11, 12, 13, 14

_JAM = frozenset(op for op, (mn, _md) in OPS.items() if mn == "JAM")


def _disassemble_table():
    tbl = [("???", "imp")] * 256
    for op, (mn, mode) in OPS.items():
        tbl[op] = (mn, _MODE_TOKEN[mode])
    return tbl


def _instruct_table():
    """py65-compatible ``instruct`` table for read-introspection only.

    Every opcode is implemented (none is ``inst_not_implemented``); execution
    runs through the validated engine, so these are never dispatched.
    """
    tbl = []
    for op in range(256):

        def _inst(self, _op=op):  # pragma: no cover - never dispatched
            raise NotImplementedError("MPU.instruct is introspection-only")

        _inst.__name__ = "inst_0x%02x" % op
        tbl.append(_inst)
    return tbl


class MPU:
    RESET = 0xFFFC
    NMI = 0xFFFA
    IRQ = 0xFFFE

    NEGATIVE = 128
    OVERFLOW = 64
    UNUSED = 32
    BREAK = 16
    DECIMAL = 8
    INTERRUPT = 4
    ZERO = 2
    CARRY = 1

    BYTE_WIDTH = 8
    BYTE_FORMAT = "%02x"
    ADDR_WIDTH = 16
    ADDR_FORMAT = "%04x"

    disassemble = _disassemble_table()
    instruct = _instruct_table()
    cycletime = list(CYCLETIME)
    extracycles = list(EXTRACYCLES)

    def __init__(self, memory=None, pc=0x0000):
        self.name = "6510"
        self.byteMask = (1 << self.BYTE_WIDTH) - 1
        self.addrMask = (1 << self.ADDR_WIDTH) - 1
        self.addrHighMask = self.byteMask << self.BYTE_WIDTH
        self.spBase = 1 << self.BYTE_WIDTH

        self.excycles = 0
        self.addcycles = False
        self.processorCycles = 0
        self.jammed = False
        self.pc = 0

        # Engine runs on the caller's memory so ObservableMemory observers fire.
        self._vm = PcodeVM(b"")
        self._cache = {}
        if memory is None:
            memory = 0x10000 * [0x00]
        self.memory = memory
        self._vm.volatile = False

        self.start_pc = pc
        self.reset()

    @property
    def memory(self):
        return self._memory

    @memory.setter
    def memory(self, mem):
        self._memory = mem
        self._vm.mem = mem

    @property
    def a(self):
        return self._vm.reg[_A]

    @a.setter
    def a(self, v):
        self._vm.reg[_A] = v & self.byteMask

    @property
    def x(self):
        return self._vm.reg[_X]

    @x.setter
    def x(self, v):
        self._vm.reg[_X] = v & self.byteMask

    @property
    def y(self):
        return self._vm.reg[_Y]

    @y.setter
    def y(self, v):
        self._vm.reg[_Y] = v & self.byteMask

    @property
    def sp(self):
        return self._vm.reg[_SP]

    @sp.setter
    def sp(self, v):
        self._vm.reg[_SP] = v & self.byteMask

    @property
    def p(self):
        r = self._vm.reg
        return (
            r[_FC]
            | (r[_FZ] << 1)
            | (r[_FI] << 2)
            | (r[_FD] << 3)
            | (r[_FV] << 6)
            | (r[_FN] << 7)
            | self.UNUSED
            | self.BREAK
        )

    @p.setter
    def p(self, v):
        r = self._vm.reg
        r[_FC] = v & 1
        r[_FZ] = (v >> 1) & 1
        r[_FI] = (v >> 2) & 1
        r[_FD] = (v >> 3) & 1
        r[_FV] = (v >> 6) & 1
        r[_FN] = (v >> 7) & 1

    def reset(self):
        self.pc = self.start_pc
        if self.pc is None:
            self.pc = self.WordAt(self.RESET)
        r = self._vm.reg
        r[_A] = r[_X] = r[_Y] = 0
        r[_SP] = self.byteMask
        for i in (_FC, _FZ, _FI, _FD, _FB, _FV, _FN):
            r[i] = 0
        self._vm.cycles = 0
        self.processorCycles = 0
        self.jammed = False

    def _push(self, value):
        self._memory[self.spBase + self._vm.reg[_SP]] = value & self.byteMask
        self._vm.reg[_SP] = (self._vm.reg[_SP] - 1) & self.byteMask

    def _push_word(self, value):
        self._push(value >> self.BYTE_WIDTH)
        self._push(value)

    def irq(self):
        if self._vm.reg[_FI]:
            return
        self._push_word(self.pc)
        self._push((self.p & ~self.BREAK) | self.UNUSED)
        self._vm.reg[_FI] = 1
        self.pc = self.WordAt(self.IRQ)
        self.processorCycles += 7

    def nmi(self):
        self._push_word(self.pc)
        self._push((self.p & ~self.BREAK) | self.UNUSED)
        self._vm.reg[_FI] = 1
        self.pc = self.WordAt(self.NMI)
        self.processorCycles += 7

    def ByteAt(self, addr):
        return self._memory[addr & self.addrMask]

    def WordAt(self, addr):
        return self.ByteAt(addr) + (self.ByteAt((addr + 1) & self.addrMask) << self.BYTE_WIDTH)

    def ProgramCounter(self):
        return self.pc

    def step(self):
        pc = self.pc & self.addrMask
        if self._memory[pc] in _JAM:
            self.jammed = True
            return self
        before = self._vm.cycles
        self.pc = self._vm.step(pc, self._cache, lift)
        self.processorCycles += self._vm.cycles - before
        return self

    def reprformat(self):
        return "%s PC  AC XR YR SP NV-BDIZC\n%s: %04x %02x %02x %02x %02x %s"

    def __repr__(self):
        flags = bin(self.p)[2:].rjust(self.BYTE_WIDTH, "0")
        indent = " " * (len(self.name) + 2)
        return self.reprformat() % (
            indent,
            self.name,
            self.pc,
            self.a,
            self.x,
            self.y,
            self.sp,
            flags,
        )


ILLEGAL_OPCODES = frozenset(ILLEGAL_OPCODES)
