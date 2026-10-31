//! 6510 register file, interrupts and the control-flow resolvers.

use crate::bus::Bus;
use crate::cpu_gen;

pub const NMI: u16 = 0xFFFA;
pub const RESET: u16 = 0xFFFC;
pub const IRQ: u16 = 0xFFFE;

/// Register file as `PcodeVM.reg`: A/X/Y/SP and one byte (0/1) per flag.
///
/// `d` is stored but ADC/SBC are binary, as in deity-informant. `cycles` is
/// the running count (`processorCycles`); `jams` counts executed JAMs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cpu {
    pub a: u8,
    pub x: u8,
    pub y: u8,
    pub sp: u8,
    pub c: u8,
    pub z: u8,
    pub i: u8,
    pub d: u8,
    pub b: u8,
    pub v: u8,
    pub n: u8,
    pub pc: u16,
    pub cycles: u64,
    pub jammed: bool,
    pub jams: u64,
}

impl Default for Cpu {
    fn default() -> Self {
        Self::new()
    }
}

/// 1 if `base + idx` leaves `base`'s page.
#[inline(always)]
pub(crate) fn cross(base: u32, idx: u8) -> u32 {
    ((base & 0xFF00) != ((base + idx as u32) & 0xFF00)) as u32
}

impl Cpu {
    pub fn new() -> Self {
        Cpu {
            a: 0,
            x: 0,
            y: 0,
            sp: 0xFF,
            c: 0,
            z: 0,
            i: 0,
            d: 0,
            b: 0,
            v: 0,
            n: 0,
            pc: 0,
            cycles: 0,
            jammed: false,
            jams: 0,
        }
    }

    /// `NV-BDIZC` with B and the unused bit set, as py65's `MPU.p` reads.
    pub fn p(&self) -> u8 {
        self.status(1)
    }

    /// Set C/Z/I/D/V/N from `p`; B and bit 5 are not stored.
    pub fn set_p(&mut self, p: u8) {
        self.c = p & 1;
        self.z = (p >> 1) & 1;
        self.i = (p >> 2) & 1;
        self.d = (p >> 3) & 1;
        self.v = (p >> 6) & 1;
        self.n = (p >> 7) & 1;
    }

    #[inline(always)]
    fn status(&self, brk: u8) -> u8 {
        self.c
            | (self.z << 1)
            | (self.i << 2)
            | (self.d << 3)
            | (self.v << 6)
            | (self.n << 7)
            | 0x20
            | (brk << 4)
    }

    /// Clear registers and flags, SP=$FF, zero the cycle count; PC from `pc` or the reset vector.
    pub fn reset<B: Bus>(&mut self, bus: &mut B, pc: Option<u16>) {
        *self = Cpu::new();
        self.pc = pc.unwrap_or_else(|| Self::vector(bus, RESET));
    }

    /// Execute one instruction; cycles taken, page-cross and branch extras included.
    #[inline]
    pub fn step<B: Bus>(&mut self, bus: &mut B) -> u32 {
        bus.sync(self);
        let n = cpu_gen::execute(self, bus);
        self.cycles += n as u64;
        n
    }

    /// Take a maskable interrupt unless I is set; cycles taken (0 if masked).
    pub fn irq<B: Bus>(&mut self, bus: &mut B) -> u32 {
        if self.i != 0 {
            return 0;
        }
        self.interrupt(bus, IRQ)
    }

    /// Take a non-maskable interrupt; cycles taken.
    pub fn nmi<B: Bus>(&mut self, bus: &mut B) -> u32 {
        self.interrupt(bus, NMI)
    }

    fn interrupt<B: Bus>(&mut self, bus: &mut B, vec: u16) -> u32 {
        bus.sync(self);
        self.push(bus, (self.pc >> 8) as u8);
        self.push(bus, self.pc as u8);
        self.push(bus, self.status(0));
        self.i = 1;
        self.pc = Self::vector(bus, vec);
        self.cycles += 7;
        7
    }

    #[inline(always)]
    fn vector<B: Bus>(bus: &mut B, addr: u16) -> u16 {
        let lo = bus.read(addr) as u16;
        lo | ((bus.read(addr.wrapping_add(1)) as u16) << 8)
    }

    #[inline(always)]
    fn push<B: Bus>(&mut self, bus: &mut B, v: u8) {
        bus.stack_write(0x100 | self.sp as u16, v);
        self.sp = self.sp.wrapping_sub(1);
    }

    #[inline(always)]
    fn pull<B: Bus>(&mut self, bus: &mut B) -> u8 {
        self.sp = self.sp.wrapping_add(1);
        bus.stack_read(0x100 | self.sp as u16)
    }

    /// Resolve a relative branch at `pc`; extra cycles (taken +1, page cross +1).
    #[inline(always)]
    pub(crate) fn branch(&mut self, taken: bool, pc: u16, off: u32) -> u32 {
        let ft = pc.wrapping_add(2);
        if !taken {
            self.pc = ft;
            return 0;
        }
        let tgt = ft.wrapping_add(off as u8 as i8 as u16);
        self.pc = tgt;
        1 + ((ft ^ tgt) & 0xFF00 != 0) as u32
    }

    #[inline(always)]
    pub(crate) fn jsr<B: Bus>(&mut self, bus: &mut B, ret: u16, tgt: u16) {
        self.push(bus, (ret >> 8) as u8);
        self.push(bus, ret as u8);
        self.pc = tgt;
    }

    #[inline(always)]
    pub(crate) fn rts<B: Bus>(&mut self, bus: &mut B) {
        let lo = self.pull(bus) as u16;
        let hi = self.pull(bus) as u16;
        self.pc = ((hi << 8) | lo).wrapping_add(1);
    }

    #[inline(always)]
    pub(crate) fn rti<B: Bus>(&mut self, bus: &mut B) {
        let p = self.pull(bus);
        self.set_p(p);
        let lo = self.pull(bus) as u16;
        let hi = self.pull(bus) as u16;
        self.pc = (hi << 8) | lo;
    }

    #[inline(always)]
    pub(crate) fn brk<B: Bus>(&mut self, bus: &mut B, ret: u16) {
        self.push(bus, (ret >> 8) as u8);
        self.push(bus, ret as u8);
        self.push(bus, self.status(1));
        self.i = 1;
        self.pc = Self::vector(bus, IRQ);
    }

    /// `JMP (ptr)` with the NMOS page-wrap of the pointer's high byte.
    #[inline(always)]
    pub(crate) fn jmp_ind<B: Bus>(&mut self, bus: &mut B, ptr: u16) {
        let lo = bus.read(ptr) as u16;
        let hi = bus.read((ptr & 0xFF00) | (ptr.wrapping_add(1) & 0xFF)) as u16;
        self.pc = (hi << 8) | lo;
    }

    /// JAM/KIL: flag the lock-up and skip a byte so drivers do not hang.
    #[inline(always)]
    pub(crate) fn jam(&mut self, pc: u16) {
        self.jammed = true;
        self.jams += 1;
        self.pc = pc.wrapping_add(1);
    }
}
