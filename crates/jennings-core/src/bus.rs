//! The CPU's view of the address space.

use crate::cpu::Cpu;

/// A 64 KiB address space the [`Cpu`](crate::Cpu) runs on.
///
/// `read`/`write` are the accesses that are neither instruction fetch, data nor
/// stack: interrupt vectors and the `JMP (ind)` pointer. The classified methods
/// default to them, so a plain memory implements only `read` and `write`;
/// a bus that tells them apart (observers see data `LOAD`/`STORE` only)
/// overrides the rest.
pub trait Bus {
    fn read(&mut self, addr: u16) -> u8;
    fn write(&mut self, addr: u16, v: u8);

    /// Opcode and operand bytes.
    #[inline(always)]
    fn fetch(&mut self, addr: u16) -> u8 {
        self.read(addr)
    }

    /// A data `LOAD` (operand reads, `(zp)` pointer bytes).
    #[inline(always)]
    fn load(&mut self, addr: u16) -> u8 {
        self.read(addr)
    }

    /// A data `STORE`.
    #[inline(always)]
    fn store(&mut self, addr: u16, v: u8) {
        self.write(addr, v)
    }

    /// A pull from page 1 (`PLA`/`PLP`/`RTS`/`RTI`).
    #[inline(always)]
    fn stack_read(&mut self, addr: u16) -> u8 {
        self.read(addr)
    }

    /// A push to page 1 (`PHA`/`PHP`/`JSR`/`BRK`, interrupt frames).
    #[inline(always)]
    fn stack_write(&mut self, addr: u16, v: u8) {
        self.write(addr, v)
    }

    /// Instruction (or interrupt) boundary: `cpu` as it is before it executes.
    #[inline(always)]
    fn sync(&mut self, _cpu: &Cpu) {}
}

impl Bus for [u8; 0x10000] {
    #[inline(always)]
    fn read(&mut self, addr: u16) -> u8 {
        self[addr as usize]
    }

    #[inline(always)]
    fn write(&mut self, addr: u16, v: u8) {
        self[addr as usize] = v
    }
}
