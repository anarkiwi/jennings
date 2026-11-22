//! A [`Bus`] adapter that logs each data access and redirects the accesses of chosen
//! instructions through per-address tables.

use crate::bus::Bus;
use crate::cpu::Cpu;

/// The address an access by a redirected instruction reaches as it is.
pub const AS_IS: i32 = -1;
/// The table entry of an address whose reads give 0.
pub const ZERO: i32 = -2;

/// Per-instruction address redirection.
///
/// An instruction at a PC with a table accesses each address whose entry is not
/// [`AS_IS`] elsewhere: a read gives the byte at the entry ([`ZERO`]: 0) and a write
/// stores nothing. The page-zero pointer bytes (operand and operand + 1) of an
/// instruction whose opcode is in `pointer_opcodes` are accessed as they are.
pub struct Remap {
    site: Box<[u32]>,
    tables: Vec<Box<[i32]>>,
    pointer: [bool; 256],
}

impl Default for Remap {
    fn default() -> Self {
        Remap {
            site: vec![0; 0x10000].into_boxed_slice(),
            tables: Vec::new(),
            pointer: [false; 256],
        }
    }
}

impl Remap {
    /// Redirect the instruction at `pc` through `table` (64 Ki entries, replacing any before).
    pub fn insert(&mut self, pc: u16, table: &[i32]) {
        assert_eq!(table.len(), 0x10000, "a table has one entry per address");
        let t = table.to_vec().into_boxed_slice();
        match self.site[pc as usize] {
            0 => {
                self.tables.push(t);
                self.site[pc as usize] = self.tables.len() as u32;
            }
            i => self.tables[i as usize - 1] = t,
        }
    }

    /// Opcodes whose page-zero pointer bytes are never redirected.
    pub fn set_pointer_opcodes(&mut self, ops: impl IntoIterator<Item = u8>) {
        self.pointer = [false; 256];
        ops.into_iter()
            .for_each(|o| self.pointer[o as usize] = true);
    }

    fn table(&self, pc: u16) -> Option<&[i32]> {
        match self.site[pc as usize] {
            0 => None,
            i => Some(&self.tables[i as usize - 1]),
        }
    }
}

/// The key [`Accesses`] logs a data access by: `pc << 17 | write << 16 | addr`.
#[inline(always)]
pub fn access_key(pc: u16, write: bool, addr: u16) -> u64 {
    (pc as u64) << 17 | (write as u64) << 16 | addr as u64
}

/// `inner` with every data LOAD/STORE logged by [`access_key`] (the instruction's start PC)
/// and the accesses of `remap`'s instructions redirected.
pub struct Accesses<'a, B: Bus> {
    pub inner: &'a mut B,
    pub log: Vec<u64>,
    logs: bool,
    remap: Option<&'a Remap>,
    pc: u16,
    table: Option<&'a [i32]>,
    keep: Option<(u16, u16)>,
}

impl<'a, B: Bus> Accesses<'a, B> {
    pub fn new(inner: &'a mut B, remap: Option<&'a Remap>) -> Self {
        Accesses {
            inner,
            log: Vec::new(),
            logs: true,
            remap,
            pc: 0,
            table: None,
            keep: None,
        }
    }

    /// `inner` with `remap`'s instructions' accesses redirected and nothing logged.
    pub fn remapped(inner: &'a mut B, remap: &'a Remap) -> Self {
        Accesses {
            logs: false,
            ..Accesses::new(inner, Some(remap))
        }
    }

    #[inline(always)]
    fn note(&mut self, write: bool, addr: u16) {
        if self.logs {
            self.log.push(access_key(self.pc, write, addr));
        }
    }

    /// Where the current instruction's access of `addr` goes: None as it is.
    #[inline(always)]
    fn target(&self, addr: u16) -> Option<i32> {
        let src = self.table?[addr as usize];
        if src == AS_IS || self.keep.is_some_and(|(a, b)| addr == a || addr == b) {
            return None;
        }
        Some(src)
    }

    #[inline(always)]
    fn redirected_read(&mut self, addr: u16) -> Option<u8> {
        self.target(addr).map(|src| match src {
            ZERO => 0,
            s => self.inner.read(s as u16),
        })
    }
}

impl<B: Bus> Bus for Accesses<'_, B> {
    #[inline(always)]
    fn read(&mut self, addr: u16) -> u8 {
        self.redirected_read(addr)
            .unwrap_or_else(|| self.inner.read(addr))
    }

    #[inline(always)]
    fn write(&mut self, addr: u16, v: u8) {
        if self.target(addr).is_none() {
            self.inner.write(addr, v)
        }
    }

    #[inline(always)]
    fn fetch(&mut self, addr: u16) -> u8 {
        self.inner.fetch(addr)
    }

    #[inline(always)]
    fn load(&mut self, addr: u16) -> u8 {
        self.note(false, addr);
        self.redirected_read(addr)
            .unwrap_or_else(|| self.inner.load(addr))
    }

    #[inline(always)]
    fn store(&mut self, addr: u16, v: u8) {
        self.note(true, addr);
        if self.target(addr).is_none() {
            self.inner.store(addr, v)
        }
    }

    #[inline(always)]
    fn stack_read(&mut self, addr: u16) -> u8 {
        self.redirected_read(addr)
            .unwrap_or_else(|| self.inner.stack_read(addr))
    }

    #[inline(always)]
    fn stack_write(&mut self, addr: u16, v: u8) {
        if self.target(addr).is_none() {
            self.inner.stack_write(addr, v)
        }
    }

    #[inline(always)]
    fn sync(&mut self, cpu: &Cpu) {
        self.inner.sync(cpu);
        self.pc = cpu.pc;
        self.table = self.remap.and_then(|r| r.table(cpu.pc));
        self.keep = None;
        if let (Some(r), Some(_)) = (self.remap, self.table) {
            if r.pointer[self.inner.fetch(cpu.pc) as usize] {
                let zp = self.inner.fetch(cpu.pc.wrapping_add(1)) as u16;
                self.keep = Some((zp, (zp + 1) & 0xFF));
            }
        }
    }
}
