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
/// stores nothing. The page-zero pointer bytes of an instruction whose opcode is in
/// `pointer_opcodes` are accessed as they are: operand + X and operand + X + 1 for
/// `(zp,X)`, else operand and operand + 1 (each wrapping in page zero).
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

/// Whether `op` is `(zp,X)`: addressing-mode bits `bbb = 000` in group `cc = 01` or `11`.
#[inline(always)]
fn indexed_by_x(op: u8) -> bool {
    op & 0x1D == 0x01
}

/// The key [`Accesses`] logs a data access by: `pc << 17 | write << 16 | addr`.
#[inline(always)]
pub fn access_key(pc: u16, write: bool, addr: u16) -> u64 {
    (pc as u64) << 17 | (write as u64) << 16 | addr as u64
}

/// How an instruction's data accesses are based: [`Base::Abs`] and [`Base::Zp`] on the
/// operand of an indexed absolute or page-zero mode, [`Base::IndY`] on the pointer word of
/// `(zp),Y` (read by its first two loads), [`Base::Addr`] on the address accessed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Base {
    Addr,
    Abs,
    Zp,
    IndY,
}

/// The [`Base`] of opcode `op` by its addressing-mode bits `bbb`: `zp,X`/`zp,Y` (101),
/// `abs,X`/`abs,Y` (111, and 110 in groups `cc = 01`/`11`), `(zp),Y` (100 in those groups).
pub fn base_of(op: u8) -> Base {
    match (op >> 2 & 7, op & 1) {
        (5, _) => Base::Zp,
        (7, _) | (6, 1) => Base::Abs,
        (4, 1) => Base::IndY,
        _ => Base::Addr,
    }
}

/// `inner` with every data LOAD/STORE logged by [`access_key`] (the instruction's start PC)
/// and the accesses of `remap`'s instructions redirected; with [`Accesses::with_bases`]
/// also each access's base address ([`Base`]) in `bases`, in step with `log`.
pub struct Accesses<'a, B: Bus> {
    pub inner: &'a mut B,
    pub log: Vec<u64>,
    pub bases: Vec<u16>,
    logs: bool,
    based: bool,
    remap: Option<&'a Remap>,
    pc: u16,
    table: Option<&'a [i32]>,
    keep: Option<(u16, u16)>,
    base: Base,
    operand: [u8; 2],
    pointer: [u8; 2],
    loads: u8,
}

impl<'a, B: Bus> Accesses<'a, B> {
    pub fn new(inner: &'a mut B, remap: Option<&'a Remap>) -> Self {
        Accesses {
            inner,
            log: Vec::new(),
            bases: Vec::new(),
            logs: true,
            based: false,
            remap,
            pc: 0,
            table: None,
            keep: None,
            base: Base::Addr,
            operand: [0; 2],
            pointer: [0; 2],
            loads: 0,
        }
    }

    /// As [`Accesses::new`], also logging each access's base in `bases`.
    pub fn with_bases(inner: &'a mut B, remap: Option<&'a Remap>) -> Self {
        Accesses {
            based: true,
            ..Accesses::new(inner, remap)
        }
    }

    /// `inner` with `remap`'s instructions' accesses redirected and nothing logged.
    pub fn remapped(inner: &'a mut B, remap: &'a Remap) -> Self {
        Accesses::unlogged(inner, Some(remap))
    }

    /// `inner`, with `remap`'s instructions' accesses redirected if given, and nothing logged.
    pub fn unlogged(inner: &'a mut B, remap: Option<&'a Remap>) -> Self {
        Accesses {
            logs: false,
            ..Accesses::new(inner, remap)
        }
    }

    #[inline(always)]
    fn note(&mut self, write: bool, addr: u16) {
        if self.logs {
            self.log.push(access_key(self.pc, write, addr));
        }
        if self.based {
            self.bases.push(match self.base {
                Base::Abs => u16::from_le_bytes(self.operand),
                Base::Zp => self.operand[0] as u16,
                Base::IndY if self.loads >= 2 => u16::from_le_bytes(self.pointer),
                _ => addr,
            });
        }
    }

    /// Where the current instruction's access of `addr` goes: None as it is.
    #[inline(always)]
    pub fn target(&self, addr: u16) -> Option<i32> {
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

    /// A data `LOAD` of `addr` as the current instruction takes it: a redirected one is a
    /// load of its source, so an observer of `inner` sees the byte actually read.
    #[inline(always)]
    fn redirected_load(&mut self, addr: u16) -> u8 {
        match self.target(addr) {
            None => self.inner.load(addr),
            Some(ZERO) => 0,
            Some(s) => self.inner.load(s as u16),
        }
    }
}

impl<B: Bus> Bus for Accesses<'_, B> {
    #[inline(always)]
    fn target(&self, addr: u16) -> Option<i32> {
        Accesses::target(self, addr)
    }

    #[inline(always)]
    fn peek(&mut self, addr: u16) -> u8 {
        self.inner.peek(addr)
    }

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
        let v = self.inner.fetch(addr);
        if self.based {
            match addr.wrapping_sub(self.pc) {
                1 => self.operand[0] = v,
                2 => self.operand[1] = v,
                _ => {}
            }
        }
        v
    }

    #[inline(always)]
    fn load(&mut self, addr: u16) -> u8 {
        self.note(false, addr);
        let v = self.redirected_load(addr);
        if self.based && self.base == Base::IndY && self.loads < 2 {
            self.pointer[self.loads as usize] = v;
            self.loads += 1;
        }
        v
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
        if self.based {
            self.base = base_of(self.inner.fetch(cpu.pc));
            self.loads = 0;
        }
        if let (Some(r), Some(_)) = (self.remap, self.table) {
            let op = self.inner.fetch(cpu.pc);
            if r.pointer[op as usize] {
                let x = if indexed_by_x(op) { cpu.x } else { 0 };
                let zp = self.inner.fetch(cpu.pc.wrapping_add(1)).wrapping_add(x) as u16;
                self.keep = Some((zp, (zp + 1) & 0xFF));
            }
        }
    }
}
