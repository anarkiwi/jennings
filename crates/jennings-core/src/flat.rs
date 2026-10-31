//! A flat 64 KiB memory: RAM, per-4 KiB read-only overlays, an I/O page,
//! address watch sets and a raster model, with a hook trait for watch hits.

use std::cell::Cell;

use crate::bus::Bus;
use crate::cpu::Cpu;

/// What a non-fetch access is, as the [`Hooks`] see it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    /// Data LOAD/STORE.
    Data,
    /// Stack-page push/pull.
    Stack,
    /// Interrupt vectors and the `JMP (ind)` pointer.
    Other,
    /// The embedder's own `peek`/`poke`.
    Host,
}

/// Embedder callbacks; [`FlatMemory`] calls them only for watched addresses.
pub trait Hooks {
    /// A read of a read-watched address; `Some(v)` substitutes `v`.
    fn read_hit(&mut self, _mem: &FlatMemory, _addr: u16, _access: Access) -> Option<u8> {
        None
    }

    /// A write to a write-watched address; returns the value to store.
    fn write_hit(&mut self, _mem: &FlatMemory, _addr: u16, v: u8, _access: Access) -> u8 {
        v
    }

    /// A data LOAD of a load-observed address, after it completed.
    fn observe_load(&mut self, _mem: &FlatMemory, _addr: u16) {}

    /// A data STORE of a store-observed address with the instruction's value.
    fn observe_store(&mut self, _mem: &FlatMemory, _addr: u16, _v: u8) {}

    /// A non-fetch read of the I/O page while it is mapped.
    fn io_read(&mut self, mem: &FlatMemory, addr: u16) -> u8 {
        mem.ram(addr)
    }

    /// A write to the I/O page while it is mapped.
    fn io_write(&mut self, mem: &FlatMemory, addr: u16, v: u8) {
        mem.set_ram(addr, v)
    }
}

/// [`Hooks`] that do nothing: watches are inert, the I/O page is RAM.
pub struct NoHooks;

impl Hooks for NoHooks {}

pub const IO_PAGE: u16 = 0xD;
pub const VIC_CONTROL_1: u16 = 0xD011;
pub const VIC_RASTER: u16 = 0xD012;

const R_SUB: u8 = 1;
const R_OBS: u8 = 2;
const R_RASTER: u8 = 4;
const R_MAPPED: u8 = 8;
const W_SUB: u8 = 1;
const W_OBS: u8 = 2;
const W_IO: u8 = 4;

/// Bitmap over the 64 KiB address space.
struct Bits(Box<[Cell<u64>]>);

impl Bits {
    fn new() -> Self {
        Bits((0..1024).map(|_| Cell::new(0)).collect())
    }

    #[inline(always)]
    fn get(&self, a: u16) -> bool {
        self.0[a as usize >> 6].get() >> (a & 63) & 1 != 0
    }

    fn set(&self, a: u16, on: bool) {
        let w = &self.0[a as usize >> 6];
        let m = 1u64 << (a & 63);
        w.set(if on { w.get() | m } else { w.get() & !m });
    }

    fn page_any(&self, page: usize) -> bool {
        self.0[page * 4..page * 4 + 4].iter().any(|w| w.get() != 0)
    }
}

fn cells(n: usize) -> Box<[Cell<u8>]> {
    (0..n).map(|_| Cell::new(0)).collect()
}

/// The 64 KiB RAM: owned, or borrowed from an external writable buffer.
enum Ram {
    Owned(Box<[Cell<u8>]>),
    /// SAFETY contract of [`FlatMemory::with_shared_ram`]: valid for 64 KiB while `self` lives.
    Shared(*const Cell<u8>),
}

// SAFETY: `Shared` is a plain pointer to memory the embedder guarantees; `Cell` access is
// already the embedder's to serialise, as for the owned box.
unsafe impl Send for Ram {}

impl std::ops::Deref for Ram {
    type Target = [Cell<u8>];

    #[inline(always)]
    fn deref(&self) -> &[Cell<u8>] {
        match self {
            Ram::Owned(b) => b,
            // SAFETY: see `Ram::Shared`.
            Ram::Shared(p) => unsafe { std::slice::from_raw_parts(*p, 0x10000) },
        }
    }
}

/// 64 KiB RAM plus a page table of sixteen 4 KiB pages.
///
/// Reads of an overlaid page come from its image, writes always reach RAM
/// (ROM under RAM, as on a C64). Mutation goes through `&self`, so a hook may
/// rebank, poke or (un)watch from inside an access.
pub struct FlatMemory {
    ram: Ram,
    shadow: Box<[Cell<u8>]>,
    rpage: [Cell<u8>; 256],
    wpage: [Cell<u8>; 256],
    rsub: Bits,
    wsub: Bits,
    robs: Bits,
    wobs: Bits,
    overlay: Cell<u16>,
    io: Cell<bool>,
    raster: Cell<Option<(u32, u32)>>,
    now: Cell<u64>,
    last_load: Cell<Option<u16>>,
    stack_obs: Cell<bool>,
}

impl Default for FlatMemory {
    fn default() -> Self {
        Self::new()
    }
}

impl FlatMemory {
    pub fn new() -> Self {
        FlatMemory {
            ram: Ram::Owned(cells(0x10000)),
            shadow: cells(0x10000),
            rpage: std::array::from_fn(|_| Cell::new(0)),
            wpage: std::array::from_fn(|_| Cell::new(0)),
            rsub: Bits::new(),
            wsub: Bits::new(),
            robs: Bits::new(),
            wobs: Bits::new(),
            overlay: Cell::new(0),
            io: Cell::new(false),
            raster: Cell::new(None),
            now: Cell::new(0),
            last_load: Cell::new(None),
            stack_obs: Cell::new(false),
        }
    }

    /// A memory whose RAM is the 64 KiB at `ram`, not a copy.
    ///
    /// # Safety
    /// `ram` must be valid for reads and writes of 0x10000 bytes for the life of the
    /// returned value, and written by nothing else while a [`Cpu`] runs on it.
    pub unsafe fn with_shared_ram(ram: *mut u8) -> Self {
        FlatMemory {
            ram: Ram::Shared(ram as *const Cell<u8>),
            ..Self::new()
        }
    }

    /// The RAM as cells (`Cell<u8>` is `u8`-layout: `as_ptr` is a byte buffer).
    pub fn ram_cells(&self) -> &[Cell<u8>] {
        &self.ram
    }

    #[inline(always)]
    pub fn ram(&self, addr: u16) -> u8 {
        self.ram[addr as usize].get()
    }

    #[inline(always)]
    pub fn set_ram(&self, addr: u16, v: u8) {
        self.ram[addr as usize].set(v)
    }

    /// Map 4 KiB page `page` to read from `image` (exactly 4096 bytes), or back to RAM.
    pub fn set_page(&self, page: u8, image: Option<&[u8]>) {
        let base = (page as usize & 15) << 12;
        let bit = 1u16 << (page & 15);
        match image {
            Some(img) => {
                assert_eq!(img.len(), 0x1000, "page image must be 4096 bytes");
                for (c, &b) in self.shadow[base..base + 0x1000].iter().zip(img) {
                    c.set(b);
                }
                self.overlay.set(self.overlay.get() | bit);
            }
            None => self.overlay.set(self.overlay.get() & !bit),
        }
        self.refresh_block(page & 15);
    }

    /// Whether page `page` reads from an overlay image.
    pub fn page_mapped(&self, page: u8) -> bool {
        self.overlay.get() >> (page & 15) & 1 != 0
    }

    /// Map the I/O page (`$D000-$DFFF`): reads/writes route to the hooks, raster is live.
    pub fn set_io(&self, on: bool) {
        self.io.set(on);
        self.refresh_block(IO_PAGE as u8);
    }

    pub fn io(&self) -> bool {
        self.io.get()
    }

    /// `$D011`/`$D012` follow the beam at the cycle count as `(cycles_per_line, lines)`.
    pub fn set_raster(&self, model: Option<(u32, u32)>) {
        self.raster.set(model);
        self.refresh((VIC_RASTER >> 8) as usize);
    }

    pub fn raster(&self) -> Option<(u32, u32)> {
        self.raster.get()
    }

    /// The cycle count raster reads use (the CPU's, at the access's instruction start).
    pub fn now(&self) -> u64 {
        self.now.get()
    }

    pub fn set_now(&self, cycles: u64) {
        self.now.set(cycles)
    }

    /// The address of the most recent data LOAD, None before the first.
    pub fn last_load(&self) -> Option<u16> {
        self.last_load.get()
    }

    pub fn watch_read(&self, addr: u16, on: bool) {
        self.rsub.set(addr, on);
        self.refresh(addr as usize >> 8);
    }

    pub fn watch_write(&self, addr: u16, on: bool) {
        self.wsub.set(addr, on);
        self.refresh(addr as usize >> 8);
    }

    /// Also report `PHA`/`PHP`/`PLA`/`PLP` page-1 accesses to the load/store observers.
    pub fn observe_stack(&self, on: bool) {
        self.stack_obs.set(on)
    }

    #[inline(always)]
    fn observed(&self, access: Access) -> bool {
        access == Access::Data || (access == Access::Stack && self.stack_obs.get())
    }

    pub fn observe_load(&self, addr: u16, on: bool) {
        self.robs.set(addr, on);
        self.refresh(addr as usize >> 8);
    }

    pub fn observe_store(&self, addr: u16, on: bool) {
        self.wobs.set(addr, on);
        self.refresh(addr as usize >> 8);
    }

    fn refresh_block(&self, block: u8) {
        for p in (block as usize) << 4..((block as usize) + 1) << 4 {
            self.refresh(p);
        }
    }

    fn refresh(&self, p: usize) {
        let block = (p >> 4) as u16;
        let io = self.io.get() && block == IO_PAGE;
        let mut r = 0;
        if self.rsub.page_any(p) {
            r |= R_SUB;
        }
        if self.robs.page_any(p) {
            r |= R_OBS;
        }
        if io && self.raster.get().is_some() && p == (VIC_RASTER >> 8) as usize {
            r |= R_RASTER;
        }
        if io || self.overlay.get() >> block & 1 != 0 {
            r |= R_MAPPED;
        }
        let mut w = 0;
        if self.wsub.page_any(p) {
            w |= W_SUB;
        }
        if self.wobs.page_any(p) {
            w |= W_OBS;
        }
        if io {
            w |= W_IO;
        }
        self.rpage[p].set(r);
        self.wpage[p].set(w);
    }

    #[inline(always)]
    fn fetch(&self, addr: u16) -> u8 {
        let src = if self.overlay.get() >> (addr >> 12) & 1 != 0 {
            &*self.shadow
        } else {
            &*self.ram
        };
        src[addr as usize].get()
    }

    fn mapped<H: Hooks>(&self, h: &mut H, addr: u16) -> u8 {
        if self.overlay.get() >> (addr >> 12) & 1 != 0 {
            self.shadow[addr as usize].get()
        } else if self.io.get() && addr >> 12 == IO_PAGE {
            h.io_read(self, addr)
        } else {
            self.ram(addr)
        }
    }

    fn beam(&self, addr: u16) -> Option<u8> {
        let (cpl, lines) = self.raster.get()?;
        let line = (self.now.get() / cpl as u64 % lines as u64) as u32;
        match addr {
            VIC_RASTER => Some(line as u8),
            VIC_CONTROL_1 => Some(self.ram(VIC_CONTROL_1) & 0x7F | ((line >> 8) as u8 & 1) << 7),
            _ => None,
        }
    }

    #[inline(always)]
    fn read_as<H: Hooks>(&self, h: &mut H, addr: u16, access: Access) -> u8 {
        let f = self.rpage[addr as usize >> 8].get();
        if access == Access::Data {
            self.last_load.set(Some(addr));
        }
        if f == 0 {
            return self.ram(addr);
        }
        self.read_slow(h, addr, access, f)
    }

    #[inline(never)]
    fn read_slow<H: Hooks>(&self, h: &mut H, addr: u16, access: Access, f: u8) -> u8 {
        let mut v = if f & R_RASTER != 0 {
            self.beam(addr)
        } else {
            None
        };
        if f & R_SUB != 0 && self.rsub.get(addr) {
            if let Some(x) = h.read_hit(self, addr, access) {
                v = Some(x);
            }
        }
        let v = v.unwrap_or_else(|| self.mapped(h, addr));
        if self.observed(access) && f & R_OBS != 0 && self.robs.get(addr) {
            h.observe_load(self, addr);
        }
        v
    }

    #[inline(always)]
    fn write_as<H: Hooks>(&self, h: &mut H, addr: u16, v: u8, access: Access) {
        let f = self.wpage[addr as usize >> 8].get();
        if f == 0 {
            return self.set_ram(addr, v);
        }
        self.write_slow(h, addr, v, access, f)
    }

    #[inline(never)]
    fn write_slow<H: Hooks>(&self, h: &mut H, addr: u16, v: u8, access: Access, f: u8) {
        let stored = if f & W_SUB != 0 && self.wsub.get(addr) {
            h.write_hit(self, addr, v, access)
        } else {
            v
        };
        if f & W_IO != 0 {
            h.io_write(self, addr, stored);
        } else {
            self.set_ram(addr, stored);
        }
        if self.observed(access) && f & W_OBS != 0 && self.wobs.get(addr) {
            h.observe_store(self, addr, v);
        }
    }

    /// A host read: page-mapped, raster and read watches apply; observers do not.
    pub fn peek<H: Hooks>(&self, h: &mut H, addr: u16) -> u8 {
        self.read_as(h, addr, Access::Host)
    }

    /// A host write: write watches apply; observers do not.
    pub fn poke<H: Hooks>(&self, h: &mut H, addr: u16, v: u8) {
        self.write_as(h, addr, v, Access::Host)
    }

    /// A [`Bus`] over this memory dispatching watch hits to `hooks`.
    pub fn bus<'a, H: Hooks>(&'a self, hooks: &'a mut H) -> FlatBus<'a, H> {
        FlatBus { mem: self, hooks }
    }
}

/// [`FlatMemory`] as the CPU's [`Bus`].
pub struct FlatBus<'a, H: Hooks> {
    pub mem: &'a FlatMemory,
    pub hooks: &'a mut H,
}

impl<H: Hooks> Bus for FlatBus<'_, H> {
    #[inline(always)]
    fn read(&mut self, addr: u16) -> u8 {
        self.mem.read_as(self.hooks, addr, Access::Other)
    }

    #[inline(always)]
    fn write(&mut self, addr: u16, v: u8) {
        self.mem.write_as(self.hooks, addr, v, Access::Other)
    }

    #[inline(always)]
    fn fetch(&mut self, addr: u16) -> u8 {
        self.mem.fetch(addr)
    }

    #[inline(always)]
    fn load(&mut self, addr: u16) -> u8 {
        self.mem.read_as(self.hooks, addr, Access::Data)
    }

    #[inline(always)]
    fn store(&mut self, addr: u16, v: u8) {
        self.mem.write_as(self.hooks, addr, v, Access::Data)
    }

    #[inline(always)]
    fn stack_read(&mut self, addr: u16) -> u8 {
        self.mem.read_as(self.hooks, addr, Access::Stack)
    }

    #[inline(always)]
    fn stack_write(&mut self, addr: u16, v: u8) {
        self.mem.write_as(self.hooks, addr, v, Access::Stack)
    }

    #[inline(always)]
    fn sync(&mut self, cpu: &Cpu) {
        self.mem.now.set(cpu.cycles)
    }
}
