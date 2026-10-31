//! `jennings._core`: the native 6510 core and memory for Python.
//!
//! State lives in `UnsafeCell`/`Cell`s behind frozen pyclasses. Every Python
//! entry point holds the GIL; a driver additionally claims its core and memory
//! for the running thread, so other threads are refused while it runs detached
//! and only the running thread's watch callbacks may touch them.

use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::ffi::{c_int, c_void};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use jennings_core::{
    run_to_rts, run_until, Access, AddrSet, Bus, Cpu, FlatBus, FlatMemory, Hooks, Stop, Until,
};
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::PySlice;

const PAL: (u32, u32) = (63, 312);
const NTSC: (u32, u32) = (65, 263);

thread_local!(static TOKEN: u8 = const { 0 });

fn token() -> usize {
    TOKEN.with(|t| t as *const u8 as usize)
}

/// Idle (0) or the token of the thread running a driver.
#[derive(Default)]
struct Owner(AtomicUsize);

impl Owner {
    fn check(&self) -> PyResult<()> {
        match self.0.load(Ordering::Acquire) {
            0 => Ok(()),
            t if t == token() => Ok(()),
            _ => Err(PyRuntimeError::new_err(
                "in use by a driver on another thread",
            )),
        }
    }

    fn here(&self) -> bool {
        self.0.load(Ordering::Acquire) == token()
    }

    fn claim(&self) -> PyResult<Claim<'_>> {
        self.0
            .compare_exchange(0, token(), Ordering::AcqRel, Ordering::Acquire)
            .map(|_| Claim(self))
            .map_err(|_| PyRuntimeError::new_err("a driver is already running on it"))
    }
}

struct Claim<'a>(&'a Owner);

impl Drop for Claim<'_> {
    fn drop(&mut self) {
        self.0 .0.store(0, Ordering::Release);
    }
}

#[derive(Default)]
struct Subs {
    read: HashMap<u16, Vec<Py<PyAny>>>,
    write: HashMap<u16, Vec<Py<PyAny>>>,
    on_load: Option<Py<PyAny>>,
    on_store: Option<Py<PyAny>>,
}

/// 64 KiB C64-style memory for the native core (`jennings.memory.NativeMemory`).
///
/// py65-style ``mem[addr]``/``mem[a:b]`` read through the page table and the
/// read watches and write through the write watches; the buffer protocol
/// exposes the RAM itself. Watches have ``ObservableMemory`` semantics and fire
/// on every non-fetch access; :meth:`observe_data_access` sees data only.
#[pyclass(module = "jennings._core", name = "NativeMemory", frozen)]
pub struct Memory {
    flat: FlatMemory,
    subs: UnsafeCell<Subs>,
    owner: Owner,
}

// SAFETY: every access holds the GIL; while a driver runs detached, `owner`
// refuses all threads but the driver's, whose callbacks re-attach first.
unsafe impl Sync for Memory {}

impl Memory {
    #[allow(clippy::mut_from_ref)]
    fn subs(&self, _py: Python<'_>) -> &mut Subs {
        // SAFETY: GIL held (token `_py`); callers never hold two borrows at once.
        unsafe { &mut *self.subs.get() }
    }

    fn host<R>(&self, f: impl FnOnce(&mut PyHooks) -> R) -> PyResult<R> {
        self.owner.check()?;
        let mut h = PyHooks::new(self, None);
        let r = f(&mut h);
        h.err.map_or(Ok(r), Err)
    }

    fn index(addr: i64) -> u16 {
        (addr & 0xFFFF) as u16
    }
}

fn byte(v: i64) -> PyResult<u8> {
    u8::try_from(v).map_err(|_| PyValueError::new_err("byte must be in range(0, 256)"))
}

/// Bytes from any buffer (bytes, bytearray, memoryview, numpy) or iterable of ints.
fn octets(obj: &Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    if let Ok(buf) = PyBuffer::<u8>::get(obj) {
        return buf.to_vec(obj.py());
    }
    obj.try_iter()?
        .map(|v| v.and_then(|v| v.extract::<i64>()).and_then(byte))
        .collect()
}

fn addrs(it: &Bound<'_, PyAny>) -> PyResult<Vec<u16>> {
    it.try_iter()?
        .map(|a| a.and_then(|a| a.extract::<i64>()).map(Memory::index))
        .collect()
}

#[pymethods]
impl Memory {
    #[new]
    #[pyo3(signature = (data=None))]
    fn new(data: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let m = Memory {
            flat: FlatMemory::new(),
            subs: UnsafeCell::default(),
            owner: Owner::default(),
        };
        if let Some(d) = data.map(octets).transpose()? {
            if d.len() > 0x10000 {
                return Err(PyValueError::new_err("more than 64 KiB"));
            }
            m.store(0, &d);
        }
        Ok(m)
    }

    fn __len__(&self) -> usize {
        0x10000
    }

    fn __getitem__(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        if let Ok(s) = key.cast::<PySlice>() {
            let ix = s.indices(0x10000)?;
            let vals = self.host(|h| {
                (0..ix.slicelength)
                    .map(|k| self.flat.peek(h, (ix.start + k as isize * ix.step) as u16))
                    .collect::<Vec<u8>>()
            })?;
            return Ok(pyo3::types::PyList::new(py, vals)?.into_any().unbind());
        }
        let a = Self::index(key.extract()?);
        Ok(self
            .host(|h| self.flat.peek(h, a))?
            .into_pyobject(py)?
            .into_any()
            .unbind())
    }

    fn __setitem__(&self, key: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        if let Ok(s) = key.cast::<PySlice>() {
            let ix = s.indices(0x10000)?;
            let vals = octets(value)?;
            return self.host(|h| {
                for (k, v) in vals.into_iter().take(ix.slicelength).enumerate() {
                    self.flat
                        .poke(h, (ix.start + k as isize * ix.step) as u16, v);
                }
            });
        }
        let (a, v) = (Self::index(key.extract()?), byte(value.extract()?)?);
        self.host(|h| self.flat.poke(h, a, v))
    }

    fn __iter__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let all = PySlice::full(py);
        let vals = self.__getitem__(py, all.as_any())?;
        Ok(vals.bind(py).try_iter()?.into_any().unbind())
    }

    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        let ram = slf.get().flat.ram_cells().as_ptr() as *mut c_void;
        // SAFETY: the RAM is a 64 KiB `Cell<u8>` box owned by `slf`, which the view keeps alive.
        if unsafe { ffi::PyBuffer_FillInfo(view, slf.as_ptr(), ram, 0x10000, 0, flags) } == -1 {
            return Err(PyErr::fetch(slf.py()));
        }
        Ok(())
    }

    /// Occurrences of byte `v` in RAM.
    fn count(&self, v: i64) -> usize {
        self.flat
            .ram_cells()
            .iter()
            .filter(|c| c.get() as i64 == v)
            .count()
    }

    /// Copy `data` into RAM at `start`, bypassing watches (``ObservableMemory.write``).
    fn write(&self, start: i64, data: &Bound<'_, PyAny>) -> PyResult<()> {
        self.owner.check()?;
        self.store(Self::index(start), &octets(data)?);
        Ok(())
    }

    /// Map 4 KiB page `n` to a read-only 4096-byte `image`, or back to RAM with None.
    #[pyo3(signature = (n, image=None))]
    fn set_page(&self, n: u8, image: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
        self.owner.check()?;
        let image = image.map(octets).transpose()?;
        if n > 15 || image.as_ref().is_some_and(|i| i.len() != 0x1000) {
            return Err(PyValueError::new_err("page 0-15 with a 4096-byte image"));
        }
        self.flat.set_page(n, image.as_deref());
        Ok(())
    }

    /// Whether page `n` reads from an overlay image.
    fn page_mapped(&self, n: u8) -> bool {
        self.flat.page_mapped(n)
    }

    /// The I/O page `$D000-$DFFF` is banked in: the raster model answers there.
    #[getter]
    fn get_io(&self) -> bool {
        self.flat.io()
    }

    #[setter]
    fn set_io(&self, on: bool) -> PyResult<()> {
        self.owner.check()?;
        self.flat.set_io(on);
        Ok(())
    }

    /// `$D011`/`$D012` read the beam at the cycle count (PAL, or NTSC if true); None disables.
    #[pyo3(signature = (ntsc=None))]
    fn raster_model(&self, ntsc: Option<bool>) -> PyResult<()> {
        self.owner.check()?;
        self.flat
            .set_raster(ntsc.map(|n| if n { NTSC } else { PAL }));
        Ok(())
    }

    /// Address of the most recent data LOAD by the CPU.
    #[getter]
    fn last_load(&self) -> u16 {
        self.flat.last_load()
    }

    /// Call `cb(addr)` on each read of `addrs`; a non-None result substitutes the value.
    fn subscribe_to_read(
        &self,
        py: Python<'_>,
        addrs_: &Bound<'_, PyAny>,
        cb: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.subscribe(py, addrs(addrs_)?, cb, true)
    }

    /// Call `cb(addr, value)` on each write of `addrs`; a non-None result is stored instead.
    fn subscribe_to_write(
        &self,
        py: Python<'_>,
        addrs_: &Bound<'_, PyAny>,
        cb: Bound<'_, PyAny>,
    ) -> PyResult<()> {
        self.subscribe(py, addrs(addrs_)?, cb, false)
    }

    /// Route the CPU's data LOAD/STORE (never fetch or stack) per byte to callbacks.
    ///
    /// ``on_read(addr)`` after a load, ``on_write(addr, value)`` with the instruction's
    /// value after a store. ``reads``/``writes`` limit them to those addresses
    /// (default: all). Replaces any previous observers; both None clears them.
    #[pyo3(signature = (on_read=None, on_write=None, reads=None, writes=None))]
    fn observe_data_access(
        &self,
        py: Python<'_>,
        on_read: Option<Py<PyAny>>,
        on_write: Option<Py<PyAny>>,
        reads: Option<&Bound<'_, PyAny>>,
        writes: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<()> {
        self.owner.check()?;
        let pick = |on: bool, set: Option<&Bound<'_, PyAny>>| -> PyResult<Vec<bool>> {
            let mut bits = vec![on && set.is_none(); 0x10000];
            if let (true, Some(s)) = (on, set) {
                addrs(s)?.into_iter().for_each(|a| bits[a as usize] = true);
            }
            Ok(bits)
        };
        let (rb, wb) = (
            pick(on_read.is_some(), reads)?,
            pick(on_write.is_some(), writes)?,
        );
        for a in 0..=0xFFFFu16 {
            self.flat.observe_load(a, rb[a as usize]);
            self.flat.observe_store(a, wb[a as usize]);
        }
        let s = self.subs(py);
        s.on_load = on_read;
        s.on_store = on_write;
        Ok(())
    }
}

impl Memory {
    fn store(&self, start: u16, data: &[u8]) {
        for (k, &b) in data.iter().enumerate() {
            self.flat.set_ram(start.wrapping_add(k as u16), b);
        }
    }

    fn subscribe(
        &self,
        py: Python<'_>,
        addrs: Vec<u16>,
        cb: Bound<'_, PyAny>,
        read: bool,
    ) -> PyResult<()> {
        self.owner.check()?;
        for a in addrs {
            let list = {
                let s = self.subs(py);
                if read { &mut s.read } else { &mut s.write }
                    .entry(a)
                    .or_default()
            };
            let mut seen = false;
            for c in list.iter() {
                if cb.eq(c.bind(py))? {
                    seen = true;
                    break;
                }
            }
            if !seen {
                list.push(cb.clone().unbind());
            }
            if read {
                self.flat.watch_read(a, true);
            } else {
                self.flat.watch_write(a, true);
            }
        }
        Ok(())
    }
}

/// [`Hooks`] calling the Python callbacks; the first exception stops the run.
struct PyHooks<'a> {
    m: &'a Memory,
    stop: Option<&'a AtomicBool>,
    err: Option<PyErr>,
}

impl<'a> PyHooks<'a> {
    fn new(m: &'a Memory, stop: Option<&'a AtomicBool>) -> Self {
        PyHooks { m, stop, err: None }
    }

    fn fail(&mut self, e: PyErr) {
        self.err.get_or_insert(e);
        if let Some(s) = self.stop {
            s.store(true, Ordering::Relaxed);
        }
    }

    fn callbacks(&self, py: Python<'_>, addr: u16, read: bool) -> Vec<Py<PyAny>> {
        let s = self.m.subs(py);
        let map = if read { &s.read } else { &s.write };
        map.get(&addr)
            .map(|v| v.iter().map(|c| c.clone_ref(py)).collect())
            .unwrap_or_default()
    }

    fn result(&mut self, py: Python<'_>, r: PyResult<Py<PyAny>>) -> Option<u8> {
        match r.and_then(|r| r.extract::<Option<i64>>(py)) {
            Ok(v) => v.map(|v| v as u8),
            Err(e) => {
                self.fail(e);
                None
            }
        }
    }
}

impl Hooks for PyHooks<'_> {
    fn read_hit(&mut self, _mem: &FlatMemory, addr: u16, _access: Access) -> Option<u8> {
        if self.err.is_some() {
            return None;
        }
        Python::attach(|py| {
            let mut out = None;
            for cb in self.callbacks(py, addr, true) {
                let r = cb.call1(py, (addr,));
                if let Some(v) = self.result(py, r) {
                    out = Some(v);
                }
                if self.err.is_some() {
                    break;
                }
            }
            out
        })
    }

    fn write_hit(&mut self, _mem: &FlatMemory, addr: u16, v: u8, _access: Access) -> u8 {
        if self.err.is_some() {
            return v;
        }
        Python::attach(|py| {
            let mut v = v;
            for cb in self.callbacks(py, addr, false) {
                let r = cb.call1(py, (addr, v));
                if let Some(x) = self.result(py, r) {
                    v = x;
                }
                if self.err.is_some() {
                    break;
                }
            }
            v
        })
    }

    fn observe_load(&mut self, _mem: &FlatMemory, addr: u16) {
        if self.err.is_none() {
            Python::attach(|py| {
                if let Some(cb) = self.m.subs(py).on_load.as_ref().map(|c| c.clone_ref(py)) {
                    let r = cb.call1(py, (addr,)).map(|_| ());
                    if let Err(e) = r {
                        self.fail(e);
                    }
                }
            })
        }
    }

    fn observe_store(&mut self, _mem: &FlatMemory, addr: u16, v: u8) {
        if self.err.is_none() {
            Python::attach(|py| {
                if let Some(cb) = self.m.subs(py).on_store.as_ref().map(|c| c.clone_ref(py)) {
                    let r = cb.call1(py, (addr, v)).map(|_| ());
                    if let Err(e) = r {
                        self.fail(e);
                    }
                }
            })
        }
    }
}

/// The memory bus plus a per-instruction register snapshot for callbacks.
struct PyBus<'a, 'h> {
    inner: FlatBus<'h, PyHooks<'a>>,
    snap: *mut Cpu,
}

impl Bus for PyBus<'_, '_> {
    #[inline(always)]
    fn read(&mut self, addr: u16) -> u8 {
        self.inner.read(addr)
    }
    #[inline(always)]
    fn write(&mut self, addr: u16, v: u8) {
        self.inner.write(addr, v)
    }
    #[inline(always)]
    fn fetch(&mut self, addr: u16) -> u8 {
        self.inner.fetch(addr)
    }
    #[inline(always)]
    fn load(&mut self, addr: u16) -> u8 {
        self.inner.load(addr)
    }
    #[inline(always)]
    fn store(&mut self, addr: u16, v: u8) {
        self.inner.store(addr, v)
    }
    #[inline(always)]
    fn stack_read(&mut self, addr: u16) -> u8 {
        self.inner.stack_read(addr)
    }
    #[inline(always)]
    fn stack_write(&mut self, addr: u16, v: u8) {
        self.inner.stack_write(addr, v)
    }
    #[inline(always)]
    fn sync(&mut self, cpu: &Cpu) {
        self.inner.sync(cpu);
        // SAFETY: `snap` is only read by this thread's callbacks, never borrowed across this.
        unsafe { self.snap.write(cpu.clone()) }
    }
}

/// `*const Core` the detached closure may carry (the claim keeps it exclusive).
#[derive(Clone, Copy)]
struct CorePtr(*const Core);
// SAFETY: the pointee is Sync and outlives the detached run, which the caller joins.
unsafe impl Send for CorePtr {}

/// Native 6510 core over a :class:`NativeMemory` (wrapped by ``jennings.devices.mpu6502.MPU``).
#[pyclass(module = "jennings._core", name = "Core", frozen)]
pub struct Core {
    cpu: UnsafeCell<Cpu>,
    snap: UnsafeCell<Cpu>,
    mem: Py<Memory>,
    stop: AtomicBool,
    owner: Owner,
}

// SAFETY: as for `Memory`: GIL plus the `owner` claim serialise all access.
unsafe impl Sync for Core {}

impl Core {
    /// Registers as Python may see them: the instruction-start snapshot while running.
    fn regs(&self) -> PyResult<&Cpu> {
        self.owner.check()?;
        // SAFETY: while this thread runs a driver only `snap` is read here; else `cpu` is idle.
        Ok(unsafe {
            &*if self.owner.here() {
                self.snap.get()
            } else {
                self.cpu.get()
            }
        })
    }

    #[allow(clippy::mut_from_ref)]
    fn regs_mut(&self) -> PyResult<&mut Cpu> {
        self.owner.check()?;
        if self.owner.here() {
            return Err(PyRuntimeError::new_err(
                "registers are read-only while the core runs",
            ));
        }
        // SAFETY: idle core, GIL held.
        Ok(unsafe { &mut *self.cpu.get() })
    }

    /// Run `f` on the claimed core and bus, detached from the interpreter if `detach`.
    fn drive<R: Send>(
        &self,
        py: Python<'_>,
        detach: bool,
        f: impl FnOnce(&mut Cpu, &mut PyBus, &AtomicBool) -> R + Send,
    ) -> PyResult<R> {
        let mem = self.mem.get();
        let _c = self.owner.claim()?;
        let _m = mem.owner.claim()?;
        self.stop.store(false, Ordering::Relaxed);
        let me = CorePtr(self);
        let run = move || {
            let me = me;
            // SAFETY: claimed above; the claim guards are dropped only after this returns.
            let core = unsafe { &*me.0 };
            let cpu = unsafe { &mut *core.cpu.get() };
            let mem = core.mem.get();
            let mut hooks = PyHooks::new(mem, Some(&core.stop));
            let r = {
                let mut bus = PyBus {
                    inner: mem.flat.bus(&mut hooks),
                    snap: core.snap.get(),
                };
                f(cpu, &mut bus, &core.stop)
            };
            mem.flat.set_now(cpu.cycles);
            (r, hooks.err)
        };
        let (r, err) = if detach { py.detach(run) } else { run() };
        err.map_or(Ok(r), Err)
    }
}

fn reason(s: Stop) -> &'static str {
    match s {
        Stop::Cycles => "cycles",
        Stop::Pc => "pc",
        Stop::Sp => "sp",
        Stop::Jam => "jam",
        Stop::Stopped => "stop",
    }
}

fn pc_set(pcs: Option<&Bound<'_, PyAny>>) -> PyResult<Option<AddrSet>> {
    pcs.map(|p| addrs(p).map(|v| v.into_iter().collect()))
        .transpose()
}

#[pymethods]
impl Core {
    #[new]
    fn new(mem: Py<Memory>) -> Self {
        Core {
            cpu: UnsafeCell::new(Cpu::new()),
            snap: UnsafeCell::new(Cpu::new()),
            mem,
            stop: AtomicBool::new(false),
            owner: Owner::default(),
        }
    }

    #[getter]
    fn memory(&self, py: Python<'_>) -> Py<Memory> {
        self.mem.clone_ref(py)
    }

    #[getter]
    fn get_a(&self) -> PyResult<u8> {
        Ok(self.regs()?.a)
    }

    #[setter]
    fn set_a(&self, v: i64) -> PyResult<()> {
        self.regs_mut()?.a = v as u8;
        Ok(())
    }

    #[getter]
    fn get_x(&self) -> PyResult<u8> {
        Ok(self.regs()?.x)
    }

    #[setter]
    fn set_x(&self, v: i64) -> PyResult<()> {
        self.regs_mut()?.x = v as u8;
        Ok(())
    }

    #[getter]
    fn get_y(&self) -> PyResult<u8> {
        Ok(self.regs()?.y)
    }

    #[setter]
    fn set_y(&self, v: i64) -> PyResult<()> {
        self.regs_mut()?.y = v as u8;
        Ok(())
    }

    #[getter]
    fn get_sp(&self) -> PyResult<u8> {
        Ok(self.regs()?.sp)
    }

    #[setter]
    fn set_sp(&self, v: i64) -> PyResult<()> {
        self.regs_mut()?.sp = v as u8;
        Ok(())
    }

    #[getter]
    fn get_p(&self) -> PyResult<u8> {
        Ok(self.regs()?.p())
    }

    #[setter]
    fn set_p(&self, v: i64) -> PyResult<()> {
        self.regs_mut()?.set_p(v as u8);
        Ok(())
    }

    #[getter]
    fn get_pc(&self) -> PyResult<u16> {
        Ok(self.regs()?.pc)
    }

    #[setter]
    fn set_pc(&self, v: i64) -> PyResult<()> {
        self.regs_mut()?.pc = v as u16;
        Ok(())
    }

    #[getter]
    fn get_cycles(&self) -> PyResult<u64> {
        Ok(self.regs()?.cycles)
    }

    #[setter]
    fn set_cycles(&self, v: u64) -> PyResult<()> {
        self.regs_mut()?.cycles = v;
        self.mem.get().flat.set_now(v);
        Ok(())
    }

    #[getter]
    fn get_jammed(&self) -> PyResult<bool> {
        Ok(self.regs()?.jammed)
    }

    #[setter]
    fn set_jammed(&self, v: bool) -> PyResult<()> {
        self.regs_mut()?.jammed = v;
        Ok(())
    }

    /// Set by a watch callback to end the current run at the next instruction boundary.
    #[getter]
    fn get_stop(&self) -> bool {
        self.stop.load(Ordering::Relaxed)
    }

    #[setter]
    fn set_stop(&self, v: bool) {
        self.stop.store(v, Ordering::Relaxed)
    }

    /// Execute one instruction; cycles taken.
    fn step(&self, py: Python<'_>) -> PyResult<u32> {
        self.drive(py, false, |cpu, bus, _| cpu.step(bus))
    }

    /// Clear registers and cycles; PC from `pc`, else the reset vector.
    #[pyo3(signature = (pc=None))]
    fn reset(&self, py: Python<'_>, pc: Option<u16>) -> PyResult<()> {
        self.drive(py, false, |cpu, bus, _| cpu.reset(bus, pc))
    }

    /// Take an IRQ unless I is set; cycles taken.
    fn irq(&self, py: Python<'_>) -> PyResult<u32> {
        self.drive(py, false, |cpu, bus, _| cpu.irq(bus))
    }

    /// Take an NMI; cycles taken.
    fn nmi(&self, py: Python<'_>) -> PyResult<u32> {
        self.drive(py, false, |cpu, bus, _| cpu.nmi(bus))
    }

    /// Run with the GIL released until a condition holds; the reason as a string.
    #[pyo3(signature = (max_cycles=None, stop_pcs=None, sp_above=None, stop_on_jam=true))]
    fn run_until(
        &self,
        py: Python<'_>,
        max_cycles: Option<u64>,
        stop_pcs: Option<&Bound<'_, PyAny>>,
        sp_above: Option<i32>,
        stop_on_jam: bool,
    ) -> PyResult<&'static str> {
        let pcs = pc_set(stop_pcs)?;
        self.drive(py, true, move |cpu, bus, stop| {
            let until = Until {
                sp_above,
                max_cycles,
                stop_pcs: pcs.as_ref(),
                stop_on_jam,
            };
            reason(run_until(cpu, bus, &until, || {
                stop.load(Ordering::Relaxed) && stop.swap(false, Ordering::Relaxed)
            }))
        })
    }

    /// `vagrant_moses.cpu.run_to_rts`: call `pc` with `A = acc` until its RTS or the cap.
    fn run_to_rts(&self, py: Python<'_>, pc: u16, acc: i64, max_cycles: u64) -> PyResult<()> {
        self.drive(py, true, move |cpu, bus, stop| {
            run_to_rts(cpu, bus, pc, acc as u8, None, max_cycles, || {
                stop.load(Ordering::Relaxed) && stop.swap(false, Ordering::Relaxed)
            });
        })
    }

    /// `pysidtracker.emu.run_to_rts_until`: as `run_to_rts`, True on entering `pcs`.
    fn run_to_rts_until(
        &self,
        py: Python<'_>,
        pc: u16,
        acc: i64,
        pcs: &Bound<'_, PyAny>,
        max_cycles: u64,
    ) -> PyResult<bool> {
        let set = pc_set(Some(pcs))?;
        self.drive(py, true, move |cpu, bus, stop| {
            let r = run_to_rts(cpu, bus, pc, acc as u8, set.as_ref(), max_cycles, || {
                stop.load(Ordering::Relaxed) && stop.swap(false, Ordering::Relaxed)
            });
            r == Stop::Pc
        })
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Memory>()?;
    m.add_class::<Core>()?;
    Ok(())
}
