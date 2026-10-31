//! Run-until drivers: step a [`Cpu`] until a stop condition, checked per instruction.

use crate::bus::Bus;
use crate::cpu::Cpu;

/// A set of 16-bit addresses (a 64 Ki-bit bitmap).
#[derive(Clone)]
pub struct AddrSet(Box<[u64]>);

impl Default for AddrSet {
    fn default() -> Self {
        AddrSet(vec![0; 1024].into_boxed_slice())
    }
}

impl AddrSet {
    pub fn insert(&mut self, a: u16) {
        self.0[a as usize >> 6] |= 1 << (a & 63);
    }

    /// Insert every address in the half-open range `[lo, hi)` (`hi` up to `0x10000`).
    pub fn insert_range(&mut self, lo: u32, hi: u32) {
        (lo..hi.min(0x10000)).for_each(|a| self.insert(a as u16));
    }

    #[inline(always)]
    pub fn contains(&self, a: u16) -> bool {
        self.0[a as usize >> 6] >> (a & 63) & 1 != 0
    }
}

impl FromIterator<u16> for AddrSet {
    fn from_iter<I: IntoIterator<Item = u16>>(it: I) -> Self {
        let mut s = AddrSet::default();
        it.into_iter().for_each(|a| s.insert(a));
        s
    }
}

/// Why a run ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    /// More than `max_cycles` elapsed.
    Cycles,
    /// PC entered `stop_pcs` (before executing it).
    Pc,
    /// SP rose above `sp_above`.
    Sp,
    /// A JAM executed.
    Jam,
    /// The `stopped` probe asked to stop after an instruction.
    Stopped,
    /// An instruction left both PC and SP unchanged (`JMP *`, a branch to itself).
    Idle,
    /// The I flag is clear (checked before an instruction).
    Irq,
}

/// Per-instruction record of a run: one row per executed instruction.
///
/// `pc` and `cycles` are those at the instruction's start; with `regs`, A/X/Y/SP/P
/// are those after it.
#[derive(Default)]
pub struct Trace {
    pub regs: bool,
    pub pc: Vec<u16>,
    pub cycles: Vec<u64>,
    pub a: Vec<u8>,
    pub x: Vec<u8>,
    pub y: Vec<u8>,
    pub sp: Vec<u8>,
    pub p: Vec<u8>,
}

impl Trace {
    pub fn new(regs: bool) -> Self {
        Trace {
            regs,
            ..Trace::default()
        }
    }

    #[inline(always)]
    fn start(&mut self, cpu: &Cpu) {
        self.pc.push(cpu.pc);
        self.cycles.push(cpu.cycles);
    }

    #[inline(always)]
    fn end(&mut self, cpu: &Cpu) {
        if self.regs {
            self.a.push(cpu.a);
            self.x.push(cpu.x);
            self.y.push(cpu.y);
            self.sp.push(cpu.sp);
            self.p.push(cpu.p());
        }
    }
}

/// A periodic IRQ: taken (unless masked) once the cycle count reaches `due`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Periodic {
    pub period: u64,
    pub due: u64,
}

/// Stop conditions; the boundary checks run in field order before each instruction.
///
/// `irq`, after each instruction, calls [`Cpu::irq`] when `due` is reached and
/// advances `due` by `period` (a masked IRQ is dropped, as `MPU.irq()` does).
#[derive(Default)]
pub struct Until<'a> {
    pub sp_above: Option<i32>,
    pub max_cycles: Option<u64>,
    pub stop_pcs: Option<&'a AddrSet>,
    pub stop_on_jam: bool,
    /// Stop after an instruction that leaves PC and SP as they were.
    pub stop_on_idle: bool,
    /// Stop before an instruction when the I flag is clear.
    pub stop_on_irq: bool,
    pub irq: Option<Periodic>,
    pub trace: Option<&'a mut Trace>,
}

/// Step until a condition in `until` holds or `stopped()` returns true after an instruction.
pub fn run_until<B: Bus, S: FnMut() -> bool>(
    cpu: &mut Cpu,
    bus: &mut B,
    until: &mut Until,
    mut stopped: S,
) -> Stop {
    let start = cpu.cycles;
    loop {
        if until.sp_above.is_some_and(|t| cpu.sp as i32 > t) {
            return Stop::Sp;
        }
        if until.max_cycles.is_some_and(|m| cpu.cycles - start > m) {
            return Stop::Cycles;
        }
        if until.stop_pcs.is_some_and(|s| s.contains(cpu.pc)) {
            return Stop::Pc;
        }
        if until.stop_on_irq && cpu.i == 0 {
            return Stop::Irq;
        }
        let jams = cpu.jams;
        let (pc0, sp0) = (cpu.pc, cpu.sp);
        if let Some(t) = until.trace.as_deref_mut() {
            t.start(cpu);
        }
        cpu.step(bus);
        if let Some(t) = until.trace.as_deref_mut() {
            t.end(cpu);
        }
        if let Some(p) = until.irq.as_mut() {
            if cpu.cycles >= p.due {
                cpu.irq(bus);
                p.due += p.period;
            }
        }
        if stopped() {
            return Stop::Stopped;
        }
        if until.stop_on_jam && cpu.jams != jams {
            return Stop::Jam;
        }
        if until.stop_on_idle && cpu.pc == pc0 && cpu.sp == sp0 {
            return Stop::Idle;
        }
    }
}

/// Call `pc` with `A = acc` until its balancing RTS lifts SP above its start.
///
/// The dummy return `$0100` is pushed first (`vagrant_moses.cpu.run_to_rts`);
/// `stop_pcs` are checked before each instruction (`run_to_rts_until`).
#[allow(clippy::too_many_arguments)]
pub fn run_to_rts<B: Bus, S: FnMut() -> bool>(
    cpu: &mut Cpu,
    bus: &mut B,
    pc: u16,
    acc: u8,
    stop_pcs: Option<&AddrSet>,
    max_cycles: u64,
    trace: Option<&mut Trace>,
    stopped: S,
) -> Stop {
    let start_sp = cpu.sp as i32;
    for byte in [0x00, 0x01] {
        bus.write(0x100 | cpu.sp as u16, byte);
        cpu.sp = cpu.sp.wrapping_sub(1);
    }
    cpu.a = acc;
    cpu.pc = pc;
    let mut until = Until {
        sp_above: Some(start_sp - 1),
        max_cycles: Some(max_cycles),
        stop_pcs,
        trace,
        ..Until::default()
    };
    run_until(cpu, bus, &mut until, stopped)
}
