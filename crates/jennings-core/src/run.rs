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
}

/// Stop conditions; the boundary checks run in field order before each instruction.
#[derive(Clone, Default)]
pub struct Until<'a> {
    pub sp_above: Option<i32>,
    pub max_cycles: Option<u64>,
    pub stop_pcs: Option<&'a AddrSet>,
    pub stop_on_jam: bool,
}

/// Step until a condition in `until` holds or `stopped()` returns true after an instruction.
pub fn run_until<B: Bus, S: FnMut() -> bool>(
    cpu: &mut Cpu,
    bus: &mut B,
    until: &Until,
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
        let jams = cpu.jams;
        cpu.step(bus);
        if stopped() {
            return Stop::Stopped;
        }
        if until.stop_on_jam && cpu.jams != jams {
            return Stop::Jam;
        }
    }
}

/// Call `pc` with `A = acc` until its balancing RTS lifts SP above its start.
///
/// The dummy return `$0100` is pushed first (`vagrant_moses.cpu.run_to_rts`);
/// `stop_pcs` are checked before each instruction (`run_to_rts_until`).
pub fn run_to_rts<B: Bus, S: FnMut() -> bool>(
    cpu: &mut Cpu,
    bus: &mut B,
    pc: u16,
    acc: u8,
    stop_pcs: Option<&AddrSet>,
    max_cycles: u64,
    stopped: S,
) -> Stop {
    let start_sp = cpu.sp as i32;
    for byte in [0x00, 0x01] {
        bus.stack_write(0x100 | cpu.sp as u16, byte);
        cpu.sp = cpu.sp.wrapping_sub(1);
    }
    cpu.a = acc;
    cpu.pc = pc;
    let until = Until {
        sp_above: Some(start_sp - 1),
        max_cycles: Some(max_cycles),
        stop_pcs,
        stop_on_jam: false,
    };
    run_until(cpu, bus, &until, stopped)
}
