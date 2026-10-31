//! NMOS 6510 core whose opcode semantics are generated from deity-informant's lifter.
//!
//! [`Cpu::step`] runs one instruction on any [`Bus`]; [`FlatMemory`] is a banked
//! 64 KiB bus with watch sets; [`run`] holds the run-until drivers.

pub mod bus;
pub mod cpu;
#[rustfmt::skip]
#[allow(clippy::all, unused_parens, unused_variables)]
mod cpu_gen;
pub mod flat;
pub mod run;

pub use bus::Bus;
pub use cpu::Cpu;
pub use flat::{Access, FlatBus, FlatMemory, Hooks, NoHooks};
pub use run::{run_to_rts, run_until, AddrSet, Periodic, Stop, Trace, Until};
