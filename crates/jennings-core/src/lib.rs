//! NMOS 6510 core whose opcode semantics are generated from deity-informant's lifter.
//!
//! [`Cpu::step`] runs one instruction on any [`Bus`]; [`FlatMemory`] is a banked
//! 64 KiB bus with watch sets; [`run`] holds the run-until drivers and [`orbit`] repeats an
//! entry's calls until its state repeats; [`taint`] learns how a call's state depends on the
//! state before it.

pub mod access;
pub mod bus;
pub mod cpu;
#[rustfmt::skip]
#[allow(clippy::all, unused_parens, unused_variables)]
mod cpu_gen;
pub mod flat;
#[rustfmt::skip]
mod flow_gen;
pub mod orbit;
pub mod run;
pub mod taint;

pub use access::{access_key, base_of, Accesses, Base, Remap, AS_IS, ZERO};
pub use bus::Bus;
pub use cpu::Cpu;
pub use flat::{Access, FlatBus, FlatMemory, Hooks, NoHooks};
pub use orbit::{orbit, End, Orbit};
pub use run::{run_to_rts, run_until, AddrSet, Periodic, Stop, Trace, Until};
pub use taint::{cells_digest, learn, trajectory, Learn, Learned, REG, STATE_REGS};
