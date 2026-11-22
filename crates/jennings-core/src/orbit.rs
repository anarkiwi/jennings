//! An entry's orbit: its calls repeated from one state until the memory a call starts from
//! repeats, with every data access and control edge those calls take.

use std::collections::HashSet;

use crate::access::{Accesses, Remap};
use crate::bus::Bus;
use crate::cpu::Cpu;
use crate::run::{run_to_rts, AddrSet, Stop, Trace};

/// Why an [`orbit`] ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum End {
    /// A call started from a state seen before.
    Closed,
    /// The call limit was reached first.
    Open,
    /// A call loaded a volatile address (that call included).
    Volatile,
    /// A call jammed (that call included).
    Jam,
    /// The stop probe interrupted a call.
    Stopped,
}

/// The result of [`orbit`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Orbit {
    pub end: End,
    /// The distinct states the calls started from.
    pub states: usize,
    /// The distinct [`crate::access_key`]s of the calls' data accesses, ascending.
    pub accesses: Vec<u64>,
    /// The distinct `pc << 16 | next` of consecutive instructions within a call, ascending.
    pub edges: Vec<u64>,
}

/// A set of keys `hi << 16 | lo`: one 64 Ki-bit page per distinct `hi`.
struct Keys {
    slot: Vec<u32>,
    pages: Vec<(u32, Box<[u64; 1024]>)>,
}

impl Keys {
    fn new(his: usize) -> Self {
        Keys {
            slot: vec![0; his],
            pages: Vec::new(),
        }
    }

    #[inline(always)]
    fn insert(&mut self, key: u64) {
        let hi = (key >> 16) as usize;
        let s = match self.slot[hi] {
            0 => {
                self.pages.push((hi as u32, Box::new([0; 1024])));
                self.slot[hi] = self.pages.len() as u32;
                self.pages.len() - 1
            }
            i => i as usize - 1,
        };
        let lo = key as usize & 0xFFFF;
        self.pages[s].1[lo >> 6] |= 1 << (lo & 63);
    }

    fn sorted(mut self) -> Vec<u64> {
        self.pages.sort_unstable_by_key(|p| p.0);
        let mut out = Vec::new();
        for (hi, bits) in &self.pages {
            for (w, &b) in bits.iter().enumerate() {
                let mut b = b;
                while b != 0 {
                    out.push((*hi as u64) << 16 | (w as u64) << 6 | b.trailing_zeros() as u64);
                    b &= b - 1;
                }
            }
        }
        out
    }
}

/// Call `pc` with `A = acc` (each call to its RTS or `max_cycles`, as [`run_to_rts`]) from the
/// state `bus` holds, again and again, until `digest` of the state a call would start from
/// was seen before, `limit` calls ran, or a call loads an address in `volatile` or jams.
/// `remap` redirects its instructions' accesses ([`Accesses`]); `stopped` is polled after
/// each instruction.
#[allow(clippy::too_many_arguments)]
pub fn orbit<B: Bus>(
    cpu: &mut Cpu,
    bus: &mut B,
    pc: u16,
    acc: u8,
    max_cycles: u64,
    limit: usize,
    remap: Option<&Remap>,
    volatile: &AddrSet,
    mut digest: impl FnMut(&mut B) -> u128,
    mut stopped: impl FnMut() -> bool,
) -> Orbit {
    let (mut accesses, mut edges) = (Keys::new(1 << 17), Keys::new(1 << 16));
    let (mut seen, mut end) = (HashSet::new(), End::Open);
    let (mut trace, mut log) = (Trace::new(false), Vec::new());
    for k in 0..=limit {
        if !seen.insert(digest(bus)) {
            end = End::Closed;
            break;
        }
        if k == limit {
            break;
        }
        trace.pc.clear();
        trace.cycles.clear();
        log.clear();
        let mut a = Accesses::new(bus, remap);
        a.log = std::mem::take(&mut log);
        let stop = run_to_rts(
            cpu,
            &mut a,
            pc,
            acc,
            None,
            max_cycles,
            Some(&mut trace),
            &mut stopped,
        );
        log = a.log;
        let mut read_volatile = false;
        for &key in &log {
            accesses.insert(key);
            read_volatile |= key & 1 << 16 == 0 && volatile.contains(key as u16);
        }
        for w in trace.pc.windows(2) {
            edges.insert((w[0] as u64) << 16 | w[1] as u64);
        }
        if stop == Stop::Stopped {
            end = End::Stopped;
            break;
        }
        if read_volatile {
            end = End::Volatile;
            break;
        }
        if cpu.jammed {
            end = End::Jam;
            break;
        }
    }
    Orbit {
        end,
        states: seen.len(),
        accesses: accesses.sorted(),
        edges: edges.sorted(),
    }
}
