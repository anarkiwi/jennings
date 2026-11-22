//! An entry's orbit: its calls repeated from one state until the memory a call starts from
//! repeats, with every data access and control edge those calls take.

use std::collections::HashSet;

use crate::access::Remap;
use crate::bus::Bus;
use crate::census::Classes;
use crate::cpu::Cpu;
use crate::run::{AddrSet, Stop};
use crate::taint::{observed_call, Observer, LOAD, STORE};

const STEP: u32 = 1 << 31;

/// A call's instructions and data accesses as [`orbit`] keeps them: [`STEP`] with each
/// instruction's PC, then `write << 16 | addr` of each data access it names.
#[derive(Default)]
struct Walk {
    ev: Vec<u32>,
    active: bool,
}

impl Observer for Walk {
    #[inline(always)]
    fn active(&self) -> bool {
        self.active
    }

    #[inline(always)]
    fn note(&mut self, kind: u8, addr: u16, _c: i32) {
        if kind == LOAD || kind == STORE {
            self.ev.push(((kind == STORE) as u32) << 16 | addr as u32);
        }
    }

    #[inline(always)]
    fn start(&mut self, pc: u16, _op: u8) {
        self.active = true;
        self.ev.push(STEP | pc as u32);
    }
}

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
#[derive(Clone)]
pub(crate) struct Keys {
    slot: Vec<u32>,
    pages: Vec<(u32, Box<[u64; 1024]>)>,
}

impl Keys {
    pub(crate) fn new(his: usize) -> Self {
        Keys {
            slot: vec![0; his],
            pages: Vec::new(),
        }
    }

    #[inline(always)]
    pub(crate) fn insert(&mut self, key: u64) {
        self.insert_new(key);
    }

    /// Insert `key`; whether it was not in the set.
    #[inline(always)]
    pub(crate) fn insert_new(&mut self, key: u64) -> bool {
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
        let (w, bit) = (&mut self.pages[s].1[lo >> 6], 1u64 << (lo & 63));
        let new = *w & bit == 0;
        *w |= bit;
        new
    }

    pub(crate) fn sorted(mut self) -> Vec<u64> {
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

/// Call `pc` with `A = acc` (each call to its RTS or `max_cycles`, as [`crate::run_to_rts`])
/// from the state `cpu` and `bus` hold, again and again, until `digest` of the state a call
/// would start from was seen before, `limit` calls ran, or a call loads an address in `volatile`
/// or jams. `remap` redirects its instructions' accesses ([`crate::Accesses`]); `stopped` is
/// polled after each instruction. A call whose instructions and accesses repeat an earlier
/// call's, in order, adds nothing, so only new sequences are added to the sets.
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
    mut digest: impl FnMut(&Cpu, &mut B) -> u128,
    mut stopped: impl FnMut() -> bool,
) -> Orbit {
    let (mut accesses, mut edges) = (Keys::new(1 << 17), Keys::new(1 << 16));
    let (mut seen, mut end) = (HashSet::new(), End::Open);
    let (mut eng, mut calls) = (Walk::default(), Classes::default());
    for k in 0..=limit {
        if !seen.insert(digest(cpu, bus)) {
            end = End::Closed;
            break;
        }
        if k == limit {
            break;
        }
        eng.ev.clear();
        eng.active = false;
        let stop = observed_call(cpu, bus, &mut eng, pc, acc, max_cycles, remap, &mut stopped);
        let mut read_volatile = false;
        if calls.intern(&eng.ev).1 {
            let mut prev: Option<u16> = None;
            for &e in &eng.ev {
                if e & STEP != 0 {
                    let at = e as u16;
                    if let Some(p) = prev {
                        edges.insert((p as u64) << 16 | at as u64);
                    }
                    prev = Some(at);
                    continue;
                }
                let key = (prev.unwrap_or(0) as u64) << 17 | e as u64;
                accesses.insert(key);
                read_volatile |= key & 1 << 16 == 0 && volatile.contains(key as u16);
            }
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
