//! One run of an entry's calls that sorts them into classes by what [`crate::Learn`] sees of
//! each: its instructions and opcodes in order, each followed by the locations its accesses
//! reach. Learning is a function of that sequence, of the control dependences and of the
//! memory the calls write, so a learner needs to taint only each class's first call, from the
//! entry state snapshotted here (see `docs/coverage.md`).

use crate::access::Remap;
use crate::bus::Bus;
use crate::cpu::Cpu;
use crate::flow_gen::FLOW;
use crate::orbit::Keys;
use crate::run::{AddrSet, Stop};
use crate::taint::{
    ended, observed_call, value, FxMap, Observer, LOAD, PC_OUT, READ, STACK_LOAD, STACK_STORE,
    STORE, STORE0, WRITE,
};

const INSTR: u32 = 1 << 31;
const CONSTANT: u32 = 0xFFFFF;
const NONE: u32 = u32::MAX;

/// What [`Census::run`] saw of the calls it ran, each first-seen item with the call it was
/// first seen in, so any prefix of the calls has its own.
pub struct Census {
    cells: Vec<u32>,
    init: Vec<u8>,
    class: Vec<u32>,
    classes: Classes,
    first: Vec<u32>,
    regs: Vec<Cpu>,
    snap: Vec<u8>,
    snap_at: Vec<usize>,
    dirty: Vec<u16>,
    edge_seen: Keys,
    exit_seen: AddrSet,
    wrote_seen: AddrSet,
    /// `(pc << 16 | next, call)`: consecutive instructions within a call, each first seen in
    /// `call`.
    pub edges: Vec<(u64, u32)>,
    /// `(pc, call)`: the instructions that ended a call.
    pub exits: Vec<(u16, u32)>,
    /// `(address, call)`: the memory a call writes, as [`crate::Learned::written`] gives it.
    pub written: Vec<(u16, u32)>,
    /// Per cell, its value at each call's entry.
    pub columns: Vec<Vec<u8>>,
    cur: Vec<u32>,
    active: bool,
    dirty_at: Vec<u32>,
    stamp: Vec<u32>,
    undo: Vec<(u16, u8)>,
}

impl Observer for Census {
    #[inline(always)]
    fn active(&self) -> bool {
        self.active
    }

    #[inline(always)]
    fn note(&mut self, kind: u8, c: i32) {
        self.cur.push((kind as u32) << 20 | (c + 2) as u32);
    }

    #[inline(always)]
    fn start(&mut self, pc: u16, op: u8) {
        self.active = true;
        self.cur.push(INSTR | (pc as u32) << 8 | op as u32);
    }

    const WRITES: bool = true;

    #[inline(always)]
    fn wrote(&mut self, addr: u16, old: u8) {
        let call = self.class.len() as u32 + 1;
        if self.stamp[addr as usize] != call {
            self.stamp[addr as usize] = call;
            self.undo.push((addr, old));
        }
    }
}

impl Census {
    /// A census recording the locations `cells` ([`value`]) at each call's entry.
    pub fn new(cells: Vec<u32>) -> Self {
        Census {
            cells,
            init: Vec::new(),
            class: Vec::new(),
            classes: Classes::default(),
            first: Vec::new(),
            regs: Vec::new(),
            snap: Vec::new(),
            snap_at: vec![0],
            dirty: Vec::new(),
            edge_seen: Keys::new(1 << 16),
            exit_seen: AddrSet::default(),
            wrote_seen: AddrSet::default(),
            edges: Vec::new(),
            exits: Vec::new(),
            written: Vec::new(),
            columns: Vec::new(),
            cur: Vec::new(),
            active: false,
            dirty_at: vec![NONE; 0x10000],
            stamp: vec![0; 0x10000],
            undo: Vec::new(),
        }
    }

    /// The calls run so far.
    pub fn calls(&self) -> usize {
        self.class.len()
    }

    /// The distinct classes of the calls so far.
    pub fn classes(&self) -> usize {
        self.first.len()
    }

    /// The cells recorded in [`Census::columns`].
    pub fn cells(&self) -> &[u32] {
        &self.cells
    }

    /// The class of call `k`, numbered in order of first call.
    pub fn class(&self, k: usize) -> Option<u32> {
        self.class.get(k).copied()
    }

    /// Run `calls` more calls of `pc` with `A = acc` (as [`crate::Learn::run`], `remap`
    /// applied) from the state `cpu` and `bus` hold, which `peek` reads; stops as it does.
    #[allow(clippy::too_many_arguments)]
    pub fn run<B: Bus>(
        &mut self,
        cpu: &mut Cpu,
        bus: &mut B,
        pc: u16,
        acc: u8,
        max_cycles: u64,
        calls: usize,
        remap: Option<&Remap>,
        mut peek: impl FnMut(&mut B, u16) -> u8,
        mut stopped: impl FnMut() -> bool,
    ) -> Stop {
        if self.init.is_empty() {
            self.init = (0..=0xFFFF).map(|a| peek(bus, a)).collect();
        }
        self.columns.resize(self.cells.len(), Vec::new());
        for col in &mut self.columns {
            col.reserve(calls);
        }
        for _ in 0..calls {
            for (col, &c) in self.columns.iter_mut().zip(&self.cells) {
                col.push(value(cpu, bus, &mut peek, c));
            }
            let (regs, dirty) = (cpu.clone(), self.dirty.len());
            self.cur.clear();
            self.undo.clear();
            self.active = false;
            let stop = observed_call(cpu, bus, self, pc, acc, max_cycles, remap, &mut stopped);
            if self.classify(regs) {
                let at = self.snap.len();
                for &a in &self.dirty[..dirty] {
                    self.snap.push(peek(bus, a));
                }
                for &(a, old) in &self.undo {
                    match self.dirty_at[a as usize] as usize {
                        i if i < dirty => self.snap[at + i] = old,
                        _ => {}
                    }
                }
                self.snap_at.push(self.snap.len());
            }
            if let Some(s) = ended(cpu, stop) {
                return s;
            }
        }
        Stop::Sp
    }

    /// Into `out`, the events of class `class` with each read of an address no call of the
    /// census writes and `constant` holds made one alike: such a read taints nothing, so
    /// [`crate::Learn`] learns calls equal so alike.
    pub(crate) fn canonical(&self, class: u32, constant: impl Fn(u16) -> bool, out: &mut Vec<u32>) {
        out.clear();
        out.extend(self.classes.events(class).iter().map(|&e| {
            let (kind, c) = (e >> 20 & 7, (e & 0xFFFFF) as i32 - 2);
            let read = e & INSTR == 0 && matches!(kind as u8, LOAD | STACK_LOAD | READ);
            match read && c >= 0 && !self.wrote_seen.contains(c as u16) && constant(c as u16) {
                true => kind << 20 | CONSTANT,
                false => e,
            }
        }));
    }

    /// Class the call just run by its events; whether it is a new class, entered with the
    /// registers `regs` (its memory snapshot is the caller's to add).
    fn classify(&mut self, regs: Cpu) -> bool {
        let k = self.class.len() as u32;
        let (c, new) = self.classes.intern(&self.cur);
        self.class.push(c);
        if !new {
            return false;
        }
        self.first.push(k);
        let sp = regs.sp;
        self.regs.push(regs);
        self.derive(k, sp);
        true
    }

    fn dirty(&mut self, a: u16) {
        if self.dirty_at[a as usize] == NONE {
            self.dirty_at[a as usize] = self.dirty.len() as u32;
            self.dirty.push(a);
        }
    }

    /// The edges, exit and written memory first seen in call `k` (entered with `sp`), from
    /// its events, and the memory it modifies.
    fn derive(&mut self, k: u32, sp: u8) {
        let cur = std::mem::take(&mut self.cur);
        let mut prev: Option<u16> = None;
        let mut i = 0;
        while i < cur.len() {
            let e = cur[i];
            i += 1;
            if e & INSTR == 0 {
                continue;
            }
            let (pc, op) = ((e >> 8) as u16, e as u8);
            if let Some(p) = prev {
                let key = (p as u64) << 16 | pc as u64;
                if self.edge_seen.insert_new(key) {
                    self.edges.push((key, k));
                }
            }
            prev = Some(pc);
            let (mut stores, mut writes) = ([-1i32; 8], [-1i32; 3]);
            let (mut ns, mut nw) = (0, 0);
            while i < cur.len() && cur[i] & INSTR == 0 {
                let (kind, c) = ((cur[i] >> 20) as u8, (cur[i] & 0xFFFFF) as i32 - 2);
                i += 1;
                if matches!(kind, STORE | STACK_STORE | WRITE) && c >= 0 {
                    self.dirty(c as u16);
                }
                match kind {
                    STORE | STACK_STORE => {
                        stores[ns] = c;
                        ns += 1;
                    }
                    WRITE => {
                        writes[nw.min(2)] = c;
                        nw += 1;
                    }
                    _ => {}
                }
            }
            let outs = FLOW[op as usize].outs.iter().map(|o| o.0);
            let stored = outs.filter(|&d| d >= STORE0 && d != PC_OUT);
            let stored = stored.map(|d| stores[(d - STORE0) as usize]);
            for c in stored.chain(writes[..nw.min(3)].iter().copied()) {
                if c >= 0 && !self.wrote_seen.contains(c as u16) {
                    self.wrote_seen.insert(c as u16);
                    self.written.push((c as u16, k));
                }
            }
        }
        if let Some(p) = prev {
            if !self.exit_seen.contains(p) {
                self.exit_seen.insert(p);
                self.exits.push((p, k));
            }
        }
        self.dirty(0x100 | sp as u16);
        self.dirty(0x100 | sp.wrapping_sub(1) as u16);
        self.cur = cur;
    }

    /// Set `cpu` and, through `poke`, the memory to the state class `class`'s first call
    /// started from; `poke` reaches every address a call modifies.
    pub fn restore(&self, class: usize, cpu: &mut Cpu, mut poke: impl FnMut(u16, u8)) {
        *cpu = self.regs[class].clone();
        let snap = &self.snap[self.snap_at[class]..self.snap_at[class + 1]];
        for (i, &a) in self.dirty.iter().enumerate() {
            poke(a, snap.get(i).copied().unwrap_or(self.init[a as usize]));
        }
    }
}

/// Distinct sequences of events, numbered in order of first sight.
#[derive(Default)]
pub(crate) struct Classes {
    index: FxMap<u64, u32>,
    chain: Vec<u32>,
    events: Vec<u32>,
    at: Vec<usize>,
}

impl Classes {
    pub(crate) fn len(&self) -> usize {
        self.chain.len()
    }

    pub(crate) fn events(&self, class: u32) -> &[u32] {
        let i = class as usize;
        &self.events[if i == 0 { 0 } else { self.at[i - 1] }..self.at[i]]
    }

    /// The number of `ev`, and whether it is new; equal hashes are compared in full.
    pub(crate) fn intern(&mut self, ev: &[u32]) -> (u32, bool) {
        let h = digest(ev);
        let mut at = self.index.get(&h).copied().unwrap_or(NONE);
        while at != NONE {
            if self.events(at) == ev {
                return (at, false);
            }
            at = self.chain[at as usize];
        }
        let c = self.len() as u32;
        self.chain.push(self.index.insert(h, c).unwrap_or(NONE));
        self.events.extend_from_slice(ev);
        self.at.push(self.events.len());
        (c, true)
    }
}

/// A 64-bit hash of `ev`, on four independent lanes.
fn digest(ev: &[u32]) -> u64 {
    const K: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut lanes = [K, K.rotate_left(17), K.rotate_left(31), K.rotate_left(47)];
    let (chunks, rest) = ev.as_chunks::<8>();
    for c in chunks {
        for (j, l) in lanes.iter_mut().enumerate() {
            let w = c[2 * j] as u64 | (c[2 * j + 1] as u64) << 32;
            *l = (*l ^ w).wrapping_mul(K).rotate_left(29);
        }
    }
    let mut h = ev.len() as u64;
    for (j, &e) in rest.iter().enumerate() {
        lanes[j & 3] = (lanes[j & 3] ^ e as u64).wrapping_mul(K).rotate_left(29);
    }
    for l in lanes {
        h = (h ^ l).wrapping_mul(K).rotate_left(23);
    }
    h
}
