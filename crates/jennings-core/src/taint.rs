//! Dependence of an entry's state on its state at the call before: each location written by a
//! call is tainted with the entry locations its value came from, by data
//! ([`crate::flow_gen`], read off the lifter's P-Code as the core is) and by control (the
//! conditions of the branches the writing instruction is control dependent on), or in explicit
//! mode by data alone, value apart from address, constant reads named by where they happen.
//!
//! Locations are the 64 KiB of memory, [`REG`] plus a `PcodeVM` register index and, in
//! explicit mode, [`CONST`] plus the PC of an instruction loading a constant.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

use crate::access::{Accesses, Remap};
use crate::bus::Bus;
use crate::census::{Census, Classes};
use crate::cpu::Cpu;
use crate::flow_gen::FLOW;
use crate::orbit::Keys;
use crate::run::{run_to_rts, AddrSet, Stop};

/// The location of `PcodeVM` register `i` is `REG + i` (A 0, X 1, Y 2, SP 3, C 8 ... N 14).
pub const REG: u32 = 0x10000;
/// The location `CONST + pc`: a constant byte an explicit [`Learn`] saw loaded at `pc`.
pub const CONST: u32 = 0x20000;
const LOCS: usize = REG as usize + 16;
const SITE_BITS: u32 = 18;
const SP: u32 = 1 << 3;
const FLAGS: u32 = 0x6F00;
/// Registers that carry state from one call into the next (X, Y and the flags but B).
pub const STATE_REGS: u32 = 0b110 | FLAGS;
const O1: u32 = 16;
const O2: u32 = 17;
const LOAD0: u32 = 20;
pub(crate) const STORE0: u8 = 16;
pub(crate) const PC_OUT: u8 = 31;

/// An opcode's control class, as `deity_informant.lifter.template` gives it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    Next,
    Branch,
    Jmp,
    Jsr,
    JmpInd,
    Rts,
    Rti,
    Brk,
    Jam,
}

/// An opcode's data flow over symbols: bit `r < 16` register `r`, 16/17 operand bytes 1/2,
/// `20 + k` the byte of load `k`. `outs` maps a register, `16 + k` store `k`'s byte or 31 a
/// branch's condition to its inputs; `load_addr`/`store_addr` give each access's address inputs;
/// `vals` is `outs` with each load depending only on the bytes it reads, not its address.
pub(crate) struct Flow {
    pub kind: Kind,
    pub len: u8,
    pub outs: &'static [(u8, u32)],
    pub load_addr: &'static [u32],
    pub store_addr: &'static [u32],
    pub vals: &'static [(u8, u32)],
}

#[derive(Default)]
pub(crate) struct Fx(pub(crate) u64);

impl Fx {
    #[inline(always)]
    pub(crate) fn add(&mut self, v: u64) {
        self.0 = (self.0.rotate_left(5) ^ v).wrapping_mul(0x517C_C1B7_2722_0A95);
    }
}

impl Hasher for Fx {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for c in bytes.chunks(8) {
            let mut w = [0u8; 8];
            w[..c.len()].copy_from_slice(c);
            self.add(u64::from_le_bytes(w));
        }
    }

    fn write_u32(&mut self, v: u32) {
        self.add(v as u64)
    }

    fn write_u64(&mut self, v: u64) {
        self.add(v)
    }

    fn write_usize(&mut self, v: usize) {
        self.add(v as u64)
    }
}

pub(crate) type FxMap<K, V> = HashMap<K, V, BuildHasherDefault<Fx>>;
pub(crate) type FxSet<K> = HashSet<K, BuildHasherDefault<Fx>>;

/// Interned sorted location sets; id 0 is the empty set and unions are memoised.
struct Sets {
    members: Vec<Box<[u32]>>,
    index: FxMap<Box<[u32]>, u32>,
    unions: FxMap<u64, u32>,
    single: Vec<u32>,
}

impl Sets {
    fn new() -> Self {
        let mut index = FxMap::default();
        index.insert(Box::from([]), 0);
        Sets {
            members: vec![Box::from([])],
            index,
            unions: FxMap::default(),
            single: vec![0; CONST as usize + 0x10000],
        }
    }

    fn intern(&mut self, v: Vec<u32>) -> u32 {
        if let Some(&i) = self.index.get(&v[..]) {
            return i;
        }
        let id = self.members.len() as u32;
        let b: Box<[u32]> = v.into();
        self.members.push(b.clone());
        self.index.insert(b, id);
        id
    }

    #[inline]
    fn single(&mut self, loc: u32) -> u32 {
        match self.single[loc as usize] {
            0 => {
                let id = self.intern(vec![loc]);
                self.single[loc as usize] = id;
                id
            }
            id => id,
        }
    }

    #[inline]
    fn union(&mut self, a: u32, b: u32) -> u32 {
        if a == b || b == 0 {
            return a;
        }
        if a == 0 {
            return b;
        }
        let key = (a.min(b) as u64) << 32 | a.max(b) as u64;
        if let Some(&u) = self.unions.get(&key) {
            return u;
        }
        let (x, y) = (&self.members[a as usize], &self.members[b as usize]);
        let (mut i, mut j, mut v) = (0, 0, Vec::with_capacity(x.len() + y.len()));
        while i < x.len() && j < y.len() {
            let (p, q) = (x[i], y[j]);
            v.push(p.min(q));
            i += (p <= q) as usize;
            j += (q <= p) as usize;
        }
        v.extend_from_slice(&x[i..]);
        v.extend_from_slice(&y[j..]);
        let u = self.intern(v);
        self.unions.insert(key, u);
        u
    }
}

pub(crate) const LOAD: u8 = 0;
pub(crate) const STACK_LOAD: u8 = 1;
pub(crate) const STORE: u8 = 2;
pub(crate) const STACK_STORE: u8 = 3;
pub(crate) const READ: u8 = 4;
pub(crate) const WRITE: u8 = 5;
pub(crate) const SUPPRESSED: i32 = -1;

/// What [`learn`] gathers over the calls it runs; [`Learn::run`] adds calls to it.
pub struct Learn {
    sets: Sets,
    written: AddrSet,
    cd_off: Vec<u32>,
    cd: Vec<u16>,
    reg_reads: [u32; 256],
    call: u32,
    stamp: Vec<u32>,
    taint: Vec<u32>,
    touched: Vec<u32>,
    cstamp: Vec<u32>,
    cond: Vec<u32>,
    dirty: Vec<u32>,
    read_first: Vec<bool>,
    wrote: Vec<bool>,
    deps: Vec<u32>,
    store_sites: FxSet<u64>,
    implicit_seen: FxSet<u64>,
    implicit: Vec<(u32, u16)>,
    sites: [Vec<u32>; 4],
    per_loc: [Vec<u32>; 3],
    edges: Keys,
    rec: Vec<(u8, i32)>,
    pc: u16,
    op: u8,
    prev: Option<u16>,
    exits: AddrSet,
    active: bool,
    explicit: bool,
    is_branch: Vec<bool>,
    ran: Vec<u16>,
    replay: Vec<(usize, Vec<(u16, u32)>)>,
    alike: Vec<u32>,
    coarse: Classes,
    scratch: Vec<u32>,
}

/// What [`Learn`] found, each list ascending.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Learned {
    /// Locations some call reads before writing (operand bytes read at fetch) and some call
    /// writes; of the registers only [`STATE_REGS`].
    pub persistent: Vec<u32>,
    /// `loc << 32 | src`: the end-of-call value of `loc` depends on `src` at the call's entry
    /// (or, explicit, on the constant `src` = [`CONST`] + pc loaded); `src` may be `loc`.
    pub deps: Vec<u64>,
    /// `pc << 16 | next` of consecutive instructions within a call.
    pub edges: Vec<u64>,
    /// `(pc << 2 | kind) << 18 | loc`: `loc` taints the address of a data load (kind 0) or
    /// store (1) at `pc`, where control goes from `pc` (2), or the byte a data store at `pc`
    /// stores (3).
    pub sites: Vec<u64>,
    /// Explicit only, `loc << 32 | src` over every data store to memory `loc`: `src` taints
    /// the stored byte.
    pub stored: Vec<u64>,
    /// Explicit only, as `stored`: `src` taints the store's address.
    pub addressed: Vec<u64>,
    /// Explicit only, as `stored`: `src` taints the conditions of the branches the store is
    /// under, transitively, and of those its store sites are control dependent on.
    pub gated: Vec<u64>,
    /// The PCs of the instructions that ended a call.
    pub exits: Vec<u16>,
    /// The memory some call wrote.
    pub written: Vec<u16>,
}

impl Learn {
    /// `control[pc]`: the branches `pc` is directly control dependent on; `written`: the
    /// memory a call may write (the rest is constant, so taints nothing). `explicit`: outputs
    /// take only their data inputs by [`Flow::vals`] (no opcode byte, no control), and a data
    /// load of a constant at `pc` is tainted with [`CONST`] + `pc`.
    pub fn new(control: &[(u16, Vec<u16>)], written: AddrSet, explicit: bool) -> Self {
        let mut by_pc: Vec<Vec<u16>> = vec![Vec::new(); 0x10000];
        let mut is_branch = vec![false; 0x10000];
        for (pc, brs) in control {
            by_pc[*pc as usize].extend_from_slice(brs);
            brs.iter().for_each(|&b| is_branch[b as usize] = true);
        }
        let (mut cd_off, mut cd) = (Vec::with_capacity(0x10001), Vec::new());
        for v in &mut by_pc {
            v.sort_unstable();
            v.dedup();
            cd_off.push(cd.len() as u32);
            cd.extend_from_slice(v);
        }
        cd_off.push(cd.len() as u32);
        let mut reg_reads = [0u32; 256];
        for (op, f) in FLOW.iter().enumerate() {
            let ins = f.outs.iter().map(|o| o.1);
            let all = ins
                .chain(f.load_addr.iter().copied())
                .chain(f.store_addr.iter().copied());
            reg_reads[op] = all.fold(0, |m, v| m | v) & 0xFFFF
                | match f.kind {
                    Kind::Jsr | Kind::Rts | Kind::Rti => SP,
                    Kind::Brk => SP | FLAGS,
                    _ => 0,
                };
        }
        Learn {
            sets: Sets::new(),
            written,
            cd_off,
            cd,
            reg_reads,
            call: 1,
            stamp: vec![0; LOCS],
            taint: vec![0; LOCS],
            touched: vec![0; LOCS],
            cstamp: vec![0; 0x10000],
            cond: vec![0; 0x10000],
            dirty: Vec::new(),
            read_first: vec![false; LOCS],
            wrote: vec![false; LOCS],
            deps: vec![0; LOCS],
            store_sites: FxSet::default(),
            implicit_seen: FxSet::default(),
            implicit: Vec::new(),
            sites: std::array::from_fn(|_| vec![0; 0x10000]),
            per_loc: std::array::from_fn(|_| {
                if explicit {
                    vec![0; 0x10000]
                } else {
                    Vec::new()
                }
            }),
            edges: Keys::new(1 << 16),
            rec: Vec::with_capacity(16),
            pc: 0,
            op: 0,
            prev: None,
            exits: AddrSet::default(),
            active: false,
            explicit,
            is_branch,
            ran: Vec::new(),
            replay: Vec::new(),
            alike: Vec::new(),
            coarse: Classes::default(),
            scratch: Vec::new(),
        }
    }

    /// Calls run so far.
    pub fn calls(&self) -> u32 {
        self.call - 1
    }

    /// The classes [`Learn::run_census`] has learned, one call of each tainted.
    pub fn classes(&self) -> usize {
        self.coarse.len()
    }

    /// The taint `loc` holds now: its own at entry unless constant or written this call.
    #[inline(always)]
    fn get(&mut self, loc: u32) -> u32 {
        if self.stamp[loc as usize] == self.call {
            return self.taint[loc as usize];
        }
        let state = if loc < REG {
            self.written.contains(loc as u16)
        } else {
            STATE_REGS >> (loc - REG) & 1 != 0
        };
        if state {
            self.sets.single(loc)
        } else {
            0
        }
    }

    #[inline(always)]
    fn get_mem(&mut self, a: u16) -> u32 {
        self.get(a as u32)
    }

    #[inline(always)]
    fn cell(&mut self, c: i32) -> u32 {
        if c < 0 {
            0
        } else {
            self.get(c as u32)
        }
    }

    /// The taint of a load of `c`: [`CONST`] + pc for an explicit data load of a constant.
    #[inline(always)]
    fn load_taint(&mut self, c: i32, data: bool) -> u32 {
        if self.explicit
            && data
            && c >= 0
            && self.stamp[c as usize] != self.call
            && !self.written.contains(c as u16)
        {
            self.sets.single(CONST + self.pc as u32)
        } else {
            self.cell(c)
        }
    }

    /// The union of the taints of `mask`'s symbols, `loads` the taints of the loads.
    fn gather(&mut self, mask: u32, loads: &[u32]) -> u32 {
        let (mut s, mut m) = (0, mask);
        while m != 0 {
            let b = m.trailing_zeros();
            m &= m - 1;
            let t = match b {
                0..=15 => self.get(REG + b),
                O1 => self.get_mem(self.pc.wrapping_add(1)),
                O2 => self.get_mem(self.pc.wrapping_add(2)),
                _ => loads[(b - LOAD0) as usize],
            };
            s = self.sets.union(s, t);
        }
        s
    }

    #[inline(always)]
    fn touch(&mut self, loc: u32, read: bool) {
        let l = loc as usize;
        if self.touched[l] != self.call {
            self.touched[l] = self.call;
            self.read_first[l] |= read;
        }
    }

    fn begin(&mut self, pc: u16, op: u8) {
        if let Some(p) = self.prev {
            self.edges.insert((p as u64) << 16 | pc as u64);
        }
        self.prev = Some(pc);
        self.pc = pc;
        self.op = op;
        self.rec.clear();
        self.active = true;
    }

    /// Taint the current instruction's outputs from its inputs as they were before it.
    fn finish(&mut self) {
        if !self.active {
            return;
        }
        self.active = false;
        let (pc, op) = (self.pc, self.op);
        let f = &FLOW[op as usize];
        for k in 0..f.len as u16 {
            self.touch(pc.wrapping_add(k) as u32, true);
        }
        let rec = std::mem::take(&mut self.rec);
        let (mut loads, mut stores, mut reads, mut writes) = (
            [SUPPRESSED; 8],
            [SUPPRESSED; 8],
            [SUPPRESSED; 3],
            [SUPPRESSED; 3],
        );
        let (mut nl, mut ns, mut nr, mut nw) = (0, 0, 0, 0);
        let (mut data_loads, mut data_stores) = (0u8, 0u8);
        for &(kind, c) in &rec {
            if c >= 0 {
                self.touch(c as u32, kind == LOAD || kind == STACK_LOAD || kind == READ);
            }
            match kind {
                LOAD | STACK_LOAD => {
                    data_loads |= ((kind == LOAD) as u8) << nl;
                    loads[nl] = c;
                    nl += 1;
                }
                STORE | STACK_STORE => {
                    data_stores |= ((kind == STORE) as u8) << ns;
                    stores[ns] = c;
                    ns += 1;
                }
                READ => {
                    reads[nr.min(2)] = c;
                    nr += 1;
                }
                _ => {
                    writes[nw.min(2)] = c;
                    nw += 1;
                }
            }
        }
        self.rec = rec;
        let mut lt = [0u32; 8];
        for (k, t) in lt[..nl].iter_mut().enumerate() {
            *t = self.load_taint(loads[k], data_loads >> k & 1 != 0);
        }
        let loads = lt;
        let mut m = self.reg_reads[op as usize];
        while m != 0 {
            self.touch(REG + m.trailing_zeros(), true);
            m &= m - 1;
        }
        let code = self.get_mem(pc);
        let mut ctl = 0;
        for k in self.cd_off[pc as usize]..self.cd_off[pc as usize + 1] {
            let br = self.cd[k as usize] as usize;
            if self.cstamp[br] == self.call {
                ctl = self.sets.union(ctl, self.cond[br]);
            }
        }
        let base = if self.explicit {
            0
        } else {
            self.sets.union(code, ctl)
        };
        let mut out: [(u32, u32); 16] = [(0, 0); 16];
        let mut n = 0;
        let mut ctrl = code;
        for &(dst, mask) in if self.explicit { f.vals } else { f.outs } {
            let s = self.gather(mask, &loads);
            match dst {
                PC_OUT => ctrl = self.sets.union(ctrl, s),
                d if d < STORE0 => {
                    out[n] = (REG + d as u32, self.sets.union(s, base));
                    n += 1;
                }
                d => {
                    let k = (d - STORE0) as usize;
                    let (c, s) = (stores[k], self.sets.union(s, base));
                    if data_stores >> k & 1 != 0 {
                        self.site(3, s);
                        if self.explicit && c >= 0 {
                            self.per(0, c, s);
                            self.per(2, c, ctl);
                        }
                    }
                    if c >= 0 {
                        out[n] = (c as u32, s);
                        n += 1;
                    }
                }
            }
        }
        for (k, &mask) in f.load_addr.iter().enumerate() {
            if data_loads >> k & 1 != 0 {
                let s = self.gather(mask, &loads);
                self.site(0, s);
            }
        }
        for (k, &mask) in f.store_addr.iter().enumerate() {
            if data_stores >> k & 1 != 0 {
                let s = self.gather(mask, &loads);
                self.site(1, s);
                if self.explicit && stores[k] >= 0 {
                    self.per(1, stores[k], s);
                }
            }
        }
        let operands = |me: &mut Self| {
            let a = me.get_mem(pc.wrapping_add(1));
            let b = me.get_mem(pc.wrapping_add(2));
            me.sets.union(a, b)
        };
        match f.kind {
            Kind::Branch => {
                let o = self.get_mem(pc.wrapping_add(1));
                ctrl = self.sets.union(ctrl, o);
            }
            Kind::Jmp | Kind::Jsr | Kind::JmpInd => {
                let o = operands(self);
                ctrl = self.sets.union(ctrl, o);
            }
            _ => {}
        }
        let first_target = (f.kind == Kind::Rti) as usize;
        for &c in &reads[first_target..nr.min(3)] {
            let t = self.cell(c);
            ctrl = self.sets.union(ctrl, t);
        }
        let sp = self.get(REG + 3);
        let sp = self.sets.union(sp, base);
        match f.kind {
            Kind::Jsr | Kind::Rts | Kind::Rti | Kind::Brk => {
                out[n] = (REG + 3, sp);
                n += 1;
            }
            _ => {}
        }
        if f.kind == Kind::Rti {
            let p = self.cell(reads[0]);
            let p = self.sets.union(p, base);
            for r in [8, 9, 10, 11, 13, 14] {
                out[n] = (REG + r, p);
                n += 1;
            }
        }
        for (k, &c) in writes[..nw.min(3)].iter().enumerate() {
            if c >= 0 {
                let s = if f.kind == Kind::Brk && k == 2 {
                    let p = self.gather(FLAGS, &loads);
                    self.sets.union(p, base)
                } else {
                    base
                };
                out[n] = (c as u32, s);
                n += 1;
            }
        }
        if f.kind == Kind::Brk {
            out[n] = (REG + 10, base);
            n += 1;
        }
        self.site(2, ctrl);
        self.cond[pc as usize] = self.sets.union(ctrl, ctl);
        if self.is_branch[pc as usize] && self.cstamp[pc as usize] != self.call {
            self.ran.push(pc);
        }
        self.cstamp[pc as usize] = self.call;
        let (lo, hi) = (self.cd_off[pc as usize], self.cd_off[pc as usize + 1]);
        for &(loc, s) in &out[..n] {
            let l = loc as usize;
            self.touch(loc, false);
            self.wrote[l] = true;
            if self.stamp[l] != self.call {
                self.stamp[l] = self.call;
                self.dirty.push(loc);
            }
            self.taint[l] = s;
            if lo < hi
                && (!self.explicit
                    || (0..ns).any(|k| stores[k] == loc as i32 && data_stores >> k & 1 != 0))
                && self.store_sites.insert((loc as u64) << 16 | pc as u64)
            {
                for k in lo..hi {
                    let br = self.cd[k as usize];
                    if self.implicit_seen.insert((loc as u64) << 16 | br as u64) {
                        self.implicit.push((loc, br));
                    }
                }
            }
        }
    }

    #[inline(always)]
    fn per(&mut self, k: usize, c: i32, s: u32) {
        let v = &mut self.per_loc[k][c as usize];
        *v = self.sets.union(*v, s);
    }

    #[inline(always)]
    fn site(&mut self, kind: usize, s: u32) {
        if s != 0 {
            let p = self.pc as usize;
            self.sites[kind][p] = self.sets.union(self.sites[kind][p], s);
        }
    }

    /// Fold the call's end-of-call taints into the dependences and start the next call.
    fn end_call(&mut self) {
        self.finish();
        for k in 0..self.dirty.len() {
            let l = self.dirty[k] as usize;
            self.deps[l] = self.sets.union(self.deps[l], self.taint[l]);
        }
        for k in 0..self.implicit.len() {
            let (loc, br) = self.implicit[k];
            if self.cstamp[br as usize] == self.call {
                let (l, c) = (loc as usize, self.cond[br as usize]);
                let d = if self.explicit {
                    &mut self.per_loc[2][l]
                } else {
                    &mut self.deps[l]
                };
                *d = self.sets.union(*d, c);
            }
        }
        self.dirty.clear();
        if let Some(p) = self.prev.take() {
            self.exits.insert(p);
        }
        self.call += 1;
    }

    /// Run `calls` calls of `pc` with `A = acc` (each as [`run_to_rts`], `remap` applied),
    /// stopping early at one that does not return or jams; that call's stop reason, else
    /// [`Stop::Sp`].
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
        mut stopped: impl FnMut() -> bool,
    ) -> Stop {
        for _ in 0..calls {
            let stop = observed_call(cpu, bus, self, pc, acc, max_cycles, remap, &mut stopped);
            self.end_call();
            if let Some(s) = ended(cpu, stop) {
                return s;
            }
        }
        Stop::Sp
    }

    /// [`Learn::run`] over the calls `census` ran from the state this learner's first call
    /// started from, which `cpu` and `bus` held then, the next `calls` from [`Learn::calls`].
    ///
    /// A call of a class seen before (the same instructions, opcodes and accesses in order)
    /// taints as its first did, so it is not run: only the implicit flows learned since then
    /// are added for the branches it ran. A first one starts from its entry state, restored
    /// by `poke` from the census' snapshot; ends as [`Learn::run`] does.
    #[allow(clippy::too_many_arguments)]
    pub fn run_census<B: Bus>(
        &mut self,
        cpu: &mut Cpu,
        bus: &mut B,
        census: &Census,
        pc: u16,
        acc: u8,
        max_cycles: u64,
        calls: usize,
        remap: Option<&Remap>,
        mut poke: impl FnMut(&mut B, u16, u8),
        mut stopped: impl FnMut() -> bool,
    ) -> Stop {
        for _ in 0..calls {
            let k = self.calls() as usize;
            let fine = census.class(k).expect("the census ran this call");
            let class = self.alike(census, fine) as usize;
            if class < self.replay.len() {
                self.replay_class(class);
                self.call += 1;
                continue;
            }
            census.restore(fine as usize, cpu, |a, v| poke(bus, a, v));
            self.ran.clear();
            let stop = observed_call(cpu, bus, self, pc, acc, max_cycles, remap, &mut stopped);
            self.end_call();
            let mut conds: Vec<(u16, u32)> = self
                .ran
                .iter()
                .map(|&b| (b, self.cond[b as usize]))
                .collect();
            conds.sort_unstable();
            self.replay.push((self.implicit.len(), conds));
            if let Some(s) = ended(cpu, stop) {
                return s;
            }
        }
        Stop::Sp
    }

    /// The class this learner learns census class `fine` as: its events with reads of memory
    /// no call writes made alike ([`Census::canonical`]), numbered in order of first call.
    fn alike(&mut self, census: &Census, fine: u32) -> u32 {
        if let Some(&c) = self.alike.get(fine as usize) {
            return c;
        }
        let mut ev = std::mem::take(&mut self.scratch);
        let written = &self.written;
        census.canonical(fine, |a| !written.contains(a), &mut ev);
        let c = self.coarse.intern(&ev).0;
        self.scratch = ev;
        debug_assert_eq!(
            self.alike.len(),
            fine as usize,
            "census classes come in order"
        );
        self.alike.push(c);
        c
    }

    /// The implicit flows learned since class `class` last ran, for the branches it ran.
    fn replay_class(&mut self, class: usize) {
        let (from, conds) = &mut self.replay[class];
        for &(loc, br) in &self.implicit[*from..] {
            if let Ok(i) = conds.binary_search_by_key(&br, |c| c.0) {
                let d = if self.explicit {
                    &mut self.per_loc[2][loc as usize]
                } else {
                    &mut self.deps[loc as usize]
                };
                *d = self.sets.union(*d, conds[i].1);
            }
        }
        *from = self.implicit.len();
    }

    /// What the calls so far found.
    pub fn learned(&self) -> Learned {
        let persistent = (0..LOCS as u32)
            .filter(|&l| {
                self.read_first[l as usize]
                    && self.wrote[l as usize]
                    && (l < REG || STATE_REGS >> (l - REG) & 1 != 0)
            })
            .collect();
        let pairs = |v: &[u32]| -> Vec<u64> {
            let mut out = Vec::new();
            for (l, &d) in v.iter().enumerate() {
                out.extend(
                    self.sets.members[d as usize]
                        .iter()
                        .map(|&s| (l as u64) << 32 | s as u64),
                );
            }
            out
        };
        let [stored, addressed, gated] = self.per_loc.each_ref().map(|v| pairs(v));
        let deps = pairs(&self.deps);
        let mut sites = Vec::new();
        for pc in 0..0x10000usize {
            for kind in 0..4 {
                let key = ((pc as u64) << 2 | kind as u64) << SITE_BITS;
                sites.extend(
                    self.sets.members[self.sites[kind][pc] as usize]
                        .iter()
                        .map(|&l| key | l as u64),
                );
            }
        }
        Learned {
            persistent,
            deps,
            edges: self.edges.clone().sorted(),
            sites,
            stored,
            addressed,
            gated,
            exits: (0..=0xFFFFu16)
                .filter(|&a| self.exits.contains(a))
                .collect(),
            written: (0..=0xFFFFu16)
                .filter(|&a| self.wrote[a as usize])
                .collect(),
        }
    }
}

/// What an [`Observed`] bus tells of a call: each instruction's start and each access after.
pub(crate) trait Observer {
    /// Whether an instruction has started (accesses before the first are not noted).
    fn active(&self) -> bool;
    /// An access of `kind` of `addr` reaching `c`: an address, [`crate::ZERO`] or
    /// [`SUPPRESSED`].
    fn note(&mut self, kind: u8, addr: u16, c: i32);
    /// The instruction at `pc` with opcode `op` starts.
    fn start(&mut self, pc: u16, op: u8);
    /// Whether [`Observer::wrote`] is told of each write that reaches memory.
    const WRITES: bool = false;
    /// A write reaching `addr`, which held `old`, even before the first instruction.
    #[inline(always)]
    fn wrote(&mut self, _addr: u16, _old: u8) {}
}

impl Observer for Learn {
    #[inline(always)]
    fn active(&self) -> bool {
        self.active
    }

    #[inline(always)]
    fn note(&mut self, kind: u8, _addr: u16, c: i32) {
        self.rec.push((kind, c));
    }

    #[inline(always)]
    fn start(&mut self, pc: u16, op: u8) {
        self.finish();
        self.begin(pc, op);
    }
}

/// `inner` with each access of the current instruction told to `eng` as the location it
/// reaches (after `inner`'s redirection, [`Bus::target`]).
pub(crate) struct Observed<'a, 'e, B: Bus, O: Observer> {
    pub(crate) inner: &'a mut B,
    pub(crate) eng: &'e mut O,
}

impl<B: Bus, O: Observer> Observed<'_, '_, B, O> {
    #[inline(always)]
    fn note_read(&mut self, kind: u8, addr: u16) {
        if self.eng.active() {
            let c = self.inner.target(addr).unwrap_or(addr as i32);
            self.eng.note(kind, addr, c);
        }
    }

    #[inline(always)]
    fn note_write(&mut self, kind: u8, addr: u16) {
        let reaches = self.inner.target(addr).is_none();
        if O::WRITES && reaches {
            let old = self.inner.peek(addr);
            self.eng.wrote(addr, old);
        }
        if self.eng.active() {
            self.eng
                .note(kind, addr, if reaches { addr as i32 } else { SUPPRESSED });
        }
    }
}

impl<B: Bus, O: Observer> Bus for Observed<'_, '_, B, O> {
    #[inline(always)]
    fn read(&mut self, addr: u16) -> u8 {
        self.note_read(READ, addr);
        self.inner.read(addr)
    }

    #[inline(always)]
    fn write(&mut self, addr: u16, v: u8) {
        self.note_write(WRITE, addr);
        self.inner.write(addr, v)
    }

    #[inline(always)]
    fn fetch(&mut self, addr: u16) -> u8 {
        self.inner.fetch(addr)
    }

    #[inline(always)]
    fn load(&mut self, addr: u16) -> u8 {
        self.note_read(LOAD, addr);
        self.inner.load(addr)
    }

    #[inline(always)]
    fn store(&mut self, addr: u16, v: u8) {
        self.note_write(STORE, addr);
        self.inner.store(addr, v)
    }

    #[inline(always)]
    fn stack_read(&mut self, addr: u16) -> u8 {
        self.note_read(STACK_LOAD, addr);
        self.inner.stack_read(addr)
    }

    #[inline(always)]
    fn stack_write(&mut self, addr: u16, v: u8) {
        self.note_write(STACK_STORE, addr);
        self.inner.stack_write(addr, v)
    }

    #[inline(always)]
    fn sync(&mut self, cpu: &Cpu) {
        let op = self.inner.fetch(cpu.pc);
        self.eng.start(cpu.pc, op);
        self.inner.sync(cpu);
    }
}

/// One call of `pc` (as [`run_to_rts`]) on `bus`, through [`Accesses`] only under a `remap`.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn plain_call<B: Bus>(
    cpu: &mut Cpu,
    bus: &mut B,
    pc: u16,
    acc: u8,
    max_cycles: u64,
    remap: Option<&Remap>,
    stopped: &mut impl FnMut() -> bool,
) -> Stop {
    match remap {
        Some(r) => {
            let mut a = Accesses::remapped(bus, r);
            run_to_rts(cpu, &mut a, pc, acc, None, max_cycles, None, stopped)
        }
        None => run_to_rts(cpu, bus, pc, acc, None, max_cycles, None, stopped),
    }
}

/// [`plain_call`] with its instructions and accesses told to `eng` ([`Observed`]).
#[allow(clippy::too_many_arguments)]
#[inline(always)]
pub(crate) fn observed_call<B: Bus, O: Observer>(
    cpu: &mut Cpu,
    bus: &mut B,
    eng: &mut O,
    pc: u16,
    acc: u8,
    max_cycles: u64,
    remap: Option<&Remap>,
    stopped: &mut impl FnMut() -> bool,
) -> Stop {
    match remap {
        Some(r) => {
            let mut a = Accesses::remapped(bus, r);
            let mut ob = Observed { inner: &mut a, eng };
            run_to_rts(cpu, &mut ob, pc, acc, None, max_cycles, None, stopped)
        }
        None => {
            let mut ob = Observed { inner: bus, eng };
            run_to_rts(cpu, &mut ob, pc, acc, None, max_cycles, None, stopped)
        }
    }
}

/// The stop that ends a run of calls after one that stopped with `stop`: None to go on.
#[inline(always)]
pub(crate) fn ended(cpu: &Cpu, stop: Stop) -> Option<Stop> {
    if stop != Stop::Sp {
        Some(stop)
    } else if cpu.jammed {
        Some(Stop::Jam)
    } else {
        None
    }
}

/// [`Learn::run`] from a fresh [`Learn::new`]: `calls` calls' [`Learned`] and the last stop.
#[allow(clippy::too_many_arguments)]
pub fn learn<B: Bus>(
    cpu: &mut Cpu,
    bus: &mut B,
    pc: u16,
    acc: u8,
    max_cycles: u64,
    calls: usize,
    control: &[(u16, Vec<u16>)],
    written: AddrSet,
    explicit: bool,
    remap: Option<&Remap>,
) -> (Learned, Stop) {
    let mut l = Learn::new(control, written, explicit);
    let stop = l.run(cpu, bus, pc, acc, max_cycles, calls, remap, || false);
    (l.learned(), stop)
}

/// The value of location `loc` ([`REG`] for registers) in `cpu` and `peek`'s memory.
pub fn value<B>(cpu: &Cpu, bus: &mut B, peek: &mut impl FnMut(&mut B, u16) -> u8, loc: u32) -> u8 {
    if loc < REG {
        return peek(bus, loc as u16);
    }
    match loc - REG {
        0 => cpu.a,
        1 => cpu.x,
        2 => cpu.y,
        3 => cpu.sp,
        8 => cpu.c,
        9 => cpu.z,
        10 => cpu.i,
        11 => cpu.d,
        12 => cpu.b,
        13 => cpu.v,
        14 => cpu.n,
        _ => 0,
    }
}

/// `calls` calls of `pc` with `A = acc` (as [`run_to_rts`], `remap` applied), appending the
/// values of `cells` (see [`value`]) at each call's entry to `out`; stops early as
/// [`Learn::run`] does.
#[allow(clippy::too_many_arguments)]
pub fn trajectory<B: Bus>(
    cpu: &mut Cpu,
    bus: &mut B,
    pc: u16,
    acc: u8,
    max_cycles: u64,
    calls: usize,
    cells: &[u32],
    remap: Option<&Remap>,
    mut peek: impl FnMut(&mut B, u16) -> u8,
    out: &mut Vec<u8>,
    mut stopped: impl FnMut() -> bool,
) -> Stop {
    out.reserve(calls * cells.len());
    for _ in 0..calls {
        for &c in cells {
            let v = value(cpu, bus, &mut peek, c);
            out.push(v);
        }
        let stop = plain_call(cpu, bus, pc, acc, max_cycles, remap, &mut stopped);
        if let Some(s) = ended(cpu, stop) {
            return s;
        }
    }
    Stop::Sp
}

/// A 16-byte digest of the values of `cells` (see [`value`]), in order: equal values give an
/// equal digest.
pub fn cells_digest<B>(
    cpu: &Cpu,
    bus: &mut B,
    peek: &mut impl FnMut(&mut B, u16) -> u8,
    cells: &[u32],
) -> u128 {
    let (mut h0, mut h1) = (Fx(0x243F_6A88_85A3_08D3), Fx(0x1319_8A2E_0370_7344));
    for (k, &c) in cells.iter().enumerate() {
        let v = (k as u64) << 8 | value(cpu, bus, peek, c) as u64;
        h0.add(v);
        h1.add(v.rotate_left(29) ^ 0xA409_3822_299F_31D0);
    }
    (h0.finish() as u128) << 64 | h1.finish() as u128
}
