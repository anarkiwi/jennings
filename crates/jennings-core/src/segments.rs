//! Calls cut into segments at chosen PCs: what each segment reads before it writes, what it
//! stores and what it leaves there, over a run of calls.
//!
//! A segment runs from the call's entry or an arrival at a cut PC to the next arrival or the
//! call's return, so the `n`th segment of every call is one segment as long as every call has
//! the same segments. What a segment does is a function of the locations it reads before
//! writing them ([`Segment::feed`]): memory by any access, registers as the generated data
//! flow names them, and what the driver of the call reads to tell that the call has returned
//! (SP, and [`BASE`], the SP the call was entered with).
//!
//! A segment may be given free locations with the values each may hold. A run that reads one
//! before writing it is then run again from its entry state with each other value, and so on
//! for what those runs read: every run the segment can make from that state, whatever the
//! free locations hold, is made, and where it leaves a location otherwise than the call's own
//! run did is told ([`Cut::differs`]).

use crate::access::{access_key, Accesses, Remap, ZERO};
use crate::bus::Bus;
use crate::cpu::Cpu;
use crate::memo::{reg_use, regs};
use crate::orbit::Keys;
use crate::run::{AddrSet, Stop};
use crate::taint::{ended, FxMap, FxSet, REG};

/// The location of the SP a call was entered with: its first segment stores it, and the test
/// for the call's return reads it before each instruction.
pub const BASE: u32 = REG + 15;
const LOCS: usize = REG as usize + 16;
const A: u32 = REG;
const SP: u32 = REG + 3;
/// The segments a call may have.
pub const MAX_SEGMENTS: usize = 256;
/// The runs made again with other values of free locations, over all calls, at most.
pub const MAX_FORKS: u64 = 1 << 24;

const UNSEEN: u8 = 0;
const ONE: u8 = 1;
const MANY: u8 = 2;

/// A location a segment read before writing it, in some run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fed {
    pub loc: u32,
    /// The value it held at each such read, None where they were not all one value.
    pub value: Option<u8>,
}

/// A location a segment wrote, in some run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Left {
    pub loc: u32,
    /// The runs of the segment that wrote it.
    pub runs: u32,
    /// The value it held where each of those runs ended, None where they were not all one.
    pub value: Option<u8>,
}

/// One segment over the calls of a [`Cut`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Segment {
    /// The PC each of its runs started at.
    pub pc: u16,
    /// Its runs: one per call, and those made again with other values of free locations.
    pub runs: u32,
    /// The most cycles a run took.
    pub cycles: u64,
    /// Ascending by location.
    pub feed: Vec<Fed>,
    /// Ascending by location.
    pub stores: Vec<Left>,
}

/// The first run of a segment that is not as the first cut call has it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Odd {
    /// Counted from the first cut call.
    pub call: usize,
    /// The segment, counted from 0, that did not start as in the first cut call.
    pub segment: usize,
    /// The PC it started at, None where the call returned before it.
    pub pc: Option<u16>,
    /// Whether the run before it was one made again with another value of a free location.
    pub fork: bool,
}

/// A location a run made again left otherwise than the call's own run did, other than a free
/// location it was given another value of and did not write.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Differs {
    pub segment: usize,
    /// The free location last given another value for that run.
    pub free: u32,
    pub loc: u32,
}

/// The result of [`cut_calls`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cut {
    /// The cut calls run whose segments are the first's (all but one that ends the run).
    pub calls: usize,
    pub segments: Vec<Segment>,
    pub odd: Option<Odd>,
    /// The distinct [`access_key`]s of the calls' data accesses, ascending.
    pub accesses: Vec<u64>,
    /// The distinct `pc << 16 | next` of consecutive instructions within a call, ascending.
    pub edges: Vec<u64>,
    /// `(pc, n)`: the instructions every cut call arrived at `n` times, ascending.
    pub arrivals: Vec<(u16, u32)>,
    /// Every location's value at the entry of the first cut call ([`BASE`]: 0).
    pub entry: Vec<u8>,
    /// The runs made again with other values of free locations.
    pub forks: u64,
    /// As `accesses` and `edges`, of those runs.
    pub fork_accesses: Vec<u64>,
    pub fork_edges: Vec<u64>,
    /// `(location, value)` each free location held where a cut call or a run ended, ascending.
    pub values: Vec<(u32, u8)>,
    /// Ascending.
    pub differs: Vec<Differs>,
}

struct Stats {
    pc: u16,
    runs: u32,
    cycles: u64,
    fed: Box<[u8]>,
    fed_value: Box<[u8]>,
    feeds: Vec<u32>,
    left: Box<[u8]>,
    left_value: Box<[u8]>,
    left_runs: Box<[u32]>,
    stored: Vec<u32>,
    free: Vec<u32>,
}

impl Stats {
    fn new(pc: u16, free: Vec<u32>) -> Self {
        let bytes = || vec![0u8; LOCS].into_boxed_slice();
        Stats {
            pc,
            runs: 0,
            cycles: 0,
            fed: bytes(),
            fed_value: bytes(),
            feeds: Vec::new(),
            left: bytes(),
            left_value: bytes(),
            left_runs: vec![0u32; LOCS].into_boxed_slice(),
            stored: Vec::new(),
            free,
        }
    }

    fn segment(mut self) -> Segment {
        self.feeds.sort_unstable();
        self.stored.sort_unstable();
        let one = |state: u8, v: u8| (state == ONE).then_some(v);
        let feed = self.feeds.iter().map(|&l| Fed {
            loc: l,
            value: one(self.fed[l as usize], self.fed_value[l as usize]),
        });
        let stores = self.stored.iter().map(|&l| Left {
            loc: l,
            runs: self.left_runs[l as usize],
            value: one(self.left[l as usize], self.left_value[l as usize]),
        });
        Segment {
            pc: self.pc,
            runs: self.runs,
            cycles: self.cycles,
            feed: feed.collect(),
            stores: stores.collect(),
        }
    }
}

/// Note `v` among the values of `state` and `value` at `loc`; whether it is the first there.
#[inline]
fn seen(state: &mut [u8], value: &mut [u8], loc: usize, v: u8) -> bool {
    match state[loc] {
        UNSEEN => {
            state[loc] = ONE;
            value[loc] = v;
            return true;
        }
        ONE if value[loc] != v => state[loc] = MANY,
        _ => {}
    }
    false
}

/// How a segment's run ended.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Exit {
    /// At an arrival at this cut PC.
    Cut(u16),
    Return,
    Stop(Stop),
}

/// The free locations and the values each may hold, as [`cut_calls`] is given them.
#[derive(Default)]
struct Free {
    /// Per segment, ascending.
    of: Vec<Vec<u32>>,
    any: Vec<u32>,
    is: Box<[bool]>,
    known: FxMap<u32, Vec<u8>>,
}

impl Free {
    fn new(free: &[(usize, u32)], values: &[(u32, u8)]) -> Self {
        let mut f = Free {
            is: vec![false; LOCS].into_boxed_slice(),
            ..Free::default()
        };
        for &(seg, loc) in free {
            if f.of.len() <= seg {
                f.of.resize(seg + 1, Vec::new());
            }
            f.of[seg].push(loc);
            f.is[loc as usize] = true;
            f.any.push(loc);
        }
        f.of.iter_mut().for_each(|v| v.sort_unstable());
        f.any.sort_unstable();
        f.any.dedup();
        for &(loc, v) in values {
            f.known.entry(loc).or_default().push(v);
        }
        f
    }
}

struct Tracker<'c> {
    cuts: &'c AddrSet,
    uses: [(u16, u16); 256],
    free: Free,
    segs: Vec<Stats>,
    calls: usize,
    odd: Option<Odd>,
    /// Whether segments are kept: not in a leading call, and not once a run is odd.
    on: bool,
    /// Whether the run is one made again with another value of a free location.
    fork: bool,
    cur: usize,
    run: u32,
    touched_in: Box<[u32]>,
    wrote: Vec<u32>,
    /// The free locations the run read before writing, in order, with what they held.
    free_read: Vec<(u32, u8)>,
    /// `(address, byte before)` of the run's writes to memory, where its segment has free
    /// locations.
    log: Vec<(u16, u8)>,
    logs: bool,
    /// A run made again: the byte each address it wrote held before.
    before: FxMap<u16, u8>,
    entry: Cpu,
    entry_prev: Option<u16>,
    base: u8,
    at: u16,
    prev: Option<u16>,
    accesses: Keys,
    edges: Keys,
    next_of: Box<[u32]>,
    access_of: Box<[u32]>,
    fork_accesses: FxSet<u64>,
    fork_edges: FxSet<u64>,
    forks: u64,
    values: FxSet<(u32, u8)>,
    differs: FxSet<Differs>,
    first: Option<Vec<u8>>,
    count: Box<[u32]>,
    touched: Vec<u16>,
    steady: Vec<(u16, u32)>,
}

impl<'c> Tracker<'c> {
    fn new(cuts: &'c AddrSet, free: Free) -> Self {
        let mut uses = [(0, 0); 256];
        for (op, u) in uses.iter_mut().enumerate() {
            *u = reg_use(op as u8);
        }
        Tracker {
            cuts,
            uses,
            free,
            segs: Vec::new(),
            calls: 0,
            odd: None,
            on: false,
            fork: false,
            cur: 0,
            run: 0,
            touched_in: vec![0; LOCS].into_boxed_slice(),
            wrote: Vec::new(),
            free_read: Vec::new(),
            log: Vec::new(),
            logs: false,
            before: FxMap::default(),
            entry: Cpu::new(),
            entry_prev: None,
            base: 0,
            at: 0,
            prev: None,
            accesses: Keys::new(1 << 17),
            edges: Keys::new(1 << 16),
            next_of: vec![u32::MAX; 0x10000].into_boxed_slice(),
            access_of: vec![u32::MAX; 0x10000].into_boxed_slice(),
            fork_accesses: FxSet::default(),
            fork_edges: FxSet::default(),
            forks: 0,
            values: FxSet::default(),
            differs: FxSet::default(),
            first: None,
            count: vec![0; 0x10000].into_boxed_slice(),
            touched: Vec::new(),
            steady: Vec::new(),
        }
    }

    /// A read of `loc`, which held `v`: of the segment's feed unless its run wrote or read
    /// it before. `touched_in` holds `run << 1` for a location read and one more once written.
    #[inline]
    fn read(&mut self, loc: u32, v: u8) {
        let l = loc as usize;
        if !self.on || self.touched_in[l] >> 1 == self.run {
            return;
        }
        self.touched_in[l] = self.run << 1;
        let s = &mut self.segs[self.cur];
        if seen(&mut s.fed, &mut s.fed_value, l, v) {
            s.feeds.push(loc);
        }
        if self.free.is[l] && s.free.binary_search(&loc).is_ok() {
            self.free_read.push((loc, v));
        }
    }

    #[inline]
    fn write(&mut self, loc: u32) {
        let l = loc as usize;
        if self.on && self.touched_in[l] != self.run << 1 | 1 {
            self.touched_in[l] = self.run << 1 | 1;
            self.wrote.push(loc);
        }
    }

    /// An edge from the instruction at `p` to the one at `pc`.
    #[inline]
    fn edge(&mut self, p: u16, pc: u16) {
        let key = (p as u64) << 16 | pc as u64;
        if self.fork {
            self.fork_edges.insert(key);
        } else if self.next_of[p as usize] != pc as u32 {
            self.next_of[p as usize] = pc as u32;
            self.edges.insert(key);
        }
    }

    /// The instruction at `pc` starts: an edge from the one before it, and an arrival.
    #[inline]
    fn arrive(&mut self, pc: u16) {
        if let Some(p) = self.prev {
            self.edge(p, pc);
        }
        self.prev = Some(pc);
        self.at = pc;
        if self.on && !self.fork {
            if self.count[pc as usize] == 0 {
                self.touched.push(pc);
            }
            self.count[pc as usize] += 1;
        }
    }

    /// A data access of `addr` by the instruction that last arrived.
    #[inline]
    fn access(&mut self, write: bool, addr: u16) {
        let key = (write as u32) << 16 | addr as u32;
        if self.fork {
            self.fork_accesses.insert(access_key(self.at, write, addr));
        } else if self.access_of[self.at as usize] != key {
            self.access_of[self.at as usize] = key;
            self.accesses.insert(access_key(self.at, write, addr));
        }
    }

    /// A run of the current segment starts from `cpu`.
    fn start(&mut self) {
        self.run += 1;
        self.wrote.clear();
        self.free_read.clear();
        self.log.clear();
    }

    /// The segment the call goes on with: the `cur`th of its call, at `pc`, from `cpu`.
    fn open(&mut self, pc: u16, cpu: &Cpu) {
        if self.calls == 0 && self.cur == self.segs.len() && self.cur < MAX_SEGMENTS {
            let free = self.free.of.get(self.cur).cloned().unwrap_or_default();
            self.segs.push(Stats::new(pc, free));
        }
        if self.segs.get(self.cur).is_none_or(|s| s.pc != pc) {
            return self.fail(Some(pc));
        }
        self.logs = !self.segs[self.cur].free.is_empty();
        self.entry = cpu.clone();
        self.entry_prev = self.prev;
        self.start();
    }

    fn fail(&mut self, pc: Option<u16>) {
        self.on = false;
        self.odd = Some(Odd {
            call: self.calls,
            segment: self.cur,
            pc,
            fork: self.fork,
        });
    }

    /// The end of a run of the current segment at `cpu`, `value` giving what each location
    /// holds: what it left where it wrote, and what the free locations hold.
    fn close(&mut self, cpu: &Cpu, mut value: impl FnMut(u32) -> u8) {
        let s = &mut self.segs[self.cur];
        s.runs += 1;
        s.cycles = s.cycles.max(cpu.cycles - self.entry.cycles);
        for &loc in &self.wrote {
            let l = loc as usize;
            if seen(&mut s.left, &mut s.left_value, l, value(loc)) {
                s.stored.push(loc);
            }
            s.left_runs[l] += 1;
        }
        for &loc in &self.free.any {
            self.values.insert((loc, value(loc)));
        }
    }

    /// A call starts at `cpu`, its segments kept if `on`.
    fn begin<B: Bus>(&mut self, cpu: &Cpu, bus: &mut B, pc: u16, on: bool) {
        self.on = on;
        self.cur = 0;
        self.prev = None;
        self.base = cpu.sp;
        if on {
            if self.first.is_none() {
                let all = (0..LOCS as u32).map(|l| held(cpu, bus, 0, l));
                self.first = Some(all.collect());
                for &loc in &self.free.any {
                    self.values.insert((loc, held(cpu, bus, 0, loc)));
                }
            }
            self.open(pc, cpu);
        }
    }

    /// A call returned: whether its segments were kept; its arrivals are compared with the
    /// first's.
    fn end(&mut self) -> bool {
        if self.on && self.cur + 1 != self.segs.len() {
            self.cur += 1;
            self.fail(None);
        }
        if self.calls == 0 {
            let counted = self.touched.iter().map(|&p| (p, self.count[p as usize]));
            self.steady = counted.collect();
        } else {
            let count = &self.count;
            self.steady.retain(|&(p, n)| count[p as usize] == n);
        }
        for p in self.touched.drain(..) {
            self.count[p as usize] = 0;
        }
        self.calls += self.on as usize;
        self.on
    }

    fn result(mut self) -> Cut {
        self.steady.sort_unstable();
        let sorted = |set: FxSet<u64>| {
            let mut v: Vec<u64> = set.into_iter().collect();
            v.sort_unstable();
            v
        };
        let mut values: Vec<(u32, u8)> = self.values.into_iter().collect();
        values.sort_unstable();
        let mut differs: Vec<Differs> = self.differs.into_iter().collect();
        differs.sort_unstable();
        Cut {
            calls: self.calls,
            segments: self.segs.into_iter().map(Stats::segment).collect(),
            odd: self.odd,
            accesses: self.accesses.sorted(),
            edges: self.edges.sorted(),
            arrivals: self.steady,
            entry: self.first.unwrap_or_default(),
            forks: self.forks,
            fork_accesses: sorted(self.fork_accesses),
            fork_edges: sorted(self.fork_edges),
            values,
            differs,
        }
    }
}

/// The value of location `loc` in `cpu` and on `bus`, [`BASE`] being `base`.
#[inline(always)]
fn held<B: Bus>(cpu: &Cpu, bus: &mut B, base: u8, loc: u32) -> u8 {
    if loc < REG {
        bus.peek(loc as u16)
    } else if loc == BASE {
        base
    } else {
        regs(cpu)[(loc - REG) as usize]
    }
}

/// `inner` with each access told to a [`Tracker`] as the location it reaches.
struct Tracked<'a, 't, 'c, B: Bus> {
    inner: &'a mut B,
    t: &'t mut Tracker<'c>,
}

impl<B: Bus> Tracked<'_, '_, '_, B> {
    #[inline]
    fn reading(&mut self, addr: u16, read: impl FnOnce(&mut B, u16) -> u8) -> u8 {
        let to = self.inner.target(addr);
        let v = read(self.inner, addr);
        match to {
            None => self.t.read(addr as u32, v),
            Some(ZERO) => {}
            Some(s) => self.t.read(s as u32, v),
        }
        v
    }

    #[inline]
    fn writing(&mut self, addr: u16) {
        if self.inner.target(addr).is_none() {
            if self.t.logs {
                let old = self.inner.peek(addr);
                self.t.log.push((addr, old));
            }
            self.t.write(addr as u32);
        }
    }
}

impl<B: Bus> Bus for Tracked<'_, '_, '_, B> {
    #[inline(always)]
    fn target(&self, addr: u16) -> Option<i32> {
        self.inner.target(addr)
    }

    #[inline(always)]
    fn peek(&mut self, addr: u16) -> u8 {
        self.inner.peek(addr)
    }

    #[inline(always)]
    fn poke(&mut self, addr: u16, v: u8) {
        self.inner.poke(addr, v)
    }

    #[inline(always)]
    fn read(&mut self, addr: u16) -> u8 {
        self.reading(addr, |b, a| b.read(a))
    }

    #[inline(always)]
    fn write(&mut self, addr: u16, v: u8) {
        self.writing(addr);
        self.inner.write(addr, v)
    }

    #[inline(always)]
    fn fetch(&mut self, addr: u16) -> u8 {
        let v = self.inner.fetch(addr);
        self.t.read(addr as u32, v);
        v
    }

    #[inline(always)]
    fn load(&mut self, addr: u16) -> u8 {
        self.t.access(false, addr);
        self.reading(addr, |b, a| b.load(a))
    }

    #[inline(always)]
    fn store(&mut self, addr: u16, v: u8) {
        self.t.access(true, addr);
        self.writing(addr);
        self.inner.store(addr, v)
    }

    #[inline(always)]
    fn stack_read(&mut self, addr: u16) -> u8 {
        self.reading(addr, |b, a| b.stack_read(a))
    }

    #[inline(always)]
    fn stack_write(&mut self, addr: u16, v: u8) {
        self.writing(addr);
        self.inner.stack_write(addr, v)
    }

    #[inline]
    fn sync(&mut self, cpu: &Cpu) {
        let (pc, t) = (cpu.pc, &mut *self.t);
        t.arrive(pc);
        t.read(BASE, t.base);
        t.read(SP, cpu.sp);
        let op = self.inner.fetch(pc);
        t.read(pc as u32, op);
        let (reads, writes) = t.uses[op as usize];
        let now = regs(cpu);
        let mut left = reads;
        while left != 0 {
            let r = left.trailing_zeros();
            t.read(REG + r, now[r as usize]);
            left &= left - 1;
        }
        let mut left = writes;
        while left != 0 {
            t.write(REG + left.trailing_zeros());
            left &= left - 1;
        }
        self.inner.sync(cpu);
    }
}

/// What every call of a run is called with: the PC, A, and the cycles it may take.
#[derive(Clone, Copy)]
struct Entry {
    pc: u16,
    acc: u8,
    max_cycles: u64,
}

/// The driver's start of a call, as [`crate::run_to_rts`] makes it: SP is read and kept, the
/// return pushed, SP and A set.
fn enter<B: Bus>(cpu: &mut Cpu, bus: &mut B, t: &mut Tracker, entry: Entry) {
    t.read(SP, cpu.sp);
    t.write(BASE);
    t.write(SP);
    t.write(A);
    let mut tracked = Tracked { inner: bus, t };
    for byte in [0x00, 0x01] {
        tracked.write(0x100 | cpu.sp as u16, byte);
        cpu.sp = cpu.sp.wrapping_sub(1);
    }
    cpu.a = entry.acc;
    cpu.pc = entry.pc;
}

/// Run `cpu` to the end of the segment it is in: the call's return, its cycle limit (`end`),
/// or with `t.on` its next arrival at a cut PC, as [`crate::run_to_rts`] checks them.
fn run<B: Bus>(
    cpu: &mut Cpu,
    bus: &mut B,
    t: &mut Tracker,
    end: u64,
    stopped: &mut impl FnMut() -> bool,
) -> Exit {
    let above = t.base as i32 - 1;
    let mut tracked = Tracked { inner: bus, t };
    let mut first = true;
    loop {
        if cpu.sp as i32 > above {
            return Exit::Return;
        }
        if cpu.cycles >= end {
            return Exit::Stop(Stop::Cycles);
        }
        if tracked.t.on && !first && tracked.t.cuts.contains(cpu.pc) {
            return Exit::Cut(cpu.pc);
        }
        first = false;
        cpu.step(&mut tracked);
        if stopped() {
            return Exit::Stop(Stop::Stopped);
        }
    }
}

/// The state a segment's own run left, kept while it is run again.
struct Own {
    /// `(address, byte)` of the memory it wrote, as it left it.
    memory: Vec<(u16, u8)>,
    left: FxMap<u16, u8>,
    exit: Exit,
    regs: [u8; 16],
}

/// Put memory back as the run that wrote `t.log` found it.
fn undo<B: Bus>(bus: &mut B, t: &mut Tracker) {
    for &(addr, old) in t.log.iter().rev() {
        bus.poke(addr, old);
    }
    t.log.clear();
}

/// Run the current segment again from its entry state with the free locations `given`
/// holding other values: the free locations it read, and whether it ended as `own` did.
#[allow(clippy::too_many_arguments)]
fn fork<B: Bus>(
    bus: &mut B,
    t: &mut Tracker,
    own: &Own,
    given: &[(u32, u8)],
    entry: Entry,
    end: u64,
    stopped: &mut impl FnMut() -> bool,
) -> (Vec<(u32, u8)>, bool) {
    undo(bus, t);
    let mut cpu = t.entry.clone();
    t.start();
    t.prev = t.entry_prev;
    t.forks += 1;
    for &(loc, v) in given {
        if loc < REG {
            t.log.push((loc as u16, bus.peek(loc as u16)));
            bus.poke(loc as u16, v);
        } else {
            let mut to = regs(&cpu);
            to[(loc - REG) as usize] = v;
            crate::memo::set_regs(&mut cpu, 1 << (loc - REG), to);
        }
    }
    if t.cur == 0 {
        enter(&mut cpu, bus, t, entry);
    }
    let exit = run(&mut cpu, bus, t, end, stopped);
    if let (Exit::Cut(pc), Some(p)) = (exit, t.prev) {
        t.edge(p, pc);
    }
    let base = t.base;
    t.close(&cpu, |loc| held(&cpu, bus, base, loc));
    let free = given.last().map_or(0, |g| g.0);
    let differ = |t: &mut Tracker, loc: u32| {
        let wrote = t.touched_in[loc as usize] == t.run << 1 | 1;
        if wrote || given.iter().all(|g| g.0 != loc) {
            let segment = t.cur;
            t.differs.insert(Differs { segment, free, loc });
        }
    };
    let mut before = std::mem::take(&mut t.before);
    before.clear();
    for &(addr, old) in t.log.iter().rev() {
        before.insert(addr, old);
    }
    for (&addr, &was) in &before {
        if bus.peek(addr) != own.left.get(&addr).copied().unwrap_or(was) {
            differ(t, addr as u32);
        }
    }
    for &(addr, left) in &own.memory {
        if !before.contains_key(&addr) && bus.peek(addr) != left {
            differ(t, addr as u32);
        }
    }
    t.before = before;
    for (r, (now, theirs)) in regs(&cpu).iter().zip(&own.regs).enumerate() {
        if now != theirs {
            differ(t, REG + r as u32);
        }
    }
    (t.free_read.clone(), exit == own.exit && !cpu.jammed)
}

/// Make every run the current segment can make from its entry state, whatever its free
/// locations hold of the values known for them, after its own run ended at `cpu` by `exit`;
/// memory is left as that run left it. False where one ends otherwise than it did.
fn explore<B: Bus>(
    cpu: &Cpu,
    bus: &mut B,
    t: &mut Tracker,
    exit: Exit,
    entry: Entry,
    end: u64,
    stopped: &mut impl FnMut() -> bool,
) -> bool {
    if t.free_read.is_empty() {
        return true;
    }
    let mut memory: Vec<(u16, u8)> = t.log.iter().map(|&(a, _)| (a, bus.peek(a))).collect();
    memory.sort_unstable();
    memory.dedup();
    let own = Own {
        left: memory.iter().copied().collect(),
        memory,
        exit,
        regs: regs(cpu),
    };
    let (prev, mut alike) = (t.prev, true);
    t.fork = true;
    let mut todo = vec![(0, t.free_read.clone())];
    while let Some((fixed, read)) = todo.pop() {
        for at in fixed..read.len() {
            let (loc, held) = read[at];
            let known = t.free.known.get(&loc).cloned().unwrap_or_default();
            for v in known.into_iter().filter(|&v| v != held) {
                let mut given = read[..at].to_vec();
                given.push((loc, v));
                let (theirs, same) = fork(bus, t, &own, &given, entry, end, stopped);
                alike &= same && t.forks < MAX_FORKS;
                if !alike {
                    break;
                }
                todo.push((at + 1, theirs));
            }
        }
        if !alike {
            break;
        }
    }
    undo(bus, t);
    for &(addr, v) in &own.memory {
        bus.poke(addr, v);
    }
    t.fork = false;
    t.prev = prev;
    alike
}

/// One call of `entry` on `bus`, told to `t`, its segments kept if `on`: how it stopped, and
/// whether its segments were kept.
fn call<B: Bus>(
    cpu: &mut Cpu,
    bus: &mut B,
    t: &mut Tracker,
    entry: Entry,
    on: bool,
    stopped: &mut impl FnMut() -> bool,
) -> (Stop, bool) {
    let end = cpu
        .cycles
        .saturating_add(entry.max_cycles)
        .saturating_add(1);
    t.begin(cpu, bus, entry.pc, on);
    enter(cpu, bus, t, entry);
    loop {
        let exit = run(cpu, bus, t, end, stopped);
        if let Exit::Stop(stop) = exit {
            return (stop, t.end());
        }
        if t.on {
            let base = t.base;
            t.close(cpu, |loc| held(cpu, bus, base, loc));
            if !explore(cpu, bus, t, exit, entry, end, stopped) {
                t.fork = true;
                t.cur += 1;
                t.fail(None);
                t.fork = false;
            }
        }
        match exit {
            Exit::Cut(pc) if t.on => {
                t.cur += 1;
                t.open(pc, cpu);
            }
            Exit::Cut(_) => {}
            _ => return (Stop::Sp, t.end()),
        }
    }
}

/// Call `pc` with `A = acc` (each call as [`crate::run_to_rts`], `remap` applied) from the
/// state `cpu` and `bus` hold: `lead` calls whose accesses and edges alone are kept, then
/// `calls` calls each cut where it arrives at a PC of `cuts` after its first instruction.
///
/// `free` gives `(segment, location)` pairs and `values` the `(location, value)` pairs each
/// may hold: a run reading a free location of its segment before writing it is made again
/// with each other value (see the module). The [`Cut`], and [`Stop::Sp`] unless the last call
/// run ended otherwise. The run ends early at a call that does not return, jams, or has a
/// run not as the first cut call has it ([`Cut::odd`]).
#[allow(clippy::too_many_arguments)]
pub fn cut_calls<B: Bus>(
    cpu: &mut Cpu,
    bus: &mut B,
    pc: u16,
    acc: u8,
    max_cycles: u64,
    lead: usize,
    calls: usize,
    cuts: &AddrSet,
    free: &[(usize, u32)],
    values: &[(u32, u8)],
    remap: Option<&Remap>,
    mut stopped: impl FnMut() -> bool,
) -> (Cut, Stop) {
    let mut t = Tracker::new(cuts, Free::new(free, values));
    let mut last = Stop::Sp;
    let entry = Entry {
        pc,
        acc,
        max_cycles,
    };
    for k in 0..lead + calls {
        let on = k >= lead;
        let (stop, kept) = match remap {
            Some(r) => {
                let mut redirected = Accesses::remapped(bus, r);
                call(cpu, &mut redirected, &mut t, entry, on, &mut stopped)
            }
            None => call(cpu, bus, &mut t, entry, on, &mut stopped),
        };
        if let Some(s) = ended(cpu, stop) {
            t.calls -= kept as usize;
            last = s;
            break;
        }
        if on && !kept {
            break;
        }
    }
    (t.result(), last)
}

#[cfg(test)]
mod tests {
    use super::*;

    const PLAY: u16 = 0x1000;
    const LOOP: u16 = 0x1002;
    const POS: u16 = 0x2000;
    const TMP: u16 = 0x00FB;

    /// `ldx #2 / loop: inc POS,x / lda POS,x / sta $D400,x / stx TMP / dex / bpl loop / rts`.
    fn voices() -> Box<[u8; 0x10000]> {
        let mut ram = Box::new([0u8; 0x10000]);
        let code = [
            0xA2, 0x02, 0xFE, 0x00, 0x20, 0xBD, 0x00, 0x20, 0x9D, 0x00, 0xD4, 0x86, 0xFB, 0xCA,
            0x10, 0xF2, 0x60,
        ];
        ram[PLAY as usize..][..code.len()].copy_from_slice(&code);
        ram
    }

    fn run(ram: &mut [u8; 0x10000], cuts: &[u16], calls: usize, max_cycles: u64) -> (Cut, Stop) {
        with(ram, cuts, (0, calls, max_cycles), (&[], &[]), None)
    }

    /// Free `(segment, location)` pairs and the `(location, value)` pairs they may hold.
    type Given<'a> = (&'a [(usize, u32)], &'a [(u32, u8)]);

    /// [`cut_calls`] of [`PLAY`] on `ram`: `calls` is `(lead, calls, max_cycles)`.
    fn with(
        ram: &mut [u8; 0x10000],
        cuts: &[u16],
        calls: (usize, usize, u64),
        free: Given,
        remap: Option<&Remap>,
    ) -> (Cut, Stop) {
        let cuts: AddrSet = cuts.iter().copied().collect();
        let (mut cpu, (lead, n, max_cycles)) = (Cpu::new(), calls);
        cut_calls(
            &mut cpu,
            ram,
            PLAY,
            0,
            max_cycles,
            lead,
            n,
            &cuts,
            free.0,
            free.1,
            remap,
            || false,
        )
    }

    fn fed(s: &Segment, loc: u32) -> Option<Fed> {
        s.feed.iter().copied().find(|f| f.loc == loc)
    }

    fn left(s: &Segment, loc: u32) -> Option<Left> {
        s.stores.iter().copied().find(|f| f.loc == loc)
    }

    #[test]
    fn a_call_is_cut_at_each_arrival_and_every_call_alike() {
        let (cut, stop) = run(&mut voices(), &[LOOP], 5, 1000);
        assert_eq!((cut.calls, stop, cut.odd), (5, Stop::Sp, None));
        let starts: Vec<u16> = cut.segments.iter().map(|s| s.pc).collect();
        assert_eq!(starts, [PLAY, LOOP, LOOP, LOOP]);
        assert!(cut.segments.iter().all(|s| s.runs == 5));
        assert_eq!(cut.segments[0].cycles, 2);
        assert_eq!(cut.segments[3].cycles, 7 + 4 + 5 + 3 + 2 + 2 + 6);
        assert!(cut.arrivals.contains(&(LOOP, 3)) && cut.arrivals.contains(&(PLAY, 1)));
        assert_eq!(cut.arrivals.len(), 8);
    }

    #[test]
    fn a_segment_feeds_on_what_it_reads_before_writing() {
        let (cut, _) = run(&mut voices(), &[LOOP], 5, 1000);
        let x = REG + 1;
        let first = &cut.segments[0];
        assert_eq!(
            fed(first, SP),
            Some(Fed {
                loc: SP,
                value: Some(0xFF)
            })
        );
        assert_eq!(fed(first, PLAY as u32).unwrap().value, Some(0xA2));
        assert_eq!(
            (fed(first, A), fed(first, x), fed(first, BASE)),
            (None, None, None)
        );
        for (n, s) in cut.segments[1..].iter().enumerate() {
            let pos = POS as u32 + 2 - n as u32;
            assert_eq!(
                fed(s, x),
                Some(Fed {
                    loc: x,
                    value: Some(2 - n as u8)
                })
            );
            assert_eq!(
                fed(s, pos),
                Some(Fed {
                    loc: pos,
                    value: None
                })
            );
            assert_eq!(
                fed(s, BASE),
                Some(Fed {
                    loc: BASE,
                    value: Some(0xFF)
                })
            );
            assert_eq!(
                fed(s, SP),
                Some(Fed {
                    loc: SP,
                    value: Some(0xFD)
                })
            );
            assert_eq!(fed(s, TMP as u32), None, "written, never read");
            assert_eq!(fed(s, 0xD402 - n as u32), None);
            assert_eq!(fed(s, LOOP as u32).unwrap().value, Some(0xFE));
        }
        assert_eq!(fed(&cut.segments[3], 0x01FE).unwrap().value, Some(0x01));
        assert_eq!(fed(&cut.segments[3], 0x01FF).unwrap().value, Some(0x00));
        assert_eq!(fed(&cut.segments[2], 0x01FE), None);
    }

    #[test]
    fn a_segment_leaves_what_its_runs_store() {
        let (cut, _) = run(&mut voices(), &[LOOP], 5, 1000);
        let x = REG + 1;
        let stored = |s: &Segment, loc: u32, value: Option<u8>| {
            assert_eq!(
                left(s, loc),
                Some(Left {
                    loc,
                    runs: 5,
                    value
                })
            );
        };
        let first = &cut.segments[0];
        stored(first, x, Some(2));
        stored(first, A, Some(0));
        stored(first, BASE, Some(0xFF));
        stored(first, SP, Some(0xFD));
        stored(first, 0x01FF, Some(0x00));
        stored(first, 0x01FE, Some(0x01));
        assert_eq!(left(first, POS as u32), None);
        for (n, s) in cut.segments[1..].iter().enumerate() {
            stored(s, POS as u32 + 2 - n as u32, None);
            stored(s, 0xD402 - n as u32, None);
            stored(s, TMP as u32, Some(2 - n as u8));
            stored(s, x, Some((1 - n as i8) as u8));
            assert_eq!(left(s, BASE), None);
        }
        stored(&cut.segments[3], SP, Some(0xFF));
        assert_eq!(left(&cut.segments[2], SP), None);
    }

    #[test]
    fn accesses_and_edges_are_those_of_the_orbit() {
        let (mut a, mut b) = (voices(), voices());
        let (cut, _) = run(&mut a, &[LOOP], 300, 1000);
        let (mut cpu, none, mut n) = (Cpu::new(), AddrSet::default(), 0u128);
        let count = |_: &Cpu, _: &mut [u8; 0x10000]| {
            n += 1;
            n
        };
        let o = crate::orbit(
            &mut cpu,
            &mut *b,
            PLAY,
            0,
            1000,
            300,
            None,
            &none,
            count,
            || false,
        );
        assert_eq!(o.states, 301);
        assert_eq!((cut.accesses, cut.edges), (o.accesses, o.edges));
        assert_eq!(a[..], b[..]);
    }

    #[test]
    fn a_call_with_other_segments_than_the_first_ends_the_run() {
        let mut ram = voices();
        ram[0x1001] = 0x00;
        ram[0x100E] = 0xD0;
        let (cut, stop) = run(&mut ram, &[LOOP], 300, 100_000);
        assert_eq!((cut.calls, stop), (0, Stop::Sp));
        assert_eq!(cut.segments.len(), MAX_SEGMENTS);
        let odd = cut.odd.unwrap();
        assert_eq!(
            (odd.call, odd.segment, odd.pc),
            (0, MAX_SEGMENTS, Some(LOOP))
        );
        let (cut, _) = run(&mut voices(), &[0x100D], 4, 1000);
        assert_eq!((cut.calls, cut.odd, cut.segments.len()), (4, None, 4));
        let mut ram = voices();
        ram[0x1008..0x100B].copy_from_slice(&[0x8D, 0x01, 0x10]);
        let (cut, stop) = run(&mut ram, &[LOOP], 9, 1000);
        assert_eq!((cut.calls, stop), (1, Stop::Sp));
        let odd = cut.odd.unwrap();
        assert_eq!(
            (odd.call, odd.segment, odd.pc),
            (1, 3, None),
            "the count of voices is stored over"
        );
        let mut ram = voices();
        ram[0x1008..0x100B].copy_from_slice(&[0x8D, 0x01, 0x10]);
        let (cut, _) = run(&mut ram, &[LOOP, 0x1010], 9, 1000);
        let odd = cut.odd.unwrap();
        assert_eq!(
            (cut.calls, odd.call, odd.segment, odd.pc),
            (1, 1, 3, Some(0x1010))
        );
    }

    #[test]
    fn a_location_written_before_it_is_read_is_not_fed_and_arrivals_that_vary_are_not_steady() {
        let mut ram = Box::new([0u8; 0x10000]);
        ram[0x1000..0x1007].copy_from_slice(&[0xA9, 0x07, 0x85, 0xFB, 0xA5, 0xFB, 0x60]);
        let (cut, _) = run(&mut ram, &[], 3, 1000);
        assert_eq!(fed(&cut.segments[0], 0xFB), None);
        assert_eq!(left(&cut.segments[0], 0xFB).unwrap().value, Some(7));
        let mut ram = voices();
        ram[0x1008..0x100B].copy_from_slice(&[0x8D, 0x01, 0x10]);
        let (cut, _) = run(&mut ram, &[], 4, 1000);
        assert!(cut.arrivals.contains(&(PLAY, 1)) && cut.arrivals.iter().all(|a| a.0 != LOOP));
    }

    #[test]
    fn leading_calls_give_accesses_and_edges_alone() {
        let mut ram = voices();
        ram[0x1008..0x100B].copy_from_slice(&[0x8E, 0x01, 0x10]);
        let (cut, stop) = with(&mut ram, &[LOOP], (1, 8, 1000), (&[], &[]), None);
        assert_eq!((cut.calls, stop, cut.odd), (8, Stop::Sp, None));
        let starts: Vec<u16> = cut.segments.iter().map(|s| s.pc).collect();
        assert_eq!(
            starts,
            [PLAY, LOOP],
            "one voice after the first call's three"
        );
        assert!(cut.segments.iter().all(|s| s.runs == 8));
        assert!(cut.accesses.contains(&access_key(0x1002, true, POS + 2)));
        assert!(cut.arrivals.contains(&(LOOP, 1)));
        assert_eq!(fed(&cut.segments[1], POS as u32 + 2), None);
        assert_eq!((cut.entry[0x1001], cut.entry[POS as usize + 2]), (0, 1));
        assert_eq!(cut.entry[SP as usize], 0xFF);
    }

    #[test]
    fn a_call_that_does_not_return_is_not_counted() {
        let (cut, stop) = run(&mut voices(), &[LOOP], 5, 40);
        assert_eq!((cut.calls, stop), (0, Stop::Cycles));
        let mut ram = voices();
        ram[0x100B..0x100D].copy_from_slice(&[0x02, 0xEA]);
        let (cut, stop) = run(&mut ram, &[LOOP], 5, 1000);
        assert_eq!((cut.calls, stop), (0, Stop::Jam));
    }

    #[test]
    fn a_redirected_read_feeds_on_its_source_and_a_suppressed_write_stores_nothing() {
        let mut ram = voices();
        ram[0x3000] = 7;
        let mut table = vec![crate::AS_IS; 0x10000];
        table[POS as usize + 2] = 0x3000;
        let mut remap = Remap::default();
        remap.insert(0x1005, &table);
        let mut dry = vec![crate::AS_IS; 0x10000];
        dry[0xD401] = ZERO;
        remap.insert(0x1008, &dry);
        let (cut, _) = with(&mut ram, &[LOOP], (0, 3, 1000), (&[], &[]), Some(&remap));
        assert_eq!(fed(&cut.segments[1], 0x3000).unwrap().value, Some(7));
        assert_eq!(left(&cut.segments[1], 0xD402).unwrap().value, Some(7));
        assert_eq!(left(&cut.segments[2], 0xD401), None);
        assert!(cut.accesses.contains(&access_key(0x1008, true, 0xD401)));
        assert!(cut.accesses.contains(&access_key(0x1005, false, POS + 2)));
    }

    const MASK: u32 = 0x2010;
    const ALL: [(u32, u8); 4] = [(MASK, 0), (MASK, 1), (MASK, 2), (MASK, 3)];
    const BOTH: [(usize, u32); 2] = [(1, MASK), (2, MASK)];

    /// `ldx #1 / loop: inc POS,x / lda MASK / eor BIT,x / sta MASK / dex / bpl loop / rts`,
    /// the three bytes at `$1008` and at `$100B` replaced by `patch`.
    fn shared(patch: [u8; 6]) -> Box<[u8; 0x10000]> {
        let mut ram = Box::new([0u8; 0x10000]);
        let code = [
            0xA2, 0x01, 0xFE, 0x00, 0x20, 0xAD, 0x10, 0x20, 0x5D, 0x20, 0x20, 0x8D, 0x10, 0x20,
            0xCA, 0x10, 0xF1, 0x60,
        ];
        ram[PLAY as usize..][..code.len()].copy_from_slice(&code);
        ram[0x1008..0x100E].copy_from_slice(&patch);
        ram[0x2020..0x2022].copy_from_slice(&[1, 2]);
        ram
    }

    const TOGGLE: [u8; 6] = [0x5D, 0x20, 0x20, 0x8D, 0x10, 0x20];

    #[test]
    fn a_run_reading_a_free_location_is_made_again_with_each_other_value() {
        let (mut a, mut b) = (shared(TOGGLE), shared(TOGGLE));
        let (plain, _) = run(&mut a, &[LOOP], 6, 1000);
        let (cut, stop) = with(&mut b, &[LOOP], (0, 6, 1000), (&BOTH, &ALL), None);
        assert_eq!(
            (cut.calls, stop, cut.odd, cut.forks),
            (6, Stop::Sp, None, 6 * 2 * 3)
        );
        assert_eq!(
            a[..],
            b[..],
            "memory is left as the calls' own runs leave it"
        );
        assert_eq!((&cut.accesses, &cut.edges), (&plain.accesses, &plain.edges));
        assert_eq!(cut.arrivals, plain.arrivals);
        assert_eq!(cut.values, ALL);
        let runs: Vec<u32> = cut.segments.iter().map(|s| s.runs).collect();
        assert_eq!(runs, [6, 24, 24]);
        assert_eq!(fed(&cut.segments[1], MASK).unwrap().value, None);
        assert_eq!(left(&cut.segments[1], MASK).unwrap().runs, 24);
        assert_eq!(left(&cut.segments[1], POS as u32 + 1).unwrap().runs, 24);
        let differ: Vec<(usize, u32)> = cut.differs.iter().map(|d| (d.segment, d.loc)).collect();
        assert!(cut.differs.iter().all(|d| d.free == MASK));
        assert!(
            differ.contains(&(1, MASK)) && differ.contains(&(2, MASK)),
            "both write it"
        );
        assert!(differ.contains(&(1, A)) && !differ.contains(&(1, POS as u32 + 1)));
        assert!(cut.fork_edges.iter().all(|e| cut.edges.contains(e)));
        assert!(cut.fork_accesses.iter().all(|e| cut.accesses.contains(e)));
        assert!(cut.fork_edges.contains(&(0x100F << 16 | 0x1002)));
        assert!(cut
            .fork_accesses
            .contains(&access_key(0x100B, true, MASK as u16)));
    }

    #[test]
    fn values_a_run_made_again_leaves_are_told_and_only_given_values_are_tried() {
        let (cut, _) = with(
            &mut shared(TOGGLE),
            &[LOOP],
            (0, 6, 1000),
            (&BOTH, &[]),
            None,
        );
        assert_eq!((cut.forks, cut.differs.len()), (0, 0));
        assert_eq!(cut.values, ALL, "the calls' own runs leave all four");
        let seven = [(MASK, 7)];
        let (cut, _) = with(
            &mut shared(TOGGLE),
            &[LOOP],
            (0, 6, 1000),
            (&BOTH, &seven),
            None,
        );
        assert_eq!(cut.forks, 6 * 2);
        let more: Vec<u8> = cut.values.iter().map(|v| v.1).collect();
        assert_eq!(
            more,
            [0, 1, 2, 3, 5, 6],
            "what 7 leaves with one bit or the other toggled"
        );
        let one = [(2, MASK)];
        let (cut, _) = with(
            &mut shared(TOGGLE),
            &[LOOP],
            (0, 6, 1000),
            (&one, &ALL),
            None,
        );
        assert_eq!(
            cut.forks,
            6 * 3,
            "the other segment's runs are not made again"
        );
    }

    #[test]
    fn a_run_made_again_that_leaves_state_otherwise_is_told() {
        let copy = [0x9D, 0x30, 0x20, 0xEE, 0x10, 0x20];
        let (mut a, mut b) = (shared(copy), shared(copy));
        run(&mut a, &[LOOP], 4, 1000);
        let known = [(MASK, 0), (MASK, 9)];
        let (cut, _) = with(&mut b, &[LOOP], (0, 4, 1000), (&BOTH, &known), None);
        assert_eq!(a[..], b[..]);
        let at = |segment, loc| Differs {
            segment,
            free: MASK,
            loc,
        };
        assert!(cut.differs.contains(&at(1, 0x2031)) && cut.differs.contains(&at(2, 0x2030)));
        assert!(!cut.differs.contains(&at(1, 0x2030)) && !cut.differs.contains(&at(1, 0x2001)));
        assert_eq!(left(&cut.segments[1], 0x2031).unwrap().value, None);
    }

    #[test]
    fn a_run_made_again_that_ends_elsewhere_is_odd() {
        let leave = [0xF0, 0x07, 0xEA, 0xEE, 0x10, 0x20];
        let mut ram = shared(leave);
        ram[MASK as usize] = 1;
        let (cut, stop) = run(&mut ram, &[LOOP], 4, 1000);
        assert_eq!((cut.calls, stop, cut.odd), (4, Stop::Sp, None));
        let mut ram = shared(leave);
        ram[MASK as usize] = 1;
        let zero = [(MASK, 0)];
        let (cut, stop) = with(&mut ram, &[LOOP], (0, 4, 1000), (&BOTH, &zero), None);
        let odd = cut.odd.unwrap();
        assert_eq!((cut.calls, stop), (0, Stop::Sp));
        assert_eq!(
            (odd.call, odd.segment, odd.pc, odd.fork),
            (0, 2, None, true)
        );
        assert_eq!(ram[MASK as usize], 3, "the call went on from its own run");
    }

    #[test]
    fn a_free_register_and_the_first_segment_are_run_again_too() {
        let mut ram = shared(TOGGLE);
        ram[0x1000..0x1002].copy_from_slice(&[0xA6, 0xFB]);
        ram[0xFB] = 1;
        let known = [(0xFB, 0), (0xFB, 1), (REG + 8, 0), (REG + 8, 1)];
        let (cut, _) = with(
            &mut ram,
            &[LOOP],
            (0, 3, 1000),
            (&[(0, 0xFB)], &known),
            None,
        );
        assert_eq!((cut.calls, cut.forks, cut.odd), (3, 3, None));
        let x = Differs {
            segment: 0,
            free: 0xFB,
            loc: REG + 1,
        };
        assert!(
            cut.differs.contains(&x),
            "the first segment leaves X as the cell has it"
        );
        assert!(cut.differs.iter().all(|d| d.loc != 0xFB), "as it was given");
        let carry = [(1, REG + 8)];
        let (cut, _) = with(
            &mut shared(TOGGLE),
            &[LOOP],
            (0, 3, 1000),
            (&carry, &known),
            None,
        );
        assert_eq!(cut.forks, 0, "no run reads the carry before writing it");
        let adds = [0x7D, 0x20, 0x20, 0x8D, 0x10, 0x20];
        let (cut, _) = with(
            &mut shared(adds),
            &[LOOP],
            (0, 3, 1000),
            (&carry, &known),
            None,
        );
        assert_eq!(cut.forks, 3);
        assert!(cut
            .differs
            .iter()
            .any(|d| (d.segment, d.free, d.loc) == (1, REG + 8, MASK)));
    }

    #[test]
    fn every_free_location_a_run_reads_is_given_every_value() {
        let free = [(1, MASK), (1, 0x2021)];
        let known = [(MASK, 0), (MASK, 3), (0x2021, 2), (0x2021, 4)];
        let (cut, _) = with(
            &mut shared(TOGGLE),
            &[LOOP],
            (0, 4, 1000),
            (&free, &known),
            None,
        );
        assert_eq!(cut.forks, 4 * 3, "two values of each: three more runs");
        assert!(cut.values.contains(&(MASK, 4)) && cut.values.contains(&(MASK, 7)));
    }
}
