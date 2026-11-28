//! [`crate::orbit`] without emulating what is already known: the calls are cut into units at
//! chosen PCs ([`Grain`]) and each unit is memoised on exactly what it read.
//!
//! A unit's instructions, accesses, writes and exit registers are a function of the values of
//! the locations it reads before writing them: registers, and memory some call writes
//! (instruction bytes included; memory no call writes is constant). A unit whose reads all
//! match a stored one's therefore repeats it, so its writes are replayed and it adds no access
//! or edge. Stored units are found through a radix trie per start PC over their reads in
//! order; whatever the trie's shape, a unit is replayed only when every read it stored matches.
//!
//! The state a call starts from is hashed incrementally to find candidate repeats; a repeat is
//! confirmed by running the calls again from the first state and comparing the states.
//!
//! With some tracked cells set apart as a fibre, calls are run by a tape of one period of
//! the rest of the written memory ([`fibre`]).

mod fibre;

use crate::access::{Accesses, Redirect, Remap};
use crate::bus::Bus;
use crate::cpu::Cpu;
use crate::flow_gen::FLOW;
use crate::orbit::{End, Keys, Orbit};
use crate::run::{AddrSet, Stop};
use crate::taint::{Fx, FxMap, FxSet, Kind, REG};
use fibre::{Fibre, Notes};

const NONE: u32 = u32::MAX;
const SP: u16 = 1 << 3;
const FLAGS: u16 = 0x6F00;
const SEED: u64 = 0x9E37_79B9_7F4A_7C15;
const PROFILE: usize = 16;
const RECENT: u32 = 12;

/// When the units of one start PC are cut and when they are given up on. Every value gives
/// the same [`Orbit`] and leaves the same state; each trades only work and memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Tuning {
    /// The units of one start PC at which cutting them is first considered, doubling for that
    /// start PC each time no cut fits. Lower walks the trie for a cut sooner and more often,
    /// on less evidence of where the reads part; higher stores more units whole before a cut
    /// lets them share.
    pub(crate) crowd: u32,
    /// The units of one start PC beyond which, no cut having fitted, it is emulated rather
    /// than memoised. Lower stops paying for stores and lookups that rarely hit sooner, and
    /// emulates what could have been replayed; higher holds more units and longer lookups.
    pub(crate) raw: u32,
    /// A cut must leave the runs of reads on each side saturated: the units sharing them at
    /// least this many times their distinct values. Higher cuts only where units would be
    /// shared often, so more units are stored whole; lower cuts more readily, each cut
    /// emptying the memo and leaving shorter units that cost more lookups per call.
    pub(crate) saturated: u32,
    /// The bytes of base cells a tape may hold as frames, which sets how many calls apart
    /// they are. Lower holds less and runs more calls again to reach the state a call
    /// starts from; higher holds a frame for more of the tape's calls.
    pub(crate) frames: usize,
    /// The calls a tape may run to learn segments, beyond twice its period, before it is
    /// given up where they are more than half of its calls. Lower gives up sooner on a
    /// fibre whose segments rarely recur; higher learns for longer first.
    pub(crate) grace: u64,
    /// The calls run before the base is next looked at after a tape is given up, at
    /// least; doubled for each tape given up. Lower tries again sooner.
    pub(crate) rest: usize,
    /// The longest period of the base a tape is recorded for, and the calls its entry
    /// states are remembered over while it is looked for. Lower holds less and misses
    /// longer periods.
    pub(crate) span: usize,
    /// The segment results a tape may hold; past that none is added and segments not
    /// held are run each time.
    pub(crate) known: usize,
}

impl Default for Tuning {
    fn default() -> Self {
        Tuning {
            crowd: 1024,
            raw: 1 << 16,
            saturated: 8,
            frames: 16 << 20,
            grace: 4096,
            rest: 1024,
            span: 1 << 22,
            known: 1 << 20,
        }
    }
}

/// Where calls are cut into the units that are memoised.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Grain {
    /// A whole call.
    Call,
    /// At each subroutine's entry and return, as they are first taken.
    Sub,
    /// As [`Grain::Sub`], and at the target of each backward branch or jump.
    Loop,
    /// As [`Grain::Loop`], and inside the units of a start PC that come to be many, at an
    /// instruction few distinct runs of reads lead to or follow from.
    Split,
}

/// What a [`memo_orbit`] did.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MemoStats {
    /// Calls run, those run again to confirm a repeat included.
    pub calls: u64,
    /// Units replayed from the memo.
    pub hits: u64,
    /// Units emulated and stored.
    pub misses: u64,
    /// Instructions emulated.
    pub steps: u64,
    /// Units stored (distinct start PC and reads), over every flush.
    pub entries: u64,
    /// Units the memo holds.
    pub live: u64,
    /// Times the memo was emptied: a PC became a cut, memory read as constant was written, or
    /// a call started with another SP.
    pub flushes: u64,
    /// Cuts made inside crowded units ([`Grain::Split`]).
    pub splits: u64,
    /// Start PCs whose units are no longer memoised, being too many to find cheaply.
    pub raw: u64,
    /// Units not stored because the trie held one on the same reads: a stored unit emulated
    /// because replaying it would pass the call's cycle limit, whose call then returned.
    pub dropped: u64,
    /// Candidate repeats whose states differed.
    pub collisions: u64,
    /// Bytes the memo, the set of state hashes and the largest tape hold.
    pub bytes: u64,
    /// The period of the base the last tape was recorded for.
    pub tape_period: u64,
    /// The call the last tape was first run from.
    pub tape_from: u64,
    /// Calls run by a tape without emulating or replaying a unit.
    pub tape_calls: u64,
    /// Segments of those calls, each answered by what it was learned to do.
    pub tape_known: u64,
    /// Calls of a tape run as ordinary calls to learn what their segments do.
    pub tape_learned: u64,
    /// Calls among those whose segment did not rejoin the tape.
    pub tape_failed: u64,
    /// Calls among those run so because the tape's call would pass the cycle limit.
    pub tape_limited: u64,
    /// Times a tape was run from its last call on to its first.
    pub tape_passes: u64,
    /// Recordings started again with more registers live or more cells near.
    pub tape_retaped: u64,
    /// Tapes given up.
    pub tape_lost: u64,
    /// Bytes of the largest tape.
    pub tape_bytes: u64,
    /// `(pc, units)`: the start PCs the memo holds most units of, most first.
    pub roots: Vec<(u16, u32)>,
    /// `(location, nodes, values)`: the locations whose bytes tell the most held units apart,
    /// by the trie nodes that branch on each and the values they branch to, most first.
    pub fanout: Vec<(u32, u32, u32)>,
}

/// The registers (bit `r` for `PcodeVM` register `r`) opcode `op` reads and those it writes.
pub(crate) fn reg_use(op: u8) -> (u16, u16) {
    let f = &FLOW[op as usize];
    let ins = f.outs.iter().map(|o| o.1);
    let all = ins
        .chain(f.load_addr.iter().copied())
        .chain(f.store_addr.iter().copied());
    let reads = all.fold(0, |m, v| m | v) as u16;
    let writes = f.outs.iter().filter(|o| o.0 < 16).map(|o| 1u16 << o.0);
    let writes = writes.fold(0, |m, v| m | v);
    match f.kind {
        Kind::Jsr | Kind::Rts => (reads | SP, writes | SP),
        Kind::Rti => (reads | SP, writes | SP | FLAGS),
        Kind::Brk => (reads | SP | FLAGS, writes | SP | 1 << 10),
        _ => (reads, writes),
    }
}

/// The registers by `PcodeVM` index.
#[inline(always)]
pub(crate) fn regs(cpu: &Cpu) -> [u8; 16] {
    [
        cpu.a, cpu.x, cpu.y, cpu.sp, 0, 0, 0, 0, cpu.c, cpu.z, cpu.i, cpu.d, cpu.b, cpu.v, cpu.n, 0,
    ]
}

#[inline(always)]
fn reg(cpu: &Cpu, r: u32) -> u8 {
    regs(cpu)[r as usize & 15]
}

/// Set the registers of `mask` (bit `r` for index `r`, B never) from `to` and leave the rest.
#[inline(always)]
pub(crate) fn set_regs(cpu: &mut Cpu, mask: u32, to: [u8; 16]) {
    let pick = |r: usize, now: u8| if mask >> r & 1 != 0 { to[r] } else { now };
    cpu.a = pick(0, cpu.a);
    cpu.x = pick(1, cpu.x);
    cpu.y = pick(2, cpu.y);
    cpu.sp = pick(3, cpu.sp);
    cpu.c = pick(8, cpu.c);
    cpu.z = pick(9, cpu.z);
    cpu.i = pick(10, cpu.i);
    cpu.d = pick(11, cpu.d);
    cpu.v = pick(13, cpu.v);
    cpu.n = pick(14, cpu.n);
}

/// A read of location `loc` by the instruction at `site`, which gave `byte`.
#[inline(always)]
fn item(site: u16, loc: u32, byte: u8) -> u64 {
    (site as u64) << 32 | (loc as u64) << 8 | byte as u64
}

const LEAF: u32 = 0;
const FEW: u32 = 1;
const MANY: u32 = 2;
/// The values a [`FEW`] node branches to at most.
const FAN: usize = 4;
/// The words of a stored unit before its writes.
const UNIT: usize = 9;

/// [`MemoStats::roots`] and [`MemoStats::fanout`].
type Profile = (Vec<(u16, u32)>, Vec<(u32, u32, u32)>);

/// Where a trie node is pointed to from.
#[derive(Clone, Copy)]
enum Slot {
    Root(usize),
    Code(usize),
    Table(usize, usize),
}

/// Stored units by start PC and the reads each made, in order, as nodes in `code`.
///
/// A node is a word `reads << 2 | kind`, that many reads `loc << 8 | byte` that must all
/// match, then by kind: [`LEAF`] a stored unit; [`FEW`] a location, a count and [`FAN`] pairs
/// of a byte and the node to go on at where the location holds it; [`MANY`] a location and
/// the index in `tables` of a node per byte. A stored unit is [`UNIT`] words (`pc`, the
/// registers written as a mask, the registers' bytes in three words, the cycles and the
/// instructions in two each) then
/// a count and that many writes `addr << 8 | byte`. `sites` holds, per word of `code` that is
/// a read or a location, the PC of the instruction that read it.
struct Trie {
    roots: Box<[u32]>,
    code: Vec<u32>,
    sites: Vec<u16>,
    tables: Vec<[u32; 256]>,
}

impl Trie {
    fn new() -> Self {
        Trie {
            roots: vec![NONE; 0x10000].into_boxed_slice(),
            code: Vec::new(),
            sites: Vec::new(),
            tables: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.roots.fill(NONE);
        self.code.clear();
        self.sites.clear();
        self.tables.clear();
    }

    fn bytes(&self) -> usize {
        self.code.capacity() * 4 + self.sites.capacity() * 2 + self.tables.capacity() * 1024
    }

    fn word(&mut self, w: u32, site: u16) {
        self.code.push(w);
        self.sites.push(site);
    }

    /// Append `code[from..to]` and its sites.
    fn copy(&mut self, from: usize, to: usize) {
        self.code.extend_from_within(from..to);
        self.sites.extend_from_within(from..to);
    }

    fn leaf(&mut self, key: &[u64], unit: &[u32]) -> u32 {
        let n = self.code.len() as u32;
        self.word((key.len() as u32) << 2 | LEAF, 0);
        for &it in key {
            self.word(it as u32, (it >> 32) as u16);
        }
        for &w in unit {
            self.word(w, 0);
        }
        n
    }

    fn point(&mut self, slot: Slot, node: u32) {
        match slot {
            Slot::Root(pc) => self.roots[pc] = node,
            Slot::Code(at) => self.code[at] = node,
            Slot::Table(t, v) => self.tables[t][v] = node,
        }
    }

    /// The words of the node at `n` from the end of its reads on.
    fn tail(&self, n: usize) -> usize {
        let at = n + 1 + (self.code[n] >> 2) as usize;
        match self.code[n] & 3 {
            LEAF => UNIT + 1 + self.code[at + UNIT] as usize,
            FEW => 2 + 2 * FAN,
            _ => 2,
        }
    }

    /// The `(byte, node)` a branching node whose reads end at `at` goes on to.
    fn kids(&self, kind: u32, at: usize) -> Vec<(u8, u32)> {
        if kind == FEW {
            let pairs = &self.code[at + 2..at + 2 + 2 * self.code[at + 1] as usize];
            let pairs = pairs.as_chunks::<2>().0.iter();
            return pairs.map(|p| (p[0] as u8, p[1])).collect();
        }
        let table = self.tables[self.code[at + 1] as usize].iter().enumerate();
        let held = table.filter(|c| *c.1 != NONE);
        held.map(|(v, &c)| (v as u8, c)).collect()
    }

    /// Store `unit` under `pc` and `key`; false where a stored unit already ends on a prefix
    /// of `key`, or reads on past it or elsewhere.
    fn insert(&mut self, pc: u16, key: &[u64], unit: &[u32]) -> bool {
        let mut slot = Slot::Root(pc as usize);
        let mut n = self.roots[pc as usize] as usize;
        if n == NONE as usize {
            let new = self.leaf(key, unit);
            self.point(slot, new);
            return true;
        }
        let mut p = 0;
        loop {
            let (len, kind) = ((self.code[n] >> 2) as usize, self.code[n] & 3);
            for j in 0..len {
                let have = self.code[n + 1 + j];
                let Some(&want) = key.get(p + j) else {
                    return false;
                };
                if have == want as u32 {
                    continue;
                }
                if have >> 8 != want as u32 >> 8 {
                    return false;
                }
                let end = n + 1 + len + self.tail(n);
                let rest = self.code.len() as u32;
                self.word(((len - j - 1) as u32) << 2 | kind, 0);
                self.copy(n + 2 + j, end);
                let new = self.leaf(&key[p + j + 1..], unit);
                let head = self.code.len() as u32;
                self.word((j as u32) << 2 | FEW, 0);
                self.copy(n + 1, n + 1 + j);
                self.word(have >> 8, self.sites[n + 1 + j]);
                for w in [2, have & 0xFF, rest, want as u32 & 0xFF, new] {
                    self.word(w, 0);
                }
                for _ in 0..2 * FAN - 4 {
                    self.word(0, 0);
                }
                self.point(slot, head);
                return true;
            }
            p += len;
            let at = n + 1 + len;
            let Some(&want) = key.get(p) else {
                return false;
            };
            let (loc, v) = (want as u32 >> 8, want as u32 & 0xFF);
            if kind == LEAF || self.code[at] != loc {
                return false;
            }
            if kind == MANY {
                let t = self.code[at + 1] as usize;
                match self.tables[t][v as usize] {
                    NONE => {
                        self.tables[t][v as usize] = self.leaf(&key[p + 1..], unit);
                        return true;
                    }
                    next => (slot, n, p) = (Slot::Table(t, v as usize), next as usize, p + 1),
                }
                continue;
            }
            let count = self.code[at + 1] as usize;
            let pair = (0..count).find(|i| self.code[at + 2 + 2 * i] == v);
            if let Some(i) = pair {
                let child = at + 3 + 2 * i;
                (slot, n, p) = (Slot::Code(child), self.code[child] as usize, p + 1);
                continue;
            }
            let new = self.leaf(&key[p + 1..], unit);
            if count < FAN {
                self.code[at + 2 + 2 * count] = v;
                self.code[at + 3 + 2 * count] = new;
                self.code[at + 1] += 1;
                return true;
            }
            let mut table = [NONE; 256];
            for (v, child) in self.kids(FEW, at) {
                table[v as usize] = child;
            }
            table[v as usize] = new;
            self.code[n] = (len as u32) << 2 | MANY;
            self.code[at + 1] = self.tables.len() as u32;
            self.tables.push(table);
            return true;
        }
    }

    /// An instruction to cut the units starting at `pc` at: one that at least half of them
    /// reach, where the distinct runs of reads before its first read or from it on are more
    /// than one and few for the units (`saturated`, as [`Tuning::saturated`]), the one with
    /// the fewest of both.
    fn cut_for(&self, pc: u16, saturated: u32) -> Option<u16> {
        let root = self.roots[pc as usize];
        if root == NONE {
            return None;
        }
        let mut probe = Probe::new(self);
        probe.visit(root as usize);
        let units = probe.units;
        let score = |s: &Site| s.before + s.after.len() as u32;
        let fits = |s: &Site| {
            let (before, after) = (s.before, s.after.len() as u32);
            s.units * 2 >= units && before.min(after) > 1 && before.min(after) * saturated <= units
        };
        let sites = probe.sites.iter().filter(|s| *s.0 != pc && fits(s.1));
        sites.min_by_key(|s| (score(s.1), *s.0)).map(|s| *s.0)
    }

    /// [`MemoStats::roots`] and [`MemoStats::fanout`], at most `top` of each.
    fn profile(&self, top: usize) -> Profile {
        let (mut roots, mut probe) = (Vec::new(), Probe::new(self));
        for (pc, &n) in self.roots.iter().enumerate() {
            if n != NONE {
                let before = probe.units;
                probe.visit(n as usize);
                roots.push((pc as u16, probe.units - before));
            }
        }
        roots.sort_unstable_by_key(|r| (std::cmp::Reverse(r.1), r.0));
        roots.truncate(top);
        let fanout = probe.fanout.into_iter().map(|(l, (n, v))| (l, n, v));
        let mut fanout: Vec<(u32, u32, u32)> = fanout.collect();
        fanout.sort_unstable_by_key(|f| (std::cmp::Reverse(f.2), f.0));
        fanout.truncate(top);
        (roots, fanout)
    }
}

/// What cutting units at an instruction would leave: the distinct runs of reads `before` its
/// first read, the `units` that reach it and their distinct runs of reads from it on.
#[derive(Default)]
struct Site {
    before: u32,
    units: u32,
    after: FxSet<u64>,
}

/// A walk of the units under trie nodes: each instruction's [`Site`] and, per location
/// branched on, the nodes that branch on it and the values they branch to.
struct Probe<'t> {
    trie: &'t Trie,
    path: Vec<u64>,
    on_path: FxMap<u16, u32>,
    sites: FxMap<u16, Site>,
    fanout: FxMap<u32, (u32, u32)>,
    units: u32,
    hashes: Vec<u64>,
    seen: FxSet<u16>,
}

impl<'t> Probe<'t> {
    fn new(trie: &'t Trie) -> Self {
        Probe {
            trie,
            path: Vec::new(),
            on_path: FxMap::default(),
            sites: FxMap::default(),
            fanout: FxMap::default(),
            units: 0,
            hashes: Vec::new(),
            seen: FxSet::default(),
        }
    }

    /// Add the read at `at` of `code` to the path, `first` unless it is another byte of a
    /// location branched on.
    fn push(&mut self, at: usize, read: u32, first: bool) {
        let site = self.trie.sites[at];
        let n = self.on_path.entry(site).or_default();
        if *n == 0 && first {
            self.sites.entry(site).or_default().before += 1;
        }
        *n += 1;
        self.path.push(item(site, read >> 8, read as u8));
    }

    fn pop(&mut self, n: usize) {
        for _ in 0..n {
            let site = (self.path.pop().expect("pushed") >> 32) as u16;
            *self.on_path.get_mut(&site).expect("pushed") -= 1;
        }
    }

    fn unit(&mut self) {
        self.units += 1;
        self.hashes.clear();
        let mut h = Fx(SEED);
        for &it in self.path.iter().rev() {
            h.add(it);
            self.hashes.push(h.0);
        }
        self.seen.clear();
        for (&it, &h) in self.path.iter().zip(self.hashes.iter().rev()) {
            let site = (it >> 32) as u16;
            if self.seen.insert(site) {
                let s = self.sites.get_mut(&site).expect("pushed");
                s.units += 1;
                s.after.insert(h);
            }
        }
    }

    fn visit(&mut self, n: usize) {
        let (len, kind) = ((self.trie.code[n] >> 2) as usize, self.trie.code[n] & 3);
        for at in n + 1..n + 1 + len {
            self.push(at, self.trie.code[at], true);
        }
        let at = n + 1 + len;
        if kind == LEAF {
            self.unit();
        } else {
            let (loc, kids) = (self.trie.code[at], self.trie.kids(kind, at));
            let fan = self.fanout.entry(loc).or_default();
            *fan = (fan.0 + 1, fan.1 + kids.len() as u32);
            for (i, (v, child)) in kids.into_iter().enumerate() {
                self.push(at, loc << 8 | v as u32, i == 0);
                self.visit(child as usize);
                self.pop(1);
            }
        }
        self.pop(len);
    }
}

/// The hashes of the states calls started from: open addressing, 0 for an empty slot.
struct Seen {
    slots: Vec<u64>,
    len: usize,
    shift: u32,
}

impl Seen {
    fn new() -> Self {
        Seen {
            slots: vec![0; 1 << 10],
            len: 0,
            shift: 64 - 10,
        }
    }

    #[inline(always)]
    fn place(&mut self, h: u64) -> bool {
        let mask = self.slots.len() - 1;
        let mut i = (h.wrapping_mul(SEED) >> self.shift) as usize;
        loop {
            match self.slots[i] {
                0 => {
                    self.slots[i] = h;
                    return true;
                }
                s if s == h => return false,
                _ => i = (i + 1) & mask,
            }
        }
    }

    /// Add `h`; whether it was not there.
    #[inline(always)]
    fn insert(&mut self, h: u64) -> bool {
        if !self.place(h.max(1)) {
            return false;
        }
        self.len += 1;
        if self.len * 8 > self.slots.len() * 5 {
            let old = std::mem::take(&mut self.slots);
            self.slots = vec![0; old.len() * 2];
            self.shift -= 1;
            for h in old.into_iter().filter(|&h| h != 0) {
                self.place(h);
            }
        }
        true
    }
}

fn splitmix(x: &mut u64) -> u64 {
    *x = x.wrapping_add(SEED);
    let z = (*x ^ *x >> 30).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    let z = (z ^ z >> 27).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ z >> 31
}

/// `bus` with what the open unit reads before writing and what it writes recorded, each
/// write's change added to the hash of the tracked state, and the memo of closed units.
///
/// `bus` must be plain memory: a read gives the byte last written and nothing else changes.
/// `T` compiles in what a tape needs noted ([`fibre`]).
pub struct Memo<'a, B: Bus, const T: bool> {
    bus: &'a mut B,
    weight: Box<[u64]>,
    reg_cells: Vec<(u32, u64)>,
    hash: u64,
    ever: Box<[bool]>,
    unkeyed: Box<[bool]>,
    mark: Box<[u32]>,
    wmark: Box<[u32]>,
    stamp: u32,
    open: bool,
    /// The memo is to be emptied, or a start PC's units are crowded.
    attention: bool,
    stale: bool,
    crowded: Option<u16>,
    key: Vec<u64>,
    wrote: Vec<u16>,
    unit: Vec<u32>,
    site: u16,
    touched: u16,
    written: u16,
    start_pc: u16,
    start_cycles: u64,
    start_steps: u64,
    uses: [(u16, u16); 256],
    held: Box<[u32]>,
    crowd: Box<[u32]>,
    tuning: Tuning,
    trie: Trie,
    stats: MemoStats,
    fib: Notes,
}

impl<'a, B: Bus, const T: bool> Memo<'a, B, T> {
    /// A memo over `bus` whose tracked state is the memory in `keep` and the registers
    /// `cells` ([`REG`] offsets), cutting and giving up on crowded units as `tuning` says,
    /// the tracked cells among `fibre` set apart.
    fn new(bus: &'a mut B, keep: &AddrSet, cells: &[u32], tuning: Tuning, fibre: &[u16]) -> Self {
        let mut seed = SEED;
        let mut weight = vec![0u64; 0x10000].into_boxed_slice();
        for (a, w) in weight.iter_mut().enumerate() {
            let r = splitmix(&mut seed) | 1;
            if keep.contains(a as u16) {
                *w = r;
            }
        }
        let reg_cells = cells
            .iter()
            .map(|&r| (r, splitmix(&mut seed) | 1))
            .collect();
        let apart: Vec<u16> = fibre
            .iter()
            .copied()
            .filter(|&a| keep.contains(a))
            .collect();
        let mut m = Memo {
            bus,
            weight,
            reg_cells,
            hash: 0,
            ever: vec![false; 0x10000].into_boxed_slice(),
            unkeyed: vec![false; 0x10000].into_boxed_slice(),
            mark: vec![0; 0x10000].into_boxed_slice(),
            wmark: vec![0; 0x10000].into_boxed_slice(),
            stamp: 0,
            open: false,
            attention: false,
            stale: false,
            crowded: None,
            key: Vec::new(),
            wrote: Vec::new(),
            unit: Vec::new(),
            site: 0,
            touched: 0,
            written: 0,
            start_pc: 0,
            start_cycles: 0,
            start_steps: 0,
            uses: std::array::from_fn(|op| reg_use(op as u8)),
            held: vec![0; 0x10000].into_boxed_slice(),
            crowd: vec![tuning.crowd; 0x10000].into_boxed_slice(),
            tuning,
            trie: Trie::new(),
            stats: MemoStats::default(),
            fib: Notes::new(&apart),
        };
        m.rehash();
        m
    }

    fn rehash(&mut self) {
        self.hash = 0;
        for a in 0..0x10000 {
            let v = self.bus.peek(a as u16) as u64;
            self.hash = self.hash.wrapping_add(self.weight[a].wrapping_mul(v));
        }
    }

    /// The hash of the tracked state: equal states hash alike.
    #[inline(always)]
    fn state_hash(&self, cpu: &Cpu) -> u64 {
        let cells = self.reg_cells.iter();
        cells.fold(self.hash, |h, &(r, w)| {
            h.wrapping_add(w.wrapping_mul(reg(cpu, r) as u64))
        })
    }

    /// The tracked state, memory then registers.
    fn state(&mut self, cpu: &Cpu) -> Vec<u8> {
        let mem = (0..0x10000).filter(|&a| self.weight[a] != 0);
        let mut out: Vec<u8> = mem.map(|a| self.bus.peek(a as u16)).collect();
        out.extend(self.reg_cells.iter().map(|&(r, _)| reg(cpu, r)));
        out
    }

    fn snapshot(&mut self) -> Vec<u8> {
        (0..=0xFFFF).map(|a| self.bus.peek(a)).collect()
    }

    fn restore(&mut self, ram: &[u8]) {
        for (a, &v) in ram.iter().enumerate() {
            self.bus.poke(a as u16, v);
        }
        self.rehash();
    }

    #[inline]
    fn on_read(&mut self, addr: u16, v: u8) -> u8 {
        let a = addr as usize;
        if T && self.fib.logging {
            self.fib.rlog.push(addr);
        }
        if self.open && self.mark[a] != self.stamp {
            self.mark[a] = self.stamp;
            if self.ever[a] {
                self.key.push(item(self.site, addr as u32, v));
            } else {
                self.unkeyed[a] = true;
            }
        }
        v
    }

    #[inline]
    fn on_write(&mut self, addr: u16, v: u8) {
        let a = addr as usize;
        if !self.ever[a] {
            self.ever[a] = true;
            self.fib.dead = true;
            if self.unkeyed[a] {
                (self.stale, self.attention) = (true, true);
            }
        }
        let old = self.bus.peek(addr);
        if T && self.fib.logging {
            self.fib.wlog.push((addr, old));
        }
        let d = (v as u64).wrapping_sub(old as u64);
        self.hash = self.hash.wrapping_add(self.weight[a].wrapping_mul(d));
        if self.open {
            self.mark[a] = self.stamp;
            if self.wmark[a] != self.stamp {
                self.wmark[a] = self.stamp;
                self.wrote.push(addr);
            }
        }
    }

    /// Empty the memo and abandon the open unit.
    fn flush(&mut self) {
        self.trie.clear();
        self.held.fill(0);
        self.unkeyed.fill(false);
        self.open = false;
        (self.stale, self.crowded, self.attention) = (false, None, false);
        self.stats.flushes += 1;
        self.stats.live = 0;
    }

    /// The memo is to be emptied before a unit is next looked up.
    fn spoil(&mut self) {
        (self.stale, self.attention) = (true, true);
    }

    /// Where in the trie's code the stored unit starting at `cpu`'s PC whose reads all match
    /// the state is.
    #[inline(always)]
    fn lookup(&mut self, cpu: &Cpu) -> Option<usize> {
        let mut n = self.trie.roots[cpu.pc as usize] as usize;
        if n == NONE as usize {
            return None;
        }
        let (bus, code, regs) = (&mut *self.bus, &self.trie.code[..], regs(cpu));
        let mut value = |loc: u32| {
            let (mem, reg) = (bus.peek(loc as u16), regs[loc as usize & 15]);
            if loc < REG {
                mem
            } else {
                reg
            }
        };
        loop {
            let head = code[n];
            let at = n + 1 + (head >> 2) as usize;
            for &it in &code[n + 1..at] {
                if value(it >> 8) != it as u8 {
                    return None;
                }
            }
            let next = match head & 3 {
                LEAF => return Some(at),
                FEW => {
                    let v = value(code[at]) as u32;
                    let pairs = &code[at + 2..at + 2 + 2 * FAN];
                    let count = code[at + 1] as usize;
                    let found = (0..count).find(|&i| pairs[2 * i] == v);
                    found.map_or(NONE, |i| pairs[2 * i + 1])
                }
                _ => self.trie.tables[code[at + 1] as usize][value(code[at]) as usize],
            };
            if next == NONE {
                return None;
            }
            n = next as usize;
        }
    }

    /// Note for the tape the stored unit at `at`, which [`Memo::lookup`] found for `cpu`:
    /// what it is keyed on and, where a tape is recorded, what it does.
    #[inline(never)]
    fn note_hit(&mut self, at: usize, cpu: &Cpu) {
        let (bus, code, fib) = (&mut *self.bus, &self.trie.code[..], &mut self.fib);
        let now = regs(cpu);
        if fib.rec {
            fib.begin(cpu.pc, now);
        }
        let mut n = self.trie.roots[cpu.pc as usize] as usize;
        loop {
            let head = code[n];
            let end = n + 1 + (head >> 2) as usize;
            for &it in &code[n + 1..end] {
                fib.read(it >> 8);
            }
            if head & 3 == LEAF {
                fib.uses(0, code[at + 1] as u16);
                break;
            }
            let loc = code[end];
            fib.read(loc);
            let (mem, reg) = (bus.peek(loc as u16), now[loc as usize & 15]);
            let v = if loc < REG { mem } else { reg } as u32;
            n = if head & 3 == FEW {
                let count = code[end + 1] as usize;
                let i = (0..count).find(|&i| code[end + 2 + 2 * i] == v);
                code[end + 3 + 2 * i.expect("found")] as usize
            } else {
                self.trie.tables[code[end + 1] as usize][v as usize] as usize
            };
        }
        if fib.rec {
            let unit = &code[at..];
            fib.write(&unit[UNIT + 1..UNIT + 1 + unit[UNIT] as usize]);
            let cycles = unit[5] as u64 | (unit[6] as u64) << 32;
            let steps = unit[7] as u64 | (unit[8] as u64) << 32;
            fib.end(unit[1] as u16, cycles, steps);
        }
    }

    /// Note for the tape the unit being closed, which took `cycles` and `steps`.
    fn note_miss(&mut self, cycles: u64, steps: u64) {
        for i in 0..self.key.len() {
            self.fib.read(self.key[i] as u32 >> 8);
        }
        for i in 0..self.wrote.len() {
            let a = self.wrote[i];
            let w = (a as u32) << 8 | self.bus.peek(a) as u32;
            self.fib.write(&[w]);
        }
        self.fib.end(self.written, cycles, steps);
    }

    /// The part of the hash of the tracked state that `cells` are.
    fn cells_hash(&mut self, cells: &[u16]) -> u64 {
        let weigh = |h: u64, &a: &u16| {
            let v = self.bus.peek(a) as u64;
            h.wrapping_add(self.weight[a as usize].wrapping_mul(v))
        };
        cells.iter().fold(0, weigh)
    }

    /// The cycles the stored unit at `at` takes.
    #[inline(always)]
    fn cycles(&self, at: usize) -> u64 {
        self.trie.code[at + 5] as u64 | (self.trie.code[at + 6] as u64) << 32
    }

    /// Apply the stored unit at `at` to `cpu` and memory as running it would.
    #[inline(always)]
    fn replay(&mut self, at: usize, cpu: &mut Cpu) {
        let code = &self.trie.code[at..];
        let writes = &code[UNIT + 1..UNIT + 1 + code[UNIT] as usize];
        for &w in writes {
            let (addr, v) = ((w >> 8) as u16, w as u8);
            let old = self.bus.peek(addr);
            if T && self.fib.logging {
                self.fib.wlog.push((addr, old));
            }
            if old != v {
                let d = (v as u64).wrapping_sub(old as u64);
                self.hash = self
                    .hash
                    .wrapping_add(self.weight[addr as usize].wrapping_mul(d));
                self.bus.poke(addr, v);
            }
        }
        let mut to = [0; 16];
        to[..4].copy_from_slice(&code[2].to_le_bytes());
        to[8..12].copy_from_slice(&code[3].to_le_bytes());
        to[12..].copy_from_slice(&code[4].to_le_bytes());
        set_regs(cpu, code[1], to);
        cpu.pc = code[0] as u16;
        cpu.cycles += code[5] as u64 | (code[6] as u64) << 32;
        cpu.steps += code[7] as u64 | (code[8] as u64) << 32;
    }

    /// Open a unit at `cpu`'s PC.
    fn begin(&mut self, cpu: &Cpu) {
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            self.mark.fill(0);
            self.wmark.fill(0);
            self.stamp = 1;
        }
        self.key.clear();
        self.wrote.clear();
        (self.touched, self.written) = (0, 0);
        (self.start_pc, self.start_cycles) = (cpu.pc, cpu.cycles);
        self.start_steps = cpu.steps;
        self.open = true;
        self.stats.misses += 1;
        if T && self.fib.rec {
            self.fib.begin(cpu.pc, regs(cpu));
        }
    }

    /// The instruction at `cpu`'s PC is about to run: its opcode, and with a unit open the
    /// registers it reads that the unit has not written, are reads of the unit.
    #[inline(always)]
    fn instruction(&mut self, cpu: &Cpu) -> u8 {
        self.site = cpu.pc;
        if T && self.fib.rec {
            self.fib.broken |= !self.open;
        }
        let op = self.bus.fetch(cpu.pc);
        self.on_read(cpu.pc, op);
        if T && self.fib.logging {
            self.fib
                .uses(self.uses[op as usize].0, self.uses[op as usize].1);
        }
        if self.open {
            let (reads, writes) = self.uses[op as usize];
            let mut new = reads & !self.touched;
            while new != 0 {
                let r = new.trailing_zeros();
                self.key.push(item(self.site, REG + r, reg(cpu, r)));
                new &= new - 1;
            }
            self.touched |= reads | writes;
            self.written |= writes;
        }
        op
    }

    /// An instruction to cut the units of the start PC grown crowded at (with `split`); where
    /// none fits they are left to grow to twice as many, and past [`Tuning::raw`] the start PC
    /// is returned to be emulated.
    fn relieve(&mut self, split: bool) -> (Option<u16>, Option<u16>) {
        let Some(pc) = self.crowded.take() else {
            return (None, None);
        };
        let site = if split {
            self.trie.cut_for(pc, self.tuning.saturated)
        } else {
            None
        };
        if site.is_some() {
            return (site, None);
        }
        let crowd = &mut self.crowd[pc as usize];
        *crowd = crowd.saturating_mul(2);
        if *crowd <= self.tuning.raw {
            return (None, None);
        }
        self.trie.roots[pc as usize] = NONE;
        self.stats.raw += 1;
        self.stats.live -= self.held[pc as usize] as u64;
        self.held[pc as usize] = 0;
        (None, Some(pc))
    }

    /// Close the open unit at `cpu`, which it has reached, and store it.
    fn close(&mut self, cpu: &Cpu) {
        self.open = false;
        let (cycles, now) = (cpu.cycles - self.start_cycles, regs(cpu));
        let steps = cpu.steps - self.start_steps;
        if T && self.fib.rec {
            self.note_miss(cycles, steps);
        }
        self.unit.clear();
        self.unit.extend([
            cpu.pc as u32,
            self.written as u32,
            u32::from_le_bytes([now[0], now[1], now[2], now[3]]),
            u32::from_le_bytes([now[8], now[9], now[10], now[11]]),
            u32::from_le_bytes([0, now[13], now[14], 0]),
            cycles as u32,
            (cycles >> 32) as u32,
            steps as u32,
            (steps >> 32) as u32,
            self.wrote.len() as u32,
        ]);
        for &a in &self.wrote {
            self.unit.push((a as u32) << 8 | self.bus.peek(a) as u32);
        }
        if !self.trie.insert(self.start_pc, &self.key, &self.unit) {
            self.stats.dropped += 1;
            return;
        }
        self.stats.entries += 1;
        self.stats.live += 1;
        let at = self.start_pc as usize;
        self.held[at] += 1;
        if self.held[at] >= self.crowd[at] {
            (self.crowded, self.attention) = (Some(self.start_pc), true);
        }
    }
}

impl<B: Bus, const T: bool> Bus for Memo<'_, B, T> {
    #[inline(always)]
    fn read(&mut self, addr: u16) -> u8 {
        let v = self.bus.read(addr);
        self.on_read(addr, v)
    }

    #[inline(always)]
    fn write(&mut self, addr: u16, v: u8) {
        self.on_write(addr, v);
        self.bus.write(addr, v)
    }

    #[inline(always)]
    fn fetch(&mut self, addr: u16) -> u8 {
        let v = self.bus.fetch(addr);
        self.on_read(addr, v)
    }

    #[inline(always)]
    fn load(&mut self, addr: u16) -> u8 {
        let v = self.bus.load(addr);
        self.on_read(addr, v)
    }

    #[inline(always)]
    fn store(&mut self, addr: u16, v: u8) {
        self.on_write(addr, v);
        self.bus.store(addr, v)
    }

    #[inline(always)]
    fn stack_read(&mut self, addr: u16) -> u8 {
        let v = self.bus.stack_read(addr);
        self.on_read(addr, v)
    }

    #[inline(always)]
    fn stack_write(&mut self, addr: u16, v: u8) {
        self.on_write(addr, v);
        self.bus.stack_write(addr, v)
    }

    #[inline(always)]
    fn peek(&mut self, addr: u16) -> u8 {
        self.bus.peek(addr)
    }

    #[inline(always)]
    fn sync(&mut self, cpu: &Cpu) {
        self.bus.sync(cpu)
    }
}

/// A bus over a [`Memo`]: the memo itself, or its accesses redirected.
trait Over<'a, B: Bus + 'a, const T: bool>: Bus {
    fn memo(&mut self) -> &mut Memo<'a, B, T>;
}

impl<'a, B: Bus, const T: bool> Over<'a, B, T> for Memo<'a, B, T> {
    #[inline(always)]
    fn memo(&mut self) -> &mut Memo<'a, B, T> {
        self
    }
}

impl<'a, B: Bus, const T: bool> Over<'a, B, T> for Redirect<'_, Memo<'a, B, T>> {
    #[inline(always)]
    fn memo(&mut self) -> &mut Memo<'a, B, T> {
        self.inner()
    }
}

/// A set of keys under 2^34 with the keys last added remembered, so adding one again costs a
/// compare.
struct Recent {
    last: Box<[u64]>,
    keys: Keys,
}

impl Recent {
    fn new(his: usize) -> Self {
        Recent {
            last: vec![u64::MAX; 1 << RECENT].into_boxed_slice(),
            keys: Keys::new(his),
        }
    }

    /// Add `key`; whether it was not in the set.
    #[inline]
    fn insert(&mut self, key: u64) -> bool {
        let slot = &mut self.last[(key.wrapping_mul(SEED) >> (64 - RECENT)) as usize];
        if *slot == key {
            return false;
        }
        *slot = key;
        self.keys.insert_new(key)
    }
}

/// `inner` with each data access noted as the instruction named it: its [`crate::access_key`]
/// and whether it loads a volatile address.
struct Tap<'t, I: Bus> {
    inner: &'t mut I,
    accesses: Recent,
    volatile: &'t AddrSet,
    read_volatile: bool,
    pc: u16,
}

impl<I: Bus> Bus for Tap<'_, I> {
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
        if self.accesses.insert((self.pc as u64) << 17 | addr as u64) {
            self.read_volatile |= self.volatile.contains(addr);
        }
        self.inner.load(addr)
    }

    #[inline(always)]
    fn store(&mut self, addr: u16, v: u8) {
        self.accesses
            .insert((self.pc as u64) << 17 | 1 << 16 | addr as u64);
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
        self.pc = cpu.pc;
        self.inner.sync(cpu)
    }
}

const CUT: u8 = 1;
const EMULATED: u8 = 2;

/// The calls of a [`memo_orbit`] over the bus `tap`.
struct Run<'t, 'r, I: Bus, S: FnMut() -> bool> {
    tap: Tap<'t, I>,
    edges: Recent,
    /// Per PC: [`CUT`] where units start, [`EMULATED`] where they would but are not memoised.
    cut: Box<[u8]>,
    /// How each opcode can make the PC it leads to a cut: 1 a call or return, 2 a jump.
    cutting: [u8; 256],
    grain: Grain,
    pc: u16,
    acc: u8,
    max_cycles: u64,
    start_sp: Option<u8>,
    stopped: &'r mut S,
    fx: Fibre,
}

impl<'a, I: Bus, S: FnMut() -> bool> Run<'_, '_, I, S> {
    /// Empty the memo if it is stale, else relieve its crowded start PC.
    #[inline(never)]
    fn attend<B: Bus + 'a, const T: bool>(&mut self)
    where
        I: Over<'a, B, T>,
    {
        let memo = self.tap.inner.memo();
        memo.open = false;
        if !memo.stale {
            let (site, raw) = memo.relieve(self.grain >= Grain::Split);
            if let Some(pc) = raw {
                self.cut[pc as usize] = EMULATED;
            }
            if let Some(site) = site {
                self.cut[site as usize] = CUT;
                memo.stats.splits += 1;
                memo.stale = true;
            }
        }
        memo.attention = false;
        if memo.stale {
            memo.flush();
        }
    }

    /// One call from `cpu`, as [`crate::run_to_rts`] runs it.
    fn call<B: Bus + 'a, const T: bool>(&mut self, cpu: &mut Cpu) -> Stop
    where
        I: Over<'a, B, T>,
    {
        let (cycles, steps) = (cpu.cycles, cpu.steps);
        let (sp_above, end) = self.setup(cpu);
        let stop = self.go(cpu, sp_above, end, None);
        if T && self.tap.inner.memo().fib.rec {
            self.record(cpu, stop, cycles, steps);
        }
        stop
    }

    /// Start a call from `cpu`: the SP its return is above and the cycles it must end by.
    #[inline(always)]
    fn setup<B: Bus + 'a, const T: bool>(&mut self, cpu: &mut Cpu) -> (i32, u64)
    where
        I: Over<'a, B, T>,
    {
        let memo = self.tap.inner.memo();
        memo.stats.calls += 1;
        if self.start_sp != Some(cpu.sp) {
            if self.start_sp.is_some() {
                memo.spoil();
            }
            self.start_sp = Some(cpu.sp);
        }
        let sp_above = cpu.sp as i32 - 1;
        if T && memo.fib.rec {
            memo.fib.entry(cpu.sp);
        }
        for byte in [0x00, 0x01] {
            Bus::write(memo, 0x100 | cpu.sp as u16, byte);
            cpu.sp = cpu.sp.wrapping_sub(1);
        }
        cpu.a = self.acc;
        cpu.pc = self.pc;
        let end = cpu.cycles.saturating_add(self.max_cycles).saturating_add(1);
        (sp_above, end)
    }

    /// Run the call from `cpu` to its end, or with `until` `(pc, n)` to the `n`th time a
    /// unit would then start at `pc`, where it is first not counted ([`Stop::Pc`]).
    #[inline(always)]
    fn go<B: Bus + 'a, const T: bool>(
        &mut self,
        cpu: &mut Cpu,
        sp_above: i32,
        end: u64,
        until: Option<(u16, u32)>,
    ) -> Stop
    where
        I: Over<'a, B, T>,
    {
        let (mut prev, mut use_memo, mut hits): (Option<u16>, bool, u64) = (None, true, 0);
        let (mut first, mut left) = (true, until.map_or(0, |u| u.1));
        let stop = loop {
            if cpu.sp as i32 > sp_above {
                break Stop::Sp;
            }
            if cpu.cycles >= end {
                break Stop::Cycles;
            }
            let at = cpu.pc;
            if let Some(p) = prev {
                self.edges.insert((p as u64) << 16 | at as u64);
            }
            if self.cut[at as usize] != 0 && use_memo {
                let memo = self.tap.inner.memo();
                if memo.open | memo.attention {
                    if memo.open && !memo.stale {
                        memo.close(cpu);
                    }
                    if memo.attention {
                        self.attend();
                    }
                }
                if T {
                    if let Some((pc, _)) = until {
                        if at == pc && !first {
                            left -= 1;
                            if left == 0 {
                                break Stop::Pc;
                            }
                        }
                        first = false;
                    }
                }
                if self.cut[at as usize] == CUT {
                    let memo = self.tap.inner.memo();
                    if let Some(unit) = memo.lookup(cpu) {
                        if cpu.cycles + memo.cycles(unit) < end {
                            if T && (memo.fib.rec | memo.fib.logging) {
                                memo.note_hit(unit, cpu);
                            }
                            memo.replay(unit, cpu);
                            hits += 1;
                            prev = None;
                            continue;
                        }
                    }
                    memo.begin(cpu);
                }
            }
            prev = Some(at);
            let memo = self.tap.inner.memo();
            let op = memo.instruction(cpu);
            memo.stats.steps += 1;
            let jams = cpu.jams;
            cpu.step(&mut self.tap);
            if (self.stopped)() {
                break Stop::Stopped;
            }
            let cutting = self.cutting[op as usize];
            if cutting != 0 {
                let to = &mut self.cut[cpu.pc as usize];
                let cuts = match cutting {
                    1 => self.grain >= Grain::Sub,
                    _ => self.grain >= Grain::Loop && cpu.pc <= at,
                };
                if cuts && *to == 0 {
                    *to = CUT;
                    self.tap.inner.memo().spoil();
                }
            }
            if cpu.jams != jams {
                use_memo = false;
                self.tap.inner.memo().open = false;
            }
        };
        let memo = self.tap.inner.memo();
        if stop == Stop::Sp && memo.open && !memo.stale {
            memo.close(cpu);
        }
        memo.open = false;
        memo.stats.hits += hits;
        stop
    }
}

/// [`crate::orbit`] by memoised units cut at `grain`: the same [`Orbit`], its state the
/// memory in `keep` and the registers `cells` ([`REG`] offsets) at each call's entry, a
/// repeated state confirmed by comparing it with the earlier one. `bus` must be plain memory
/// (every read gives the byte last written there).
#[allow(clippy::too_many_arguments)]
pub fn memo_orbit<B: Bus>(
    cpu: &mut Cpu,
    bus: &mut B,
    pc: u16,
    acc: u8,
    max_cycles: u64,
    limit: usize,
    remap: Option<&Remap>,
    volatile: &AddrSet,
    keep: &AddrSet,
    cells: &[u32],
    grain: Grain,
    stopped: impl FnMut() -> bool,
) -> (Orbit, MemoStats) {
    let (tuning, fibre) = (Tuning::default(), []);
    memo_orbit_tuned::<B, false>(
        cpu, bus, pc, acc, max_cycles, limit, remap, volatile, keep, cells, grain, tuning, &fibre,
        stopped,
    )
}

/// [`memo_orbit`] with the tracked memory cells among `fibre` set apart: once the rest of
/// the written memory repeats at a call's entry, calls are run by a tape of that period
/// and only where they read `fibre`. The same [`Orbit`], memory and registers.
#[allow(clippy::too_many_arguments)]
pub fn memo_orbit_fibre<B: Bus>(
    cpu: &mut Cpu,
    bus: &mut B,
    pc: u16,
    acc: u8,
    max_cycles: u64,
    limit: usize,
    remap: Option<&Remap>,
    volatile: &AddrSet,
    keep: &AddrSet,
    cells: &[u32],
    grain: Grain,
    fibre: &[u16],
    stopped: impl FnMut() -> bool,
) -> (Orbit, MemoStats) {
    let tuning = Tuning::default();
    memo_orbit_tuned::<B, true>(
        cpu, bus, pc, acc, max_cycles, limit, remap, volatile, keep, cells, grain, tuning, fibre,
        stopped,
    )
}

/// [`memo_orbit_fibre`] as `tuning` says; without `T` no tape is recorded.
#[allow(clippy::too_many_arguments)]
pub(crate) fn memo_orbit_tuned<B: Bus, const T: bool>(
    cpu: &mut Cpu,
    bus: &mut B,
    pc: u16,
    acc: u8,
    max_cycles: u64,
    limit: usize,
    remap: Option<&Remap>,
    volatile: &AddrSet,
    keep: &AddrSet,
    cells: &[u32],
    grain: Grain,
    tuning: Tuning,
    fibre: &[u16],
    mut stopped: impl FnMut() -> bool,
) -> (Orbit, MemoStats) {
    let mut memo = Memo::<B, T>::new(bus, keep, cells, tuning, fibre);
    match remap {
        Some(r) => {
            let mut over = Accesses::remapped(&mut memo, r);
            walk(
                cpu,
                &mut over,
                pc,
                acc,
                max_cycles,
                limit,
                volatile,
                grain,
                &mut stopped,
            )
        }
        None => walk(
            cpu,
            &mut memo,
            pc,
            acc,
            max_cycles,
            limit,
            volatile,
            grain,
            &mut stopped,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn walk<'a, B: Bus + 'a, I: Over<'a, B, T>, const T: bool>(
    cpu: &mut Cpu,
    over: &mut I,
    pc: u16,
    acc: u8,
    max_cycles: u64,
    limit: usize,
    volatile: &AddrSet,
    grain: Grain,
    stopped: &mut impl FnMut() -> bool,
) -> (Orbit, MemoStats) {
    let mut cut = vec![0; 0x10000].into_boxed_slice();
    cut[pc as usize] = CUT;
    let memo = over.memo();
    let fx = Fibre::new(T && memo.fib.is_f.contains(&true), memo.tuning);
    let cutting = std::array::from_fn(|op| match FLOW[op].kind {
        Kind::Jsr | Kind::Rts => 1,
        Kind::Branch | Kind::Jmp => 2,
        _ => 0,
    });
    let mut run = Run {
        tap: Tap {
            inner: over,
            accesses: Recent::new(1 << 17),
            volatile,
            read_volatile: false,
            pc: 0,
        },
        edges: Recent::new(1 << 16),
        cut,
        cutting,
        grain,
        pc,
        acc,
        max_cycles,
        start_sp: None,
        stopped,
        fx,
    };
    let first = (run.tap.inner.memo().snapshot(), cpu.clone());
    let (mut seen, mut end, mut states, mut at) = (Seen::new(), End::Open, 0, 0);
    'calls: for k in 0..=limit {
        at = k;
        if T && run.enter(cpu, k).is_some() {
            end = End::Stopped;
            break;
        }
        let taped = T && run.fx.taped();
        let h = match taped {
            true => run.tape_hash(),
            false => run.tap.inner.memo().state_hash(cpu),
        };
        if !seen.insert(h) {
            if taped && run.settle(cpu, k).is_some() {
                end = End::Stopped;
                break;
            }
            let paused = run.fx.pause();
            let memo = run.tap.inner.memo();
            let rec = std::mem::take(&mut memo.fib.rec);
            let here = (memo.snapshot(), cpu.clone(), memo.state(cpu));
            memo.restore(&first.0);
            *cpu = first.1.clone();
            let mut repeats = false;
            for _ in 0..k {
                let memo = run.tap.inner.memo();
                if memo.state_hash(cpu) == h && memo.state(cpu) == here.2 {
                    repeats = true;
                    break;
                }
                if run.call(cpu) == Stop::Stopped {
                    end = End::Stopped;
                    break 'calls;
                }
            }
            run.tap.inner.memo().restore(&here.0);
            *cpu = here.1;
            run.tap.inner.memo().fib.rec = rec;
            run.fx.resume(paused);
            if repeats {
                end = End::Closed;
                break;
            }
            run.tap.inner.memo().stats.collisions += 1;
        }
        states += 1;
        if k == limit {
            break;
        }
        let stop = match T {
            true => run.step(cpu, k),
            false => run.call(cpu),
        };
        if stop == Stop::Stopped {
            end = End::Stopped;
            break;
        }
        at = k + 1;
        if run.tap.read_volatile {
            end = End::Volatile;
            break;
        }
        if cpu.jammed {
            end = End::Jam;
            break;
        }
    }
    if T {
        run.wind_up(cpu, at);
    }
    let memo = run.tap.inner.memo();
    let mut stats = memo.stats.clone();
    stats.bytes = (memo.trie.bytes() + seen.slots.capacity() * 8) as u64 + stats.tape_bytes;
    (stats.roots, stats.fanout) = memo.trie.profile(PROFILE);
    let orbit = Orbit {
        end,
        states,
        accesses: run.tap.accesses.keys.sorted(),
        edges: run.edges.keys.sorted(),
    };
    (orbit, stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    const REGISTERS: [u32; 10] = [0, 1, 2, 3, 8, 9, 10, 11, 13, 14];

    fn set_reg(cpu: &mut Cpu, r: u32, v: u8) {
        set_regs(cpu, 1 << r, [v; 16]);
    }

    /// A RAM logging the addresses read and the bytes written, each write undone by `undo`.
    struct Logged {
        ram: Box<[u8; 0x10000]>,
        reads: Vec<u16>,
        writes: Vec<(u16, u8)>,
        old: Vec<(u16, u8)>,
    }

    impl Bus for Logged {
        fn read(&mut self, addr: u16) -> u8 {
            self.reads.push(addr);
            self.ram[addr as usize]
        }

        fn write(&mut self, addr: u16, v: u8) {
            self.old.push((addr, self.ram[addr as usize]));
            self.writes.push((addr, v));
            self.ram[addr as usize] = v;
        }
    }

    impl Logged {
        fn step(&mut self, cpu: &Cpu) -> (Cpu, Vec<u16>, Vec<(u16, u8)>) {
            let mut c = cpu.clone();
            c.step(self);
            while let Some((a, v)) = self.old.pop() {
                self.ram[a as usize] = v;
            }
            (
                c,
                std::mem::take(&mut self.reads),
                std::mem::take(&mut self.writes),
            )
        }
    }

    #[test]
    fn register_use_bounds_what_each_opcode_reads_and_names_what_it_writes() {
        let mut seed = 7;
        let mut bus = Logged {
            ram: Box::new([0; 0x10000]),
            reads: Vec::new(),
            writes: Vec::new(),
            old: Vec::new(),
        };
        for b in bus.ram.iter_mut() {
            *b = splitmix(&mut seed) as u8;
        }
        for op in 0..=255u8 {
            if FLOW[op as usize].kind == Kind::Jam {
                continue;
            }
            let (reads, writes) = reg_use(op);
            for _ in 0..48 {
                let mut cpu = Cpu::new();
                cpu.pc = splitmix(&mut seed) as u16;
                bus.ram[cpu.pc as usize] = op;
                for r in REGISTERS {
                    let v = splitmix(&mut seed) as u8;
                    set_reg(&mut cpu, r, if r < 8 { v } else { v & 1 });
                }
                assert_eq!(regs(&cpu)[12], 0);
                let (base, base_reads, base_writes) = bus.step(&cpu);
                for r in REGISTERS {
                    if writes >> r & 1 == 0 {
                        assert_eq!(reg(&base, r), reg(&cpu, r), "{op:02X} writes {r}");
                    }
                    if reads >> r & 1 != 0 {
                        continue;
                    }
                    let mut other = cpu.clone();
                    let v = reg(&cpu, r) ^ if r < 8 { 0x5B } else { 1 };
                    set_reg(&mut other, r, v);
                    let (got, got_reads, got_writes) = bus.step(&other);
                    assert_eq!(got_reads, base_reads, "{op:02X} reads {r}");
                    assert_eq!(got_writes, base_writes, "{op:02X} reads {r}");
                    assert_eq!((got.pc, got.cycles), (base.pc, base.cycles), "{op:02X} {r}");
                    for q in REGISTERS {
                        let want = if q == r && writes >> r & 1 == 0 {
                            v
                        } else {
                            reg(&base, q)
                        };
                        assert_eq!(reg(&got, q), want, "{op:02X} reads {r} into {q}");
                    }
                }
            }
        }
    }

    /// The PC of the unit stored under `pc` whose reads `vals` (of locations 0, 1, ...) match.
    fn find(t: &Trie, pc: u16, vals: &[u8]) -> Option<u32> {
        let mut ram = Box::new([0u8; 0x10000]);
        ram[..vals.len()].copy_from_slice(vals);
        let none = AddrSet::default();
        let mut memo = Memo::<_, false>::new(&mut *ram, &none, &[], Tuning::default(), &[]);
        memo.trie = Trie {
            roots: t.roots.clone(),
            code: t.code.clone(),
            sites: t.sites.clone(),
            tables: t.tables.clone(),
        };
        let mut cpu = Cpu::new();
        cpu.pc = pc;
        memo.lookup(&cpu).map(|at| memo.trie.code[at])
    }

    #[test]
    fn trie_finds_a_unit_only_by_all_its_reads() {
        let mut t = Trie::new();
        let unit = |pc: u32| [pc, 0, 0, 0, 0, 0, 0, 0, 0, 1, 7 << 8 | 9];
        let key = |vals: &[u8]| -> Vec<u64> {
            let at = vals.iter().enumerate();
            at.map(|(loc, &v)| item(9, loc as u32, v)).collect()
        };
        assert!(t.insert(5, &key(&[1, 2, 3]), &unit(0)));
        assert!(t.insert(5, &key(&[1, 2, 4]), &unit(1)));
        assert!(t.insert(5, &key(&[1, 7]), &unit(2)));
        for v in 0..20 {
            assert!(t.insert(5, &key(&[1, 9 + v, v, v]), &unit(3 + v as u32)));
        }
        assert!(t.insert(5, &key(&[1, 9, 0, 1]), &unit(23)));
        assert!(!t.insert(5, &key(&[1, 2]), &unit(99)));
        assert!(!t.insert(5, &key(&[1, 2, 3, 4]), &unit(99)));
        assert!(!t.insert(5, &key(&[1, 2, 3]), &unit(99)));
        assert!(!t.insert(5, &[item(9, 1, 1), item(9, 7, 2)], &unit(99)));
        assert!(!t.insert(5, &[item(9, 0, 1), item(9, 7, 2)], &unit(99)));
        assert_eq!(find(&t, 5, &[1, 2, 3]), Some(0));
        assert_eq!(find(&t, 5, &[1, 2, 4, 8]), Some(1));
        assert_eq!(find(&t, 5, &[1, 7, 3]), Some(2));
        assert_eq!(find(&t, 5, &[1, 9, 0, 1]), Some(23));
        for v in 0..20 {
            assert_eq!(find(&t, 5, &[1, 9 + v, v, v]), Some(3 + v as u32));
        }
        for miss in [
            &[1, 2, 5][..],
            &[1, 8],
            &[2, 2, 3],
            &[1, 9, 1, 0],
            &[1, 30, 1, 1],
        ] {
            assert_eq!(find(&t, 5, miss), None);
        }
        assert_eq!(find(&t, 6, &[1, 2, 3]), None);
        let (roots, fanout) = t.profile(4);
        assert_eq!(roots, [(5, 24)]);
        assert_eq!(fanout, [(1, 1, 22), (2, 1, 2), (3, 1, 2)]);
        assert!(t.bytes() > 0);
        t.clear();
        assert!(t.insert(5, &[], &unit(0)) && !t.insert(5, &key(&[1]), &unit(1)));
        assert_eq!(find(&t, 5, &[4]), Some(0));
    }

    #[test]
    fn a_cut_is_where_few_runs_of_reads_meet() {
        let mut t = Trie::new();
        let unit = [0; UNIT + 1];
        for a in 0..4 {
            for b in 0..40 {
                let key = [
                    item(1, 0, a),
                    item(1, 1, 7),
                    item(2, 2, b),
                    item(3, 3, b / 2),
                ];
                assert!(t.insert(5, &key, &unit));
            }
        }
        assert_eq!(t.cut_for(5, 8), Some(2));
        assert_eq!(t.cut_for(6, 8), None);
        let mut flat = Trie::new();
        for a in 0..=255 {
            assert!(flat.insert(5, &[item(1, 0, a), item(2, 1, a)], &unit));
        }
        assert_eq!(flat.cut_for(5, 8), None);
    }

    #[test]
    fn seen_keeps_every_hash_through_growth() {
        let (mut s, mut seed) = (Seen::new(), 3);
        let hashes: Vec<u64> = (0..5000).map(|_| splitmix(&mut seed)).collect();
        assert!(hashes.iter().all(|&h| s.insert(h)));
        assert!(hashes.iter().all(|&h| !s.insert(h)));
        assert!(s.insert(0) && !s.insert(1) && !s.insert(0));
        assert_eq!(s.len, 5001);
    }

    pub(super) const PLAY: u16 = 0x1000;
    pub(super) const MAX_CYCLES: u64 = 10_000;
    pub(super) const GRAINS: [Grain; 4] = [Grain::Call, Grain::Sub, Grain::Loop, Grain::Split];

    /// $1000 INC $10 ; LDA $10 ; AND #3 ; STA $10 ; NOP ; INC $11 ; LDA $11 ; CMP #5 ;
    /// BNE + ; LDA #0 ; STA $11 ; + RTS: counters of periods 4 and 5 read at two instructions.
    pub(super) const COUNTERS: &[u8] = &[
        0xE6, 0x10, 0xA5, 0x10, 0x29, 0x03, 0x85, 0x10, 0xEA, 0xE6, 0x11, 0xA5, 0x11, 0xC9, 0x05,
        0xD0, 0x04, 0xA9, 0x00, 0x85, 0x11, 0x60,
    ];

    /// $1000 INC $10 ; RTS: one counter, read by the call's first instruction.
    const COUNTER: &[u8] = &[0xE6, 0x10, 0x60];

    /// $1000 LDX $10 ; LDA $1020,X ; STA $12 ; INX ; TXA ; AND #7 ; STA $10 ; ADC $11 ;
    /// STA $11 ; RTS, then the table at $1020: an index walking a table into a sum that also
    /// reads the carry the call before left.
    pub(super) fn table() -> Vec<u8> {
        let mut code = vec![
            0xA6, 0x10, 0xBD, 0x20, 0x10, 0x85, 0x12, 0xE8, 0x8A, 0x29, 0x07, 0x85, 0x10, 0x65,
            0x11, 0x85, 0x11, 0x60,
        ];
        code.resize(0x20, 0);
        code.extend([3, 1, 4, 1, 5, 9, 2, 6]);
        code
    }

    /// $1000 LDX #2 ; - JSR tick ; DEX ; BPL - ; RTS ; NOP ; tick DEC $10,X ; BPL + ;
    /// LDA $18,X ; STA $10,X ; + RTS: three counters of periods `$18,X` + 1 ticked by a
    /// subroutine.
    pub(super) const VOICES: &[u8] = &[
        0xA2, 0x02, 0x20, 0x0A, 0x10, 0xCA, 0x10, 0xFA, 0x60, 0xEA, 0xD6, 0x10, 0x10, 0x04, 0xB5,
        0x18, 0x95, 0x10, 0x60,
    ];

    /// $1000 LDY $10 ; INY ; - INC $11 ; DEY ; BNE - ; INC $10 ; LDA $10 ; AND #7 ; STA $10 ;
    /// RTS: a backward loop whose trip count is state.
    pub(super) const LOOP: &[u8] = &[
        0xA4, 0x10, 0xC8, 0xE6, 0x11, 0x88, 0xD0, 0xFB, 0xE6, 0x10, 0xA5, 0x10, 0x29, 0x07, 0x85,
        0x10, 0x60,
    ];

    /// RAM holding `code` at [`PLAY`] and the bytes `data`.
    pub(super) fn image(code: &[u8], data: &[(u16, u8)]) -> Box<[u8; 0x10000]> {
        let mut m = Box::new([0u8; 0x10000]);
        m[PLAY as usize..PLAY as usize + code.len()].copy_from_slice(code);
        for &(a, v) in data {
            m[a as usize] = v;
        }
        m
    }

    /// The memory an orbit here takes as its state: page zero, the stack and the code's page.
    pub(super) fn tracked() -> Vec<u32> {
        (0..0x200).chain(PLAY as u32..PLAY as u32 + 0x100).collect()
    }

    /// The brute-force [`crate::orbit::orbit`] of `m` over [`tracked`], the RAM and registers
    /// it leaves.
    fn brute(m: &[u8; 0x10000], limit: usize, max_cycles: u64) -> (Orbit, Box<[u8; 0x10000]>, Cpu) {
        brute_over(m, limit, max_cycles, &[])
    }

    /// [`brute`] with the registers `regs` ([`REG`] offsets) state too.
    pub(super) fn brute_over(
        m: &[u8; 0x10000],
        limit: usize,
        max_cycles: u64,
        regs: &[u32],
    ) -> (Orbit, Box<[u8; 0x10000]>, Cpu) {
        let (mut ram, mut cpu, mut cells) = (Box::new(*m), Cpu::new(), tracked());
        cells.extend(regs.iter().map(|r| REG + r));
        let peek = &mut |b: &mut [u8; 0x10000], a: u16| b[a as usize];
        let digest = |c: &Cpu, b: &mut [u8; 0x10000]| crate::cells_digest(c, b, peek, &cells);
        let none = AddrSet::default();
        let o = crate::orbit::orbit(
            &mut cpu,
            &mut *ram,
            PLAY,
            0,
            max_cycles,
            limit,
            None,
            &none,
            digest,
            || false,
        );
        (o, ram, cpu)
    }

    /// [`memo_orbit_tuned`] of `m` over [`tracked`] after asserting it is the brute-force
    /// orbit and leaves the RAM and registers that leaves.
    fn agree(
        m: &[u8; 0x10000],
        limit: usize,
        max_cycles: u64,
        grain: Grain,
        tuning: Tuning,
    ) -> MemoStats {
        let want = brute(m, limit, max_cycles);
        let (mut ram, mut cpu) = (Box::new(*m), Cpu::new());
        let keep: AddrSet = tracked().into_iter().map(|a| a as u16).collect();
        let none = AddrSet::default();
        let (got, stats) = memo_orbit_tuned::<_, false>(
            &mut cpu,
            &mut *ram,
            PLAY,
            0,
            max_cycles,
            limit,
            None,
            &none,
            &keep,
            &[],
            grain,
            tuning,
            &[],
            || false,
        );
        assert_eq!(got, want.0, "{grain:?} {tuning:?}");
        assert!(ram == want.1, "{grain:?} {tuning:?}: RAM");
        assert_eq!(cpu, want.2, "{grain:?} {tuning:?}");
        stats
    }

    /// $1000 INC $10 ; LDA $10 ; AND #3 ; STA $10 ; EOR #3 ; STA $11 ; RTS: `$10` counts up
    /// as `$11` counts down, so their sum, and every other tracked byte, is the same at each
    /// call's entry after the first.
    const SEESAW: &[u8] = &[
        0xE6, 0x10, 0xA5, 0x10, 0x29, 0x03, 0x85, 0x10, 0x49, 0x03, 0x85, 0x11, 0x60,
    ];

    #[test]
    fn colliding_state_hashes_are_told_apart_by_the_states() {
        let m = image(SEESAW, &[(0x11, 3)]);
        for (limit, end, states) in [(100, End::Closed, 5), (3, End::Open, 4)] {
            let want = brute(&m, limit, MAX_CYCLES);
            assert_eq!((want.0.end, want.0.states), (end, states));
            for grain in GRAINS {
                let (mut ram, mut cpu) = (Box::new(*m), Cpu::new());
                let keep: AddrSet = tracked().into_iter().map(|a| a as u16).collect();
                let none = AddrSet::default();
                let mut memo = Memo::<_, false>::new(&mut *ram, &keep, &[], Tuning::default(), &[]);
                for w in memo.weight.iter_mut().filter(|w| **w != 0) {
                    *w = SEED | 1;
                }
                memo.rehash();
                let (got, stats) = walk(
                    &mut cpu,
                    &mut memo,
                    PLAY,
                    0,
                    MAX_CYCLES,
                    limit,
                    &none,
                    grain,
                    &mut || false,
                );
                assert!(stats.collisions > 0, "{grain:?} {limit}");
                assert_eq!(got, want.0, "{grain:?} {limit}");
                assert!(ram == want.1, "{grain:?} {limit}: RAM");
                assert_eq!(cpu, want.2, "{grain:?} {limit}");
            }
        }
    }

    #[test]
    fn a_start_pc_with_more_units_than_raw_is_emulated_from_then_on() {
        let m = image(COUNTER, &[]);
        let tuning = Tuning {
            crowd: 2,
            raw: 8,
            ..Tuning::default()
        };
        for grain in [Grain::Call, Grain::Sub, Grain::Split] {
            let stats = agree(&m, 40, MAX_CYCLES, grain, tuning);
            assert_eq!((stats.raw, stats.splits), (1, 0), "{grain:?}");
            assert!(stats.misses < 20 && stats.live == 0, "{grain:?}");
        }
        let stats = agree(&m, 40, MAX_CYCLES, Grain::Split, Tuning::default());
        assert_eq!(stats.raw, 0);
        assert!(stats.misses > 30);
    }

    #[test]
    fn crowded_units_are_split_between_independent_counters() {
        let m = image(COUNTERS, &[]);
        let tuning = Tuning {
            crowd: 8,
            raw: 1 << 16,
            saturated: 2,
            ..Tuning::default()
        };
        let split = agree(&m, 100, MAX_CYCLES, Grain::Split, tuning);
        assert!(split.splits >= 1 && split.raw == 0);
        let whole = agree(&m, 100, MAX_CYCLES, Grain::Loop, tuning);
        assert_eq!(whole.splits, 0);
        // Cut between the counters, the halves recur within the period and are replayed.
        assert!(split.hits > whole.hits);
    }

    #[test]
    fn a_unit_replayed_past_the_cycle_limit_is_emulated_and_dropped() {
        // $1000 INC $10 ; JSR $1006 ; RTS ; $1006 RTS: the call's last RTS starts within the
        // limit and ends past it, so its stored unit is emulated again on the same reads.
        let m = image(&[0xE6, 0x10, 0x20, 0x06, 0x10, 0x60, 0x60], &[]);
        let stats = agree(&m, 20, 20, Grain::Sub, Tuning::default());
        assert!(stats.dropped > 0);
        assert_eq!(agree(&m, 20, 30, Grain::Sub, Tuning::default()).dropped, 0);
    }

    #[test]
    fn where_units_are_cut_changes_nothing_found() {
        let tunings = [
            Tuning::default(),
            Tuning {
                crowd: 2,
                raw: 4,
                saturated: 1,
                ..Tuning::default()
            },
            Tuning {
                crowd: 3,
                raw: 1 << 20,
                saturated: 64,
                ..Tuning::default()
            },
        ];
        let routines = [
            image(COUNTERS, &[]),
            image(&table(), &[]),
            image(VOICES, &[(0x18, 2), (0x19, 3), (0x1A, 4)]),
            image(LOOP, &[]),
        ];
        let (mut splits, mut raw) = (0, 0);
        for m in &routines {
            for grain in GRAINS {
                for tuning in tunings {
                    let stats = agree(m, 150, MAX_CYCLES, grain, tuning);
                    (splits, raw) = (splits + stats.splits, raw + stats.raw);
                }
            }
        }
        assert!(splits > 0 && raw > 0);
    }
}
