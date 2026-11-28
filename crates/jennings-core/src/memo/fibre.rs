//! Calls run by a tape of one period of the base, the written memory outside the fibre.
//!
//! A call at the tape's phase is the tape's units again but for its segments, the parts that
//! read the fibre; what a segment does is learned from one ordinary call (`docs/memo.md`).

use super::{regs, set_regs, Memo, MemoStats, Over, Run, Tuning, SP};
use crate::bus::Bus;
use crate::cpu::Cpu;
use crate::run::Stop;
use crate::taint::{FxMap, REG};

/// A and the flags.
const VALUES: u16 = 0x6F01;
/// Every register.
const REGS: u16 = 0x6F0F;

/// A unit a recorded call ran.
#[derive(Clone, Default)]
struct Ev {
    pc: u16,
    regs: [u8; 16],
    /// Registers read before written, registers written.
    rmask: u16,
    wmask: u16,
    /// Its key reads a fibre cell.
    fread: bool,
    /// The memory its key reads, in [`Notes::evr`], and its writes, in [`Notes::evw`].
    reads: (u32, u32),
    writes: (u32, u32),
    cycles: u64,
    steps: u64,
    done: bool,
}

/// What the memo notes for a tape: the units of a call being recorded, or what the part of
/// a call being learned reads and writes.
pub(super) struct Notes {
    /// The tracked cells set apart.
    pub(super) is_f: Box<[bool]>,
    pub(super) rec: bool,
    pub(super) logging: bool,
    /// An instruction of the call being recorded ran outside any unit.
    pub(super) broken: bool,
    /// A cell was written for the first time.
    pub(super) dead: bool,
    evs: Vec<Ev>,
    evr: Vec<u16>,
    evw: Vec<u32>,
    /// What the part being learned did: the memory read, each write's address and the
    /// byte it replaced, the registers read before written and the registers written.
    pub(super) rlog: Vec<u16>,
    pub(super) wlog: Vec<(u16, u8)>,
    pub(super) rmask: u16,
    pub(super) wmask: u16,
}

impl Notes {
    pub(super) fn new(cells: &[u16]) -> Self {
        let mut is_f = vec![false; 0x10000].into_boxed_slice();
        for &c in cells {
            is_f[c as usize] = true;
        }
        Notes {
            is_f,
            rec: false,
            logging: false,
            broken: false,
            dead: false,
            evs: Vec::new(),
            evr: Vec::new(),
            evw: Vec::new(),
            rlog: Vec::new(),
            wlog: Vec::new(),
            rmask: 0,
            wmask: 0,
        }
    }

    /// Start or stop noting what a part being learned does.
    fn log(&mut self, on: bool) {
        self.logging = on;
        if on {
            self.rlog.clear();
            self.wlog.clear();
            (self.rmask, self.wmask) = (0, 0);
        }
    }

    /// The part being learned runs something that reads the registers `reads` and then
    /// writes the registers `writes`.
    #[inline(always)]
    pub(super) fn uses(&mut self, reads: u16, writes: u16) {
        self.rmask |= reads & !self.wmask;
        self.wmask |= writes;
    }

    fn clear(&mut self) {
        self.evs.clear();
        self.evr.clear();
        self.evw.clear();
        self.broken = false;
    }

    /// A unit starts at `pc` with the registers `regs`.
    pub(super) fn begin(&mut self, pc: u16, regs: [u8; 16]) {
        let at = (self.evr.len() as u32, self.evw.len() as u32);
        self.evs.push(Ev {
            pc,
            regs,
            reads: (at.0, at.0),
            writes: (at.1, at.1),
            ..Ev::default()
        });
    }

    /// The call's start from `sp`, as a unit: the return address pushed and A set.
    pub(super) fn entry(&mut self, sp: u8) {
        self.begin(0, [0; 16]);
        let low = sp.wrapping_sub(1);
        self.write(&[(0x100 | sp as u32) << 8, (0x100 | low as u32) << 8 | 1]);
        self.evs.last_mut().expect("begun").rmask = SP;
        self.end(SP | 1, 0, 0);
    }

    /// The unit begun, or the part being learned, is keyed on `loc`.
    #[inline]
    pub(super) fn read(&mut self, loc: u32) {
        if self.logging {
            match loc < REG {
                true => self.rlog.push(loc as u16),
                false => self.uses(1 << (loc - REG), 0),
            }
            return;
        }
        let Some(ev) = self.evs.last_mut() else {
            return;
        };
        if loc >= REG {
            ev.rmask |= 1 << (loc - REG);
        } else {
            ev.fread |= self.is_f[loc as usize];
            self.evr.push(loc as u16);
        }
    }

    /// The unit begun writes `writes` (`addr << 8 | byte`).
    pub(super) fn write(&mut self, writes: &[u32]) {
        self.evw.extend(writes);
    }

    /// The unit begun wrote the registers `wmask` and took `cycles` and `steps`.
    pub(super) fn end(&mut self, wmask: u16, cycles: u64, steps: u64) {
        let Some(ev) = self.evs.last_mut() else {
            self.broken = true;
            return;
        };
        (ev.reads.1, ev.writes.1) = (self.evr.len() as u32, self.evw.len() as u32);
        (ev.wmask, ev.cycles, ev.steps, ev.done) = (wmask, cycles, steps, true);
    }
}

/// The registers whose bytes are not the tape's: their bytes and the tape's.
#[derive(Clone, Copy, Default, Debug)]
struct Stale {
    mask: u16,
    now: [u8; 16],
    was: [u8; 16],
}

impl Stale {
    /// The registers of `now` that differ from `was`.
    fn between(now: &[u8; 16], was: &[u8; 16]) -> Self {
        let mut s = Stale::default();
        for r in (0..16).filter(|&r| REGS >> r & 1 != 0 && now[r] != was[r]) {
            s.mask |= 1 << r;
            (s.now[r], s.was[r]) = (now[r], was[r]);
        }
        s
    }

    /// Whether the same registers are stale with the same bytes.
    fn same(&self, other: &Stale) -> bool {
        let regs = (0..16).filter(|&r| self.mask >> r & 1 != 0);
        self.mask == other.mask && regs.into_iter().all(|r| self.now[r] == other.now[r])
    }
}

/// Interned lists of words; list 0 is empty.
struct Lists {
    data: Vec<u32>,
    at: Vec<u32>,
    ids: FxMap<Box<[u32]>, u32>,
}

impl Lists {
    fn new() -> Self {
        let mut lists = Lists {
            data: Vec::new(),
            at: vec![0],
            ids: FxMap::default(),
        };
        lists.intern(&[]);
        lists
    }

    fn intern(&mut self, items: &[u32]) -> u32 {
        if let Some(&id) = self.ids.get(items) {
            return id;
        }
        let id = self.at.len() as u32 - 1;
        self.data.extend(items);
        self.at.push(self.data.len() as u32);
        self.ids.insert(items.into(), id);
        id
    }

    #[inline(always)]
    fn get(&self, id: u32) -> &[u32] {
        &self.data[self.at[id as usize] as usize..self.at[id as usize + 1] as usize]
    }

    fn bytes(&self) -> usize {
        let held: usize = self.ids.keys().map(|k| k.len() * 4 + 24).sum();
        self.data.capacity() * 4 + self.at.capacity() * 4 + held
    }
}

/// The base cells a segment has been seen to read or write.
#[derive(Default)]
struct Near {
    cells: Vec<u16>,
    has: Vec<bool>,
}

impl Near {
    /// Add `a`; whether it was not there.
    fn add(&mut self, a: u16) -> bool {
        if self.has.is_empty() {
            self.has = vec![false; 0x10000];
        }
        let new = !std::mem::replace(&mut self.has[a as usize], true);
        if new {
            self.cells.push(a);
        }
        new
    }
}

/// Where a recorded call's segment starts and ends and what the tape's units do in it.
#[derive(Clone, PartialEq, Eq, Hash)]
struct Shape {
    /// The unit it starts at, as [`Run::go`] counts from the boundary before.
    start: (u16, u32),
    /// The unit after it, counted from its start; None where it ends with the call, at
    /// the PC `last`.
    end: Option<(u16, u32)>,
    last: u16,
    /// The base cells the tape's units wrote in it, as a list of `addr << 8 | byte`.
    over: u32,
}

/// The tape's registers either side of a recorded call's segment.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct Ends {
    regs: [u8; 16],
    end_regs: [u8; 16],
    /// The registers that must be the tape's at its end.
    must: u16,
}

/// A segment of a recorded call.
#[derive(Clone, Copy)]
struct Seg {
    /// The fibre writes of the units between the boundary before and it, as a list, and
    /// the registers those units leave as they are.
    pre: u32,
    kept: u16,
    /// Its [`Shape`] and the bytes of the near cells at its start, numbered, and its
    /// [`Ends`], numbered.
    ctx: u32,
    ends: u32,
}

/// A recorded call.
struct Call {
    /// Where its segments end in [`Tape::segs`].
    segs: u32,
    /// The fibre writes of the units after its last segment, as a list, and the registers
    /// those units leave as they are.
    post: u32,
    kept: u16,
    /// Cycles and instructions outside its segments.
    cycles: u32,
    steps: u32,
}

/// What a segment does from the registers and fibre cells it reads.
struct Done {
    /// Its writes in [`Tape::fw`], from the first: this many of the fibre, then this many
    /// of cells outside the tracked state.
    writes: (u32, u16, u16),
    /// The registers it writes and their bytes at its end.
    wmask: u16,
    regs: [u8; 16],
    cycles: u32,
    steps: u32,
}

/// What segments did, by key: a key under 16 bytes is held in place.
#[derive(Default)]
struct Known {
    short: FxMap<[u8; 16], Done>,
    long: FxMap<Box<[u8]>, Done>,
}

impl Known {
    /// `key` and its length, where it is under 16 bytes.
    #[inline(always)]
    fn short(key: &[u8]) -> Option<[u8; 16]> {
        let mut packed = [0; 16];
        packed.get_mut(..key.len())?.copy_from_slice(key);
        packed[15] = u8::try_from(key.len()).ok().filter(|&n| n < 16)?;
        Some(packed)
    }

    #[inline(always)]
    fn get(&self, key: &[u8]) -> Option<&Done> {
        match Known::short(key) {
            Some(packed) => self.short.get(&packed),
            None => self.long.get(key),
        }
    }

    fn insert(&mut self, key: &[u8], done: Done) {
        match Known::short(key) {
            Some(packed) => self.short.insert(packed, done),
            None => self.long.insert(key.into(), done),
        };
    }

    fn len(&self) -> usize {
        self.short.len() + self.long.len()
    }

    fn bytes(&self) -> usize {
        let long: usize = self.long.keys().map(|k| k.len() + 16).sum();
        self.short.capacity() * std::mem::size_of::<([u8; 16], Done)>()
            + self.long.capacity() * std::mem::size_of::<(Box<[u8]>, Done)>()
            + long
    }
}

/// What recording a call came to.
enum Joined {
    Yes,
    No,
    /// To be recorded again from the next call, with these registers taken as live at a
    /// call's entry.
    Again(u16),
}

/// The calls of one period of the base.
struct Tape {
    period: usize,
    /// The calls between frames.
    every: usize,
    /// The base cells (written, tracked, outside the fibre) and the written cells outside
    /// the tracked state; per address, which it is ([`BASE`], [`OUT`]).
    w: Vec<u16>,
    out: Vec<u16>,
    kind: Vec<u8>,
    /// The registers taken as live at every call's entry.
    guess: u16,
    calls: Vec<Call>,
    /// Per call the hash of the state it was recorded from, less the fibre's part.
    hb: Vec<u64>,
    segs: Vec<Seg>,
    lists: Lists,
    shapes: Vec<Shape>,
    shape_ids: FxMap<Shape, u32>,
    ends: Vec<Ends>,
    ends_ids: FxMap<Ends, u32>,
    /// Per context its shape, the fibre cells its segments have read or written and the
    /// registers they have read before writing.
    ctx_shape: Vec<u32>,
    uses: Vec<Vec<u16>>,
    reads: Vec<u16>,
    ctx_ids: FxMap<Box<[u8]>, u32>,
    /// What segments did, by context and the bytes of those registers and fibre cells.
    done: Known,
    fw: Vec<u32>,
    /// Per frame the base cells, the registers and the registers live at its call's entry.
    frames: Vec<u8>,
    frame_cpu: Vec<Cpu>,
    frame_live: Vec<u16>,
    /// Per frame and cell outside the tracked state, the last write to it before the
    /// frame's call by a unit outside a segment: when ([`when`], 0 for none) and the byte.
    frame_when: Vec<u32>,
    frame_byte: Vec<u8>,
    /// The base cells while recording, by address, and likewise those last writes.
    shadow: Vec<u8>,
    when: Vec<u32>,
    byte: Vec<u8>,
    /// The registers the last call leaves other than the first call's.
    wrap: Stale,
}

const BASE: u8 = 1;
const OUT: u8 = 2;
/// The segments a recorded call has at most, and the calls a tape has.
const SEGS: usize = 127;
const SPAN: usize = 1 << 22;

/// When in a tape a write is made: by the call at `phase`, by the units before its
/// segment `gap` (even `2 * gap`) or by segment `gap` (odd), later being greater.
fn when(phase: usize, gap: usize, segment: bool) -> u32 {
    (phase as u32 + 1) << 8 | (2 * gap as u32 + segment as u32)
}

/// Put `w` (`addr << 8 | byte`) in `list` in place of a write to its address.
fn put(list: &mut Vec<u32>, w: u32) {
    match list.iter_mut().find(|l| **l >> 8 == w >> 8) {
        Some(l) => *l = w,
        None => list.push(w),
    }
}

impl Tape {
    /// A tape of `period` calls whose frames hold `budget` bytes at most.
    fn new(period: usize, w: Vec<u16>, out: Vec<u16>, guess: u16, budget: usize) -> Self {
        let mut kind = vec![0; 0x10000];
        for (cells, k) in [(&w, BASE), (&out, OUT)] {
            for &a in cells {
                kind[a as usize] = k;
            }
        }
        let frame = w.len() + 5 * out.len() + std::mem::size_of::<Cpu>() + 2;
        let every = (period * frame).div_ceil(budget.max(1)).max(1);
        let (frames, cells, outs) = (period.div_ceil(every), w.len(), out.len());
        Tape {
            period,
            every,
            w,
            out,
            kind,
            guess,
            calls: Vec::with_capacity(period),
            hb: Vec::with_capacity(period),
            segs: Vec::new(),
            lists: Lists::new(),
            shapes: Vec::new(),
            shape_ids: FxMap::default(),
            ends: Vec::new(),
            ends_ids: FxMap::default(),
            ctx_shape: Vec::new(),
            uses: Vec::new(),
            reads: Vec::new(),
            ctx_ids: FxMap::default(),
            done: Known::default(),
            fw: Vec::new(),
            frames: Vec::with_capacity(frames * cells),
            frame_cpu: Vec::with_capacity(frames),
            frame_live: Vec::with_capacity(frames),
            frame_when: Vec::with_capacity(frames * outs),
            frame_byte: Vec::with_capacity(frames * outs),
            shadow: vec![0; 0x10000],
            when: vec![0; 0x10000],
            byte: vec![0; 0x10000],
            wrap: Stale::default(),
        }
    }

    fn bytes(&self) -> usize {
        let size = std::mem::size_of::<Cpu>();
        let contexts: usize = self.ctx_ids.keys().map(|k| k.len() + 16).sum();
        self.calls.capacity() * std::mem::size_of::<Call>()
            + self.hb.capacity() * 8
            + self.segs.capacity() * std::mem::size_of::<Seg>()
            + self.lists.bytes()
            + self.shapes.capacity() * 2 * std::mem::size_of::<Shape>()
            + self.ends.capacity() * 2 * std::mem::size_of::<Ends>()
            + self.done.bytes()
            + contexts
            + self.ctx_shape.capacity() * 34
            + self.fw.capacity() * 4
            + self.frames.capacity()
            + self.frame_cpu.capacity() * (size + 2)
            + self.frame_when.capacity() * 4
            + self.frame_byte.capacity()
            + (self.w.capacity() + self.out.capacity()) * 2
            + self.kind.capacity()
            + self.shadow.capacity()
            + self.when.capacity() * 4
            + self.byte.capacity()
    }

    /// The segments of the call at `phase`, in `segs`.
    #[inline(always)]
    fn segs_of(&self, phase: usize) -> std::ops::Range<usize> {
        let from = phase.checked_sub(1).map_or(0, |p| self.calls[p].segs);
        from as usize..self.calls[phase].segs as usize
    }

    #[inline(always)]
    fn shape(&self, ctx: u32) -> &Shape {
        &self.shapes[self.ctx_shape[ctx as usize] as usize]
    }

    /// The number of the segments of `shape` whose near cells are as the shadow has them.
    fn context(&mut self, shape: Shape, near: &Near) -> u32 {
        let next = self.shapes.len() as u32;
        let id = *self.shape_ids.entry(shape.clone()).or_insert(next);
        if id == next {
            self.shapes.push(shape);
        }
        let mut key = id.to_le_bytes().to_vec();
        key.extend(near.cells.iter().map(|&a| self.shadow[a as usize]));
        let next = self.ctx_shape.len() as u32;
        let ctx = *self.ctx_ids.entry(key.into()).or_insert(next);
        if ctx == next {
            self.ctx_shape.push(id);
            self.uses.push(Vec::new());
            self.reads.push(0);
        }
        ctx
    }

    /// The number of `ends`.
    fn ends(&mut self, ends: Ends) -> u32 {
        let next = self.ends.len() as u32;
        let id = *self.ends_ids.entry(ends).or_insert(next);
        if id == next {
            self.ends.push(ends);
        }
        id
    }

    /// Add the call whose units are `notes`', left with the registers and PC `end` after
    /// `cycles` and `steps`. A segment runs from a unit reading the fibre to the end of the last
    /// unit after it that reads the fibre or starts with A or a flag live that it wrote.
    fn finish(
        &mut self,
        notes: &Notes,
        near: &mut Near,
        end: ([u8; 16], u16),
        cycles: u64,
        steps: u64,
    ) -> Joined {
        let (evs, n) = (&notes.evs, notes.evs.len());
        let outside = notes.evr.iter().any(|&a| self.kind[a as usize] == OUT);
        if notes.broken || notes.dead || outside || n < 2 || evs.iter().any(|e| !e.done) {
            return Joined::No;
        }
        let (phase, first) = (self.calls.len(), self.segs.len());
        let mut live = vec![self.guess; n + 1];
        for k in (0..n).rev() {
            live[k] = evs[k].rmask | (live[k + 1] & !evs[k].wmask);
        }
        if live[0] & !self.guess != 0 {
            return Joined::Again(self.guess | live[0]);
        }
        let writes = |ev: &Ev| &notes.evw[ev.writes.0 as usize..ev.writes.1 as usize];
        let count = |from: usize, to: usize| {
            let at = evs[from..=to].iter().filter(|v| v.pc == evs[to].pc);
            at.count() as u32
        };
        let (mut cycles, mut steps) = (cycles, steps);
        let (mut k, mut bound, mut wrote, mut pre) = (0, 1, 0u16, Vec::new());
        while k < n {
            let ev = &evs[k];
            if !ev.fread {
                for &w in writes(ev) {
                    let a = (w >> 8) as usize;
                    match self.kind[a] {
                        BASE => self.shadow[a] = w as u8,
                        OUT => {
                            let gap = self.segs.len() - first;
                            (self.when[a], self.byte[a]) = (when(phase, gap, false), w as u8);
                        }
                        _ => put(&mut pre, w),
                    }
                }
                wrote |= ev.wmask;
                k += 1;
                continue;
            }
            let (i, mut e, mut values) = (k, k + 1, ev.wmask & VALUES);
            while e < n && (evs[e].fread || values & live[e] != 0) {
                values |= evs[e].wmask & VALUES;
                e += 1;
            }
            let (mut over, mut grew) = (Vec::new(), false);
            for ev in &evs[i..e] {
                for &a in &notes.evr[ev.reads.0 as usize..ev.reads.1 as usize] {
                    grew |= self.kind[a as usize] == BASE && near.add(a);
                }
                for &w in writes(ev)
                    .iter()
                    .filter(|&&w| self.kind[(w >> 8) as usize] == BASE)
                {
                    grew |= near.add((w >> 8) as u16);
                    put(&mut over, w);
                }
                cycles -= ev.cycles;
                steps -= ev.steps;
            }
            if grew {
                return Joined::Again(self.guess);
            }
            over.sort_unstable();
            let shape = Shape {
                start: (ev.pc, if i > bound { count(bound + 1, i) } else { 0 }),
                end: (e < n).then(|| (evs[e].pc, count(i + 1, e))),
                last: if e < n { 0 } else { end.1 },
                over: self.lists.intern(&over),
            };
            let ends = Ends {
                regs: ev.regs,
                end_regs: if e < n { evs[e].regs } else { end.0 },
                must: live[e],
            };
            let seg = Seg {
                pre: self.lists.intern(&pre),
                kept: !wrote,
                ctx: self.context(shape, near),
                ends: self.ends(ends),
            };
            self.segs.push(seg);
            for &w in &over {
                self.shadow[(w >> 8) as usize] = w as u8;
            }
            pre.clear();
            (k, bound, wrote) = (e, e, 0);
        }
        let (Ok(cycles), Ok(steps)) = (u32::try_from(cycles), u32::try_from(steps)) else {
            return Joined::No;
        };
        if self.segs.len() - first > SEGS {
            return Joined::No;
        }
        if self.calls.len().is_multiple_of(self.every) {
            self.frame_live.push(live[0]);
        }
        self.calls.push(Call {
            segs: self.segs.len() as u32,
            post: self.lists.intern(&pre),
            kept: !wrote,
            cycles,
            steps,
        });
        Joined::Yes
    }
}

/// How calls are run with a fibre.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub(super) enum Mode {
    /// As without one.
    #[default]
    Off,
    /// As without one, until this call.
    Idle(usize),
    /// As without one, looking for the base's entry state to repeat.
    Detect,
    /// As without one, recording this many more calls of the tape.
    Record(usize),
    /// By the tape.
    Tape,
    /// As without one, to go on by the tape if the base is the tape's at its next frame.
    Resync,
}

/// A call whose entry state can be put back: its registers and fibre, the base cells being
/// those in memory or those of frame `at`.
#[derive(Default)]
struct Anchor {
    k: usize,
    at: usize,
    cpu: Option<Cpu>,
    cells: Vec<u8>,
}

/// The key of what a segment of context `ctx` does from the bytes `regs` of the registers
/// `reads` and the bytes `value` gives of the fibre cells `uses`.
fn key(
    buf: &mut Vec<u8>,
    ctx: u32,
    (reads, regs): (u16, &[u8; 16]),
    uses: &[u16],
    value: impl FnMut(&u16) -> u8,
) {
    buf.clear();
    buf.extend(ctx.to_le_bytes());
    buf.extend(reads.to_le_bytes());
    buf.extend((uses.len() as u16).to_le_bytes());
    buf.extend((0..16).filter(|&r| reads >> r & 1 != 0).map(|r| regs[r]));
    buf.extend(uses.iter().map(value));
}

/// How a [`Run`]'s calls are run with a fibre, and its tape.
#[derive(Default)]
pub(super) struct Fibre {
    mode: Mode,
    tape: Option<Tape>,
    tuning: Tuning,
    /// The written fibre cells, and their part of the hash of the tracked state.
    cells: Vec<u16>,
    hash: u64,
    near: Near,
    /// The calls the base's entry states were last seen at, by their hash.
    seen: FxMap<u64, usize>,
    /// The period and live registers to record a tape with from the next call.
    pending: Option<(usize, u16)>,
    /// The call about to be run, and its phase in the tape.
    k: usize,
    phase: usize,
    stale: Stale,
    /// The fibre writes of the call being run by the tape, to be undone.
    undo: Vec<(u16, u8)>,
    key: Vec<u8>,
    /// The call the base cells in memory are the entry state of, and the last call run by
    /// the tape at a frame.
    live: Anchor,
    frame: Anchor,
    /// Tapes given up, and the calls this tape ran without and with learning.
    fails: u32,
    fast: u64,
    slow: u64,
    /// Per address the stamp of the part last learned that wrote it and the byte replaced.
    stamp: Vec<u32>,
    old: Vec<u8>,
    tick: u32,
    /// Per address outside the tracked state, the call and segment (`call << 8 | 2 *
    /// segment + 1`) a segment run by the tape last wrote it at; those of the call being
    /// run, to be undone.
    wrote: Vec<u64>,
    unwrite: Vec<(u16, u64)>,
}

impl Fibre {
    pub(super) fn new(on: bool, tuning: Tuning) -> Self {
        Fibre {
            mode: if on { Mode::Detect } else { Mode::Off },
            tuning,
            stamp: vec![0; if on { 0x10000 } else { 0 }],
            old: vec![0; if on { 0x10000 } else { 0 }],
            wrote: vec![0; if on { 0x10000 } else { 0 }],
            ..Fibre::default()
        }
    }

    #[inline(always)]
    pub(super) fn taped(&self) -> bool {
        self.mode == Mode::Tape
    }

    /// Run calls as without a fibre until [`Fibre::resume`].
    pub(super) fn pause(&mut self) -> Mode {
        std::mem::replace(&mut self.mode, Mode::Off)
    }

    pub(super) fn resume(&mut self, mode: Mode) {
        self.mode = mode;
    }

    /// The tape's next call is the one about to be run.
    fn advance(&mut self, stats: &mut MemoStats) {
        let tape = self.tape.as_ref().expect("taped");
        self.phase += 1;
        if self.phase < tape.period {
            return;
        }
        self.phase = 0;
        stats.tape_passes += 1;
        for r in (0..16).filter(|&r| tape.wrap.mask >> r & 1 != 0) {
            let stale = self.stale.mask >> r & 1 != 0;
            let now = if stale {
                self.stale.now[r]
            } else {
                tape.wrap.now[r]
            };
            (self.stale.now[r], self.stale.was[r]) = (now, tape.wrap.was[r]);
            self.stale.mask &= !(1 << r);
            self.stale.mask |= ((now != tape.wrap.was[r]) as u16) << r;
        }
    }

    /// Whether the part just learned left the base as the tape's units of `shape` do:
    /// each base cell it wrote and each they wrote holds what they leave there.
    fn rejoins<B: Bus, const T: bool>(&mut self, memo: &mut Memo<'_, B, T>, shape: &Shape) -> bool {
        let over = self.tape.as_ref().expect("taped").lists.get(shape.over);
        self.tick = self.tick.wrapping_add(1);
        if self.tick == 0 {
            self.stamp.fill(0);
            self.tick = 1;
        }
        let kind = &self.tape.as_ref().expect("taped").kind;
        let mut same = true;
        for &(addr, old) in &memo.fib.wlog {
            let a = addr as usize;
            if self.stamp[a] == self.tick {
                continue;
            }
            (self.stamp[a], self.old[a]) = (self.tick, old);
            if kind[a] == BASE {
                let theirs = over.binary_search_by_key(&(a as u32), |w| w >> 8);
                let want = theirs.map_or(old, |i| over[i] as u8);
                same &= memo.bus.peek(addr) == want;
            }
        }
        let held = |w: &u32| memo.bus.peek((w >> 8) as u16) == *w as u8;
        same && over.iter().all(held)
    }

    /// Take the base cells the part just learned read or wrote as near; whether any is new.
    fn widen<B: Bus, const T: bool>(&mut self, memo: &Memo<'_, B, T>) -> bool {
        let tape = self.tape.as_ref().expect("taped");
        let touched = memo
            .fib
            .rlog
            .iter()
            .chain(memo.fib.wlog.iter().map(|w| &w.0));
        let mut grew = false;
        for &a in touched {
            grew |= tape.kind[a as usize] == BASE && self.near.add(a);
        }
        grew
    }

    /// Hold what the part just learned did as what a segment of `ctx` does from the
    /// registers `before`: it took `cycles` and `steps` and left the registers `after`.
    fn keep<B: Bus, const T: bool>(
        &mut self,
        memo: &mut Memo<'_, B, T>,
        ctx: u32,
        before: &[u8; 16],
        after: &[u8; 16],
        (cycles, steps): (u64, u64),
    ) {
        let tape = self.tape.as_mut().expect("taped");
        if memo.fib.rlog.iter().any(|&a| tape.kind[a as usize] == OUT) {
            return;
        }
        let touched = memo
            .fib
            .rlog
            .iter()
            .chain(memo.fib.wlog.iter().map(|w| &w.0));
        let fibre = touched.filter(|&&a| memo.fib.is_f[a as usize] && memo.ever[a as usize]);
        let mut used: Vec<u16> = fibre.copied().collect();
        used.sort_unstable();
        used.dedup();
        let uses = &mut tape.uses[ctx as usize];
        if used.iter().any(|a| !uses.contains(a)) {
            uses.extend(&used);
            uses.sort_unstable();
            uses.dedup();
        }
        let wrote = |a: u16| self.stamp[a as usize] == self.tick;
        let first = |&a: &u16| match wrote(a) {
            true => self.old[a as usize],
            false => memo.bus.peek(a),
        };
        tape.reads[ctx as usize] |= memo.fib.rmask;
        key(
            &mut self.key,
            ctx,
            (tape.reads[ctx as usize], before),
            uses,
            first,
        );
        let from = tape.fw.len() as u32;
        for &a in used.iter().filter(|&&a| wrote(a)) {
            tape.fw.push((a as u32) << 8 | memo.bus.peek(a) as u32);
        }
        let mid = tape.fw.len() as u32;
        for &a in tape.out.iter().filter(|&&a| wrote(a)) {
            tape.fw.push((a as u32) << 8 | memo.bus.peek(a) as u32);
        }
        let mut regs = [0; 16];
        for r in (0..16).filter(|&r| memo.fib.wmask >> r & 1 != 0) {
            regs[r] = after[r];
        }
        let unwritten = (0..16).filter(|&r| REGS & !memo.fib.wmask >> r & 1 != 0);
        debug_assert!(unwritten.into_iter().all(|r| before[r] == after[r]));
        let counts = (mid - from, tape.fw.len() as u32 - mid);
        let sized = (u16::try_from(counts.0), u16::try_from(counts.1));
        let took = (u32::try_from(cycles), u32::try_from(steps));
        let ((Ok(fibre), Ok(out)), (Ok(cycles), Ok(steps))) = (sized, took) else {
            tape.fw.truncate(from as usize);
            return;
        };
        let done = Done {
            writes: (from, fibre, out),
            wmask: memo.fib.wmask,
            regs,
            cycles,
            steps,
        };
        if let Some(held) = tape.done.get(&self.key) {
            let theirs = held.writes.0 as usize..;
            debug_assert!(
                tape.fw[theirs][..tape.fw.len() - from as usize] == tape.fw[from as usize..]
            );
            debug_assert!((held.wmask, held.regs) == (done.wmask, done.regs));
            debug_assert!((held.cycles, held.steps) == (cycles, steps));
            tape.fw.truncate(from as usize);
        } else if tape.done.len() < self.tuning.known {
            tape.done.insert(&self.key, done);
        } else {
            tape.fw.truncate(from as usize);
        }
    }
}

impl<'a, I: Bus, S: FnMut() -> bool> Run<'_, '_, I, S> {
    /// The hash of the state the call about to be run by the tape starts from.
    pub(super) fn tape_hash<B: Bus + 'a, const T: bool>(&mut self) -> u64
    where
        I: Over<'a, B, T>,
    {
        let (memo, fx) = (self.tap.inner.memo(), &self.fx);
        let tape = fx.tape.as_ref().expect("taped");
        let stale = memo
            .reg_cells
            .iter()
            .filter(|c| fx.stale.mask >> c.0 & 1 != 0);
        let h = tape.hb[fx.phase].wrapping_add(fx.hash);
        stale.fold(h, |h, &(r, w)| {
            let (now, was) = (fx.stale.now[r as usize], fx.stale.was[r as usize]);
            h.wrapping_add(w.wrapping_mul((now as u64).wrapping_sub(was as u64)))
        })
    }

    /// Give the tape, or looking for one, up for a number of calls from call `k` that
    /// doubles each time. The memory and registers must be those call `k` starts from.
    fn lose<B: Bus + 'a, const T: bool>(&mut self, k: usize)
    where
        I: Over<'a, B, T>,
    {
        let (memo, fx) = (self.tap.inner.memo(), &mut self.fx);
        memo.fib.rec = false;
        memo.fib.clear();
        let period = fx.tape.take().map_or(0, |tape| {
            memo.stats.tape_bytes = memo.stats.tape_bytes.max(tape.bytes() as u64);
            tape.period
        });
        let wait = period.max(fx.tuning.rest) << fx.fails.min(24);
        fx.mode = Mode::Idle(k.saturating_add(wait));
        fx.fails += 1;
        (fx.pending, fx.seen) = (None, FxMap::default());
        memo.stats.tape_lost += 1;
    }

    /// Start recording a tape of `period` calls from the call about to be run.
    fn start<B: Bus + 'a, const T: bool>(&mut self, period: usize, guess: u16)
    where
        I: Over<'a, B, T>,
    {
        let (memo, fx) = (self.tap.inner.memo(), &mut self.fx);
        let written = |a: &u16| memo.ever[*a as usize] && !memo.fib.is_f[*a as usize];
        let (w, out): (Vec<u16>, Vec<u16>) = (0..=0xFFFF)
            .filter(written)
            .partition(|&a| memo.weight[a as usize] != 0);
        let mut tape = Tape::new(period, w, out, guess, fx.tuning.frames);
        for &a in &tape.w {
            tape.shadow[a as usize] = memo.bus.peek(a);
        }
        if let Some(old) = fx.tape.replace(tape) {
            memo.stats.tape_bytes = memo.stats.tape_bytes.max(old.bytes() as u64);
        }
        fx.mode = Mode::Record(period);
        memo.fib.clear();
        memo.fib.rec = true;
        memo.stats.tape_period = period as u64;
    }

    /// Note what the call about to be recorded starts from.
    fn capture<B: Bus + 'a, const T: bool>(&mut self, cpu: &Cpu)
    where
        I: Over<'a, B, T>,
    {
        let (memo, fx) = (self.tap.inner.memo(), &mut self.fx);
        let tape = fx.tape.as_mut().expect("recording");
        let h = memo
            .state_hash(cpu)
            .wrapping_sub(memo.cells_hash(&fx.cells));
        if tape.hb.len() % tape.every == 0 {
            tape.frames.extend(tape.w.iter().map(|&a| memo.bus.peek(a)));
            tape.frame_cpu.push(cpu.clone());
            tape.frame_when
                .extend(tape.out.iter().map(|&a| tape.when[a as usize]));
            tape.frame_byte
                .extend(tape.out.iter().map(|&a| tape.byte[a as usize]));
        }
        tape.hb.push(h);
        let shadowed = |&a: &u16| tape.shadow[a as usize] == memo.bus.peek(a);
        debug_assert!(tape.w.iter().all(shadowed));
    }

    /// Whether the base cells are those of the tape's frame `at`, and the registers that
    /// are not the frame's: none of them live there.
    fn framed<B: Bus + 'a, const T: bool>(&mut self, cpu: &Cpu, at: usize) -> Option<Stale>
    where
        I: Over<'a, B, T>,
    {
        let (memo, tape) = (self.tap.inner.memo(), self.fx.tape.as_ref().expect("taped"));
        let n = tape.w.len();
        let frame = tape.w.iter().zip(&tape.frames[at * n..][..n]);
        let stale = Stale::between(&regs(cpu), &regs(&tape.frame_cpu[at]));
        let same = stale.mask & tape.frame_live[at] == 0;
        (same && frame.into_iter().all(|(&a, &v)| memo.bus.peek(a) == v)).then_some(stale)
    }

    /// Run calls by the tape from call `k`, the one about to be run, which starts from the
    /// tape's frame `at` but for the registers `stale`.
    fn join<B: Bus + 'a, const T: bool>(&mut self, cpu: &Cpu, k: usize, at: usize, stale: Stale)
    where
        I: Over<'a, B, T>,
    {
        let (memo, fx) = (self.tap.inner.memo(), &mut self.fx);
        (fx.mode, fx.stale, fx.hash) = (Mode::Tape, stale, memo.cells_hash(&fx.cells));
        let cells = fx.cells.iter().map(|&a| memo.bus.peek(a)).collect();
        fx.live = Anchor {
            k,
            at,
            cpu: Some(cpu.clone()),
            cells,
        };
        fx.frame = Anchor::default();
    }

    /// What the call about to be run as call `k` from `cpu` needs done first; how the
    /// calls run again to do it stopped, if they did not all return.
    pub(super) fn enter<B: Bus + 'a, const T: bool>(
        &mut self,
        cpu: &mut Cpu,
        k: usize,
    ) -> Option<Stop>
    where
        I: Over<'a, B, T>,
    {
        if self.fx.mode == Mode::Off {
            return None;
        }
        self.fx.k = k;
        if self.tap.inner.memo().fib.dead {
            if self.fx.taped() {
                if let Some(stop) = self.settle(cpu, k) {
                    return Some(stop);
                }
            }
            if self.fx.tape.is_some() {
                self.lose(k);
            }
            let (memo, fx) = (self.tap.inner.memo(), &mut self.fx);
            memo.fib.dead = false;
            fx.seen.clear();
            let written = |&a: &u16| memo.fib.is_f[a as usize] && memo.ever[a as usize];
            fx.cells = (0..=0xFFFF).filter(written).collect();
        }
        let (memo, fx) = (self.tap.inner.memo(), &mut self.fx);
        if matches!(fx.mode, Mode::Idle(until) if k >= until) {
            fx.mode = Mode::Detect;
        }
        if fx.mode == Mode::Detect {
            let base = memo.hash.wrapping_sub(memo.cells_hash(&fx.cells));
            if let Some(before) = fx.seen.insert(base, k) {
                (fx.pending, fx.seen) = (Some((k - before, SP)), FxMap::default());
            } else if fx.seen.len() > fx.tuning.span.min(SPAN) {
                self.lose(k);
            }
        }
        if let Some((period, guess)) = self.fx.pending.take() {
            self.start(period, guess);
        }
        match self.fx.mode {
            Mode::Record(0) => {
                self.tap.inner.memo().fib.rec = false;
                let tape = self.fx.tape.as_mut().expect("recorded");
                let whole = tape.calls.len() == tape.period;
                let last = tape
                    .out
                    .iter()
                    .map(|&a| (tape.when[a as usize], tape.byte[a as usize]));
                let frames = tape.frame_when.iter_mut().zip(&mut tape.frame_byte);
                for ((at, byte), last) in frames.zip(last.cycle()).filter(|f| *f.0 .0 == 0) {
                    (*at, *byte) = last;
                }
                (tape.shadow, tape.when, tape.byte) = (Vec::new(), Vec::new(), Vec::new());
                tape.segs.shrink_to_fit();
                match self.framed(cpu, 0).filter(|_| whole) {
                    Some(wrap) => {
                        let tape = self.fx.tape.as_mut().expect("recorded");
                        (tape.wrap, tape.frame_cpu[0].pc) = (wrap, cpu.pc);
                        (self.fx.phase, self.fx.fast, self.fx.slow) = (0, 0, 0);
                        self.tap.inner.memo().stats.tape_from = k as u64;
                        self.join(cpu, k, 0, wrap);
                    }
                    None => self.lose(k),
                }
            }
            Mode::Record(n) => {
                self.capture(cpu);
                self.fx.mode = Mode::Record(n - 1);
            }
            Mode::Resync => {
                let tape = self.fx.tape.as_ref().expect("taped");
                let (at, between) = (self.fx.phase / tape.every, self.fx.phase % tape.every);
                if between == 0 {
                    match self.framed(cpu, at) {
                        Some(stale) => self.join(cpu, k, at, stale),
                        None => self.lose(k),
                    }
                }
            }
            _ => {}
        }
        let (memo, fx) = (self.tap.inner.memo(), &mut self.fx);
        let tape = fx.tape.as_ref().filter(|_| fx.mode == Mode::Tape)?;
        if fx.phase % tape.every == 0 {
            let mut at = tape.frame_cpu[fx.phase / tape.every].clone();
            set_regs(&mut at, fx.stale.mask as u32, fx.stale.now);
            (at.cycles, at.steps) = (cpu.cycles, cpu.steps);
            if let Some(joined) = fx.live.cpu.as_ref().filter(|_| fx.live.k == k) {
                at.pc = joined.pc;
            }
            (fx.frame.k, fx.frame.at, fx.frame.cpu) = (k, fx.phase / tape.every, Some(at));
            fx.frame.cells.clear();
            let cells = fx.cells.iter().map(|&a| memo.bus.peek(a));
            fx.frame.cells.extend(cells);
        }
        None
    }

    /// Make the memory and `cpu` those call `k`, the one about to be run by the tape,
    /// starts from: those of the nearest call before it that can be put back, and the
    /// calls from there run as ordinary calls. How they stopped, if one did not return.
    pub(super) fn settle<B: Bus + 'a, const T: bool>(
        &mut self,
        cpu: &mut Cpu,
        k: usize,
    ) -> Option<Stop>
    where
        I: Over<'a, B, T>,
    {
        let h = self.tape_hash();
        let (memo, fx) = (self.tap.inner.memo(), &mut self.fx);
        let tape = fx.tape.as_ref().expect("taped");
        let framed = fx.frame.cpu.is_some() && fx.frame.k > fx.live.k;
        let from = if framed { &fx.frame } else { &fx.live };
        if framed {
            let (n, outs) = (tape.w.len(), tape.out.len());
            for (&a, &v) in tape.w.iter().zip(&tape.frames[from.at * n..][..n]) {
                memo.bus.poke(a, v);
            }
            let last = tape.frame_when[from.at * outs..][..outs].iter();
            let last = last.zip(&tape.frame_byte[from.at * outs..][..outs]);
            let phase = from.at * tape.every;
            for (&a, (&at, &v)) in tape.out.iter().zip(last).filter(|o| *o.1 .0 != 0) {
                let then = (at >> 8) as usize - 1;
                let ago = if then < phase {
                    phase - then
                } else {
                    phase + tape.period - then
                };
                let call = from.k.checked_sub(ago).filter(|&c| c >= fx.live.k);
                if call.is_some_and(|c| (c as u64) << 8 | (at & 0xFF) as u64 > fx.wrote[a as usize])
                {
                    memo.bus.poke(a, v);
                }
            }
        }
        let want: Vec<u8> = fx.cells.iter().map(|&a| memo.bus.peek(a)).collect();
        for (&a, &v) in fx.cells.iter().zip(&from.cells) {
            memo.bus.poke(a, v);
        }
        let (cycles, steps, behind) = (cpu.cycles, cpu.steps, k - from.k);
        *cpu = from.cpu.clone().expect("anchored");
        let mode = fx.pause();
        for _ in 0..behind {
            let stop = self.call(cpu);
            if stop == Stop::Stopped {
                return Some(stop);
            }
            debug_assert!(stop == Stop::Sp);
        }
        let (memo, fx) = (self.tap.inner.memo(), &mut self.fx);
        fx.resume(mode);
        let cells: Vec<u8> = fx.cells.iter().map(|&a| memo.bus.peek(a)).collect();
        debug_assert!(cells == want && (cpu.cycles, cpu.steps) == (cycles, steps));
        debug_assert!(
            Stale::between(&regs(cpu), &fx.stale.was).mask & fx.stale.mask == fx.stale.mask
        );
        let cpu = Some(cpu.clone());
        fx.live = Anchor {
            k,
            at: 0,
            cpu,
            cells,
        };
        let of_regs = memo.state_hash(fx.live.cpu.as_ref().expect("set"));
        memo.hash = h.wrapping_sub(of_regs.wrapping_sub(memo.hash));
        None
    }

    /// Add the call just run from `cycles` and `steps` to the tape being recorded, or
    /// start it again, or give it up.
    pub(super) fn record<B: Bus + 'a, const T: bool>(
        &mut self,
        cpu: &Cpu,
        stop: Stop,
        cycles: u64,
        steps: u64,
    ) where
        I: Over<'a, B, T>,
    {
        let (memo, fx) = (self.tap.inner.memo(), &mut self.fx);
        let tape = fx.tape.as_mut().expect("recording");
        let took = (cpu.cycles - cycles, cpu.steps - steps);
        let joined = match stop {
            Stop::Sp => tape.finish(&memo.fib, &mut fx.near, (regs(cpu), cpu.pc), took.0, took.1),
            _ => Joined::No,
        };
        memo.fib.clear();
        match joined {
            Joined::Yes => {}
            Joined::Again(guess) => {
                memo.fib.rec = false;
                fx.pending = Some((tape.period, guess));
                memo.stats.tape_retaped += 1;
            }
            Joined::No => self.lose(self.fx.k + 1),
        }
    }

    /// Run the call at the tape's phase by the tape: the fibre writes of its units
    /// between segments, and what each segment is held to do from the fibre as it then
    /// is. False, with nothing done, where a segment is not held or the call would not
    /// end within the cycle limit.
    fn fast<B: Bus + 'a, const T: bool>(&mut self, cpu: &mut Cpu) -> bool
    where
        I: Over<'a, B, T>,
    {
        let end = cpu.cycles.saturating_add(self.max_cycles).saturating_add(1);
        let (memo, fx) = (self.tap.inner.memo(), &mut self.fx);
        let tape = fx.tape.as_ref().expect("taped");
        let before = (fx.stale, fx.hash, cpu.cycles, cpu.steps);
        fx.undo.clear();
        fx.unwrite.clear();
        let (weight, undo, hash) = (&memo.weight, &mut fx.undo, &mut fx.hash);
        let mut poke = |bus: &mut B, w: u32| {
            let (addr, v) = ((w >> 8) as u16, w as u8);
            let old = bus.peek(addr);
            if old != v {
                undo.push((addr, old));
                let d = (v as u64).wrapping_sub(old as u64);
                *hash = hash.wrapping_add(weight[addr as usize].wrapping_mul(d));
                bus.poke(addr, v);
            }
        };
        let (call, segs) = (&tape.calls[fx.phase], tape.segs_of(fx.phase));
        let (known, mut held) = (segs.len() as u64, true);
        for (s, seg) in tape.segs[segs].iter().enumerate() {
            for &w in tape.lists.get(seg.pre) {
                poke(memo.bus, w);
            }
            fx.stale.mask &= seg.kept;
            let (uses, ends) = (&tape.uses[seg.ctx as usize], &tape.ends[seg.ends as usize]);
            let mut regs = ends.regs;
            for r in (0..16).filter(|&r| fx.stale.mask >> r & 1 != 0) {
                regs[r] = fx.stale.now[r];
            }
            let reads = (tape.reads[seg.ctx as usize], &regs);
            key(&mut fx.key, seg.ctx, reads, uses, |&a| memo.bus.peek(a));
            let Some(done) = tape.done.get(&fx.key) else {
                held = false;
                break;
            };
            for r in (0..16).filter(|&r| done.wmask >> r & 1 != 0) {
                regs[r] = done.regs[r];
            }
            let after = Stale::between(&regs, &ends.end_regs);
            if after.mask & ends.must != 0 {
                held = false;
                break;
            }
            let (from, out) = (done.writes.0 as usize, done.writes.1 as usize);
            let writes = &tape.fw[from..from + out + done.writes.2 as usize];
            for &w in writes {
                poke(memo.bus, w);
            }
            for &w in &writes[out..] {
                let a = (w >> 8) as usize;
                fx.unwrite.push((a as u16, fx.wrote[a]));
                fx.wrote[a] = (fx.k as u64) << 8 | (2 * s as u64 + 1);
            }
            fx.stale = after;
            cpu.cycles += done.cycles as u64;
            cpu.steps += done.steps as u64;
        }
        if held {
            for &w in tape.lists.get(call.post) {
                poke(memo.bus, w);
            }
            fx.stale.mask &= call.kept;
            cpu.cycles += call.cycles as u64;
            cpu.steps += call.steps as u64;
            held = cpu.cycles < end;
            memo.stats.tape_limited += !held as u64;
        }
        if !held {
            for &(addr, old) in fx.undo.iter().rev() {
                memo.bus.poke(addr, old);
            }
            for &(addr, at) in fx.unwrite.iter().rev() {
                fx.wrote[addr as usize] = at;
            }
            (fx.stale, fx.hash, cpu.cycles, cpu.steps) = before;
            return false;
        }
        memo.stats.calls += 1;
        memo.stats.tape_calls += 1;
        memo.stats.tape_known += known;
        fx.fast += 1;
        fx.advance(&mut memo.stats);
        true
    }

    /// Run call `k`, the one at the tape's phase, as an ordinary call from the state it
    /// starts from, holding what each of its segments does where it rejoins the tape:
    /// where it ends, the registers live there and the base cells either wrote are the
    /// tape's. A call with a segment that does not is followed by ordinary calls.
    fn learn<B: Bus + 'a, const T: bool>(&mut self, cpu: &mut Cpu, k: usize) -> Stop
    where
        I: Over<'a, B, T>,
    {
        let (sp_above, end) = self.setup(cpu);
        let (memo, fx) = (self.tap.inner.memo(), &mut self.fx);
        let tape = fx.tape.as_ref().expect("taped");
        memo.stats.tape_learned += 1;
        fx.slow += 1;
        let (period, guess, kept) = (tape.period, tape.guess, tape.calls[fx.phase].kept);
        let (mut stale, mut stop) = (fx.stale, Stop::Pc);
        let (mut rejoined, mut grew) = (true, false);
        for s in tape.segs_of(fx.phase) {
            let tape = self.fx.tape.as_ref().expect("taped");
            let (seg, shape) = (tape.segs[s], tape.shape(tape.segs[s].ctx).clone());
            let ends = tape.ends[seg.ends as usize];
            if shape.start.1 > 0 {
                stop = self.go(cpu, sp_above, end, Some(shape.start));
            }
            stale.mask &= seg.kept;
            if stop != Stop::Pc {
                rejoined = false;
                break;
            }
            debug_assert!(cpu.pc == shape.start.0);
            debug_assert!(stale.same(&Stale::between(&regs(cpu), &ends.regs)));
            self.tap.inner.memo().fib.log(true);
            let from = (cpu.cycles, cpu.steps, regs(cpu));
            stop = self.go(cpu, sp_above, end, shape.end);
            let (memo, fx) = (self.tap.inner.memo(), &mut self.fx);
            memo.fib.log(false);
            let ended = match shape.end {
                Some(_) => stop == Stop::Pc,
                None => stop == Stop::Sp && cpu.pc == shape.last,
            };
            if !(ended && !memo.fib.dead && fx.rejoins(memo, &shape)) {
                rejoined = false;
                break;
            }
            grew |= fx.widen(memo);
            if !grew {
                let took = (cpu.cycles - from.0, cpu.steps - from.1);
                fx.keep(memo, seg.ctx, &from.2, &regs(cpu), took);
            }
            stale = Stale::between(&regs(cpu), &ends.end_regs);
            if stale.mask & ends.must != 0 {
                rejoined = false;
                break;
            }
        }
        if stop == Stop::Pc {
            stop = self.go(cpu, sp_above, end, None);
        }
        let (memo, fx) = (self.tap.inner.memo(), &mut self.fx);
        if !(rejoined && stop == Stop::Sp) {
            memo.stats.tape_failed += 1;
            (fx.phase, fx.mode) = ((fx.phase + 1) % period, Mode::Resync);
            return stop;
        }
        stale.mask &= kept;
        fx.stale = stale;
        fx.advance(&mut memo.stats);
        let (at, stale) = (k + 1, fx.stale);
        self.join(cpu, at, 0, stale);
        let fx = &mut self.fx;
        if grew {
            fx.pending = Some((period, guess));
            self.tap.inner.memo().stats.tape_retaped += 1;
        } else if fx.slow > fx.tuning.grace + 2 * period as u64 && fx.slow > fx.fast {
            self.lose(at);
        }
        stop
    }

    /// The next call, `k`, by the tape where calls are run by it.
    pub(super) fn step<B: Bus + 'a, const T: bool>(&mut self, cpu: &mut Cpu, k: usize) -> Stop
    where
        I: Over<'a, B, T>,
    {
        if self.fx.mode == Mode::Resync {
            let tape = self.fx.tape.as_ref().expect("taped");
            self.fx.phase = (self.fx.phase + 1) % tape.period;
        }
        if !self.fx.taped() {
            return self.call(cpu);
        }
        let stop = match (self.stopped)() {
            true => self.settle(cpu, k).unwrap_or(Stop::Stopped),
            false if self.fast(cpu) => return Stop::Sp,
            false => match self.settle(cpu, k) {
                Some(stop) => stop,
                None => self.learn(cpu, k),
            },
        };
        if stop == Stop::Stopped {
            self.fx.mode = Mode::Off;
        }
        stop
    }

    /// Leave the memory and `cpu` as call `k`, the one that would be run next, starts.
    pub(super) fn wind_up<B: Bus + 'a, const T: bool>(&mut self, cpu: &mut Cpu, k: usize)
    where
        I: Over<'a, B, T>,
    {
        if self.fx.taped() {
            self.settle(cpu, k);
        }
        if let Some(tape) = &self.fx.tape {
            let stats = &mut self.tap.inner.memo().stats;
            stats.tape_bytes = stats.tape_bytes.max(tape.bytes() as u64);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{brute_over, image, table, tracked};
    use super::super::tests::{COUNTERS, GRAINS, LOOP, MAX_CYCLES, PLAY, VOICES};
    use super::super::{memo_orbit_tuned, Grain};
    use super::*;
    use crate::orbit::{End, Orbit};
    use crate::run::AddrSet;

    /// $1000 JSR fast ; JSR slow ; JSR other ; JSR tail ; RTS ; fast as in [`RAMP`] but
    /// STA $2000 ; RTS ; slow INC $11 ; LDA $11 ; AND #7 ; STA $11 ; STA $2001 ; STA $2000 ;
    /// STA $2002 ; RTS ; other DEC $12 ; BPL + ; LDA #6 ; STA $12 ; + RTS ; tail LDA $10 ;
    /// STA $2002 ; RTS: counters of periods 3, 8 and 7, two copied outside the tracked state.
    const TICKS: &[u8] = &[
        0x20, 0x0D, 0x10, 0x20, 0x1D, 0x10, 0x20, 0x2F, 0x10, 0x20, 0x38, 0x10, 0x60, 0xE6, 0x10,
        0xA5, 0x10, 0xC9, 0x03, 0xD0, 0x04, 0xA9, 0x00, 0x85, 0x10, 0x8D, 0x00, 0x20, 0x60, 0xE6,
        0x11, 0xA5, 0x11, 0x29, 0x07, 0x85, 0x11, 0x8D, 0x01, 0x20, 0x8D, 0x00, 0x20, 0x8D, 0x02,
        0x20, 0x60, 0xC6, 0x12, 0x10, 0x04, 0xA9, 0x06, 0x85, 0x12, 0x60, 0xA5, 0x10, 0x8D, 0x02,
        0x20, 0x60,
    ];

    /// $1000 JSR fast ; JSR five ; JSR mix ; RTS ; mix LDA $10 ; ASL ; ADC $11 ; AND #7 ;
    /// STA $11 ; RTS ; fast as in [`RAMP`] ; five DEC $13 ; BPL + ; LDA #4 ; STA $13 ;
    /// `+` RTS: `$11` steps by what the counter of period 3 is, beside one of period 5.
    const MIXED: &[u8] = &[
        0x20, 0x14, 0x10, 0x20, 0x21, 0x10, 0x20, 0x0A, 0x10, 0x60, 0xA5, 0x10, 0x0A, 0x65, 0x11,
        0x29, 0x07, 0x85, 0x11, 0x60, 0xE6, 0x10, 0xA5, 0x10, 0xC9, 0x03, 0xD0, 0x04, 0xA9, 0x00,
        0x85, 0x10, 0x60, 0xC6, 0x13, 0x10, 0x04, 0xA9, 0x04, 0x85, 0x13, 0x60,
    ];

    /// $1000 JSR fast ; JSR five ; JSR one ; LDA $10 ; JSR two ; RTS ; one INC $11 ; LDA $11 ;
    /// AND #3 ; STA $11 ; RTS ; two SEC ; ADC $12 ; AND #7 ; STA $12 ; RTS ; fast and five
    /// as in [`MIXED`]: `one` leaves its counter in A, which is loaded again with the
    /// counter of period 3 for `two` to step `$12` by.
    const CHAIN: &[u8] = &[
        0x20, 0x20, 0x10, 0x20, 0x2D, 0x10, 0x20, 0x0F, 0x10, 0xA5, 0x10, 0x20, 0x18, 0x10, 0x60,
        0xE6, 0x11, 0xA5, 0x11, 0x29, 0x03, 0x85, 0x11, 0x60, 0x38, 0x65, 0x12, 0x29, 0x07, 0x85,
        0x12, 0x60, 0xE6, 0x10, 0xA5, 0x10, 0xC9, 0x03, 0xD0, 0x04, 0xA9, 0x00, 0x85, 0x10, 0x60,
        0xC6, 0x13, 0x10, 0x04, 0xA9, 0x04, 0x85, 0x13, 0x60,
    ];

    /// $1000 LDX #0 ; again DEC $11,X ; BPL + ; LDA $1011,X ; STA $11,X ; + INX ; CPX #2 ;
    /// BNE again ; RTS ; !byte 7, 2: counters of periods 8 and 3 stepped by one loop.
    const TWICE: &[u8] = &[
        0xA2, 0x00, 0xD6, 0x11, 0x10, 0x05, 0xBD, 0x11, 0x10, 0x95, 0x11, 0xE8, 0xE0, 0x02, 0xD0,
        0xF2, 0x60, 0x07, 0x02,
    ];

    /// $1000 JSR fast ; JSR five ; INC $11 ; LDA $11 ; AND #3 ; STA $11 ; BEQ + ; LDA #$10 ;
    /// RTS ; + LDA #$10 ; PHA ; RTI ; fast and five as in [`MIXED`]: one call in four
    /// returns by an RTI, to another PC, over the byte the JSR left on the stack.
    const ENDS: &[u8] = &[
        0x20, 0x17, 0x10, 0x20, 0x24, 0x10, 0xE6, 0x11, 0xA5, 0x11, 0x29, 0x03, 0x85, 0x11, 0xF0,
        0x03, 0xA9, 0x10, 0x60, 0xA9, 0x10, 0x48, 0x40, 0xE6, 0x10, 0xA5, 0x10, 0xC9, 0x03, 0xD0,
        0x04, 0xA9, 0x00, 0x85, 0x10, 0x60, 0xC6, 0x13, 0x10, 0x04, 0xA9, 0x04, 0x85, 0x13, 0x60,
    ];

    /// $1000 JSR fast ; JSR five ; JSR other ; JSR slow ; CPY #2 ; RTS ; slow INC $11 ;
    /// LDA $11 ; AND #3 ; STA $11 ; TAY ; RTS ; other DEC $12 ; BPL + ; LDA #6 ; STA $12 ;
    /// `+` RTS ; fast and five as in [`MIXED`]: the counter of period 4 is left in Y, which
    /// the unit after its subroutine reads.
    const CARRY: &[u8] = &[
        0x20, 0x22, 0x10, 0x20, 0x2F, 0x10, 0x20, 0x19, 0x10, 0x20, 0x0F, 0x10, 0xC0, 0x02, 0x60,
        0xE6, 0x11, 0xA5, 0x11, 0x29, 0x03, 0x85, 0x11, 0xA8, 0x60, 0xC6, 0x12, 0x10, 0x04, 0xA9,
        0x06, 0x85, 0x12, 0x60, 0xE6, 0x10, 0xA5, 0x10, 0xC9, 0x03, 0xD0, 0x04, 0xA9, 0x00, 0x85,
        0x10, 0x60, 0xC6, 0x13, 0x10, 0x04, 0xA9, 0x04, 0x85, 0x13, 0x60,
    ];

    /// $1000 JSR fast ; JSR ramp ; RTS ; ramp LDA $11 ; CMP #40 ; BCC + ; LDA #29 ;
    /// STA $11 ; + INC $11 ; RTS ; fast INC $10 ; LDA $10 ; CMP #3 ; BNE + ; LDA #0 ;
    /// STA $10 ; + RTS: `$11` counts to 40 and from then on from 30 to 40, beside a counter
    /// of period 3.
    const RAMP: &[u8] = &[
        0x20, 0x14, 0x10, 0x20, 0x07, 0x10, 0x60, 0xA5, 0x11, 0xC9, 0x28, 0x90, 0x04, 0xA9, 0x1D,
        0x85, 0x11, 0xE6, 0x11, 0x60, 0xE6, 0x10, 0xA5, 0x10, 0xC9, 0x03, 0xD0, 0x04, 0xA9, 0x00,
        0x85, 0x10, 0x60,
    ];

    /// $1000 JSR fast ; JSR slow ; RTS ; slow INC $11 ; LDA $11 ; AND #15 ; STA $11 ;
    /// CMP #9 ; BNE + ; INC $13 ; + RTS ; fast as in [`RAMP`]: `$13` steps once in each
    /// period of `$11`.
    const LEAK: &[u8] = &[
        0x20, 0x16, 0x10, 0x20, 0x07, 0x10, 0x60, 0xE6, 0x11, 0xA5, 0x11, 0x29, 0x0F, 0x85, 0x11,
        0xC9, 0x09, 0xD0, 0x02, 0xE6, 0x13, 0x60, 0xE6, 0x10, 0xA5, 0x10, 0xC9, 0x03, 0xD0, 0x04,
        0xA9, 0x00, 0x85, 0x10, 0x60,
    ];

    /// $1000 JSR slow ; JSR wait ; RTS ; slow INC $11 ; LDA $11 ; AND #7 ; STA $11 ; TAX ;
    /// loop DEX ; BPL loop ; RTS ; wait INC $10 ; LDA $10 ; CMP #3 ; BNE + ; LDA #0 ;
    /// STA $10 ; + TAY ; loop DEY ; BPL loop ; RTS: each counter also sets how long its
    /// subroutine takes.
    const DELAY: &[u8] = &[
        0x20, 0x07, 0x10, 0x20, 0x14, 0x10, 0x60, 0xE6, 0x11, 0xA5, 0x11, 0x29, 0x07, 0x85, 0x11,
        0xAA, 0xCA, 0x10, 0xFD, 0x60, 0xE6, 0x10, 0xA5, 0x10, 0xC9, 0x03, 0xD0, 0x04, 0xA9, 0x00,
        0x85, 0x10, 0xA8, 0x88, 0x10, 0xFD, 0x60,
    ];

    /// How an orbit is run: its call limit and cycle limit, the registers that are state
    /// beside [`tracked`], the grain and the tuning.
    #[derive(Clone, Copy)]
    struct How {
        limit: usize,
        max_cycles: u64,
        regs: &'static [u32],
        grain: Grain,
        tuning: Tuning,
    }

    const HOW: How = How {
        limit: 400,
        max_cycles: MAX_CYCLES,
        regs: &[],
        grain: Grain::Sub,
        tuning: Tuning {
            crowd: 1024,
            raw: 1 << 16,
            saturated: 8,
            frames: 16 << 20,
            grace: 4096,
            rest: 1024,
            span: 1 << 22,
            known: 1 << 20,
        },
    };

    /// What the brute-force orbit finds and the RAM and registers it leaves.
    type Left = (Orbit, Box<[u8; 0x10000]>, Cpu);

    /// The memo orbit of `m` with the cells `fibre` set apart, after asserting it is the
    /// brute-force orbit and leaves the RAM and registers that leaves.
    fn apart(m: &[u8; 0x10000], how: How, fibre: &[u16]) -> (Orbit, MemoStats) {
        let want = brute_over(m, how.limit, how.max_cycles, how.regs);
        check(m, how, fibre, &want)
    }

    /// [`apart`] against the brute-force orbit's `want`.
    fn check(m: &[u8; 0x10000], how: How, fibre: &[u16], want: &Left) -> (Orbit, MemoStats) {
        let (mut ram, mut cpu) = (Box::new(*m), Cpu::new());
        let keep: AddrSet = tracked().into_iter().map(|a| a as u16).collect();
        let none = AddrSet::default();
        let (got, stats) = memo_orbit_tuned::<_, true>(
            &mut cpu,
            &mut *ram,
            PLAY,
            0,
            how.max_cycles,
            how.limit,
            None,
            &none,
            &keep,
            how.regs,
            how.grain,
            how.tuning,
            fibre,
            || false,
        );
        let what = format!(
            "{:?} {:?} {:04X?}",
            how.grain,
            how.tuning,
            &fibre[..fibre.len().min(8)]
        );
        assert_eq!((got.end, got.states), (want.0.end, want.0.states), "{what}");
        assert_eq!(got, want.0, "{what}");
        let differ: Vec<usize> = (0..0x10000).filter(|&a| ram[a] != want.1[a]).collect();
        assert_eq!(differ, [0usize; 0], "{what}: RAM");
        assert_eq!(cpu, want.2, "{what}");
        (got, stats)
    }

    fn ticks() -> Box<[u8; 0x10000]> {
        image(TICKS, &[(0x12, 6)])
    }

    #[test]
    fn segments_are_answered_by_what_they_were_learned_to_do() {
        let (orbit, stats) = apart(&ticks(), HOW, &[0x11, 0x12]);
        assert_eq!((orbit.end, orbit.states), (End::Closed, 169));
        // Each segment reads one counter: the longer period's 8 values are learned once.
        assert_eq!((stats.tape_period, stats.tape_learned), (3, 8));
        assert_eq!((stats.tape_calls, stats.tape_known), (153, 306));
        assert_eq!((stats.tape_failed, stats.tape_lost), (0, 0));
        let (_, stats) = apart(&ticks(), HOW, &[0x11]);
        assert_eq!((stats.tape_period, stats.tape_learned), (21, 8));
        assert_eq!((stats.tape_calls, stats.tape_known), (117, 117));
        // A whole call is one unit that reads every counter, so nothing recurs.
        let whole = How {
            grain: Grain::Call,
            ..HOW
        };
        assert_eq!(apart(&ticks(), whole, &[0x11, 0x12]).1.tape_calls, 0);
    }

    #[test]
    fn a_tape_is_run_round_with_the_registers_its_last_call_leaves() {
        let regs = How {
            regs: REGISTERS,
            ..HOW
        };
        for fibre in [&[0x11, 0x12][..], &[0x11], &[0x12]] {
            for how in [HOW, regs] {
                let (orbit, stats) = apart(&ticks(), how, fibre);
                assert_eq!(orbit.end, End::Closed);
                assert!(
                    stats.tape_passes > 2 && stats.tape_known > 100,
                    "{fibre:04X?}"
                );
            }
        }
    }

    #[test]
    fn the_first_repeat_of_two_calls_run_by_the_tape_is_found() {
        let m = image(RAMP, &[]);
        let (orbit, stats) = apart(&m, HOW, &[0x11]);
        // The state call 63 starts from is the one call 30 started from.
        assert_eq!((orbit.end, orbit.states), (End::Closed, 63));
        assert!(stats.tape_from < 30 && stats.tape_calls > 20);
        assert_eq!((stats.tape_lost, stats.tape_failed), (0, 0));
        for limit in 55..70 {
            let (orbit, _) = apart(&m, How { limit, ..HOW }, &[0x11]);
            assert_eq!(orbit.states, 63.min(limit + 1));
        }
    }

    #[test]
    fn a_run_ends_with_the_memory_and_registers_of_its_last_call() {
        for frames in [16 << 20, 400, 200, 1] {
            for limit in 0..75 {
                let tuning = Tuning {
                    frames,
                    ..Tuning::default()
                };
                let how = How {
                    limit,
                    tuning,
                    ..HOW
                };
                apart(&ticks(), how, &[0x11]);
                apart(&ticks(), how, &[0x11, 0x12]);
            }
        }
    }

    const REGISTERS: &[u32] = &[0, 1, 2, 3, 8, 9, 10, 11, 13, 14];

    #[test]
    fn a_segment_is_answered_by_the_base_it_reads_too() {
        let m = image(MIXED, &[(0x13, 4)]);
        let (orbit, stats) = apart(&m, HOW, &[0x11]);
        assert_eq!((orbit.end, orbit.states), (End::Closed, 61));
        // What the segment does is learned per value of the counter of period 3 it reads.
        assert_eq!((stats.tape_period, stats.tape_learned), (15, 12));
        assert_eq!((stats.tape_calls, stats.tape_failed), (16, 0));
        for limit in 40..62 {
            apart(&m, How { limit, ..HOW }, &[0x11]);
        }
    }

    #[test]
    fn a_segment_is_answered_by_the_registers_it_reads_as_the_units_before_leave_them() {
        let m = image(CHAIN, &[(0x13, 4)]);
        let (orbit, stats) = apart(&m, HOW, &[0x11, 0x12]);
        assert_eq!((orbit.end, orbit.states), (End::Closed, 61));
        assert_eq!((stats.tape_calls, stats.tape_known), (16, 32));
        assert_eq!((stats.tape_learned, stats.tape_failed), (12, 0));
        for limit in 60..100 {
            apart(&m, How { limit, ..HOW }, &[0x11, 0x12]);
        }
    }

    #[test]
    fn a_segment_may_start_and_end_at_one_pc() {
        let m = image(TWICE, &[(0x11, 7), (0x12, 2)]);
        let how = How {
            grain: Grain::Loop,
            ..HOW
        };
        let (orbit, stats) = apart(&m, how, &[0x11]);
        assert_eq!((orbit.end, orbit.states), (End::Closed, 25));
        assert_eq!((stats.tape_calls, stats.tape_learned), (10, 8));
        assert_eq!((stats.tape_failed, stats.tape_lost), (0, 0));
    }

    #[test]
    fn a_segment_that_ends_the_call_at_another_pc_does_not_rejoin() {
        let m = image(ENDS, &[(0x13, 4)]);
        let (orbit, stats) = apart(&m, HOW, &[0x11]);
        assert_eq!((orbit.end, orbit.states), (End::Closed, 61));
        // The tape's call returned the other way in 12 of the calls run by it.
        assert_eq!((stats.tape_calls, stats.tape_failed), (11, 12));
        assert_eq!(stats.tape_lost, 0);
        for limit in 30..62 {
            apart(&m, How { limit, ..HOW }, &[0x11]);
        }
    }

    #[test]
    fn a_segment_that_leaves_a_live_register_other_than_the_tapes_does_not_rejoin() {
        let m = image(CARRY, &[(0x12, 6), (0x13, 4)]);
        for regs in [&[][..], REGISTERS] {
            let how = How {
                limit: 600,
                regs,
                ..HOW
            };
            let (orbit, stats) = apart(&m, how, &[0x11, 0x12]);
            assert_eq!((orbit.end, orbit.states), (End::Closed, 421));
            // Y is the tape's in one pass of four: the others' calls are ordinary calls.
            assert_eq!((stats.tape_calls, stats.tape_failed), (90, 299));
            assert_eq!((stats.tape_learned, stats.tape_lost), (299, 0));
            for limit in 100..140 {
                let how = How { limit, regs, ..HOW };
                apart(&m, how, &[0x11, 0x12]);
            }
        }
    }

    #[test]
    fn a_segment_that_changes_the_base_is_followed_by_ordinary_calls() {
        let m = image(LEAK, &[(0x11, 12)]);
        let (orbit, stats) = apart(&m, How { limit: 100, ..HOW }, &[0x11]);
        assert_eq!((orbit.end, orbit.states), (End::Open, 101));
        assert_eq!((stats.tape_failed, stats.tape_lost), (1, 1));
        assert!(stats.tape_learned > 1);
    }

    #[test]
    fn a_tape_given_up_is_tried_again_after_ever_more_calls() {
        let m = image(LEAK, &[]);
        let tuning = Tuning {
            rest: 4,
            ..Tuning::default()
        };
        let how = How {
            limit: 4000,
            tuning,
            ..HOW
        };
        let (_, stats) = apart(&m, how, &[0x11]);
        assert!((6..=10).contains(&stats.tape_lost), "{}", stats.tape_lost);
        // The base's period is past the span it is looked for over.
        let tuning = Tuning { span: 2, ..tuning };
        let (_, stats) = apart(&ticks(), How { tuning, ..how }, &[0x11]);
        assert!((6..=12).contains(&stats.tape_lost), "{}", stats.tape_lost);
        assert_eq!(stats.tape_period, 0);
    }

    #[test]
    fn a_call_the_tape_would_take_past_the_cycle_limit_is_run_as_an_ordinary_call() {
        let m = image(DELAY, &[]);
        // The longest call runs out of cycles in the subroutine after its segment.
        let how = How {
            max_cycles: 104,
            limit: 300,
            ..HOW
        };
        let (orbit, stats) = apart(&m, how, &[0x11]);
        assert_eq!((orbit.end, orbit.states), (End::Open, 301));
        assert!(stats.tape_limited == 1 && stats.tape_known > 0 && stats.tape_failed == 1);
        // Its last RTS starts within the limit, so it returns.
        let (orbit, stats) = apart(
            &m,
            How {
                max_cycles: 108,
                ..how
            },
            &[0x11],
        );
        assert_eq!((orbit.end, orbit.states), (End::Closed, 25));
        assert_eq!((stats.tape_limited, stats.tape_failed), (1, 0));
    }

    #[test]
    fn a_fibre_changes_nothing_found() {
        let small = Tuning {
            frames: 300,
            grace: 4,
            rest: 2,
            span: 40,
            known: 3,
            ..Tuning::default()
        };
        let tunings = [
            Tuning::default(),
            small,
            Tuning {
                crowd: 2,
                raw: 4,
                saturated: 1,
                ..small
            },
            Tuning {
                crowd: 3,
                raw: 1 << 20,
                saturated: 64,
                frames: 1,
                ..Tuning::default()
            },
        ];
        let routines = [
            image(COUNTERS, &[]),
            image(&table(), &[]),
            image(VOICES, &[(0x18, 2), (0x19, 3), (0x1A, 4)]),
            image(LOOP, &[]),
            ticks(),
            image(RAMP, &[]),
            image(LEAK, &[]),
            image(DELAY, &[]),
            image(MIXED, &[(0x13, 4)]),
            image(CARRY, &[(0x12, 6), (0x13, 4)]),
            image(CHAIN, &[(0x13, 4)]),
            image(TWICE, &[(0x11, 7), (0x12, 2)]),
            image(ENDS, &[(0x13, 4)]),
        ];
        let whole: Vec<u16> = tracked().into_iter().map(|a| a as u16).collect();
        let fibres: [&[u16]; 7] = [
            &[0x10],
            &[0x11],
            &[0x11, 0x12],
            &[0x1FF, 0x1FE],
            &[0x40, 0x5000],
            &[PLAY, 0x11, 0x18],
            &whole,
        ];
        let mut sum = [0; 8];
        let short = [(&routines[4], 104), (&routines[7], 104)];
        for (m, max_cycles) in routines.iter().map(|m| (m, MAX_CYCLES)).chain(short) {
            let want = brute_over(m, 150, max_cycles, &[]);
            for (grain, tuning) in GRAINS.iter().flat_map(|g| tunings.map(|t| (*g, t))) {
                let how = How {
                    limit: 150,
                    max_cycles,
                    grain,
                    tuning,
                    ..HOW
                };
                for fibre in fibres {
                    let s = check(m, how, fibre, &want).1;
                    let tape = [s.tape_calls, s.tape_known, s.tape_learned, s.tape_failed];
                    let ends = [s.tape_limited, s.tape_passes, s.tape_retaped, s.tape_lost];
                    for (sum, n) in sum.iter_mut().zip(tape.into_iter().chain(ends)) {
                        *sum += n;
                    }
                }
            }
        }
        // Every way a tape's call can go was gone.
        assert!(sum.iter().all(|&n| n > 10), "{sum:?}");
    }
}
