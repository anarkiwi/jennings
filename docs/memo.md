# Memoising orbit

`jennings.memo.orbit(mpu, pc, acc, max_cycles, limit, grain="split", keep=None, volatile=None,
remap=None, cells=None)` returns what `NativeMPU.orbit` returns for the same arguments
([native.md](native.md#drivers)), `(end, states, accesses, edges)`, plus a dict of counts, without
emulating what it has already seen. It rests on determinism alone: it learns no dependences
and does not use `jennings.coverage.components`. `NativeMPU.orbit` is the oracle the
tests compare it with. The core is `jennings_core::memo_orbit` (`crates/jennings-core/src/memo.rs`).

## Units

Each call is cut into units at chosen PCs. A unit runs from one cut to the next (or to the
call's end), so units never nest and never span calls. `grain` chooses the cuts:

| grain | cuts |
|---|---|
| `call` | the entry only: a unit is a whole call |
| `sub` | also the target of each `JSR` and the PC each `RTS` returns to, as they are first taken |
| `loop` | also the target of each backward branch or `JMP` |
| `split` | also inside the units of a start PC once they are many (below) |

Where the cuts fall changes only how much is emulated, never the result: every unit stored is
a true record of an execution, whatever its extent.

## What a unit is keyed on

A unit's instructions, accesses, writes, exit registers, cycles and instruction count are a
function of the values of the locations it reads before writing them:

- memory bytes, by any access: opcode and operand fetches, data loads, stack pulls, vectors
  and the `JMP (ind)` pointer, and under a `Remap` the bytes its redirection decides by and
  the byte a redirected load takes;
- registers and flags, read off the generated per-opcode data flow (`flow_gen.rs`) before each
  instruction: a register an opcode may read is a read unless the unit wrote it first.

Memory that no call has written so far is constant and is left out of keys. The first write
to an address that some stored unit read as constant empties the memo, so no stored unit ever
depends on a byte its key lacks.

Stored units live in a radix trie per start PC over their reads in order. The trie is only an
index: a unit is replayed when, and only when, every `(location, value)` it stored equals the
current state, which by determinism means the execution from here is the stored one. A replay
writes the bytes the unit left in the memory it wrote, sets the registers it wrote, the PC,
the cycles and the instruction count, and adds no access or edge: those were added when the
unit was recorded, including the edge from its last instruction to the next.

Units are not recorded across a jam, a call that runs out of `max_cycles`, or a stop; a call
that starts with another SP than the one before empties the memo.

A stored unit whose cycles would take the call past `max_cycles` is emulated instead of
replayed. Emulated, it reads what it read before and runs as it ran before. If it ends at a
cut inside the call, the call runs out of `max_cycles` there and nothing is stored. If it ends
at the call's `RTS`, that `RTS` started within the limit and ended past it, and the call
returns as the brute-force call does, which checks for the return before the cycles. The unit
is then closed with the reads of the unit already stored, the trie refuses it, and it is
counted as `dropped`. That is the only way a unit is dropped: at a cut, any stored unit whose
reads all match is replayed unless its cycles forbid it, and adding a cut empties the
memo. `dropped` is therefore 0 unless some call's final `RTS` straddles `max_cycles`.

## Splitting crowded units

With `grain="split"`, when the units of one start PC reach `crowd` (doubling each time
nothing fits), the trie is walked and a cut is added at an instruction other than the start
PC that at least half of them reach, where of the distinct runs of reads before its first
read and the distinct runs of reads from it on the fewer are more than one and at most the
units over `saturated`; among those, the one with the fewest of both together. This separates
state that varies independently (a voice's note position from its pulse-width sweep, say),
whose product would otherwise be stored unit by unit. Adding a cut empties the memo.

At every grain, a start PC whose units reach a crowd that doubling would take past `raw`, no
cut having fitted (without `split` none is looked for), is emulated from then on and its units
are let go (`raw`). Units that end at it still end there.

## Tuning

`crowd`, `raw` and `saturated` (1024, 65536 and 8) are the core's `Tuning`; `memo_orbit` and
`jennings.memo.orbit` use those defaults. Every value gives the same result and leaves the same
state, because every unit stored is a true record whatever its extent and a start PC given up
on is emulated. They trade work and memory only:

- `crowd`: lower walks the trie for a cut sooner and more often, on fewer units; higher stores
  more units whole before a cut lets them share.
- `raw`: lower gives up sooner on a start PC whose units rarely recur, emulating what could
  have been replayed; higher holds more units and makes their lookups longer.
- `saturated`: higher cuts only where the runs of reads on each side would be shared often,
  storing more units whole; lower cuts more readily, each cut emptying the memo and leaving
  shorter units that cost more lookups per call.

The tests in `memo.rs` run small routines at every grain under small and large values of each
and compare every result with the brute-force orbit's.

A tape ([below](#fibre-and-tape)) has five more, with the same property (defaults 16 MiB, 4096,
1024, 2^22 and 2^20):

- `frames`: the bytes of base cells a tape may hold as frames, which sets how many calls apart
  they are. Lower holds less and runs more calls again to reach a call's entry state.
- `grace`: the calls a tape may run to learn segments, beyond twice its period, before it is
  given up where they are more than half of its calls.
- `rest`: the calls run, at least, before the base is next looked at after a tape is given up.
- `span`: the longest period of the base a tape is recorded for, and the calls its entry states
  are remembered over while a period is looked for.
- `known`: the segment results a tape may hold.

The tests in `memo/fibre.rs` run the same routines, and ones made for each path of a tape, with
fibres good and bad at every grain under small and large values of all eight.
`python tools/break_check.py tools/breaks/memo_fibre.toml` breaks each of those paths on purpose,
one at a time, and checks that the tests then fail.

## Closure

The state a call starts from is the RAM at `keep` (default all), or the `cells`, exactly as
`NativeMPU.orbit` defines it; registers are state only as `cells`. Each write, emulated or
replayed, updates a 64-bit hash of that state (a sum of per-address weights times bytes), and
the hashes seen are kept in a set. A call whose entry hash was seen is only a candidate: the
calls are run again from the first state, through the memo, and at each earlier call whose
entry has that hash the two states are compared byte for byte. The first equal state ends the
orbit as `"closed"`; if none is equal, the candidate is a hash collision, counted
(`collisions`), and the run goes on from the candidate's state. Each candidate costs a rerun
of the calls before it. Equal states always hash alike, so the first repeat is never missed.

`states` is counted as the brute-force orbit counts it: the calls to the first repeated state
when closed, `limit + 1` when open, and the calls run when a call loads a volatile address or
jams. The memory and registers left behind are those the brute-force orbit leaves.

## Fibre and tape

The `fibre` of `jennings_core::memo_orbit_fibre` is a list of addresses. Those of them that are
tracked memory are set apart from the rest of the memory the calls write, the base: the written
tracked cells outside the fibre. The written cells outside the tracked state are the third kind,
outside cells. When the base's entry state repeats after some calls while the fibre goes on
changing, as it does where the fibre holds the slow counters of a routine whose other state has a
short period, the calls of one period of the base are recorded as a tape, and later calls are run by
the tape: only the parts of a call that read the fibre are in question, and what such a part does
from given fibre bytes is run once and then looked up. The orbit, the memory and the registers left
are those without a fibre, whatever the cells are: a fibre that does not fit costs time, not
exactness. `fibre` is for the caller to choose; nothing chooses it.

### Recording

The hash of the tracked state less the fibre's part is noted at each call's entry. When it is
one noted `n` calls before, the next `n` calls are recorded: per call, the units it ran as the
memo ran them (replayed or emulated), each with the registers it read before writing, the
registers it wrote, the memory its key reads, its writes, cycles and instructions. A recording
is given up if a call does not return, runs an instruction outside any unit, has a unit whose key
reads an outside cell, writes a cell for the first time, or has the memo emptied under it.

A unit whose key reads a fibre cell starts a segment, which runs to the end of the last unit after
it that reads the fibre or starts with A or a flag live that the segment wrote. The tape keeps, per
call: per segment the fibre writes of the units between the boundary before and it, the registers
those units leave alone, a context and the tape's registers either side of it; the fibre writes and
registers left alone after the last segment; the cycles and instructions outside segments; and the
entry state's hash less the fibre's part. A context is where the segment starts and ends (the nth
time a unit starts at a PC, counted from the boundary before, or the PC the call is left at), the
base cells the tape's units wrote in it, and the bytes at its start of the near cells: the base
cells any segment has been seen to read or write. A segment that reads or writes a base cell not yet
near makes it near, and the tape is recorded again from the next call, since contexts are then told
apart by one more byte. Registers live at a call's entry (read by the tape's units before written)
are found the same way: the recording starts again with them.

Every `K` calls the tape keeps a frame: the base cells, the registers, and per outside cell the
last write to it by a unit outside a segment. `K` is 1 unless the frames would pass `frames`
bytes. After `n` calls the tape is joined if the base cells are byte for byte those of its first
frame and no register that differs from that frame's is live there; otherwise it is given up.

### Running a call by the tape

A call at the tape's phase `p` starts with the base cells of the tape's call `p` and its
registers but for some, the stale ones, which no unit of the tape reads before writing. Per
segment, in order: the fibre writes of the units before it are applied; the segment's result is
looked up by its context, the bytes of the registers its runs have read before writing and the
bytes of the fibre cells its runs have read or written; its fibre writes, its writes to outside
cells and its registers are applied; and the registers that then differ from the tape's must not
be live after it. The fibre writes after the last segment follow, and the cycles and
instructions are those outside segments plus the segments'. No unit is replayed or emulated and
the base cells in memory are not touched.

Where a result is not held, the registers do not fit, or the call's cycles would reach `max_cycles`,
nothing of the call is applied and the call is run as an ordinary call instead. The memory and
registers it starts from are first put back: from the nearest earlier call whose entry state is at
hand (a frame the tape passed, with the fibre and registers noted as it passed, or the last call so
put back, whose base cells are still in memory), by running the calls from there as ordinary calls.
The call is then run through the memo up to each segment's start, the segment with its reads and
writes noted, and so on to its end. A segment rejoins the tape if it ends where the tape's does
(leaving the call at the tape's PC, if it ends the call), each base cell it wrote and each the
tape's units wrote in it holds what the tape leaves there, and no cell was written for the first
time; its result is then held (unless a new cell became near), and if its registers fit the call
goes on. A call with a segment that does not rejoin ends as the ordinary call it is, and ordinary
calls follow to the tape's next frame: there the tape is joined again if the base is the frame's,
else given up.

A tape given up, or a search for a period that passes `span`, is followed by ordinary calls,
`rest` or the tape's period of them the first time and twice as many each time after, before
the base is looked at again. A tape whose calls are mostly run to learn, past `grace`, is given
up the same way. A first write to any cell ends the tape, since the cells are no longer those it
was recorded over.

### Why this is exact

- A unit's execution is a function of its key. A unit of the tape outside segments has no fibre
  cell and no outside cell in its key, so its key is base cells and registers. Up to the first
  segment the base cells are the tape's (checked byte for byte when the tape is joined, and kept
  so by this argument), and the registers its key reads are not stale. So it is the tape's unit
  again: the same instructions, writes, cycles, and exit registers, and its accesses and edges
  were added when it was recorded. Its writes to the fibre are applied; its writes to the base
  leave the base the tape's.
- A segment's execution is a function of the registers it reads before writing, the fibre cells
  it reads, the base cells it reads, and memory never written. A held result was learned from a
  run whose base reads and writes were all near cells, in a context with the same near bytes,
  and is keyed on those registers and fibre cells, so a segment with the same key runs as that
  run did: same writes, same end, same cycles. That run rejoined, and the test it passed reads
  only its writes, the near bytes and the context's base writes, so this one rejoins too; the
  registers are tested again for each segment, against its own tape registers.
- After a segment that rejoins, the base is the tape's and the registers that differ are not read
  before written by any later unit of the tape, or by the next call's, so the first point holds
  again up to the next segment and through the following calls.
- A call run as an ordinary call starts from the true state: the state it is put back to is a
  true entry state (a frame's base is the true base by the points above, the fibre and registers
  were noted from the true ones), and ordinary calls from a true state give the true state.
  Outside cells are written by segments as they are applied and otherwise only read off a frame
  when a state is put back from it: a frame's last write to a cell is taken if it was made after
  the last call put back and after the last segment write to the cell.
- The cycle limit only stops calls: a call whose cycles stay under it runs as if there were
  none, and any other is run as an ordinary call, limit, `dropped` unit and all.

### Closure under a tape

The entry state of a call run by the tape is the base of its phase, the fibre, and the tape's
registers but for the stale ones. Its hash is therefore the hash stored for the phase, plus the
fibre cells' part, plus the difference the stale tracked registers make, and it is the hash the
state would have without a tape, since the hash is a sum over cells. It goes into the same set
as every other call's, so a call run by the tape is a candidate repeat of any earlier call and
is confirmed as any candidate is, after its state is put back: by running the calls again from
the first state and comparing states. Two calls at one phase of a tape have equal states exactly
when their fibre bytes and stale tracked registers are equal; nothing rests on that beyond their
hashes then being equal, as equal states' hashes always are.

## What the result depends on

- The memory is plain RAM: nothing mapped, watched or observed (`ValueError` otherwise), so a
  read gives the byte last written and nothing else changes.
- The per-opcode register use derived from `flow_gen.rs` names every register an opcode may
  read and exactly those it writes. `memo.rs` tests this against the generated core for every
  opcode: a register outside an opcode's reads changes nothing it does, and a register outside
  its writes is left as it was.
- `mpu.stop` set from another thread is polled per emulated instruction, and per call run by a
  tape, so it ends the run after a different instruction than it would end the brute-force
  orbit.
- `mem.last_load` is that of the last emulated load.

## Counts

| key | |
|---|---|
| `calls` | calls run, those run again to confirm a repeat included |
| `hits`, `misses` | units replayed, units emulated and stored |
| `steps` | instructions emulated |
| `entries`, `live` | units stored over the whole run, units held at its end |
| `flushes` | times the memo was emptied |
| `splits`, `raw` | cuts added inside crowded units, start PCs no longer memoised |
| `dropped` | units not stored because a unit on the same reads was stored: 0 unless a call's final `RTS` straddles `max_cycles` (above) |
| `collisions` | candidate repeats whose states differed |
| `bytes` | bytes held by the memo and the hash set |
| `roots` | `(pc, units)`: the start PCs holding most units |
| `fanout` | `(location, nodes, values)`: the locations that tell most held units apart |

Measurements on real play routines are in [memo-measurements.md](memo-measurements.md).
