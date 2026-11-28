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

## What the result depends on

- The memory is plain RAM: nothing mapped, watched or observed (`ValueError` otherwise), so a
  read gives the byte last written and nothing else changes.
- The per-opcode register use derived from `flow_gen.rs` names every register an opcode may
  read and exactly those it writes. `memo.rs` tests this against the generated core for every
  opcode: a register outside an opcode's reads changes nothing it does, and a register outside
  its writes is left as it was.
- `mpu.stop` set from another thread is polled per emulated instruction, so it ends the run
  after a different instruction than it would end the brute-force orbit.
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
