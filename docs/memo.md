# Memoising orbit

`jennings.memo.orbit(mpu, pc, acc, max_cycles, limit, grain="split", keep=None, volatile=None,
remap=None, cells=None)` returns what `NativeMPU.orbit` returns for the same arguments
([native.md](native.md#drivers)), `(end, states, accesses, edges)`, plus a dict of counts, without
emulating what it has already seen. It rests on determinism alone: it learns no dependences
and does not use `jennings.coverage.components`. `NativeMPU.orbit` is unchanged and is the
oracle the tests compare it with. The core is `jennings_core::memo_orbit`
(`crates/jennings-core/src/memo.rs`).

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

Units are not recorded across a jam, a call that runs out of `max_cycles`, or a stop; a unit
whose cycles would pass the call's cycle limit is emulated instead of replayed; a call that
starts with another SP than the one before empties the memo.

## Splitting crowded units

With `grain="split"`, when the units of one start PC reach 1024 (doubling each time nothing
fits), the trie is walked and a cut is added at the instruction that at least half of them
reach where the distinct runs of reads before its first read, or from it on, are more than
one and at most an eighth of the units; among those, the one with the fewest of both. This
separates state that varies independently (a voice's note position from its pulse-width
sweep, say), whose product would otherwise be stored unit by unit. A start PC whose units
pass 65536 with no cut found is emulated from then on. Adding a cut empties the memo.

## Closure

The state a call starts from is the RAM at `keep` (default all), or the `cells`, exactly as
`NativeMPU.orbit` defines it; registers are state only as `cells`. Each write, emulated or
replayed, updates a 64-bit hash of that state (a sum of per-address weights times bytes), and
the hashes seen are kept in a set. A call whose entry hash was seen is only a candidate: the
calls are run again from the first state, through the memo, to the earliest call with that
hash, and the two states are compared byte for byte. Only equal states end the orbit as
`"closed"`; a hash collision is counted (`collisions`) and the run goes on. Equal states
always hash alike, so the first repeat is never missed.

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
| `dropped` | units not stored because a stored unit ended elsewhere on the same reads (expected 0) |
| `collisions` | candidate repeats whose states differed |
| `bytes` | bytes held by the memo and the hash set |
| `roots` | `(pc, units)`: the start PCs holding most units |
| `fanout` | `(location, nodes, values)`: the locations that tell most held units apart |

## Measurements

`tools/memo_orbit.py --brute` on five C64 play routines from their post-init states, `$D000-$DFFF`
volatile and the rest of RAM the state, `grain="split"`, one run each on a shared host at load
average 29 (wall seconds; the brute-force column is `NativeMPU.orbit` in the same process):

| routine | calls to closure | units stored | units held | instructions emulated | bytes | memo s | brute s |
|---|---|---|---|---|---|---|---|
| Emomyst | 8,084 | 21,000 | 675 | 1,307,907 | 3.8 M | 0.13 | 0.16 |
| Incoherent Nightmare 2 | 13,380 | 51,140 | 5,468 | 1,198,442 | 2.4 M | 0.14 | 0.21 |
| Automatas | 149,025 | 317,148 | 9,560 | 3,774,865 | 7.3 M | 0.81 | 2.20 |
| Ghouls'n Ghosts, 32 songs | 79,359 at most | 159,990 | 5,229 at most | 10,520,166 | 2.4 M at most | 1.00 | 2.4 |
| Commando | 10,580,744 | 113,767 | 7,167 | 3,154,328 | 272 M | 24.1 | 163.8 |

Every row's end, states, access set and edge set equalled the brute-force orbit's. Commando's
bytes are the set of state hashes (256 MiB at 10.6 million states); its memo is under 2 MiB.

Commando's calls are almost all replayed: 228.7 million units, 21.6 per call, against 3.2
million instructions emulated of some 2.6 billion. The time is the replays themselves, about
100 ns each on that host. Its voices keep a note position, a vibrato phase, a pulse-width
sweep and several effect states that vary independently, so units small enough to recur are
only a handful of instructions long, and looking one up costs about what emulating it would.
Distinct read keys over its first 1.5 million calls, counted with nothing emulated unmemoised:
682,317 whole calls (`call` and `sub`: it has no subroutines) and 443,412 at `loop`; `split`
stored 113,763 units in all and held 7,163.
