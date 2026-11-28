# Memoising orbit: measurements

The memoising orbit ([memo.md](memo.md)) against the brute-force orbit on real play routines.

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

The Automatas count of calls to closure is for the tool's state definition: all RAM outside
`$D000-$DFFF`, and no registers. Another choice of `keep` or `cells` closes after another count.

Commando's calls are almost all replayed: 228.7 million units, 21.6 per call, against 3.2
million instructions emulated of some 2.6 billion. The time is the replays themselves, about
100 ns each on that host. Its voices keep a note position, a vibrato phase, a pulse-width
sweep and several effect states that vary independently, so units small enough to recur are
only a handful of instructions long, and looking one up costs about what emulating it would.
Distinct read keys over its first 1.5 million calls, counted with nothing emulated unmemoised:
682,317 whole calls (`call` and `sub`: it has no subroutines) and 443,412 at `loop`; `split`
stored 113,763 units in all and held 7,163.

## With a fibre

`tools/memo_orbit.py --fibre 0x1b17,0x1aff` on the Commando state above: its two slowest
counters set apart by hand, three runs each on the same shared host (72 CPUs) at load averages
10.1, 7.9 and 5.8, wall seconds for `jennings.memo.orbit` and what it builds:

| | calls to closure | seconds | calls run by the tape | calls run to learn | bytes |
|---|---|---|---|---|---|
| no fibre | 10,580,744 | 15.97, 16.37, 15.74 | | | 272 M |
| `$1B17,$1AFF` | 10,580,744 | 6.66, 6.66, 6.63 | 9,243,681 | 9,188 | 319 M |

The base repeats every 661,248 calls and the tape is run from call 1,327,875: the calls before
are those in which the period is found and then recorded, at the plain engine's pace. A call run
by the tape has 1.09 segments, each looked up; 9,188 calls were run as ordinary calls to learn
them, and 6,305 more to reach their entry states from a frame. Every result (end, states, access
set, edge set, memory and registers) equalled the run without a fibre, which equalled the
brute-force orbit.

The tape is 46.5 MB: 20 bytes per call (13.2 MB), 16 per segment (11.6 MB), 8 per call for its
state's hash (5.3 MB) and the frames, within their 16 MiB. The set of state hashes is the other
256 MiB, as without a fibre.

Other fibres on the same state, one run each (the tool's `--fibre`, each checked the same way):

| fibre | base period | tape run from | calls run by the tape | calls run to learn | tapes given up | seconds |
|---|---|---|---|---|---|---|
| `$1B17` | 2,644,992 | 5,295,364 | 5,281,885 | 3,495 | 0 | 15.31 |
| `$1B17,$1AFF,$1496` | 661,248 | 1,327,875 | 8,903,485 | 349,384 | 0 | 11.74 |
| `$1B17,$1AFF,$1479` | 94,464 | never | 0 | 0 | 7 | 17.29 |
| `$F0,$F1,$148B,$1499` | none found | never | 0 | 0 | 2 | 17.26 |
| `$148E,$149A` (fast cells) | none found | never | 0 | 0 | 2 | 18.03 |
| `$5000` (never written) | none found | never | 0 | 0 | 2 | 17.28 |

A fibre that does not fit costs up to an eighth over no fibre here: the search for a period and
the recordings that come to nothing. `$1479` is read by code that writes the base, so every tape
recorded with it fails to join; the wait before the next doubles each time.

On the other four routines of the table above (Ghouls'n Ghosts' 32 songs each), four fibres
each, among them their slowest cells, their fastest and cells never written: every result
equalled the run without a fibre. None of them is run by a tape: each closes before one period
of any base tried has been recorded twice over.

Without a fibre the engine is the one measured above. `perf stat` of the core's `memo` example
on the Commando state, this engine against the one before it, five runs each, interleaved, at
load averages 2.0 to 4.5: 81.45 to 81.51 against 81.43 to 81.45 billion instructions, 39.5 to
40.4 against 39.7 to 40.7 billion cycles, 14.11 to 14.42 against 14.18 to 14.55 seconds.
