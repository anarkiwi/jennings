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
