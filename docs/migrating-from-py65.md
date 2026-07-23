# Migrating from py65

jennings mirrors py65's module layout, so migration is a mechanical import swap:

```python
from py65.devices.mpu6502 import MPU        # ->  from jennings.devices.mpu6502 import MPU
from py65.memory import ObservableMemory    # ->  from jennings.memory import ObservableMemory
from py65.disassembler import Disassembler  # ->  from jennings.disassembler import Disassembler
from py65.assembler import Assembler        # ->  from jennings.assembler import Assembler
from py65.utils.addressing import AddressParser  # -> from jennings.utils.addressing ...
```

Then replace the `py65` dependency with `jennings` in `pyproject.toml` /
`requirements.txt`. jennings pulls `deity-informant` (a pure-Python, no-deps
engine) transitively.

## Behavioural differences

- **Illegal opcodes now execute.** Code that relied on py65 silently skipping an
  illegal (a 2-byte, 0-cycle no-op) will instead run it. This is the point of the
  migration, but audit any code that fed py65 illegal bytes expecting a skip.
- **Ad-hoc illegal patches become obsolete.** Any local monkey-patch that mutated
  `mpu.instruct` / `mpu.cycletime` to add illegals (e.g. a `patch_illegals()`
  helper) should be deleted — jennings has them built in, and mutating those
  tables has no effect on execution.
- **No decimal mode.** ADC/SBC ignore the D flag (the C64 domain runs `CLD`).
- **`py65.monitor` is not provided.**

## anarkiwi repositories to migrate

Direct py65 users:

| Repo | What to change |
|---|---|
| pysidtracker | swap imports in `emu.py`; delete `patch_illegals()` |
| preframr-tokens | swap import in `bacc/sidemu.py`; dep py65 -> jennings |
| pysoundmonitor | swap `MPU` + `ObservableMemory` imports in `player.py` |
| pydexomizer | swap import in `sfx.py` |
| sentinel-solver | swap imports in `sentinel/tests/oracle.py` |
| re-sentinel | swap imports in `scripts/_emu.py`, `roundtrip_test.py` (uses Assembler) |
| tumbler-snapper | swap imports in `tools/disasm.py` (uses Disassembler) |
| sid-wizard | swap imports in `test/sidwizard/run_protocol_test.py` |

Transitive users (via pysidtracker) generally need no code change, but drop any
explicit `py65` pin: pyjch, pygoattracker, pydefmon, pydmcsid, pysidwizard,
pymusicassembler, pyfuturecomposer.

Keep py65 in **deity-informant** — there it is the external differential oracle
for legal opcodes, and jennings depends on deity-informant (switching would be
circular).
