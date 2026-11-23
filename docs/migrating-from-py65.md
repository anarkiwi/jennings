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
- **`py65.monitor` is not provided.**
