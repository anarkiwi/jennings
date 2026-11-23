"""jennings: a py65-compatible 6502/6510 toolkit with NMOS illegal opcodes.

Execution is delegated to deity-informant's hardware-validated engine, so every
opcode py65 stubs as ``inst_not_implemented`` runs byte- and cycle-exact. The stable
API (:data:`EXPORTS`) is importable from here, each name loaded on first use.
"""

import importlib
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("jennings")
except PackageNotFoundError:  # pragma: no cover - source tree without install metadata
    __version__ = "0+unknown"

#: Top-level name -> defining module.
EXPORTS = {
    "MPU": "jennings.devices.mpu6502",
    "Assembler": "jennings.assembler",
    "Disassembler": "jennings.disassembler",
    "AddressParser": "jennings.utils.addressing",
    "ObservableMemory": "jennings.memory",
    "NativeMemory": "jennings.memory",
    "OPCODES": "jennings.opcodes",
    "Instruction": "jennings.decoding",
    "Roles": "jennings.decoding",
    "UNKNOWN": "jennings.decoding",
    "accesses": "jennings.decoding",
    "as_mask": "jennings.decoding",
    "as_values": "jennings.decoding",
    "decode": "jennings.decoding",
    "decode_many": "jennings.decoding",
    "roles": "jennings.decoding",
    "taken": "jennings.decoding",
    "transfer": "jennings.decoding",
}

__all__ = ["EXPORTS", "__version__", *EXPORTS]


def __getattr__(name):
    if name not in EXPORTS:
        raise AttributeError("module 'jennings' has no attribute %r" % name)
    value = getattr(importlib.import_module(EXPORTS[name]), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(EXPORTS))
