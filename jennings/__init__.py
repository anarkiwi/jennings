"""jennings: a py65-compatible 6502/6510 toolkit with NMOS illegal opcodes.

Execution is delegated to deity-informant's hardware-validated engine, so every
opcode py65 stubs as ``inst_not_implemented`` runs byte- and cycle-exact.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("jennings")
except PackageNotFoundError:  # pragma: no cover - source tree without install metadata
    __version__ = "0+unknown"

__all__ = ["__version__"]
