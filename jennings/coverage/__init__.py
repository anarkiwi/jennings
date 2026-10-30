"""Code coverage for 6502 execution: per-address execute/operand/branch/data flags,
execution counts, dynamic control-transfer edges and interrupt entries.

``CoverageVM`` records inside the engine, so deity-informant's drivers
(``run_sub``/``run_irq``/``run_irq_driven``) are covered as well as ``CoverageMPU.step``.
Names load from :mod:`jennings.coverage._core` on first use, so importing a submodule such
as the pytest plugin stays light.
"""

import importlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from jennings.coverage._core import (
        EXEC,
        FALLTHRU,
        OPERAND,
        READ,
        RETURN,
        TAKEN,
        WRITE,
        CoverageMap,
        CoverageMPU,
        CoverageVM,
        call,
    )

__all__ = [
    "EXEC",
    "OPERAND",
    "TAKEN",
    "FALLTHRU",
    "READ",
    "WRITE",
    "RETURN",
    "CoverageMap",
    "CoverageVM",
    "CoverageMPU",
    "call",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    core = importlib.import_module("jennings.coverage._core")
    globals().update((n, getattr(core, n)) for n in __all__)
    return globals()[name]
