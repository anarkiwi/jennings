"""Code coverage for 6502 execution: per-address execute/operand/branch/data flags,
execution counts, dynamic control-transfer edges and interrupt entries.

``CoverageVM`` records inside the engine, so deity-informant's drivers
(``run_sub``/``run_irq``/``run_irq_driven``) are covered as well as ``CoverageMPU.step``.
Names load from :mod:`jennings.coverage._core` (``one_way`` from
:mod:`jennings.coverage.cfg`) on first use, so importing a submodule such as the pytest plugin
stays light.
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
    from jennings.coverage.cfg import one_way

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
    "one_way",
]

_FROM = {"one_way": "cfg"}


def __getattr__(name):
    if name not in __all__:
        raise AttributeError("module %r has no attribute %r" % (__name__, name))
    src = _FROM.get(name, "_core")
    mod = importlib.import_module("jennings.coverage." + src)
    globals().update((n, getattr(mod, n)) for n in __all__ if _FROM.get(n, "_core") == src)
    return globals()[name]
