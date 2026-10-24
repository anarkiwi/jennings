"""A player-agnostic 6502 superoptimiser: z3 semantics from deity-informant's p-code and a
minimal-sequence search over test vectors (numpy, or CUDA through CuPy) with z3 proofs."""

from jennings.superopt.search import Found, Problem, cuda_available, search
from jennings.superopt.symbolic import Semantics, Spec, SymbolicError, Table, Var, execute, prove

__all__ = [
    "Found",
    "Problem",
    "Semantics",
    "Spec",
    "SymbolicError",
    "Table",
    "Var",
    "cuda_available",
    "execute",
    "prove",
    "search",
]
