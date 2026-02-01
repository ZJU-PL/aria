"""Unsat core computation: MUS/MSS enumeration and extraction.

Provides UnsatCoreComputer and Algorithm (MARCO, MUSX, OPTUX) for computing
minimal unsatisfiable cores and maximal satisfying subsets.
"""

from aria.unsat_core.unsat_core import (
    Algorithm,
    UnsatCoreComputer,
    UnsatCoreResult,
)

__all__ = ["Algorithm", "UnsatCoreComputer", "UnsatCoreResult"]
