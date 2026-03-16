"""
Termination verification package for bit-vector programs.

This package provides template-based termination and non-termination verification
using ranking functions and recurrence sets.
"""

from .result_types import TerminationResult, NonTerminationResult
from .termination_prover import TerminationProver
from .api import (
    prove_termination_with_ranking_functions,
    prove_non_termination_with_recurrence_sets,
    analyze_termination,
)

__all__ = [
    "TerminationResult",
    "NonTerminationResult",
    "TerminationProver",
    "prove_termination_with_ranking_functions",
    "prove_non_termination_with_recurrence_sets",
    "analyze_termination",
]
