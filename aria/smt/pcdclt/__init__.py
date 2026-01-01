"""
Parallel CDCL(T) SMT Solver

A parallel implementation of the CDCL(T) algorithm that combines Boolean SAT solving
with theory reasoning to solve Satisfiability Modulo Theories problems.

Usage:
    from aria.smt.pcdclt import CDCLTSolver, solve

    # Simple function call
    result = solve(smt2_string, logic="QF_LRA")

    # Or use the solver class
    solver = CDCLTSolver()
    result = solver.solve_smt2_file("problem.smt2", logic="QF_LRA")
"""

from .solver import CDCLTSolver, solve
from .preprocessor import FormulaAbstraction
from .theory_solver import TheorySolver
from .config import (
    NUM_SAMPLES_PER_ROUND,
    MAX_T_CHECKING_PROCESSES,
    SIMPLIFY_CLAUSES,
    ENABLE_QUERY_LOGGING,
    WORKER_SHUTDOWN_TIMEOUT,
)

__all__ = [
    "CDCLTSolver",
    "solve",
    "FormulaAbstraction",
    "TheorySolver",
    "NUM_SAMPLES_PER_ROUND",
    "MAX_T_CHECKING_PROCESSES",
    "SIMPLIFY_CLAUSES",
    "ENABLE_QUERY_LOGGING",
    "WORKER_SHUTDOWN_TIMEOUT",
]
