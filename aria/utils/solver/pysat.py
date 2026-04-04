"""Helpers for solving DIMACS CNF formulas with PySAT."""

import logging

from pysat.formula import CNF
from pysat.solvers import Solver

logger = logging.getLogger(__name__)

sat_solvers_in_pysat = [
    "cd",
    "cd15",
    "gc3",
    "gc4",
    "g3",
    "g4",
    "lgl",
    "mcb",
    "mpl",
    "mg3",
    "mc",
    "m22",
    "mgh",
]


def solve_with_sat_solver(dimacs_str: str, solver_name: str) -> str:
    """Solve a given DIMACS CNF formula using a PySAT backend."""
    assert solver_name in sat_solvers_in_pysat
    print(f"Calling SAT solver {solver_name}")
    pos = CNF(from_string=dimacs_str)
    aux = Solver(name=solver_name, bootstrap_with=pos)
    if aux.solve():
        return "sat"
    return "unsat"
