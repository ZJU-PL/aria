from typing import List

from .io import NumericClausesReader, PySATCNFReader
from .simplifier import (
    cnf_subsumption_elimination,
    cnf_hidden_subsumption_elimination,
    cnf_asymmetric_subsumption_elimination,
    cnf_asymmetric_tautoly_elimination,
    cnf_tautoly_elimination,
    cnf_hidden_tautoly_elimination,
    cnf_blocked_clause_elimination,
    cnf_hidden_blocked_clause_elimination,
)

try:
    from . import rust_backend as _rust_backend
    _rust_available = _rust_backend.is_available()
except ImportError:
    _rust_available = False


def simplify_numeric_clauses(clauses: List[List[int]]) -> List[List[int]]:
    """
    Simplify numerical clauses (subsumption elimination).
    Uses the Rust backend when ``cnfsimplifier-rs`` is installed, else Python.

    :param clauses: numerical clauses
    :return: simplified clauses
    """
    if _rust_available:
        return _rust_backend.simplify_numeric_clauses(clauses)
    cnf = NumericClausesReader().read(clauses)
    new_cnf = cnf_subsumption_elimination(cnf)  # why only subsumption?
    return new_cnf.get_numeric_clauses()


def rust_backend_available() -> bool:
    """Return True if the Rust CNF simplifier extension is installed."""
    return _rust_available
