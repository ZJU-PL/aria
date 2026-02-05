# coding: utf-8
"""
Optional Rust backend for CNF simplification.

When the ``cnfsimplifier-rs`` extension is installed (build with
``maturin develop -C aria/bool/cnfsimplifier_rs``), this module exposes
the same simplification API over numeric clauses (list of list of int).
All functions take and return ``List[List[int]]``.
"""

from typing import List

_rs = None


def _get_rust():
    global _rs
    if _rs is None:
        try:
            import cnfsimplifier_rs as _rs  # type: ignore[import-untyped]
        except ImportError:
            pass
    return _rs


def is_available() -> bool:
    """Return True if the Rust extension is installed."""
    return _get_rust() is not None


def simplify_numeric_clauses(clauses: List[List[int]]) -> List[List[int]]:
    """Subsumption elimination on numeric clauses (Rust)."""
    rs = _get_rust()
    if rs is None:
        raise RuntimeError("Rust backend not available; install cnfsimplifier-rs")
    return rs.simplify_numeric_clauses(clauses)


def cnf_tautology_elimination(clauses: List[List[int]]) -> List[List[int]]:
    """Remove tautological clauses (Rust)."""
    rs = _get_rust()
    if rs is None:
        raise RuntimeError("Rust backend not available; install cnfsimplifier-rs")
    return rs.cnf_tautology_elimination(clauses)


def cnf_subsumption_elimination(clauses: List[List[int]]) -> List[List[int]]:
    """Remove subsumed clauses (Rust)."""
    rs = _get_rust()
    if rs is None:
        raise RuntimeError("Rust backend not available; install cnfsimplifier-rs")
    return rs.cnf_subsumption_elimination(clauses)


def cnf_blocked_clause_elimination(clauses: List[List[int]]) -> List[List[int]]:
    """Remove blocked clauses (Rust)."""
    rs = _get_rust()
    if rs is None:
        raise RuntimeError("Rust backend not available; install cnfsimplifier-rs")
    return rs.cnf_blocked_clause_elimination(clauses)


def cnf_hidden_tautology_elimination(clauses: List[List[int]]) -> List[List[int]]:
    """Remove hidden tautological clauses (Rust)."""
    rs = _get_rust()
    if rs is None:
        raise RuntimeError("Rust backend not available; install cnfsimplifier-rs")
    return rs.cnf_hidden_tautology_elimination(clauses)


def cnf_hidden_subsumption_elimination(clauses: List[List[int]]) -> List[List[int]]:
    """Remove hidden subsumed clauses (Rust)."""
    rs = _get_rust()
    if rs is None:
        raise RuntimeError("Rust backend not available; install cnfsimplifier-rs")
    return rs.cnf_hidden_subsumption_elimination(clauses)


def cnf_hidden_blocked_clause_elimination(
    clauses: List[List[int]],
) -> List[List[int]]:
    """Remove hidden blocked clauses (Rust)."""
    rs = _get_rust()
    if rs is None:
        raise RuntimeError("Rust backend not available; install cnfsimplifier-rs")
    return rs.cnf_hidden_blocked_clause_elimination(clauses)


def cnf_asymmetric_tautology_elimination(
    clauses: List[List[int]],
) -> List[List[int]]:
    """Remove asymmetric tautological clauses (Rust)."""
    rs = _get_rust()
    if rs is None:
        raise RuntimeError("Rust backend not available; install cnfsimplifier-rs")
    return rs.cnf_asymmetric_tautology_elimination(clauses)


def cnf_asymmetric_subsumption_elimination(
    clauses: List[List[int]],
) -> List[List[int]]:
    """Remove asymmetric subsumed clauses (Rust)."""
    rs = _get_rust()
    if rs is None:
        raise RuntimeError("Rust backend not available; install cnfsimplifier-rs")
    return rs.cnf_asymmetric_subsumption_elimination(clauses)


def cnf_asymmetric_blocked_clause_elimination(
    clauses: List[List[int]],
) -> List[List[int]]:
    """Remove asymmetric blocked clauses (Rust)."""
    rs = _get_rust()
    if rs is None:
        raise RuntimeError("Rust backend not available; install cnfsimplifier-rs")
    return rs.cnf_asymmetric_blocked_clause_elimination(clauses)
