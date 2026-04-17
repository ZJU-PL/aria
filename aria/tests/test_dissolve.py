from pysat.formula import CNF

from aria.bool.dissolve import Dissolve, DissolveConfig
from aria.bool.dissolve.ubtree import UBTree
from aria.utils.types import SolverResult


def test_dissolve_reports_sat_for_simple_instance() -> None:
    cnf = CNF(from_clauses=[[1], [1, 2]])
    result = Dissolve(DissolveConfig(k_split_vars=1, max_rounds=5)).solve(cnf)

    assert result.result == SolverResult.SAT
    assert result.model is not None
    assert 1 in result.model


def test_dissolve_reports_unsat_for_simple_instance() -> None:
    cnf = CNF(from_clauses=[[1], [-1]])
    result = Dissolve(DissolveConfig(k_split_vars=1, max_rounds=5)).solve(cnf)

    assert result.result == SolverResult.UNSAT
    assert result.rounds >= 1
    assert result.unsat_core is None


def test_dissolve_keeps_base_unsat_core_when_no_splitting() -> None:
    cnf = CNF(from_clauses=[[1], [-1]])
    result = Dissolve(DissolveConfig(k_split_vars=0, max_rounds=2)).solve(cnf)

    assert result.result == SolverResult.UNSAT
    assert result.unsat_core is not None


def test_ubtree_normalizes_duplicate_clauses() -> None:
    ubtree = UBTree()

    first = ubtree.insert_clause([2, 1], 0)
    second = ubtree.insert_clause([1, 2], 0)

    assert first is second
    assert first.subsumed_by is None


def test_ubtree_respects_literal_signs_in_subsumption() -> None:
    ubtree = UBTree()

    negative = ubtree.insert_clause([-1], 0)
    positive = ubtree.insert_clause([1], 0)
    mixed_negative = ubtree.insert_clause([-1, 2], 0)
    mixed_positive = ubtree.insert_clause([1, 2], 0)

    assert negative.subsumed_by is None
    assert positive.subsumed_by is None
    assert mixed_negative.subsumed_by is negative
    assert mixed_positive.subsumed_by is positive
    assert negative is not positive
