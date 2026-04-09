"""Tests for Boolean structural analysis helpers."""

from pysat.formula import CNF

from aria.bool.analysis import analyze_cnf, analyze_qdimacs
from aria.bool.qbf import parse_qdimacs_string


def test_analyze_cnf_reports_basic_structure() -> None:
    formula = CNF(from_clauses=[[1, 2], [-1, 3], [3]])

    summary = analyze_cnf(formula)

    assert summary.num_vars == 3
    assert summary.num_clauses == 3
    assert summary.min_clause_size == 1
    assert summary.max_clause_size == 2
    assert abs(summary.mean_clause_size - (5.0 / 3.0)) < 1e-9
    assert summary.unit_clauses == 1
    assert summary.binary_clauses == 2
    assert summary.horn_clauses == 2
    assert summary.primal_edges == 2
    assert summary.primal_components == 1


def test_analyze_qdimacs_reports_prefix_and_span_metrics() -> None:
    instance = parse_qdimacs_string(
        """
        c sample
        p cnf 4 2
        a 1 2 0
        e 3 0
        e 4 0
        1 3 0
        -2 4 0
        """
    )

    summary = analyze_qdimacs(instance)

    assert summary.num_vars == 4
    assert summary.num_clauses == 2
    assert summary.quantifier_blocks == 2
    assert summary.alternation_depth == 2
    assert summary.universal_vars == 2
    assert summary.existential_vars == 2
    assert summary.free_vars == 0
    assert summary.prefix_pattern == "ae"
    assert summary.max_level_span == 1
    assert abs(summary.mean_clause_size - 2.0) < 1e-9
