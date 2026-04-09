"""Tests for new Boolean analysis and encoding helpers."""

from pysat.formula import CNF

from aria.bool.analysis import cnf_analysis_report
from aria.bool.encodings import (
    encode_at_least_one,
    encode_at_most_one_pairwise,
    encode_exactly_one_pairwise,
)


def test_cnf_analysis_report_summarizes_clause_shape():
    formula = CNF(from_clauses=[[1, -2], [2, 3], [-1]])

    report = cnf_analysis_report(formula)

    assert report["num_variables"] == 3
    assert report["num_clauses"] == 3
    assert report["clause_length_histogram"] == {1: 1, 2: 2}
    assert report["literal_balance"] == {
        "positive_literals": 3,
        "negative_literals": 2,
        "total_literals": 5,
    }
    assert report["variable_clause_degrees"] == {1: 2, 2: 2, 3: 1}


def test_pairwise_cardinality_encodings():
    assert encode_at_least_one([1, 2, 3]).clauses == [[1, 2, 3]]
    assert encode_at_most_one_pairwise([1, 2, 3]).clauses == [
        [-1, -2],
        [-1, -3],
        [-2, -3],
    ]
    assert encode_exactly_one_pairwise([1, 2, 3]).clauses == [
        [1, 2, 3],
        [-1, -2],
        [-1, -3],
        [-2, -3],
    ]
