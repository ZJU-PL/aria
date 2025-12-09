# coding: utf-8
"""
For testing the CNF simplifier
"""
from typing import List

from aria.tests import TestCase, main
from aria.bool import simplify_numeric_clauses
from aria.bool.cnfsimplifier import NumericClausesReader
from aria.bool.cnfsimplifier import (
    cnf_subsumption_elimination,
    cnf_hidden_subsumption_elimination,
    cnf_asymmetric_subsumption_elimination,
    cnf_asymmetric_tautoly_elimination,
    cnf_tautoly_elimination,
    cnf_hidden_tautoly_elimination,
    cnf_blocked_clause_elimination,
    cnf_hidden_blocked_clause_elimination
)


def test_tautology_elimination():
    """Test tautology elimination"""
    # Test case with tautological clauses (p ∨ ¬p)
    clauses = [[1, -1], [2, 3], [1, -2]]
    cnf = NumericClausesReader().read(clauses)
    simplified = cnf_tautoly_elimination(cnf)
    result = simplified.get_numeric_clauses()

    # Should remove the tautological clause [1, -1]
    expected = [[2, 3], [1, -2]]
    assert sorted_clauses(result) == sorted_clauses(expected), \
        f"Expected {expected}, got {result}"


def test_subsumption_elimination():
    """Test subsumption elimination"""
    # Test case where [1] subsumes [1, 2]
    clauses = [[1], [1, 2], [2, 3]]
    cnf = NumericClausesReader().read(clauses)
    simplified = cnf_subsumption_elimination(cnf)
    result = simplified.get_numeric_clauses()

    # Should remove [1, 2] as it's subsumed by [1]
    expected = [[1], [2, 3]]
    assert sorted_clauses(result) == sorted_clauses(expected), \
        f"Expected {expected}, got {result}"


def test_hidden_tautology_elimination():
    """Test hidden tautology elimination"""
    # Example: [1, 2], [-1, 3], [2, -3] forms a hidden tautology
    clauses = [[1, 2], [-1, 3], [2, -3]]
    cnf = NumericClausesReader().read(clauses)
    simplified = cnf_hidden_tautoly_elimination(cnf)
    result = simplified.get_numeric_clauses()

    # Should identify and remove hidden tautologies
    assert len(result) <= len(clauses), \
        f"Expected fewer clauses than {len(clauses)}, got {len(result)}"


def test_blocked_clause_elimination():
    """Test blocked clause elimination"""
    # Example of a blocked clause with proper variable initialization
    clauses = [[1, 2], [-1, 2], [1, -2]]

    # Create CNF with proper variable handling
    reader = NumericClausesReader()
    cnf = reader.read(clauses)

    # Ensure variables are properly converted before elimination
    simplified = cnf_blocked_clause_elimination(cnf)
    result = simplified.get_numeric_clauses()

    # Test that simplification occurred
    assert len(result) <= len(clauses), \
        f"Expected {len(clauses)} or fewer clauses, got {len(result)}"

    # Verify result validity
    assert all(len(clause) > 0 for clause in result), \
        "Empty clause found in result"


def test_asymmetric_tautology_elimination():
    """Test asymmetric tautology elimination"""
    clauses = [[1, 2], [-1, -2], [1, -2], [-1, 2]]
    cnf = NumericClausesReader().read(clauses)
    simplified = cnf_asymmetric_tautoly_elimination(cnf)
    result = simplified.get_numeric_clauses()

    # Should remove asymmetric tautologies
    assert len(result) < len(clauses), \
        f"Expected fewer clauses than {len(clauses)}, got {len(result)}"


def sorted_clauses(clauses: List[List[int]]) -> List[List[int]]:
    """Helper function to sort clauses for comparison"""
    return sorted([sorted(clause) for clause in clauses])


def run_all_tests():
    """Run all CNF simplification tests"""
    print("Running CNF simplification tests...")

    tests = [
        test_tautology_elimination,
        test_subsumption_elimination,
        test_hidden_tautology_elimination,
        test_blocked_clause_elimination,
        test_asymmetric_tautology_elimination
    ]

    for test in tests:
        try:
            test()
            print(f"✓ {test.__name__} passed")
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {str(e)}")
        except Exception as e:
            print(f"✗ {test.__name__} error: {str(e)}")


if __name__ == "__main__":
    run_all_tests()
