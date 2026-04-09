# coding: utf-8
"""Tests for prime implicant and implicate enumeration."""

from itertools import combinations
from typing import Iterable, List

from pysat.formula import CNF
from pysat.solvers import Solver

from aria.bool.prime import enumerate_prime_implicants, enumerate_prime_implicates


def _is_satisfiable_with_assumptions(formula: CNF, assumptions: Iterable[int]) -> bool:
    with Solver(name="cd", bootstrap_with=formula.clauses) as solver:
        return solver.solve(assumptions=list(assumptions))


def _assert_all_prime_implicants(formula: CNF, implicants: List[List[int]]) -> None:
    seen = set()
    for implicant in implicants:
        normalized = tuple(sorted(implicant, key=lambda lit: (abs(lit), lit < 0)))
        assert normalized not in seen
        seen.add(normalized)
        assert not _is_satisfiable_with_assumptions(formula.negate(), implicant)
        for size in range(len(implicant)):
            for subset in combinations(implicant, size):
                assert subset == tuple() or _is_satisfiable_with_assumptions(
                    formula.negate(), subset
                )


def _assert_all_prime_implicates(formula: CNF, implicates: List[List[int]]) -> None:
    seen = set()
    for implicate in implicates:
        normalized = tuple(sorted(implicate, key=lambda lit: (abs(lit), lit < 0)))
        assert normalized not in seen
        seen.add(normalized)
        complement_term = [-literal for literal in implicate]
        assert not _is_satisfiable_with_assumptions(formula, complement_term)
        for size in range(len(implicate)):
            for subset in combinations(implicate, size):
                reduced_term = [-literal for literal in subset]
                assert _is_satisfiable_with_assumptions(formula, reduced_term)


def test_prime_implicants_for_disjunction():
    formula = CNF(from_clauses=[[1, 2]])
    assert enumerate_prime_implicants(formula) == [[1], [2]]


def test_prime_implicates_for_conjunction():
    formula = CNF(from_clauses=[[1], [2]])
    assert enumerate_prime_implicates(formula) == [[1], [2]]


def test_prime_implicants_and_implicates_for_xor():
    formula = CNF(from_clauses=[[1, 2], [-1, -2]])
    assert enumerate_prime_implicants(formula) == [[-1, 2], [1, -2]]
    assert enumerate_prime_implicates(formula) == [[-1, -2], [1, 2]]


def test_prime_enumeration_handles_constant_formulas():
    assert enumerate_prime_implicants(CNF()) == [[]]
    assert enumerate_prime_implicates(CNF()) == []
    assert enumerate_prime_implicants(CNF(from_clauses=[[]])) == []
    assert enumerate_prime_implicates(CNF(from_clauses=[[]])) == [[]]


def test_prime_enumeration_minimality_on_mixed_formula():
    formula = CNF(from_clauses=[[1, 2], [-1, 3]])
    implicants = enumerate_prime_implicants(formula)
    implicates = enumerate_prime_implicates(formula)

    assert implicants == [[-1, 2], [1, 3], [2, 3]]
    assert implicates == [[-1, 3], [1, 2], [2, 3]]

    _assert_all_prime_implicants(formula, implicants)
    _assert_all_prime_implicates(formula, implicates)
