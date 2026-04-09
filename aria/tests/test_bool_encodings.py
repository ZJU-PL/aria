"""Tests for reusable Boolean encodings."""

from pysat.solvers import Solver

from aria.bool.encodings import (
    at_least_k_sequential_counter,
    at_most_k_sequential_counter,
    at_most_k_totalizer,
    exactly_k_totalizer,
    exactly_one,
    equivalent,
    implies,
    pb_at_least,
    pb_at_most,
    pb_equals,
    reify_and,
    reify_or,
)


def _is_sat(clauses, assumptions=None) -> bool:
    with Solver(name="cd", bootstrap_with=clauses) as solver:
        return solver.solve(assumptions=list(assumptions or []))


def test_exactly_one_blocks_zero_and_multi_selection() -> None:
    encoding = exactly_one([1, 2, 3])
    clauses = encoding.clauses

    assert _is_sat(clauses, assumptions=[1, -2, -3])
    assert not _is_sat(clauses, assumptions=[-1, -2, -3])
    assert not _is_sat(clauses, assumptions=[1, 2])


def test_reified_and_and_or_match_expected_models() -> None:
    and_clauses = reify_and(4, [1, 2])
    assert _is_sat(and_clauses, assumptions=[1, 2, 4])
    assert not _is_sat(and_clauses, assumptions=[1, -2, 4])
    assert _is_sat(and_clauses, assumptions=[1, -2, -4])

    or_clauses = reify_or(5, [1, 2])
    assert _is_sat(or_clauses, assumptions=[1, -5]) is False
    assert _is_sat(or_clauses, assumptions=[-1, -2, -5])
    assert not _is_sat(or_clauses, assumptions=[-1, -2, 5])


def test_implication_and_equivalence_gadgets() -> None:
    implication = implies(1, 2)
    equivalence = equivalent(3, 4)

    assert not _is_sat(implication, assumptions=[1, -2])
    assert _is_sat(implication, assumptions=[-1, -2])
    assert _is_sat(equivalence, assumptions=[3, 4])
    assert not _is_sat(equivalence, assumptions=[3, -4])


def test_sequential_counter_cardinality_bounds() -> None:
    at_most = at_most_k_sequential_counter([1, 2, 3], 1)
    assert _is_sat(at_most.clauses, assumptions=[1, -2, -3])
    assert not _is_sat(at_most.clauses, assumptions=[1, 2])

    at_least = at_least_k_sequential_counter([1, 2, 3], 2)
    assert _is_sat(at_least.clauses, assumptions=[1, 2, -3])
    assert not _is_sat(at_least.clauses, assumptions=[1, -2, -3])


def test_totalizer_cardinality_bounds() -> None:
    at_most = at_most_k_totalizer([1, 2, 3, 4], 2)
    assert _is_sat(at_most.clauses, assumptions=[1, 2, -3, -4])
    assert not _is_sat(at_most.clauses, assumptions=[1, 2, 3])

    exactly = exactly_k_totalizer([1, 2, 3], 2)
    assert _is_sat(exactly.clauses, assumptions=[1, 2, -3])
    assert not _is_sat(exactly.clauses, assumptions=[1, -2, -3])
    assert not _is_sat(exactly.clauses, assumptions=[1, 2, 3])


def test_weighted_pseudo_boolean_encodings() -> None:
    at_most = pb_at_most([1, 2], [2, 1], 2)
    assert _is_sat(at_most.clauses, assumptions=[1, -2])
    assert not _is_sat(at_most.clauses, assumptions=[1, 2])

    at_least = pb_at_least([1, 2, 3], [2, 1, 1], 3)
    assert _is_sat(at_least.clauses, assumptions=[1, 2, -3])
    assert not _is_sat(at_least.clauses, assumptions=[1, -2, -3])

    equals = pb_equals([1, 2, 3], [2, 1, 1], 2)
    assert _is_sat(equals.clauses, assumptions=[1, -2, -3])
    assert not _is_sat(equals.clauses, assumptions=[1, 2, -3])
