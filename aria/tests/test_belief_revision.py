"""Tests for belief-base revision and contraction."""

from typing import cast

import z3

from aria.abduction import (
    RevisionOperator,
    contract_belief_base,
    enumerate_optimal_contractions,
    enumerate_optimal_revisions,
    expand_belief_base,
    revise_belief_base,
)


def _is_sat(formulas) -> bool:
    solver = z3.Solver()
    solver.add(*formulas)
    return solver.check() == z3.sat


def _expr(formula: object) -> z3.ExprRef:
    return cast(z3.ExprRef, formula)


def test_expand_belief_base_appends_new_belief() -> None:
    a = z3.Bool("a")
    b = z3.Bool("b")

    expanded = expand_belief_base([a], b)

    assert expanded == [a, b]


def test_revision_keeps_consistent_base() -> None:
    x = z3.Int("x")
    belief_base = [x >= 0, x <= 10]
    new_belief = x >= 3

    result = revise_belief_base(belief_base, new_belief)

    assert result.kept_indices == [0, 1]
    assert result.removed_indices == []
    assert len(result.result_base) == 3
    assert _is_sat(result.result_base)


def test_revision_drops_conflicting_belief() -> None:
    x = z3.Int("x")
    belief_base = [x >= 0, x <= 1, x >= 5]
    new_belief = x <= 0

    result = revise_belief_base(belief_base, new_belief)

    assert result.kept_indices == [0, 1]
    assert result.removed_indices == [2]
    assert result.incision_indices == [2]
    assert result.incision_cost == 1
    assert result.conflict_sets == [[2]]
    assert len(result.rank_strata) == 1
    assert result.rank_strata[0].removed_indices == [2]
    assert _is_sat(result.result_base)


def test_weighted_revision_prefers_heavier_belief() -> None:
    x = z3.Int("x")
    belief_base = [x >= 5, x <= 1]
    new_belief = x >= 0

    result = revise_belief_base(belief_base, new_belief, weights=[5, 1])

    assert result.kept_indices == [0]
    assert result.removed_indices == [1]
    assert result.objective_value == 5
    assert _is_sat(result.result_base)


def test_contract_base_by_formula() -> None:
    a = z3.Bool("a")
    b = z3.Bool("b")
    belief_base = [a, z3.Implies(a, b)]

    result = contract_belief_base(belief_base, b)

    assert len(result.kept_indices) == 1
    assert len(result.removed_indices) == 1
    solver = z3.Solver()
    solver.add(*result.result_base)
    solver.add(z3.Not(b))
    assert solver.check() == z3.sat
    assert result.conflict_sets == [[0, 1]]
    assert result.incision_indices == [1]


def test_lexicographic_contraction_prefers_earlier_beliefs() -> None:
    a = z3.Bool("a")
    b = z3.Bool("b")
    belief_base = [a, z3.Implies(a, b)]

    result = contract_belief_base(
        belief_base,
        b,
        operator=RevisionOperator.LEXICOGRAPHIC,
    )

    assert result.kept_indices == [0]
    assert result.removed_indices == [1]
    assert result.incision_indices == [1]
    assert result.operator == "lexicographic"


def test_kernel_contraction_uses_weighted_incision() -> None:
    a = z3.Bool("a")
    b = z3.Bool("b")
    c = z3.Bool("c")
    belief_base = [a, b, c, z3.Implies(a, z3.Not(b)), z3.Implies(b, z3.Not(c))]

    result = contract_belief_base(
        belief_base,
        _expr(z3.Or(z3.Not(a), z3.Not(b), z3.Not(c))),
        weights=[10, 10, 10, 1, 1],
        operator=RevisionOperator.KERNEL,
    )

    assert result.kept_indices == [0, 1, 2]
    assert result.removed_indices == [3, 4]
    assert result.incision_indices == [3, 4]
    assert result.incision_cost == 2
    assert sorted(result.conflict_sets) == [[3], [4]]


def test_enumerate_optimal_revisions_returns_alternatives() -> None:
    a = z3.Bool("a")
    b = z3.Bool("b")
    belief_base = [a, b]
    new_belief = _expr(z3.Not(z3.And(a, b)))

    results = enumerate_optimal_revisions(belief_base, new_belief)
    kept_sets = {tuple(result.kept_indices) for result in results}

    assert kept_sets == {(0,), (1,)}
    assert all(_is_sat(result.result_base) for result in results)
    assert all(result.objective_value == 1 for result in results)


def test_lexicographic_revision_prefers_earlier_beliefs() -> None:
    a = z3.Bool("a")
    b = z3.Bool("b")
    belief_base = [a, b]
    new_belief = _expr(z3.Not(z3.And(a, b)))

    result = revise_belief_base(
        belief_base,
        new_belief,
        operator=RevisionOperator.LEXICOGRAPHIC,
    )

    assert result.kept_indices == [0]
    assert result.removed_indices == [1]
    assert result.incision_indices == [1]
    assert result.operator == "lexicographic"
    assert result.conflict_sets == [[0, 1]]


def test_kernel_revision_uses_conflict_sets() -> None:
    x = z3.Int("x")
    belief_base = [x >= 0, x <= 1, x >= 5]
    new_belief = x <= 0

    result = revise_belief_base(
        belief_base,
        new_belief,
        operator=RevisionOperator.KERNEL,
    )

    assert result.kept_indices == [0, 1]
    assert result.removed_indices == [2]
    assert result.incision_indices == [2]
    assert result.operator == "kernel"
    assert result.conflict_sets == [[2]]
    assert _is_sat(result.result_base)


def test_kernel_revision_respects_weighted_incision() -> None:
    a = z3.Bool("a")
    b = z3.Bool("b")
    c = z3.Bool("c")
    belief_base = [a, b, c, z3.Not(z3.And(a, b)), z3.Not(z3.And(b, c))]
    new_belief = _expr(z3.And(a, b, c))

    result = revise_belief_base(
        belief_base,
        new_belief,
        weights=[10, 10, 10, 1, 1],
        operator=RevisionOperator.KERNEL,
    )

    assert result.kept_indices == [0, 1, 2]
    assert result.removed_indices == [3, 4]
    assert result.incision_indices == [3, 4]
    assert sorted(result.conflict_sets) == [[3], [4]]


def test_partial_order_revision_uses_rank_strata() -> None:
    a = z3.Bool("a")
    b = z3.Bool("b")
    c = z3.Bool("c")
    belief_base = [a, b, c]
    new_belief = _expr(z3.Not(z3.And(a, b)))

    result = revise_belief_base(
        belief_base,
        new_belief,
        ranks=[0, 1, 1],
        operator=RevisionOperator.PARTIAL_ORDER,
    )

    assert result.kept_indices == [0, 2]
    assert result.incision_indices == [1]
    assert result.incision_cost == 1
    assert result.operator == "partial_order"
    assert [summary.rank for summary in result.rank_strata] == [0, 1]
    assert result.rank_strata[0].kept_indices == [0]
    assert result.rank_strata[1].kept_indices == [2]
    assert result.rank_strata[1].removed_indices == [1]


def test_epistemic_revision_prefers_higher_ranked_beliefs() -> None:
    a = z3.Bool("a")
    b = z3.Bool("b")
    belief_base = [a, b]
    new_belief = _expr(z3.Not(z3.And(a, b)))

    result = revise_belief_base(
        belief_base,
        new_belief,
        ranks=[1, 0],
        operator=RevisionOperator.EPISTEMIC,
    )

    assert result.kept_indices == [1]
    assert result.incision_indices == [0]
    assert result.incision_cost == 1
    assert result.operator == "epistemic"
    assert [summary.rank for summary in result.rank_strata] == [0, 1]
    assert result.rank_strata[0].kept_indices == [1]
    assert result.rank_strata[1].removed_indices == [0]


def test_epistemic_contraction_prefers_higher_ranked_beliefs() -> None:
    a = z3.Bool("a")
    b = z3.Bool("b")
    belief_base = [a, z3.Implies(a, b)]

    result = contract_belief_base(
        belief_base,
        b,
        ranks=[0, 1],
        operator=RevisionOperator.EPISTEMIC,
    )

    assert result.kept_indices == [0]
    assert result.incision_indices == [1]
    assert result.incision_cost == 1
    assert result.operator == "epistemic"


def test_non_max_retention_enumeration_returns_single_result() -> None:
    a = z3.Bool("a")
    b = z3.Bool("b")
    belief_base = [a, b]
    new_belief = _expr(z3.Not(z3.And(a, b)))

    results = enumerate_optimal_revisions(
        belief_base,
        new_belief,
        operator=RevisionOperator.LEXICOGRAPHIC,
    )

    assert len(results) == 1
    assert results[0].kept_indices == [0]


def test_enumerate_kernel_revisions_returns_all_optimal_incisions() -> None:
    a = z3.Bool("a")
    b = z3.Bool("b")
    c = z3.Bool("c")
    belief_base = [a, b, c]
    new_belief = _expr(z3.Not(z3.And(a, b, c)))

    results = enumerate_optimal_revisions(
        belief_base,
        new_belief,
        operator=RevisionOperator.KERNEL,
    )

    kept_sets = {tuple(result.kept_indices) for result in results}
    assert kept_sets == {(1, 2), (0, 2), (0, 1)}
    assert all(result.operator == "kernel" for result in results)
    assert {tuple(result.incision_indices) for result in results} == {
        (0,),
        (1,),
        (2,),
    }


def test_enumerate_kernel_contractions_returns_all_optimal_incisions() -> None:
    a = z3.Bool("a")
    b = z3.Bool("b")
    c = z3.Bool("c")
    belief_base = [a, b, c]
    target = _expr(z3.And(a, b, c))

    results = enumerate_optimal_contractions(
        belief_base,
        target,
        operator=RevisionOperator.KERNEL,
    )

    kept_sets = {tuple(result.kept_indices) for result in results}
    assert kept_sets == {(1, 2), (0, 2), (0, 1)}
    assert all(result.operator == "kernel" for result in results)
    assert {tuple(result.incision_indices) for result in results} == {
        (0,),
        (1,),
        (2,),
    }
