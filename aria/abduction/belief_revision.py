"""Belief-base revision and contraction utilities.

This module implements belief change for finite belief bases represented as
lists of Z3 formulas. The operations are consistency-based rather than defined
over deductively closed belief sets.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Set, Union, cast

import z3

from aria.unsat_core import enumerate_minimal_unsat_subsets


class RevisionOperator(Enum):
    """Supported belief revision operators."""

    MAX_RETENTION = "max_retention"
    KERNEL = "kernel"
    LEXICOGRAPHIC = "lexicographic"
    PARTIAL_ORDER = "partial_order"
    EPISTEMIC = "epistemic"

    @classmethod
    def from_value(cls, value: Union["RevisionOperator", str]) -> "RevisionOperator":
        if isinstance(value, cls):
            return value
        normalized = str(value).lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "maxichoice": cls.MAX_RETENTION,
            "max_retention": cls.MAX_RETENTION,
            "kernel": cls.KERNEL,
            "lexicographic": cls.LEXICOGRAPHIC,
            "lex": cls.LEXICOGRAPHIC,
            "partial_order": cls.PARTIAL_ORDER,
            "partial": cls.PARTIAL_ORDER,
            "epistemic": cls.EPISTEMIC,
            "epistemic_rank": cls.EPISTEMIC,
            "ranked": cls.EPISTEMIC,
        }
        if normalized not in aliases:
            raise ValueError(f"Unsupported revision operator: {value}")
        return aliases[normalized]


@dataclass(frozen=True)
class RankStratumSummary:
    """Summary of one belief-rank stratum."""

    rank: int
    belief_indices: List[int]
    kept_indices: List[int]
    removed_indices: List[int]
    kept_weight: int
    removed_weight: int


@dataclass(frozen=True)
class BeliefRevisionResult:
    """Result of belief-base change."""

    kept_indices: List[int]
    removed_indices: List[int]
    kept_beliefs: List[z3.ExprRef]
    removed_beliefs: List[z3.ExprRef]
    result_base: List[z3.ExprRef]
    objective_value: int
    operator: str
    conflict_sets: List[List[int]]
    conflict_belief_sets: List[List[z3.ExprRef]]
    incision_indices: List[int]
    incision_beliefs: List[z3.ExprRef]
    incision_cost: int
    rank_strata: List[RankStratumSummary]


def expand_belief_base(
    belief_base: Sequence[z3.ExprRef], new_belief: z3.ExprRef
) -> List[z3.ExprRef]:
    """Return the belief base expanded with a new belief."""

    return list(belief_base) + [new_belief]


def revise_belief_base(
    belief_base: Sequence[z3.ExprRef],
    new_belief: z3.ExprRef,
    weights: Optional[Sequence[int]] = None,
    ranks: Optional[Sequence[int]] = None,
    operator: Union[RevisionOperator, str] = RevisionOperator.MAX_RETENTION,
    max_conflicts: Optional[int] = None,
) -> BeliefRevisionResult:
    """Revise a finite belief base by a new belief."""

    _validate_weights(belief_base, weights)
    _validate_ranks(belief_base, ranks)
    hard_new_belief: List[z3.ExprRef] = [new_belief]
    if _check_unsat(hard_new_belief):
        raise ValueError("Cannot revise by an inconsistent belief")

    revision_operator = RevisionOperator.from_value(operator)
    conflict_sets = _conflicts_with_new_belief(
        belief_base, new_belief, max_conflicts=max_conflicts
    )

    if revision_operator == RevisionOperator.MAX_RETENTION:
        return _optimize_subset(
            formulas=belief_base,
            hard_constraints=hard_new_belief,
            trailing_formulas=hard_new_belief,
            weights=weights,
            operator=revision_operator,
            conflict_sets=conflict_sets,
        )

    if revision_operator == RevisionOperator.LEXICOGRAPHIC:
        return _lexicographic_revision(
            belief_base=belief_base,
            new_belief=new_belief,
            conflict_sets=conflict_sets,
        )

    if revision_operator in (RevisionOperator.PARTIAL_ORDER, RevisionOperator.EPISTEMIC):
        return _ranked_change(
            belief_base=belief_base,
            hard_constraints=hard_new_belief,
            trailing_formulas=hard_new_belief,
            weights=weights,
            ranks=ranks,
            operator=revision_operator,
            conflict_sets=conflict_sets,
        )

    return _kernel_revision(
        belief_base=belief_base,
        new_belief=new_belief,
        weights=weights,
        conflict_sets=conflict_sets,
    )


def contract_belief_base(
    belief_base: Sequence[z3.ExprRef],
    target_belief: z3.ExprRef,
    weights: Optional[Sequence[int]] = None,
    ranks: Optional[Sequence[int]] = None,
    operator: Union[RevisionOperator, str] = RevisionOperator.MAX_RETENTION,
    max_conflicts: Optional[int] = None,
) -> BeliefRevisionResult:
    """Contract a finite belief base by a target belief."""

    _validate_weights(belief_base, weights)
    _validate_ranks(belief_base, ranks)
    negated_target = _negate(target_belief)
    hard_constraints: List[z3.ExprRef] = [negated_target]
    if _check_unsat(hard_constraints):
        raise ValueError("Cannot contract by a valid belief")

    contraction_operator = RevisionOperator.from_value(operator)
    conflict_sets = _conflicts_with_new_belief(
        belief_base, negated_target, max_conflicts=max_conflicts
    )

    if contraction_operator == RevisionOperator.MAX_RETENTION:
        return _optimize_subset(
            formulas=belief_base,
            hard_constraints=hard_constraints,
            trailing_formulas=None,
            weights=weights,
            operator=contraction_operator,
            conflict_sets=conflict_sets,
        )

    if contraction_operator == RevisionOperator.LEXICOGRAPHIC:
        return _lexicographic_change(
            belief_base=belief_base,
            hard_constraints=hard_constraints,
            trailing_formulas=None,
            operator=RevisionOperator.LEXICOGRAPHIC,
            conflict_sets=conflict_sets,
        )

    if contraction_operator in (
        RevisionOperator.PARTIAL_ORDER,
        RevisionOperator.EPISTEMIC,
    ):
        return _ranked_change(
            belief_base=belief_base,
            hard_constraints=hard_constraints,
            trailing_formulas=None,
            weights=weights,
            ranks=ranks,
            operator=contraction_operator,
            conflict_sets=conflict_sets,
        )

    results = _enumerate_kernel_changes(
        formulas=belief_base,
        trailing_formulas=None,
        conflict_sets=conflict_sets,
        weights=weights,
        operator=RevisionOperator.KERNEL,
        limit=1,
    )
    return results[0]


def enumerate_optimal_revisions(
    belief_base: Sequence[z3.ExprRef],
    new_belief: z3.ExprRef,
    weights: Optional[Sequence[int]] = None,
    ranks: Optional[Sequence[int]] = None,
    limit: Optional[int] = None,
    operator: Union[RevisionOperator, str] = RevisionOperator.MAX_RETENTION,
    max_conflicts: Optional[int] = None,
) -> List[BeliefRevisionResult]:
    """Enumerate optimal revision outcomes for a finite belief base."""

    revision_operator = RevisionOperator.from_value(operator)
    if revision_operator in (
        RevisionOperator.LEXICOGRAPHIC,
        RevisionOperator.PARTIAL_ORDER,
        RevisionOperator.EPISTEMIC,
    ):
        return [
            revise_belief_base(
                belief_base,
                new_belief,
                weights=weights,
                ranks=ranks,
                operator=revision_operator,
                max_conflicts=max_conflicts,
            )
        ]

    if revision_operator == RevisionOperator.KERNEL:
        _validate_weights(belief_base, weights)
        _validate_ranks(belief_base, ranks)
        if _check_unsat([new_belief]):
            raise ValueError("Cannot revise by an inconsistent belief")
        conflict_sets = _conflicts_with_new_belief(
            belief_base, new_belief, max_conflicts=max_conflicts
        )
        return _enumerate_kernel_changes(
            formulas=belief_base,
            trailing_formulas=[new_belief],
            conflict_sets=conflict_sets,
            weights=weights,
            operator=RevisionOperator.KERNEL,
            limit=limit,
        )

    _validate_weights(belief_base, weights)
    hard_new_belief: List[z3.ExprRef] = [new_belief]
    if _check_unsat(hard_new_belief):
        raise ValueError("Cannot revise by an inconsistent belief")

    normalized_weights = _normalize_weights(belief_base, weights)
    conflict_sets = _conflicts_with_new_belief(
        belief_base, new_belief, max_conflicts=max_conflicts
    )
    optimum = revise_belief_base(
        belief_base,
        new_belief,
        normalized_weights,
        operator=RevisionOperator.MAX_RETENTION,
        max_conflicts=max_conflicts,
    )
    return _enumerate_optimal_subsets(
        formulas=belief_base,
        hard_constraints=hard_new_belief,
        trailing_formulas=hard_new_belief,
        weights=normalized_weights,
        optimum=optimum.objective_value,
        limit=limit,
        operator=RevisionOperator.MAX_RETENTION,
        conflict_sets=conflict_sets,
    )


def enumerate_optimal_contractions(
    belief_base: Sequence[z3.ExprRef],
    target_belief: z3.ExprRef,
    weights: Optional[Sequence[int]] = None,
    ranks: Optional[Sequence[int]] = None,
    limit: Optional[int] = None,
    operator: Union[RevisionOperator, str] = RevisionOperator.MAX_RETENTION,
    max_conflicts: Optional[int] = None,
) -> List[BeliefRevisionResult]:
    """Enumerate optimal contraction outcomes for a finite belief base."""

    contraction_operator = RevisionOperator.from_value(operator)
    _validate_weights(belief_base, weights)
    _validate_ranks(belief_base, ranks)
    negated_target = _negate(target_belief)
    hard_constraints: List[z3.ExprRef] = [negated_target]
    if _check_unsat(hard_constraints):
        raise ValueError("Cannot contract by a valid belief")

    conflict_sets = _conflicts_with_new_belief(
        belief_base, negated_target, max_conflicts=max_conflicts
    )

    if contraction_operator in (
        RevisionOperator.LEXICOGRAPHIC,
        RevisionOperator.PARTIAL_ORDER,
        RevisionOperator.EPISTEMIC,
    ):
        return [
            contract_belief_base(
                belief_base,
                target_belief,
                weights=weights,
                ranks=ranks,
                operator=contraction_operator,
                max_conflicts=max_conflicts,
            )
        ]

    if contraction_operator == RevisionOperator.KERNEL:
        return _enumerate_kernel_changes(
            formulas=belief_base,
            trailing_formulas=None,
            conflict_sets=conflict_sets,
            weights=weights,
            operator=RevisionOperator.KERNEL,
            limit=limit,
        )

    normalized_weights = _normalize_weights(belief_base, weights)
    optimum = contract_belief_base(
        belief_base,
        target_belief,
        normalized_weights,
        operator=RevisionOperator.MAX_RETENTION,
        max_conflicts=max_conflicts,
    )
    return _enumerate_optimal_subsets(
        formulas=belief_base,
        hard_constraints=hard_constraints,
        trailing_formulas=None,
        weights=normalized_weights,
        optimum=optimum.objective_value,
        limit=limit,
        operator=RevisionOperator.MAX_RETENTION,
        conflict_sets=conflict_sets,
    )


def _validate_weights(
    belief_base: Sequence[z3.ExprRef], weights: Optional[Sequence[int]]
) -> None:
    if weights is None:
        return
    if len(weights) != len(belief_base):
        raise ValueError("weights must have one entry per belief")
    if any(weight <= 0 for weight in weights):
        raise ValueError("weights must be positive integers")


def _validate_ranks(
    belief_base: Sequence[z3.ExprRef], ranks: Optional[Sequence[int]]
) -> None:
    if ranks is None:
        return
    if len(ranks) != len(belief_base):
        raise ValueError("ranks must have one entry per belief")


def _normalize_weights(
    belief_base: Sequence[z3.ExprRef], weights: Optional[Sequence[int]]
) -> List[int]:
    if weights is None:
        return [1 for _ in belief_base]
    return list(weights)


def _check_unsat(formulas: Sequence[z3.ExprRef]) -> bool:
    solver = z3.Solver()
    solver.add(*formulas)
    return solver.check() == z3.unsat


def _negate(formula: z3.ExprRef) -> z3.ExprRef:
    return cast(z3.ExprRef, z3.Not(formula))


def _conflicts_with_new_belief(
    belief_base: Sequence[z3.ExprRef],
    new_belief: z3.ExprRef,
    max_conflicts: Optional[int],
) -> List[List[int]]:
    augmented_base = list(belief_base) + [new_belief]
    new_index = len(belief_base)
    conflicts = enumerate_minimal_unsat_subsets(
        augmented_base,
        max_cores=max_conflicts,
    )
    result = []
    for conflict in conflicts:
        if new_index in conflict:
            result.append(sorted(index for index in conflict if index != new_index))
    return result


def _make_result(
    formulas: Sequence[z3.ExprRef],
    kept_indices: List[int],
    trailing_formulas: Optional[Sequence[z3.ExprRef]],
    objective_value: int,
    operator: RevisionOperator,
    conflict_sets: List[List[int]],
    incision_indices: Optional[List[int]] = None,
    weights: Optional[Sequence[int]] = None,
    ranks: Optional[Sequence[int]] = None,
) -> BeliefRevisionResult:
    removed_indices = [idx for idx in range(len(formulas)) if idx not in kept_indices]
    kept_beliefs = [formulas[idx] for idx in kept_indices]
    removed_beliefs = [formulas[idx] for idx in removed_indices]
    result_base = list(kept_beliefs)
    if trailing_formulas is not None:
        result_base.extend(trailing_formulas)
    conflict_belief_sets = [
        [formulas[idx] for idx in conflict_indices] for conflict_indices in conflict_sets
    ]
    actual_incision_indices = (
        removed_indices if incision_indices is None else sorted(incision_indices)
    )
    incision_beliefs = [formulas[idx] for idx in actual_incision_indices]
    normalized_weights = _normalize_weights(formulas, weights)
    normalized_ranks = _normalize_result_ranks(formulas, operator, ranks)
    return BeliefRevisionResult(
        kept_indices=kept_indices,
        removed_indices=removed_indices,
        kept_beliefs=kept_beliefs,
        removed_beliefs=removed_beliefs,
        result_base=result_base,
        objective_value=objective_value,
        operator=operator.value,
        conflict_sets=conflict_sets,
        conflict_belief_sets=conflict_belief_sets,
        incision_indices=actual_incision_indices,
        incision_beliefs=incision_beliefs,
        incision_cost=sum(normalized_weights[idx] for idx in actual_incision_indices),
        rank_strata=_build_rank_strata(
            formulas=formulas,
            kept_indices=kept_indices,
            removed_indices=removed_indices,
            weights=normalized_weights,
            ranks=normalized_ranks,
        ),
    )


def _normalize_result_ranks(
    formulas: Sequence[z3.ExprRef],
    operator: RevisionOperator,
    ranks: Optional[Sequence[int]],
) -> List[int]:
    if ranks is not None:
        return list(ranks)
    if operator in (
        RevisionOperator.LEXICOGRAPHIC,
        RevisionOperator.PARTIAL_ORDER,
        RevisionOperator.EPISTEMIC,
    ):
        return list(range(len(formulas)))
    return [0 for _ in formulas]


def _build_rank_strata(
    formulas: Sequence[z3.ExprRef],
    kept_indices: Sequence[int],
    removed_indices: Sequence[int],
    weights: Sequence[int],
    ranks: Sequence[int],
) -> List[RankStratumSummary]:
    kept_set = set(kept_indices)
    removed_set = set(removed_indices)
    summaries = []
    for rank in sorted(set(ranks)):
        belief_indices = [idx for idx, current_rank in enumerate(ranks) if current_rank == rank]
        rank_kept = [idx for idx in belief_indices if idx in kept_set]
        rank_removed = [idx for idx in belief_indices if idx in removed_set]
        summaries.append(
            RankStratumSummary(
                rank=rank,
                belief_indices=belief_indices,
                kept_indices=rank_kept,
                removed_indices=rank_removed,
                kept_weight=sum(weights[idx] for idx in rank_kept),
                removed_weight=sum(weights[idx] for idx in rank_removed),
            )
        )
    return summaries


def _optimize_subset(
    formulas: Sequence[z3.ExprRef],
    hard_constraints: Sequence[z3.ExprRef],
    trailing_formulas: Optional[Sequence[z3.ExprRef]],
    weights: Optional[Sequence[int]],
    operator: RevisionOperator,
    conflict_sets: List[List[int]],
) -> BeliefRevisionResult:
    normalized_weights = _normalize_weights(formulas, weights)
    optimize = z3.Optimize()
    selectors = [z3.Bool(f"keep_{idx}") for idx in range(len(formulas))]

    optimize.add(*hard_constraints)
    for selector, formula in zip(selectors, formulas):
        optimize.add(z3.Implies(selector, formula))

    objective = z3.Sum(
        [
            z3.If(selector, weight, 0)
            for selector, weight in zip(selectors, normalized_weights)
        ]
    )
    optimize.maximize(objective)

    if optimize.check() != z3.sat:
        raise ValueError("No consistent belief-base change exists")

    model = optimize.model()
    kept_indices = [
        idx
        for idx, selector in enumerate(selectors)
        if z3.is_true(model.eval(selector, True))
    ]
    return _make_result(
        formulas=formulas,
        kept_indices=kept_indices,
        trailing_formulas=trailing_formulas,
        objective_value=sum(normalized_weights[idx] for idx in kept_indices),
        operator=operator,
        conflict_sets=conflict_sets,
        weights=normalized_weights,
    )


def _lexicographic_change(
    belief_base: Sequence[z3.ExprRef],
    hard_constraints: Sequence[z3.ExprRef],
    trailing_formulas: Optional[Sequence[z3.ExprRef]],
    operator: RevisionOperator,
    conflict_sets: List[List[int]],
) -> BeliefRevisionResult:
    solver = z3.Optimize()
    selectors = [z3.Bool(f"lex_keep_{idx}") for idx in range(len(belief_base))]
    solver.add(*hard_constraints)
    for selector, formula in zip(selectors, belief_base):
        solver.add(z3.Implies(selector, formula))

    for idx, selector in enumerate(selectors):
        priority = 2 ** (len(belief_base) - idx - 1)
        solver.maximize(z3.If(selector, priority, 0))

    if solver.check() != z3.sat:
        raise ValueError("No consistent belief-base change exists")

    model = solver.model()
    kept_indices = [
        idx
        for idx, selector in enumerate(selectors)
        if z3.is_true(model.eval(selector, True))
    ]
    objective_value = sum(2 ** (len(belief_base) - idx - 1) for idx in kept_indices)
    return _make_result(
        formulas=belief_base,
        kept_indices=kept_indices,
        trailing_formulas=trailing_formulas,
        objective_value=objective_value,
        operator=operator,
        conflict_sets=conflict_sets,
        weights=[1 for _ in belief_base],
        ranks=list(range(len(belief_base))),
    )


def _ranked_change(
    belief_base: Sequence[z3.ExprRef],
    hard_constraints: Sequence[z3.ExprRef],
    trailing_formulas: Optional[Sequence[z3.ExprRef]],
    weights: Optional[Sequence[int]],
    ranks: Optional[Sequence[int]],
    operator: RevisionOperator,
    conflict_sets: List[List[int]],
) -> BeliefRevisionResult:
    normalized_weights = _normalize_weights(belief_base, weights)
    normalized_ranks = list(ranks) if ranks is not None else list(range(len(belief_base)))

    optimize = z3.Optimize()
    selectors = [z3.Bool(f"rank_keep_{idx}") for idx in range(len(belief_base))]
    optimize.add(*hard_constraints)
    for selector, formula in zip(selectors, belief_base):
        optimize.add(z3.Implies(selector, formula))

    for rank in sorted(set(normalized_ranks)):
        optimize.maximize(
            z3.Sum(
                [
                    z3.If(selector, weight, 0)
                    for idx, (selector, weight) in enumerate(
                        zip(selectors, normalized_weights)
                    )
                    if normalized_ranks[idx] == rank
                ]
            )
        )

    if optimize.check() != z3.sat:
        raise ValueError("No consistent belief-base change exists")

    model = optimize.model()
    kept_indices = [
        idx
        for idx, selector in enumerate(selectors)
        if z3.is_true(model.eval(selector, True))
    ]
    return _make_result(
        formulas=belief_base,
        kept_indices=kept_indices,
        trailing_formulas=trailing_formulas,
        objective_value=sum(normalized_weights[idx] for idx in kept_indices),
        operator=operator,
        conflict_sets=conflict_sets,
        weights=normalized_weights,
        ranks=normalized_ranks,
    )


def _lexicographic_revision(
    belief_base: Sequence[z3.ExprRef],
    new_belief: z3.ExprRef,
    conflict_sets: List[List[int]],
) -> BeliefRevisionResult:
    return _lexicographic_change(
        belief_base=belief_base,
        hard_constraints=[new_belief],
        trailing_formulas=[new_belief],
        operator=RevisionOperator.LEXICOGRAPHIC,
        conflict_sets=conflict_sets,
    )


def _kernel_revision(
    belief_base: Sequence[z3.ExprRef],
    new_belief: z3.ExprRef,
    weights: Optional[Sequence[int]],
    conflict_sets: List[List[int]],
) -> BeliefRevisionResult:
    normalized_weights = _normalize_weights(belief_base, weights)
    if not conflict_sets:
        return _make_result(
            formulas=belief_base,
            kept_indices=list(range(len(belief_base))),
            trailing_formulas=[new_belief],
            objective_value=sum(normalized_weights),
            operator=RevisionOperator.KERNEL,
            conflict_sets=[],
            incision_indices=[],
            weights=normalized_weights,
        )

    optimize = z3.Optimize()
    remove_vars = [z3.Bool(f"remove_{idx}") for idx in range(len(belief_base))]
    for conflict in conflict_sets:
        optimize.add(z3.Or([remove_vars[idx] for idx in conflict]))

    removal_cost = z3.Sum(
        [
            z3.If(remove_var, normalized_weights[idx], 0)
            for idx, remove_var in enumerate(remove_vars)
        ]
    )
    optimize.minimize(removal_cost)

    if optimize.check() != z3.sat:
        raise ValueError("No kernel incision exists")

    model = optimize.model()
    removed: Set[int] = {
        idx
        for idx, remove_var in enumerate(remove_vars)
        if z3.is_true(model.eval(remove_var, True))
    }
    kept_indices = [idx for idx in range(len(belief_base)) if idx not in removed]
    return _make_result(
        formulas=belief_base,
        kept_indices=kept_indices,
        trailing_formulas=[new_belief],
        objective_value=sum(normalized_weights[idx] for idx in kept_indices),
        operator=RevisionOperator.KERNEL,
        conflict_sets=conflict_sets,
        incision_indices=sorted(removed),
        weights=normalized_weights,
    )


def _enumerate_kernel_changes(
    formulas: Sequence[z3.ExprRef],
    trailing_formulas: Optional[Sequence[z3.ExprRef]],
    conflict_sets: List[List[int]],
    weights: Optional[Sequence[int]],
    operator: RevisionOperator,
    limit: Optional[int],
) -> List[BeliefRevisionResult]:
    normalized_weights = _normalize_weights(formulas, weights)
    if not conflict_sets:
        return [
            _make_result(
                formulas=formulas,
                kept_indices=list(range(len(formulas))),
                trailing_formulas=trailing_formulas,
                objective_value=sum(normalized_weights),
                operator=operator,
                conflict_sets=[],
                incision_indices=[],
                weights=normalized_weights,
            )
        ]

    optimize = z3.Optimize()
    remove_vars = [z3.Bool(f"remove_enum_{idx}") for idx in range(len(formulas))]
    for conflict in conflict_sets:
        optimize.add(z3.Or([remove_vars[idx] for idx in conflict]))

    removal_cost = z3.Sum(
        [
            z3.If(remove_var, normalized_weights[idx], 0)
            for idx, remove_var in enumerate(remove_vars)
        ]
    )
    handle = optimize.minimize(removal_cost)
    if optimize.check() != z3.sat:
        raise ValueError("No kernel incision exists")

    optimum = optimize.lower(handle)
    if not isinstance(optimum, z3.IntNumRef):
        raise ValueError("Kernel optimum is not a concrete integer")

    solver = z3.Solver()
    enum_vars = [z3.Bool(f"remove_enum_sat_{idx}") for idx in range(len(formulas))]
    for conflict in conflict_sets:
        solver.add(z3.Or([enum_vars[idx] for idx in conflict]))
    solver.add(
        z3.Sum(
            [
                z3.If(enum_var, normalized_weights[idx], 0)
                for idx, enum_var in enumerate(enum_vars)
            ]
        )
        == optimum.as_long()
    )

    results = []
    while solver.check() == z3.sat:
        model = solver.model()
        removed = {
            idx
            for idx, enum_var in enumerate(enum_vars)
            if z3.is_true(model.eval(enum_var, True))
        }
        kept_indices = [idx for idx in range(len(formulas)) if idx not in removed]
        results.append(
            _make_result(
                formulas=formulas,
                kept_indices=kept_indices,
                trailing_formulas=trailing_formulas,
                objective_value=sum(normalized_weights[idx] for idx in kept_indices),
                operator=operator,
                conflict_sets=conflict_sets,
                incision_indices=sorted(removed),
                weights=normalized_weights,
            )
        )
        solver.add(
            z3.Or(
                [
                    enum_var != z3.BoolVal(idx in removed)
                    for idx, enum_var in enumerate(enum_vars)
                ]
            )
        )
        if limit is not None and len(results) >= limit:
            break

    return results


def _enumerate_optimal_subsets(
    formulas: Sequence[z3.ExprRef],
    hard_constraints: Sequence[z3.ExprRef],
    trailing_formulas: Optional[Sequence[z3.ExprRef]],
    weights: Sequence[int],
    optimum: int,
    limit: Optional[int],
    operator: RevisionOperator,
    conflict_sets: List[List[int]],
) -> List[BeliefRevisionResult]:
    solver = z3.Solver()
    selectors = [z3.Bool(f"keep_enum_{idx}") for idx in range(len(formulas))]

    solver.add(*hard_constraints)
    for selector, formula in zip(selectors, formulas):
        solver.add(z3.Implies(selector, formula))

    total_weight = z3.Sum(
        [z3.If(selector, weight, 0) for selector, weight in zip(selectors, weights)]
    )
    solver.add(total_weight == optimum)

    results = []
    while solver.check() == z3.sat:
        model = solver.model()
        kept_indices = [
            idx
            for idx, selector in enumerate(selectors)
            if z3.is_true(model.eval(selector, True))
        ]
        results.append(
            _make_result(
                formulas=formulas,
                kept_indices=kept_indices,
                trailing_formulas=trailing_formulas,
                objective_value=optimum,
                operator=operator,
                conflict_sets=conflict_sets,
                weights=weights,
            )
        )

        solver.add(
            z3.Or(
                [
                    selector != z3.BoolVal(idx in kept_indices)
                    for idx, selector in enumerate(selectors)
                ]
            )
        )
        if limit is not None and len(results) >= limit:
            break

    return results


__all__ = [
    "BeliefRevisionResult",
    "RankStratumSummary",
    "RevisionOperator",
    "contract_belief_base",
    "enumerate_optimal_contractions",
    "enumerate_optimal_revisions",
    "expand_belief_base",
    "revise_belief_base",
]
