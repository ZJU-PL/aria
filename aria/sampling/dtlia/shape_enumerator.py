"""Enumerate datatype constructor shapes for DTLIA sampling."""

import random
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import z3


def _ast_id(expr: z3.ExprRef) -> int:
    """Return a stable AST id for a Z3 expression."""
    return z3.Z3_get_ast_id(expr.ctx.ref(), expr.as_ast())


def _is_datatype_sort(sort: z3.SortRef) -> bool:
    """Check whether a sort is a datatype sort."""
    return sort.kind() == z3.Z3_DATATYPE_SORT


def _dedupe_terms(terms: Iterable[z3.ExprRef]) -> List[z3.ExprRef]:
    """Deduplicate Z3 terms while preserving first occurrence order.

    Uses Z3 AST identity (``_ast_id``) rather than string representation so
    that structurally identical but separately allocated expressions are treated
    as the same term. Insertion order is preserved, making this suitable for
    constraint lists where position matters. For a sorted, string-keyed variant
    used when deterministic human-readable ordering is required, see
    ``_dedupe_sorted_terms`` in sampler.py.
    """
    unique_terms: List[z3.ExprRef] = []
    seen_ids: Set[int] = set()

    for term in terms:
        term_id = _ast_id(term)
        if term_id in seen_ids:
            continue
        seen_ids.add(term_id)
        unique_terms.append(term)

    return unique_terms


def _value_to_shape_signature(value: z3.ExprRef) -> Tuple[Any, ...]:
    """Convert a datatype value into a constructor-shape signature."""
    if not _is_datatype_sort(value.sort()):
        if z3.is_int_value(value):
            return ("Int",)
        return (str(value.sort()),)

    field_signatures: List[Tuple[Any, ...]] = []
    for child in value.children():
        if _is_datatype_sort(child.sort()):
            field_signatures.append(_value_to_shape_signature(child))
        elif z3.is_int_value(child):
            field_signatures.append(("Int",))
        else:
            field_signatures.append((str(child.sort()),))

    return (value.decl().name(), tuple(field_signatures))


@dataclass(frozen=True)
class ShapeSample:
    """A sampled datatype shape with residual constraints and payload terms."""

    signature: Tuple[Any, ...]
    constraints: Tuple[z3.ExprRef, ...]
    payload_terms: Tuple[z3.ExprRef, ...]


@dataclass(frozen=True)
class ShapeEnumerationResult:
    """Complete result for shape exploration, including pruning statistics."""

    shapes: List[ShapeSample]
    stats: Dict[str, Any]


@dataclass(frozen=True)
class BranchChoice:
    """A feasible constructor branch discovered for a pending datatype term."""

    constructor_idx: int
    constructor_constraint: z3.ExprRef
    child_terms: Tuple[z3.ExprRef, ...]
    scalar_terms: Tuple[z3.ExprRef, ...]
    preferred_by_model: bool
    tie_breaker: float


def _pending_term_key(term: z3.ExprRef) -> Tuple[int, str]:
    """Return a deterministic key for pending datatype terms."""
    return (_ast_id(term), str(term))


def _append_pending_term(
    pending_terms: List[z3.ExprRef],
    pending_ids: Set[int],
    assigned_ids: Set[int],
    term: z3.ExprRef,
) -> None:
    """Append a datatype child term to the pending queue once."""
    term_id = _ast_id(term)
    if term_id in pending_ids or term_id in assigned_ids:
        return
    pending_terms.append(term)
    pending_ids.add(term_id)


def _branch_order_key(choice: BranchChoice) -> Tuple[int, float, int]:
    """Order feasible branches reproducibly while reducing decl-order bias."""
    return (
        0 if choice.preferred_by_model else 1,
        choice.tie_breaker,
        choice.constructor_idx,
    )


def enumerate_datatype_shapes(
    formula: z3.ExprRef,
    root_terms: Sequence[z3.ExprRef],
    max_shapes: int,
    random_seed: Optional[int] = None,
    timeout: Optional[float] = None,
) -> ShapeEnumerationResult:
    """Enumerate distinct constructor shapes via partial constructor search."""
    if not root_terms or max_shapes <= 0:
        return ShapeEnumerationResult(
            shapes=[],
            stats={
                "shape_enumerator": "partial_feasibility",
                "shape_solver_checks": 0,
                "shape_pruned_branches": 0,
                "shape_partial_states": 0,
                "shape_duplicate_signatures": 0,
                "shape_complete_assignments": 0,
                "shape_unknown_branches": 0,
                "shape_enumeration_complete": True,
                "shape_enumeration_termination_reason": "exhausted",
                "shape_enumeration_time_ms": 0,
            },
        )

    solver = z3.Solver()
    rng = random.Random(random_seed)
    if random_seed is not None:
        solver.set("random_seed", random_seed)
        solver.set("seed", random_seed)
    if timeout is not None:
        solver.set("timeout", max(1, int(timeout * 1000)))
    solver.add(formula)

    started_at = perf_counter()
    ordered_root_terms = _dedupe_terms(root_terms)
    shapes: List[ShapeSample] = []
    seen_signatures: Set[Tuple[Any, ...]] = set()
    solver_checks = 0
    pruned_branches = 0
    partial_states = 0
    duplicate_signatures = 0
    complete_assignments = 0
    unknown_branches = 0
    enumerator_complete = True
    termination_reason = "exhausted"

    initial_pending = sorted(ordered_root_terms, key=_pending_term_key)

    def explore(
        pending_terms: List[z3.ExprRef],
        pending_ids: Set[int],
        assigned_ids: Set[int],
        shape_constraints: List[z3.ExprRef],
        payload_terms: List[z3.ExprRef],
    ) -> None:
        nonlocal solver_checks
        nonlocal pruned_branches
        nonlocal partial_states
        nonlocal duplicate_signatures
        nonlocal complete_assignments
        nonlocal unknown_branches
        nonlocal enumerator_complete
        nonlocal termination_reason

        if len(shapes) >= max_shapes:
            enumerator_complete = False
            termination_reason = "max_shapes"
            return

        solver.push()
        solver.add(*shape_constraints)
        check_result = solver.check()
        solver_checks += 1
        if check_result == z3.unsat:
            pruned_branches += 1
            solver.pop()
            return
        if check_result != z3.sat:
            unknown_branches += 1
            enumerator_complete = False
            termination_reason = str(check_result)
            solver.pop()
            return

        if not pending_terms:
            complete_assignments += 1
            model = solver.model()
            signature = tuple(
                _value_to_shape_signature(
                    model.evaluate(root_term, model_completion=True)
                )
                for root_term in ordered_root_terms
            )
            if signature in seen_signatures:
                duplicate_signatures += 1
                solver.pop()
                return

            seen_signatures.add(signature)
            shapes.append(
                ShapeSample(
                    signature=signature,
                    constraints=tuple(_dedupe_terms(shape_constraints)),
                    payload_terms=tuple(_dedupe_terms(payload_terms)),
                )
            )
            solver.pop()
            return

        partial_states += 1
        term = pending_terms[0]
        remaining_pending = list(pending_terms[1:])
        remaining_ids = set(pending_ids)
        remaining_ids.discard(_ast_id(term))

        sort = term.sort()
        if not isinstance(sort, z3.DatatypeSortRef):
            solver.pop()
            explore(
                remaining_pending,
                remaining_ids,
                set(assigned_ids),
                list(shape_constraints),
                list(payload_terms),
            )
            return

        term_id = _ast_id(term)
        next_assigned_ids = set(assigned_ids)
        next_assigned_ids.add(term_id)

        model = solver.model()
        branch_choices: List[BranchChoice] = []
        for constructor_idx in range(sort.num_constructors()):
            recognizer = sort.recognizer(constructor_idx)
            constructor_constraint = recognizer(term)

            solver.push()
            solver.add(constructor_constraint)
            branch_result = solver.check()
            solver_checks += 1
            if branch_result == z3.unsat:
                pruned_branches += 1
                solver.pop()
                continue
            if branch_result != z3.sat:
                unknown_branches += 1
                enumerator_complete = False
                termination_reason = str(branch_result)
                solver.pop()
                break

            constructor_decl = sort.constructor(constructor_idx)
            child_terms: List[z3.ExprRef] = []
            scalar_terms: List[z3.ExprRef] = []
            for field_idx in range(constructor_decl.arity()):
                accessor = sort.accessor(constructor_idx, field_idx)
                child_term = accessor(term)
                if _is_datatype_sort(child_term.sort()):
                    child_terms.append(child_term)
                else:
                    scalar_terms.append(child_term)

            constructor_value = model.evaluate(recognizer(term), model_completion=True)
            branch_choices.append(
                BranchChoice(
                    constructor_idx=constructor_idx,
                    constructor_constraint=constructor_constraint,
                    child_terms=tuple(child_terms),
                    scalar_terms=tuple(scalar_terms),
                    preferred_by_model=z3.is_true(constructor_value),
                    tie_breaker=rng.random(),
                )
            )
            solver.pop()

        solver.pop()

        if termination_reason == "unknown":
            return

        ordered_branch_choices = sorted(branch_choices, key=_branch_order_key)
        for branch_index, branch_choice in enumerate(ordered_branch_choices):
            branch_constraints = list(shape_constraints)
            branch_constraints.append(branch_choice.constructor_constraint)
            branch_payload_terms = list(payload_terms)
            branch_payload_terms.extend(branch_choice.scalar_terms)
            branch_pending = list(remaining_pending)
            branch_pending_ids = set(remaining_ids)

            for child_term in branch_choice.child_terms:
                _append_pending_term(
                    branch_pending,
                    branch_pending_ids,
                    next_assigned_ids,
                    child_term,
                )

            branch_pending.sort(key=_pending_term_key)
            explore(
                branch_pending,
                branch_pending_ids,
                next_assigned_ids,
                branch_constraints,
                branch_payload_terms,
            )
            if len(shapes) >= max_shapes:
                if branch_index < len(ordered_branch_choices) - 1:
                    enumerator_complete = False
                    termination_reason = "max_shapes"
                return
            if termination_reason == "unknown":
                return

    explore(
        initial_pending,
        {_ast_id(term) for term in initial_pending},
        set(),
        [],
        [],
    )

    elapsed_ms = int((perf_counter() - started_at) * 1000)
    return ShapeEnumerationResult(
        shapes=shapes,
        stats={
            "shape_enumerator": "partial_feasibility",
            "shape_solver_checks": solver_checks,
            "shape_pruned_branches": pruned_branches,
            "shape_partial_states": partial_states,
            "shape_duplicate_signatures": duplicate_signatures,
            "shape_complete_assignments": complete_assignments,
            "shape_unknown_branches": unknown_branches,
            "shape_enumeration_complete": enumerator_complete,
            "shape_enumeration_termination_reason": termination_reason,
            "shape_enumeration_time_ms": elapsed_ms,
        },
    )
