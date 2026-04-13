"""
Enumerate datatype constructor shapes for DTLIA sampling.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

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


def enumerate_datatype_shapes(
    formula: z3.ExprRef,
    root_terms: Sequence[z3.ExprRef],
    max_shapes: int,
    random_seed: int | None = None,
    timeout: float | None = None,
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
                "shape_enumeration_time_ms": 0,
            },
        )

    solver = z3.Solver()
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

        if len(shapes) >= max_shapes:
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
                return

            seen_signatures.add(signature)
            shapes.append(
                ShapeSample(
                    signature=signature,
                    constraints=tuple(_dedupe_terms(shape_constraints)),
                    payload_terms=tuple(_dedupe_terms(payload_terms)),
                )
            )
            return

        partial_states += 1
        term = pending_terms[0]
        remaining_pending = list(pending_terms[1:])
        remaining_ids = set(pending_ids)
        remaining_ids.discard(_ast_id(term))

        sort = term.sort()
        if not isinstance(sort, z3.DatatypeSortRef):
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

        for constructor_idx in range(sort.num_constructors()):
            recognizer = sort.recognizer(constructor_idx)
            constructor_constraint = recognizer(term)

            solver.push()
            solver.add(constructor_constraint)
            solver_checks += 1

            if solver.check() != z3.sat:
                pruned_branches += 1
                solver.pop()
                continue

            constructor_decl = sort.constructor(constructor_idx)
            branch_constraints = list(shape_constraints)
            branch_constraints.append(constructor_constraint)
            branch_payload_terms = list(payload_terms)
            branch_pending = list(remaining_pending)
            branch_pending_ids = set(remaining_ids)

            for field_idx in range(constructor_decl.arity()):
                accessor = sort.accessor(constructor_idx, field_idx)
                child_term = accessor(term)
                if _is_datatype_sort(child_term.sort()):
                    _append_pending_term(
                        branch_pending,
                        branch_pending_ids,
                        next_assigned_ids,
                        child_term,
                    )
                elif child_term.sort().kind() == z3.Z3_INT_SORT:
                    branch_payload_terms.append(child_term)

            branch_pending.sort(key=_pending_term_key)
            explore(
                branch_pending,
                branch_pending_ids,
                next_assigned_ids,
                branch_constraints,
                branch_payload_terms,
            )
            solver.pop()

            if len(shapes) >= max_shapes:
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
            "shape_enumeration_time_ms": elapsed_ms,
        },
    )
