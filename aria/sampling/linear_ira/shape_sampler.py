"""
Shape-oriented helpers for ADT + linear arithmetic sampling.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

import z3


def _ast_id(expr: z3.ExprRef) -> int:
    """Return a stable AST id for a Z3 expression."""
    return z3.Z3_get_ast_id(expr.ctx.ref(), expr.as_ast())


def _is_datatype_sort(sort: z3.SortRef) -> bool:
    """Check whether a sort is a datatype sort."""
    return sort.kind() == z3.Z3_DATATYPE_SORT


def _dedupe_terms(terms: Iterable[z3.ExprRef]) -> List[z3.ExprRef]:
    """Deduplicate Z3 terms while preserving first occurrence order."""
    unique_terms: List[z3.ExprRef] = []
    seen_ids: Set[int] = set()

    for term in terms:
        term_id = _ast_id(term)
        if term_id in seen_ids:
            continue
        seen_ids.add(term_id)
        unique_terms.append(term)

    return unique_terms


def _find_constructor_index(
    sort: z3.DatatypeSortRef, decl: z3.FuncDeclRef
) -> int:
    """Return the constructor index for a datatype declaration."""
    for idx in range(sort.num_constructors()):
        if decl.eq(sort.constructor(idx)):
            return idx
    raise ValueError(f"Constructor {decl} does not belong to sort {sort}")


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


def _extract_shape_constraints(
    term: z3.ExprRef,
    value: z3.ExprRef,
) -> Tuple[List[z3.ExprRef], List[z3.ExprRef]]:
    """Lower a datatype model value to constructor-shape constraints."""
    if not _is_datatype_sort(term.sort()):
        return [], []

    sort = term.sort()
    if not isinstance(sort, z3.DatatypeSortRef):
        return [], []

    constructor_idx = _find_constructor_index(sort, value.decl())
    constraints: List[z3.ExprRef] = [sort.recognizer(constructor_idx)(term)]
    payload_terms: List[z3.ExprRef] = []

    for field_idx in range(value.num_args()):
        accessor = sort.accessor(constructor_idx, field_idx)
        child_term = accessor(term)
        child_value = value.arg(field_idx)

        if _is_datatype_sort(child_term.sort()):
            child_constraints, child_payload_terms = _extract_shape_constraints(
                child_term, child_value
            )
            constraints.extend(child_constraints)
            payload_terms.extend(child_payload_terms)
        elif child_term.sort().kind() == z3.Z3_INT_SORT:
            payload_terms.append(child_term)
        else:
            constraints.append(child_term == child_value)

    return constraints, payload_terms


@dataclass(frozen=True)
class ShapeSample:
    """A sampled datatype shape with residual constraints and payload terms."""

    signature: Tuple[Any, ...]
    constraints: Tuple[z3.ExprRef, ...]
    payload_terms: Tuple[z3.ExprRef, ...]


def enumerate_datatype_shapes(
    formula: z3.ExprRef,
    root_terms: Sequence[z3.ExprRef],
    max_shapes: int,
    random_seed: int | None = None,
    timeout: float | None = None,
) -> List[ShapeSample]:
    """Enumerate distinct constructor shapes for root datatype terms."""
    if not root_terms or max_shapes <= 0:
        return []

    solver = z3.Solver()
    if random_seed is not None:
        solver.set("random_seed", random_seed)
        solver.set("seed", random_seed)
    if timeout is not None:
        solver.set("timeout", max(1, int(timeout * 1000)))
    solver.add(formula)

    shapes: List[ShapeSample] = []
    seen_signatures: Set[Tuple[Any, ...]] = set()

    while len(shapes) < max_shapes and solver.check() == z3.sat:
        model = solver.model()
        signature_parts: List[Tuple[Any, ...]] = []
        shape_constraints: List[z3.ExprRef] = []
        payload_terms: List[z3.ExprRef] = []

        for root_term in root_terms:
            root_value = model.evaluate(root_term, model_completion=True)
            signature_parts.append(_value_to_shape_signature(root_value))
            root_constraints, root_payload_terms = _extract_shape_constraints(
                root_term, root_value
            )
            shape_constraints.extend(root_constraints)
            payload_terms.extend(root_payload_terms)

        signature = tuple(signature_parts)
        if signature in seen_signatures:
            break
        seen_signatures.add(signature)

        deduped_constraints = tuple(_dedupe_terms(shape_constraints))
        shapes.append(
            ShapeSample(
                signature=signature,
                constraints=deduped_constraints,
                payload_terms=tuple(_dedupe_terms(payload_terms)),
            )
        )

        if not deduped_constraints:
            break
        solver.add(z3.Not(z3.And(*deduped_constraints)))

    return shapes


def sample_distance(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    """Compute a mixed-theory distance between two sampled models."""

    def value_distance(left_value: Any, right_value: Any) -> float:
        if isinstance(left_value, dict) and isinstance(right_value, dict):
            distance = 0.0
            if left_value.get("constructor") != right_value.get("constructor"):
                distance += 1.0

            left_fields = left_value.get("fields", [])
            right_fields = right_value.get("fields", [])
            for left_field, right_field in zip(left_fields, right_fields):
                distance += value_distance(left_field, right_field)
            distance += abs(len(left_fields) - len(right_fields))
            return distance

        if isinstance(left_value, bool) and isinstance(right_value, bool):
            return 0.0 if left_value == right_value else 1.0

        if isinstance(left_value, int) and isinstance(right_value, int):
            return float(abs(left_value - right_value))

        return 0.0 if left_value == right_value else 1.0

    keys = sorted(set(left) | set(right))
    return sum(value_distance(left.get(key), right.get(key)) for key in keys)


def select_max_distance_subset(
    candidates: Sequence[Dict[str, Any]], num_samples: int
) -> List[Dict[str, Any]]:
    """Greedily select a diverse subset by maximizing minimum distance."""
    if num_samples <= 0 or not candidates:
        return []
    if len(candidates) <= num_samples:
        return list(candidates)

    remaining = list(candidates)
    first_candidate = max(
        remaining,
        key=lambda candidate: sum(
            sample_distance(candidate, other)
            for other in remaining
            if other is not candidate
        ),
    )
    selected = [first_candidate]
    remaining.remove(first_candidate)

    while remaining and len(selected) < num_samples:
        next_candidate = max(
            remaining,
            key=lambda candidate: min(
                sample_distance(candidate, selected_candidate)
                for selected_candidate in selected
            ),
        )
        selected.append(next_candidate)
        remaining.remove(next_candidate)

    return selected
