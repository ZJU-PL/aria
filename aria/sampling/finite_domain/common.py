"""
Common utilities for finite-domain samplers.
"""

from typing import Any, Dict, Iterable, List, Sequence, Union, cast

import z3


def expr_to_sample_name(expr: z3.ExprRef) -> str:
    """Return a stable sample key for a tracked expression."""
    return str(expr)


def z3_value_to_sample(value: z3.ExprRef) -> Any:
    """Convert a Z3 value into a Python-friendly sample value."""
    if z3.is_true(value):
        return True
    if z3.is_false(value):
        return False
    if z3.is_int_value(value):
        numeric_value: Any = value
        return numeric_value.as_long()
    if z3.is_rational_value(value):
        rational_value: Any = value
        return float(rational_value.numerator_as_long()) / float(
            rational_value.denominator_as_long()
        )
    if z3.is_bv_value(value):
        bitvec_value: Any = value
        return bitvec_value.as_long()
    if z3.is_string_value(value):
        string_value: Any = value
        return string_value.as_string()

    if value.sort().kind() == z3.Z3_DATATYPE_SORT:
        return _datatype_value_to_sample(value)

    if z3.is_const(value) and value.decl().kind() == z3.Z3_OP_UNINTERPRETED:
        return str(value)

    return str(value)


def _datatype_value_to_sample(value: z3.ExprRef) -> Any:
    """Convert a datatype value into a Python-friendly representation."""
    if value.num_args() == 0:
        return str(value)

    return {
        "constructor": value.decl().name(),
        "fields": [z3_value_to_sample(child) for child in value.children()],
    }


def block_model_on_terms(
    solver: z3.Solver, model: z3.ModelRef, terms: Sequence[z3.ExprRef]
) -> bool:
    """Add a blocking clause for the tracked terms.

    Returns True when a blocking clause was added and False when there are no
    tracked terms to block on.
    """
    block = []
    for term in terms:
        value = model.evaluate(term, model_completion=True)
        block.append(term != value)

    if not block:
        return False

    solver.add(z3.Or(block))
    return True


def build_sample(model: z3.ModelRef, terms: Sequence[z3.ExprRef]) -> Dict[str, Any]:
    """Build a sample dictionary for the given tracked terms."""
    sample: Dict[str, Any] = {}
    for term in terms:
        value = model.evaluate(term, model_completion=True)
        sample[expr_to_sample_name(term)] = z3_value_to_sample(value)
    return sample


def resolve_projection_terms(
    available_terms: Sequence[z3.ExprRef],
    projection_terms: Union[None, Iterable[Union[str, z3.ExprRef]]],
) -> List[z3.ExprRef]:
    """Resolve projection terms against the tracked term set.

    If no projection is given, the full tracked term set is returned.
    Terms may be provided either as Z3 expressions or by their string form.
    """
    if projection_terms is None:
        return list(available_terms)

    available_by_name = {str(term): term for term in available_terms}
    resolved: List[z3.ExprRef] = []
    seen_names = set()

    for projection_term in projection_terms:
        if isinstance(projection_term, str):
            term_name = projection_term
            resolved_term = available_by_name.get(term_name)
            if resolved_term is None:
                raise ValueError(f"Unknown projection term: {term_name}")
        else:
            term_name = str(projection_term)
            resolved_term = available_by_name.get(term_name)
            if resolved_term is None:
                raise ValueError(f"Unknown projection term: {projection_term}")

        if term_name in seen_names:
            continue

        resolved.append(resolved_term)
        seen_names.add(term_name)

    return resolved


def resolve_output_terms(
    available_terms: Sequence[z3.ExprRef],
    projection_terms: Union[None, Iterable[Union[str, z3.ExprRef]]],
    tracked_terms: Union[None, Iterable[Union[str, z3.ExprRef]]],
    return_full_model: bool,
) -> List[z3.ExprRef]:
    """Resolve which terms should be returned in each sample.

    `tracked_terms` takes precedence when provided. Otherwise, the output is the
    full tracked model when `return_full_model` is true, and the projection when
    it is false.
    """
    if tracked_terms is not None:
        return resolve_projection_terms(available_terms, tracked_terms)

    if return_full_model:
        return list(available_terms)

    return resolve_projection_terms(available_terms, projection_terms)


def collect_ground_uf_terms(formula: z3.ExprRef) -> List[z3.ExprRef]:
    """Collect ground uninterpreted function applications from a formula."""
    terms: Dict[int, z3.ExprRef] = {}
    stack = [formula]
    seen = set()

    while stack:
        expr = stack.pop()
        expr_id = z3.Z3_get_ast_id(expr.ctx.ref(), expr.as_ast())
        if expr_id in seen:
            continue
        seen.add(expr_id)

        if z3.is_quantifier(expr):
            quantifier_expr: Any = expr
            stack.append(quantifier_expr.body())
            continue

        if not z3.is_app(expr):
            continue

        if expr.num_args() > 0 and expr.decl().kind() == z3.Z3_OP_UNINTERPRETED:
            terms[expr_id] = expr

        stack.extend(expr.children())

    return sorted(terms.values(), key=str)
