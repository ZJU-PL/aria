"""
Common utilities for finite-domain samplers.
"""

from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Union

import z3

from aria.sampling.base import SamplingOptions, SamplingResult
from aria.utils.z3.expr import get_variables


def _ast_id(expr: z3.ExprRef) -> int:
    """Return a stable AST id for a Z3 expression."""
    return z3.Z3_get_ast_id(expr.ctx.ref(), expr.as_ast())


def _is_datatype_sort(sort: z3.SortRef) -> bool:
    """Check whether a sort is a datatype sort."""
    return sort.kind() == z3.Z3_DATATYPE_SORT


def _unique_terms(terms: Iterable[z3.ExprRef]) -> List[z3.ExprRef]:
    """Deduplicate Z3 terms while preserving first occurrence order."""
    unique: List[z3.ExprRef] = []
    seen_ids: Set[int] = set()

    for term in terms:
        term_id = _ast_id(term)
        if term_id in seen_ids:
            continue
        seen_ids.add(term_id)
        unique.append(term)

    return unique


def _sorted_unique_terms(terms: Iterable[z3.ExprRef]) -> List[z3.ExprRef]:
    """Deduplicate and sort Z3 terms by their string form."""
    return sorted(_unique_terms(terms), key=str)


def _walk_exprs(formula: z3.ExprRef) -> List[z3.ExprRef]:
    """Return all reachable application nodes in deterministic DFS order."""
    stack = [formula]
    seen_ids: Set[int] = set()
    ordered: List[z3.ExprRef] = []

    while stack:
        expr = stack.pop()
        expr_id = _ast_id(expr)
        if expr_id in seen_ids:
            continue
        seen_ids.add(expr_id)
        ordered.append(expr)

        if z3.is_quantifier(expr):
            quantifier_expr: Any = expr
            stack.append(quantifier_expr.body())
            continue

        if not z3.is_app(expr):
            continue

        stack.extend(expr.children())

    return ordered


def _positive_conjuncts(formula: z3.ExprRef) -> List[z3.ExprRef]:
    """Return atoms that occur as positive top-level conjuncts."""
    stack = [formula]
    conjuncts: List[z3.ExprRef] = []

    while stack:
        expr = stack.pop()
        if z3.is_and(expr):
            stack.extend(expr.children())
            continue
        conjuncts.append(expr)

    return conjuncts


def _find_constructor_index(
    sort: z3.DatatypeSortRef, decl: z3.FuncDeclRef
) -> Optional[int]:
    """Return the constructor index matching a declaration, if any."""
    for idx in range(sort.num_constructors()):
        if decl.eq(sort.constructor(idx)):
            return idx
    return None


def _find_recognizer_constructor_index(
    sort: z3.DatatypeSortRef, decl: z3.FuncDeclRef
) -> Optional[int]:
    """Return the constructor index associated with a recognizer declaration."""
    for idx in range(sort.num_constructors()):
        if decl.eq(sort.recognizer(idx)):
            return idx
    return None


def _collect_datatype_aliases_and_hints(
    formula: z3.ExprRef,
) -> Dict[int, Dict[str, Any]]:
    """Collect datatype equality classes and positive constructor evidence."""
    datatype_terms: Dict[int, z3.ExprRef] = {}
    adjacency: Dict[int, Set[int]] = {}
    constructor_hints: Dict[int, Set[int]] = {}

    def register_term(term: z3.ExprRef) -> None:
        if not _is_datatype_sort(term.sort()):
            return
        term_id = _ast_id(term)
        datatype_terms[term_id] = term
        adjacency.setdefault(term_id, set())
        constructor_hints.setdefault(term_id, set())

    def add_edge(lhs: z3.ExprRef, rhs: z3.ExprRef) -> None:
        lhs_id = _ast_id(lhs)
        rhs_id = _ast_id(rhs)
        adjacency.setdefault(lhs_id, set()).add(rhs_id)
        adjacency.setdefault(rhs_id, set()).add(lhs_id)

    for expr in _positive_conjuncts(formula):
        if not z3.is_app(expr):
            continue

        if z3.is_eq(expr):
            lhs, rhs = expr.children()
            if _is_datatype_sort(lhs.sort()) and _is_datatype_sort(rhs.sort()):
                register_term(lhs)
                register_term(rhs)
                add_edge(lhs, rhs)

                if rhs.decl().kind() == z3.Z3_OP_DT_CONSTRUCTOR:
                    sort = rhs.sort()
                    if isinstance(sort, z3.DatatypeSortRef):
                        idx = _find_constructor_index(sort, rhs.decl())
                        if idx is not None:
                            constructor_hints[_ast_id(lhs)].add(idx)

                if lhs.decl().kind() == z3.Z3_OP_DT_CONSTRUCTOR:
                    sort = lhs.sort()
                    if isinstance(sort, z3.DatatypeSortRef):
                        idx = _find_constructor_index(sort, lhs.decl())
                        if idx is not None:
                            constructor_hints[_ast_id(rhs)].add(idx)

        if expr.decl().kind() == z3.Z3_OP_DT_IS and expr.num_args() == 1:
            subject = expr.arg(0)
            if not _is_datatype_sort(subject.sort()):
                continue
            register_term(subject)
            sort = subject.sort()
            if isinstance(sort, z3.DatatypeSortRef):
                idx = _find_recognizer_constructor_index(sort, expr.decl())
                if idx is not None:
                    constructor_hints[_ast_id(subject)].add(idx)

    components: Dict[int, Dict[str, Any]] = {}
    visited: Set[int] = set()

    for root_id, root_term in datatype_terms.items():
        if root_id in visited:
            continue

        queue = [root_id]
        visited.add(root_id)
        member_ids: List[int] = []
        hints: Set[int] = set()

        while queue:
            current_id = queue.pop()
            member_ids.append(current_id)
            hints.update(constructor_hints.get(current_id, set()))

            for neighbor in adjacency.get(current_id, set()):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)

        component_terms = [datatype_terms[member_id] for member_id in member_ids]
        component = {
            "terms": _sorted_unique_terms(component_terms),
            "constructor_indices": sorted(hints),
        }
        for member_id in member_ids:
            components[member_id] = component

    return components


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
    default_terms: Optional[Sequence[z3.ExprRef]] = None,
) -> List[z3.ExprRef]:
    """Resolve projection terms against the tracked term set.

    If no projection is given, the full tracked term set is returned.
    Terms may be provided either as Z3 expressions or by their string form.
    """
    if projection_terms is None:
        if default_terms is not None:
            return list(default_terms)
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
    default_terms: Optional[Sequence[z3.ExprRef]] = None,
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

    return resolve_projection_terms(
        available_terms,
        projection_terms,
        default_terms=default_terms,
    )


def collect_ground_uf_terms(formula: z3.ExprRef) -> List[z3.ExprRef]:
    """Collect ground uninterpreted function applications from a formula."""
    terms: List[z3.ExprRef] = []
    for expr in _walk_exprs(formula):
        if not z3.is_app(expr):
            continue
        if expr.num_args() > 0 and expr.decl().kind() == z3.Z3_OP_UNINTERPRETED:
            terms.append(expr)
    return _sorted_unique_terms(terms)


def collect_datatype_observable_terms(
    formula: z3.ExprRef,
    include_selector_closure: bool = False,
) -> List[z3.ExprRef]:
    """Collect datatype observables from a formula.

    Observables include datatype-typed variables plus selector and tester terms
    that syntactically occur in the formula. When `include_selector_closure` is
    enabled, constructor evidence is propagated across datatype equalities and
    used to expose one level of selector terms for aliased datatype expressions.
    """
    datatype_variables = [
        var for var in get_variables(formula) if _is_datatype_sort(var.sort())
    ]
    observed_terms: List[z3.ExprRef] = list(datatype_variables)

    for expr in _walk_exprs(formula):
        if not z3.is_app(expr):
            continue
        if expr.decl().kind() in (z3.Z3_OP_DT_ACCESSOR, z3.Z3_OP_DT_IS):
            observed_terms.append(expr)

    if include_selector_closure:
        components = _collect_datatype_aliases_and_hints(formula)
        expanded_terms: List[z3.ExprRef] = []
        seen_component_terms: Set[tuple] = set()

        for component in components.values():
            component_term_ids = tuple(_ast_id(term) for term in component["terms"])
            if component_term_ids in seen_component_terms:
                continue
            seen_component_terms.add(component_term_ids)

            terms = component["terms"]
            if not terms:
                continue
            sort = terms[0].sort()
            if not isinstance(sort, z3.DatatypeSortRef):
                continue

            for constructor_idx in component["constructor_indices"]:
                constructor_decl = sort.constructor(constructor_idx)
                for accessor_idx in range(constructor_decl.arity()):
                    accessor_decl = sort.accessor(constructor_idx, accessor_idx)
                    for term in terms:
                        expanded_terms.append(accessor_decl(term))

        observed_terms.extend(expanded_terms)

    return _sorted_unique_terms(observed_terms)


def enumerate_projected_models(
    formula: z3.ExprRef,
    options: SamplingOptions,
    available_terms: Sequence[z3.ExprRef],
    default_terms: Optional[Sequence[z3.ExprRef]] = None,
    solver_factory: Optional[Callable[[], z3.Solver]] = None,
) -> SamplingResult:
    """Enumerate projected models for a finite-domain term space."""
    solver = solver_factory() if solver_factory is not None else z3.Solver()

    if options.random_seed is not None:
        solver.set("random_seed", options.random_seed)
        solver.set("seed", options.random_seed)
    if options.timeout is not None:
        solver.set("timeout", max(1, int(options.timeout * 1000)))

    solver.add(formula)

    projection_terms = resolve_projection_terms(
        available_terms,
        options.additional_options.get("projection_terms"),
        default_terms=default_terms,
    )
    output_terms = resolve_output_terms(
        available_terms,
        options.additional_options.get("projection_terms"),
        options.additional_options.get("tracked_terms"),
        bool(options.additional_options.get("return_full_model", False)),
        default_terms=default_terms,
    )

    samples: List[Dict[str, Any]] = []
    solver_checks = 0
    started_at = perf_counter()
    termination_reason = "exhausted"

    for _ in range(options.num_samples):
        check_result = solver.check()
        solver_checks += 1
        if check_result != z3.sat:
            termination_reason = str(check_result)
            break

        model = solver.model()
        samples.append(build_sample(model, output_terms))

        if not block_model_on_terms(solver, model, projection_terms):
            termination_reason = "no_projection_terms"
            break
    else:
        termination_reason = "sample_limit"

    elapsed_ms = int((perf_counter() - started_at) * 1000)
    stats: Dict[str, Any] = {
        "time_ms": elapsed_ms,
        "iterations": len(samples),
        "solver_checks": solver_checks,
        "tracked_term_count": len(available_terms),
        "projection_term_count": len(projection_terms),
        "output_term_count": len(output_terms),
        "projection_terms": [str(term) for term in projection_terms],
        "output_terms": [str(term) for term in output_terms],
        "termination_reason": termination_reason,
    }

    return SamplingResult(samples, stats)
