"""Typed expression generation for programming by example."""

from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .expressions import Expression, Theory, ValueType, default_value_type_for_theory
from .grammar import (
    TypedGrammar,
    default_grammar_for_signature,
    default_grammar_for_task,
)
from .task import (
    PBETask,
    get_output_type as _get_output_type,
    get_theory_from_variables as _get_theory_from_variables,
    get_variable_names as _get_variable_names,
    validate_examples as _validate_examples,
)

__all__ = [
    "generate_expressions_for_task",
    "generate_expressions_for_theory",
    "get_output_type",
    "get_theory_from_variables",
    "get_variable_names",
    "rank_expressions",
    "validate_examples",
]


def validate_examples(examples: List[Dict[str, Any]]) -> None:
    """Validate that examples form a well-typed PBE task."""
    _validate_examples(examples)


def get_theory_from_variables(
    examples: List[Dict[str, Any]], theory_hint: Optional[Theory] = None
) -> Theory:
    """Infer the synthesis theory for a set of examples."""
    return _get_theory_from_variables(examples, theory_hint=theory_hint)


def get_output_type(examples: List[Dict[str, Any]], theory: Theory) -> ValueType:
    """Infer the target output sort for a set of examples."""
    return _get_output_type(examples, theory)


def get_variable_names(examples: List[Dict[str, Any]]) -> List[str]:
    """Return variable names in deterministic order."""
    return _get_variable_names(examples)


def rank_expressions(expressions: Iterable[Expression]) -> List[Expression]:
    """Sort expressions by simplicity, then deterministically."""
    return sorted(expressions, key=lambda expr: (expr.structural_cost(), str(expr)))


def _observational_signature(
    expr: Expression, examples: Optional[List[Dict[str, Any]]]
) -> Optional[Tuple[Any, ...]]:
    if not examples:
        return None

    outputs: List[Any] = []
    for example in examples:
        try:
            outputs.append(expr.evaluate(example))
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return None
    return tuple(outputs)


def _all_expressions_for_type(
    by_depth: Dict[int, Dict[ValueType, List[Expression]]], value_type: ValueType
) -> List[Expression]:
    expressions: List[Expression] = []
    for type_map in by_depth.values():
        expressions.extend(type_map.get(value_type, []))
    return expressions


def _limit_for_argument(
    grammar: TypedGrammar,
    production_index: int,
    argument_index: int,
    max_candidates: int,
) -> int:
    del grammar, production_index
    return max(1, min(max_candidates, 16))


def _enumerate_with_grammar(
    grammar: TypedGrammar,
    *,
    output_type: ValueType,
    max_depth: int,
    max_candidates: int,
    examples: Optional[List[Dict[str, Any]]],
    deduplicate_observationally: bool,
) -> List[Expression]:
    if max_depth < 0:
        return []

    expansion_limit = max(1, min(max_candidates, 16))
    by_depth: Dict[int, Dict[ValueType, List[Expression]]] = {
        depth: {} for depth in range(max_depth + 1)
    }
    signature_best: Dict[Tuple[ValueType, Tuple[Any, ...]], Expression] = {}
    structural_seen: Set[Tuple[ValueType, str]] = set()

    def add_candidate(depth: int, expr: Expression) -> None:
        if depth > max_depth:
            return

        struct_key = (expr.value_type, str(expr))
        if struct_key in structural_seen:
            return

        if deduplicate_observationally:
            signature = _observational_signature(expr, examples)
            if signature is not None:
                signature_key = (expr.value_type, signature)
                incumbent = signature_best.get(signature_key)
                if incumbent is not None:
                    current_rank = (expr.structural_cost(), str(expr))
                    incumbent_rank = (incumbent.structural_cost(), str(incumbent))
                    if current_rank >= incumbent_rank:
                        return
                signature_best[signature_key] = expr

        structural_seen.add(struct_key)
        bucket = by_depth[depth].setdefault(expr.value_type, [])
        bucket.append(expr)
        bucket.sort(key=lambda current: (current.structural_cost(), str(current)))
        if len(bucket) > max_candidates:
            del bucket[max_candidates:]

    for production in grammar.terminal_productions():
        try:
            add_candidate(0, production.builder(tuple()))
        except ValueError:
            continue

    nonterminals = grammar.nonterminal_productions()
    for depth in range(1, max_depth + 1):
        for production_index, production in enumerate(nonterminals):
            arg_pools: List[List[Expression]] = []
            for argument_index, arg_type in enumerate(production.arg_types):
                limit = expansion_limit
                if production.arg_limits and argument_index < len(production.arg_limits):
                    arg_limit = production.arg_limits[argument_index]
                    if arg_limit is not None:
                        limit = max(1, min(limit, arg_limit))
                else:
                    limit = _limit_for_argument(
                        grammar,
                        production_index,
                        argument_index,
                        max_candidates,
                    )
                pool = _all_expressions_for_type(by_depth, arg_type)[:limit]
                if not pool:
                    arg_pools = []
                    break
                arg_pools.append(pool)

            if not arg_pools and production.arg_types:
                continue

            for args in product(*arg_pools):
                if production.predicate is not None and not production.predicate(args):
                    continue
                try:
                    add_candidate(depth, production.builder(tuple(args)))
                except ValueError:
                    continue

    return rank_expressions(_all_expressions_for_type(by_depth, output_type))


def generate_expressions_for_task(
    task: PBETask,
    max_depth: int = 3,
    max_candidates: int = 2000,
    deduplicate_observationally: bool = True,
) -> List[Expression]:
    """Generate ranked expressions for a typed task using the internal grammar."""
    grammar = default_grammar_for_task(task)
    return _enumerate_with_grammar(
        grammar,
        output_type=task.output_type,
        max_depth=max_depth,
        max_candidates=max_candidates,
        examples=task.as_examples(),
        deduplicate_observationally=deduplicate_observationally,
    )


def generate_expressions_for_theory(
    theory: Theory,
    variables: List[str],
    max_depth: int = 3,
    output_type: Optional[ValueType] = None,
    examples: Optional[List[Dict[str, Any]]] = None,
    bitwidth: int = 32,
    max_candidates: int = 2000,
    variable_types: Optional[Dict[str, ValueType]] = None,
    deduplicate_observationally: bool = True,
) -> List[Expression]:
    """Generate ranked expressions for a theory via the internal grammar."""
    resolved_output_type = output_type or default_value_type_for_theory(theory)
    resolved_variable_types = variable_types or {
        name: default_value_type_for_theory(theory) for name in variables
    }

    if examples:
        try:
            task = PBETask.from_examples(
                examples,
                theory_hint=theory,
                bitwidth=bitwidth,
            )
            return generate_expressions_for_task(
                task,
                max_depth=max_depth,
                max_candidates=max_candidates,
                deduplicate_observationally=deduplicate_observationally,
            )
        except ValueError:
            pass

    grammar = default_grammar_for_signature(
        theory,
        variables,
        resolved_variable_types,
        bitwidth=bitwidth,
        examples=examples,
    )
    return _enumerate_with_grammar(
        grammar,
        output_type=resolved_output_type,
        max_depth=max_depth,
        max_candidates=max_candidates,
        examples=examples,
        deduplicate_observationally=deduplicate_observationally,
    )
