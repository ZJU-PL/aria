"""Typed expression generation for programming by example."""

from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .expressions import (
    BinaryExpr,
    BinaryOp,
    Constant,
    Expression,
    FunctionCallExpr,
    IfExpr,
    Theory,
    UnaryExpr,
    UnaryOp,
    ValueType,
    const,
    default_value_type_for_theory,
    infer_value_type,
    var,
)


def validate_examples(examples: List[Dict[str, Any]]) -> None:
    """Validate that examples form a well-typed PBE task."""
    if not examples:
        raise ValueError("No examples provided")

    required_inputs: Optional[Set[str]] = None
    input_types: Dict[str, type] = {}
    output_type: Optional[type] = None

    for index, example in enumerate(examples):
        if "output" not in example:
            raise ValueError(f"Example {index + 1} is missing an 'output' field")

        input_names = {key for key in example if key != "output"}
        if required_inputs is None:
            required_inputs = input_names
        elif required_inputs != input_names:
            raise ValueError(
                "All examples must have the same input variables; "
                f"example {index + 1} differs"
            )

        for name in input_names:
            value = example[name]
            current_type = type(value)
            if name not in input_types:
                input_types[name] = current_type
            elif input_types[name] is not current_type:
                raise ValueError(
                    f"Input variable '{name}' has inconsistent types across examples"
                )

        current_output_type = type(example["output"])
        if output_type is None:
            output_type = current_output_type
        elif output_type is not current_output_type:
            raise ValueError("Example outputs must all have the same type")

    if not required_inputs:
        raise ValueError("At least one input variable is required")


def get_theory_from_variables(
    examples: List[Dict[str, Any]], theory_hint: Optional[Theory] = None
) -> Theory:
    """Infer the theory from example inputs, optionally honoring a hint."""
    if theory_hint is not None:
        return theory_hint

    validate_examples(examples)
    values = []
    for example in examples:
        for key, value in example.items():
            if key != "output":
                values.append(value)

    if all(isinstance(value, str) for value in values):
        return Theory.STRING
    if all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        return Theory.LIA
    raise ValueError(
        "Could not infer a supported theory from the provided examples; "
        "pass a theory hint explicitly"
    )


def get_output_type(
    examples: List[Dict[str, Any]], theory: Theory
) -> ValueType:
    """Infer the target output type from example outputs."""
    validate_examples(examples)
    outputs = [example["output"] for example in examples]
    first_type = infer_value_type(outputs[0], theory)

    for output in outputs[1:]:
        if infer_value_type(output, theory) != first_type:
            raise ValueError("All outputs must have the same type")

    return first_type


def get_variable_names(examples: List[Dict[str, Any]]) -> List[str]:
    """Return variable names in deterministic order."""
    validate_examples(examples)
    return sorted(key for key in examples[0] if key != "output")


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


def _example_constants(
    theory: Theory, examples: Optional[List[Dict[str, Any]]]
) -> Dict[ValueType, List[Any]]:
    observed: Dict[ValueType, Set[Any]] = {
        ValueType.INT: set(),
        ValueType.STRING: set(),
        ValueType.BV: set(),
    }

    if not examples:
        return {value_type: [] for value_type in observed}

    for example in examples:
        for value in example.values():
            if isinstance(value, bool):
                continue
            if isinstance(value, str) and len(value) <= 4:
                observed[ValueType.STRING].add(value)
            elif isinstance(value, int):
                if theory == Theory.BV:
                    observed[ValueType.BV].add(value)
                elif abs(value) <= 16:
                    observed[ValueType.INT].add(value)

    return {
        value_type: sorted(values, key=lambda item: (str(type(item)), item))
        for value_type, values in observed.items()
    }


def _base_constants(
    theory: Theory,
    bitwidth: int,
    examples: Optional[List[Dict[str, Any]]],
) -> Dict[ValueType, List[Expression]]:
    observed = _example_constants(theory, examples)
    if theory == Theory.STRING:
        return {
            ValueType.STRING: [
                const("", theory),
                const("a", theory),
                const("b", theory),
                const("0", theory),
                const("1", theory),
            ]
            + [const(value, theory) for value in observed[ValueType.STRING]],
            ValueType.INT: [
                const(0, theory, value_type=ValueType.INT),
                const(1, theory, value_type=ValueType.INT),
                const(2, theory, value_type=ValueType.INT),
            ]
            + [
                const(value, theory, value_type=ValueType.INT)
                for value in observed[ValueType.INT]
            ],
        }

    if theory == Theory.BV:
        return {
            ValueType.BV: [
                const(0, theory, value_type=ValueType.BV, bitwidth=bitwidth),
                const(1, theory, value_type=ValueType.BV, bitwidth=bitwidth),
                const(0x0F, theory, value_type=ValueType.BV, bitwidth=bitwidth),
                const(0xF0, theory, value_type=ValueType.BV, bitwidth=bitwidth),
                const(0xFF, theory, value_type=ValueType.BV, bitwidth=bitwidth),
            ]
            + [
                const(
                    value,
                    theory,
                    value_type=ValueType.BV,
                    bitwidth=bitwidth,
                )
                for value in observed[ValueType.BV]
            ]
        }

    return {
        ValueType.INT: [
            const(-1, theory),
            const(0, theory),
            const(1, theory),
            const(2, theory),
            const(10, theory),
        ]
        + [const(value, theory) for value in observed[ValueType.INT]]
    }


def _variable_type_for_theory(theory: Theory) -> ValueType:
    return default_value_type_for_theory(theory)


def _all_expressions_for_type(
    by_depth: Dict[int, Dict[ValueType, List[Expression]]], value_type: ValueType
) -> List[Expression]:
    expressions: List[Expression] = []
    for type_map in by_depth.values():
        expressions.extend(type_map.get(value_type, []))
    return expressions


def generate_expressions_for_theory(
    theory: Theory,
    variables: List[str],
    max_depth: int = 3,
    output_type: Optional[ValueType] = None,
    examples: Optional[List[Dict[str, Any]]] = None,
    bitwidth: int = 32,
    max_candidates: int = 2000,
) -> List[Expression]:
    """Generate ranked, typed expressions for a theory."""
    output_type = output_type or default_value_type_for_theory(theory)
    if max_depth < 0:
        return []
    expansion_limit = min(max_candidates, 16)

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

        signature = _observational_signature(expr, examples)
        if signature is not None:
            signature_key = (expr.value_type, signature)
            incumbent = signature_best.get(signature_key)
            if incumbent is not None:
                current_rank = (expr.structural_cost(), str(expr))
                incumbent_rank = (
                    incumbent.structural_cost(),
                    str(incumbent),
                )
                if current_rank >= incumbent_rank:
                    return
            signature_best[signature_key] = expr

        structural_seen.add(struct_key)
        bucket = by_depth[depth].setdefault(expr.value_type, [])
        bucket.append(expr)
        bucket.sort(key=lambda current: (current.structural_cost(), str(current)))
        if len(bucket) > max_candidates:
            del bucket[max_candidates:]

    variable_type = _variable_type_for_theory(theory)
    for name in variables:
        add_candidate(
            0,
            var(name, theory, value_type=variable_type, bitwidth=bitwidth),
        )

    for value_type, expressions in _base_constants(theory, bitwidth, examples).items():
        for expr in expressions:
            if expr.value_type == value_type:
                add_candidate(0, expr)

    for depth in range(1, max_depth + 1):
        ints = _all_expressions_for_type(by_depth, ValueType.INT)[:expansion_limit]
        bools = _all_expressions_for_type(by_depth, ValueType.BOOL)[:expansion_limit]
        strings = _all_expressions_for_type(by_depth, ValueType.STRING)[
            :expansion_limit
        ]
        bitvectors = _all_expressions_for_type(by_depth, ValueType.BV)[
            :expansion_limit
        ]

        if theory == Theory.LIA:
            for operand in ints:
                add_candidate(depth, UnaryExpr(UnaryOp.NEG, operand))
                add_candidate(depth, FunctionCallExpr("abs", [operand], theory))

            for left in ints:
                for right in ints:
                    add_candidate(depth, BinaryExpr(left, BinaryOp.ADD, right))
                    add_candidate(depth, BinaryExpr(left, BinaryOp.SUB, right))
                    add_candidate(depth, BinaryExpr(left, BinaryOp.MUL, right))
                    if not (
                        isinstance(right, Constant) and getattr(right, "value", None) == 0
                    ):
                        add_candidate(depth, BinaryExpr(left, BinaryOp.DIV, right))
                        add_candidate(depth, BinaryExpr(left, BinaryOp.MOD, right))
                    add_candidate(depth, FunctionCallExpr("min", [left, right], theory))
                    add_candidate(depth, FunctionCallExpr("max", [left, right], theory))
                    add_candidate(depth, BinaryExpr(left, BinaryOp.EQ, right))
                    add_candidate(depth, BinaryExpr(left, BinaryOp.NEQ, right))
                    add_candidate(depth, BinaryExpr(left, BinaryOp.LT, right))
                    add_candidate(depth, BinaryExpr(left, BinaryOp.LE, right))
                    add_candidate(depth, BinaryExpr(left, BinaryOp.GT, right))
                    add_candidate(depth, BinaryExpr(left, BinaryOp.GE, right))

        if theory == Theory.STRING:
            for operand in strings:
                add_candidate(depth, UnaryExpr(UnaryOp.LENGTH, operand))
                add_candidate(
                    depth,
                    FunctionCallExpr(
                        "str_substring",
                        [
                            operand,
                            const(0, theory, value_type=ValueType.INT),
                            const(1, theory, value_type=ValueType.INT),
                        ],
                        theory,
                    ),
                )

            for left in strings:
                for right in strings:
                    add_candidate(depth, BinaryExpr(left, BinaryOp.CONCAT, right))
                    add_candidate(depth, BinaryExpr(left, BinaryOp.EQ, right))
                    add_candidate(depth, BinaryExpr(left, BinaryOp.NEQ, right))
                    add_candidate(
                        depth,
                        FunctionCallExpr("str_indexof", [left, right], theory),
                    )

            for left in ints:
                for right in ints:
                    add_candidate(depth, BinaryExpr(left, BinaryOp.EQ, right))
                    add_candidate(depth, BinaryExpr(left, BinaryOp.NEQ, right))
                    add_candidate(depth, BinaryExpr(left, BinaryOp.LT, right))
                    add_candidate(depth, BinaryExpr(left, BinaryOp.LE, right))
                    add_candidate(depth, BinaryExpr(left, BinaryOp.GT, right))
                    add_candidate(depth, BinaryExpr(left, BinaryOp.GE, right))

        if theory == Theory.BV:
            for operand in bitvectors:
                add_candidate(depth, UnaryExpr(UnaryOp.BVNOT, operand))

            for left in bitvectors:
                for right in bitvectors:
                    add_candidate(depth, BinaryExpr(left, BinaryOp.BVAND, right))
                    add_candidate(depth, BinaryExpr(left, BinaryOp.BVOR, right))
                    add_candidate(depth, BinaryExpr(left, BinaryOp.BVXOR, right))
                    add_candidate(depth, BinaryExpr(left, BinaryOp.BVSLL, right))
                    add_candidate(depth, BinaryExpr(left, BinaryOp.BVSLR, right))
                    add_candidate(depth, BinaryExpr(left, BinaryOp.BVSRA, right))
                    add_candidate(depth, BinaryExpr(left, BinaryOp.EQ, right))
                    add_candidate(depth, BinaryExpr(left, BinaryOp.NEQ, right))

        if bools:
            for left in bools:
                add_candidate(depth, UnaryExpr(UnaryOp.NOT, left))
                for right in bools:
                    add_candidate(depth, BinaryExpr(left, BinaryOp.AND, right))
                    add_candidate(depth, BinaryExpr(left, BinaryOp.OR, right))
                    add_candidate(depth, BinaryExpr(left, BinaryOp.EQ, right))
                    add_candidate(depth, BinaryExpr(left, BinaryOp.NEQ, right))

        for condition in bools:
            for result_type in (ValueType.INT, ValueType.STRING, ValueType.BV):
                branch_pool = _all_expressions_for_type(by_depth, result_type)[
                    :expansion_limit
                ]
                for then_expr in branch_pool:
                    for else_expr in branch_pool:
                        try:
                            add_candidate(depth, IfExpr(condition, then_expr, else_expr))
                        except ValueError:
                            continue

    return rank_expressions(_all_expressions_for_type(by_depth, output_type))
