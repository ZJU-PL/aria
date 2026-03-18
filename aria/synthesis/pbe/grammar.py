"""Internal typed grammar definitions for programming by example."""

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

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
    var,
)
from .task import PBETask, VariableSignature

ExpressionTuple = Tuple[Expression, ...]
ExpressionBuilder = Callable[[ExpressionTuple], Expression]
ExpressionPredicate = Callable[[ExpressionTuple], bool]


@dataclass(frozen=True)
class GrammarProduction:
    """A typed grammar production for constructing expressions."""

    name: str
    kind: str
    theories: Tuple[Theory, ...]
    result_type: ValueType
    arg_types: Tuple[ValueType, ...]
    builder: ExpressionBuilder
    structural_cost: int = 1
    arg_limits: Optional[Tuple[Optional[int], ...]] = None
    predicate: Optional[ExpressionPredicate] = None

    def supports_theory(self, theory: Theory) -> bool:
        """Return whether the production applies to the selected theory."""
        return theory in self.theories


@dataclass(frozen=True)
class TypedGrammar:
    """A typed grammar for a single synthesis task."""

    theory: Theory
    productions: Tuple[GrammarProduction, ...]

    def terminal_productions(self) -> List[GrammarProduction]:
        """Return zero-arity productions."""
        return [production for production in self.productions if not production.arg_types]

    def nonterminal_productions(self) -> List[GrammarProduction]:
        """Return productions that combine existing expressions."""
        return [production for production in self.productions if production.arg_types]


def _deduplicate(values: List[Any]) -> List[Any]:
    deduplicated: List[Any] = []
    seen: Set[Any] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        deduplicated.append(value)
    return deduplicated


def _example_constants(
    theory: Theory, examples: Optional[List[Dict[str, Any]]]
) -> Dict[ValueType, List[Any]]:
    observed: Dict[ValueType, Set[Any]] = {
        ValueType.INT: set(),
        ValueType.STRING: set(),
        ValueType.BV: set(),
        ValueType.BOOL: set(),
    }

    if not examples:
        return {value_type: [] for value_type in observed}

    for example in examples:
        for value in example.values():
            if isinstance(value, bool):
                observed[ValueType.BOOL].add(value)
            elif isinstance(value, str) and len(value) <= 4:
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
            ValueType.STRING: _deduplicate(
                [
                    const("", theory),
                    const("a", theory),
                    const("b", theory),
                    const("0", theory),
                    const("1", theory),
                ]
                + [const(value, theory) for value in observed[ValueType.STRING]]
            ),
            ValueType.INT: _deduplicate(
                [
                    const(0, theory, value_type=ValueType.INT),
                    const(1, theory, value_type=ValueType.INT),
                    const(2, theory, value_type=ValueType.INT),
                ]
                + [
                    const(value, theory, value_type=ValueType.INT)
                    for value in observed[ValueType.INT]
                ]
            ),
            ValueType.BOOL: [
                const(False, theory, value_type=ValueType.BOOL),
                const(True, theory, value_type=ValueType.BOOL),
            ],
        }

    if theory == Theory.BV:
        return {
            ValueType.BV: _deduplicate(
                [
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
            ),
            ValueType.BOOL: [
                const(False, theory, value_type=ValueType.BOOL),
                const(True, theory, value_type=ValueType.BOOL),
            ],
        }

    return {
        ValueType.INT: _deduplicate(
            [
                const(-1, theory),
                const(0, theory),
                const(1, theory),
                const(2, theory),
                const(10, theory),
            ]
            + [const(value, theory) for value in observed[ValueType.INT]]
        ),
        ValueType.BOOL: [
            const(False, theory, value_type=ValueType.BOOL),
            const(True, theory, value_type=ValueType.BOOL),
        ],
    }


def _terminal_production(
    expr: Expression,
    *,
    name: str,
    kind: str,
) -> GrammarProduction:
    return GrammarProduction(
        name=name,
        kind=kind,
        theories=(expr.theory,),
        result_type=expr.value_type,
        arg_types=(),
        builder=lambda args, expr=expr: expr,
        structural_cost=expr.structural_cost(),
    )


def _operator_production(
    *,
    name: str,
    kind: str,
    theory: Theory,
    result_type: ValueType,
    arg_types: Tuple[ValueType, ...],
    builder: ExpressionBuilder,
    structural_cost: int = 1,
    arg_limits: Optional[Tuple[Optional[int], ...]] = None,
    predicate: Optional[ExpressionPredicate] = None,
) -> GrammarProduction:
    return GrammarProduction(
        name=name,
        kind=kind,
        theories=(theory,),
        result_type=result_type,
        arg_types=arg_types,
        builder=builder,
        structural_cost=structural_cost,
        arg_limits=arg_limits,
        predicate=predicate,
    )


def _default_inputs_for_signature(
    variable_names: List[str],
    variable_types: Dict[str, ValueType],
    bitwidth: int,
) -> Tuple[VariableSignature, ...]:
    return tuple(
        VariableSignature(
            name=name,
            value_type=variable_types[name],
            bitwidth=bitwidth if variable_types[name] == ValueType.BV else None,
        )
        for name in variable_names
    )


def _build_default_grammar(
    theory: Theory,
    inputs: Sequence[VariableSignature],
    *,
    bitwidth: int,
    examples: Optional[List[Dict[str, Any]]] = None,
) -> TypedGrammar:
    productions: List[GrammarProduction] = []

    for signature in inputs:
        productions.append(
            _terminal_production(
                var(
                    signature.name,
                    theory,
                    value_type=signature.value_type,
                    bitwidth=signature.bitwidth,
                ),
                name=signature.name,
                kind="variable",
            )
        )

    for expressions in _base_constants(theory, bitwidth, examples).values():
        for expr in expressions:
            productions.append(
                _terminal_production(expr, name=str(expr), kind="constant")
            )

    productions.extend(
        [
            _operator_production(
                name="bool-not",
                kind="unary",
                theory=theory,
                result_type=ValueType.BOOL,
                arg_types=(ValueType.BOOL,),
                builder=lambda args: UnaryExpr(UnaryOp.NOT, args[0]),
            ),
            _operator_production(
                name="bool-and",
                kind="binary",
                theory=theory,
                result_type=ValueType.BOOL,
                arg_types=(ValueType.BOOL, ValueType.BOOL),
                builder=lambda args: BinaryExpr(args[0], BinaryOp.AND, args[1]),
            ),
            _operator_production(
                name="bool-or",
                kind="binary",
                theory=theory,
                result_type=ValueType.BOOL,
                arg_types=(ValueType.BOOL, ValueType.BOOL),
                builder=lambda args: BinaryExpr(args[0], BinaryOp.OR, args[1]),
            ),
            _operator_production(
                name="bool-eq",
                kind="binary",
                theory=theory,
                result_type=ValueType.BOOL,
                arg_types=(ValueType.BOOL, ValueType.BOOL),
                builder=lambda args: BinaryExpr(args[0], BinaryOp.EQ, args[1]),
            ),
            _operator_production(
                name="bool-neq",
                kind="binary",
                theory=theory,
                result_type=ValueType.BOOL,
                arg_types=(ValueType.BOOL, ValueType.BOOL),
                builder=lambda args: BinaryExpr(args[0], BinaryOp.NEQ, args[1]),
            ),
        ]
    )

    for result_type in (ValueType.INT, ValueType.STRING, ValueType.BV):
        productions.append(
            _operator_production(
                name=f"if-{result_type.value}",
                kind="if",
                theory=theory,
                result_type=result_type,
                arg_types=(ValueType.BOOL, result_type, result_type),
                builder=lambda args: IfExpr(args[0], args[1], args[2]),
            )
        )

    if theory == Theory.LIA:
        productions.extend(
            [
                _operator_production(
                    name="neg",
                    kind="unary",
                    theory=theory,
                    result_type=ValueType.INT,
                    arg_types=(ValueType.INT,),
                    builder=lambda args: UnaryExpr(UnaryOp.NEG, args[0]),
                ),
                _operator_production(
                    name="abs",
                    kind="function",
                    theory=theory,
                    result_type=ValueType.INT,
                    arg_types=(ValueType.INT,),
                    builder=lambda args: FunctionCallExpr("abs", list(args), theory),
                ),
                _operator_production(
                    name="add",
                    kind="binary",
                    theory=theory,
                    result_type=ValueType.INT,
                    arg_types=(ValueType.INT, ValueType.INT),
                    builder=lambda args: BinaryExpr(args[0], BinaryOp.ADD, args[1]),
                ),
                _operator_production(
                    name="sub",
                    kind="binary",
                    theory=theory,
                    result_type=ValueType.INT,
                    arg_types=(ValueType.INT, ValueType.INT),
                    builder=lambda args: BinaryExpr(args[0], BinaryOp.SUB, args[1]),
                ),
                _operator_production(
                    name="mul",
                    kind="binary",
                    theory=theory,
                    result_type=ValueType.INT,
                    arg_types=(ValueType.INT, ValueType.INT),
                    builder=lambda args: BinaryExpr(args[0], BinaryOp.MUL, args[1]),
                ),
                _operator_production(
                    name="div",
                    kind="binary",
                    theory=theory,
                    result_type=ValueType.INT,
                    arg_types=(ValueType.INT, ValueType.INT),
                    builder=lambda args: BinaryExpr(args[0], BinaryOp.DIV, args[1]),
                    predicate=lambda args: not (
                        isinstance(args[1], Constant)
                        and getattr(args[1], "value", None) == 0
                    ),
                ),
                _operator_production(
                    name="mod",
                    kind="binary",
                    theory=theory,
                    result_type=ValueType.INT,
                    arg_types=(ValueType.INT, ValueType.INT),
                    builder=lambda args: BinaryExpr(args[0], BinaryOp.MOD, args[1]),
                    predicate=lambda args: not (
                        isinstance(args[1], Constant)
                        and getattr(args[1], "value", None) == 0
                    ),
                ),
                _operator_production(
                    name="min",
                    kind="function",
                    theory=theory,
                    result_type=ValueType.INT,
                    arg_types=(ValueType.INT, ValueType.INT),
                    builder=lambda args: FunctionCallExpr("min", list(args), theory),
                ),
                _operator_production(
                    name="max",
                    kind="function",
                    theory=theory,
                    result_type=ValueType.INT,
                    arg_types=(ValueType.INT, ValueType.INT),
                    builder=lambda args: FunctionCallExpr("max", list(args), theory),
                ),
            ]
        )

        for name, op in (
            ("eq", BinaryOp.EQ),
            ("neq", BinaryOp.NEQ),
            ("lt", BinaryOp.LT),
            ("le", BinaryOp.LE),
            ("gt", BinaryOp.GT),
            ("ge", BinaryOp.GE),
        ):
            productions.append(
                _operator_production(
                    name=f"int-{name}",
                    kind="binary",
                    theory=theory,
                    result_type=ValueType.BOOL,
                    arg_types=(ValueType.INT, ValueType.INT),
                    builder=lambda args, op=op: BinaryExpr(args[0], op, args[1]),
                )
            )

    if theory == Theory.STRING:
        productions.extend(
            [
                _operator_production(
                    name="length",
                    kind="unary",
                    theory=theory,
                    result_type=ValueType.INT,
                    arg_types=(ValueType.STRING,),
                    builder=lambda args: UnaryExpr(UnaryOp.LENGTH, args[0]),
                ),
                _operator_production(
                    name="concat",
                    kind="binary",
                    theory=theory,
                    result_type=ValueType.STRING,
                    arg_types=(ValueType.STRING, ValueType.STRING),
                    builder=lambda args: BinaryExpr(args[0], BinaryOp.CONCAT, args[1]),
                ),
                _operator_production(
                    name="string-eq",
                    kind="binary",
                    theory=theory,
                    result_type=ValueType.BOOL,
                    arg_types=(ValueType.STRING, ValueType.STRING),
                    builder=lambda args: BinaryExpr(args[0], BinaryOp.EQ, args[1]),
                ),
                _operator_production(
                    name="string-neq",
                    kind="binary",
                    theory=theory,
                    result_type=ValueType.BOOL,
                    arg_types=(ValueType.STRING, ValueType.STRING),
                    builder=lambda args: BinaryExpr(args[0], BinaryOp.NEQ, args[1]),
                ),
                _operator_production(
                    name="str-substring",
                    kind="function",
                    theory=theory,
                    result_type=ValueType.STRING,
                    arg_types=(ValueType.STRING, ValueType.INT, ValueType.INT),
                    builder=lambda args: FunctionCallExpr(
                        "str_substring", list(args), theory
                    ),
                    arg_limits=(None, 6, 6),
                ),
                _operator_production(
                    name="str-indexof",
                    kind="function",
                    theory=theory,
                    result_type=ValueType.INT,
                    arg_types=(ValueType.STRING, ValueType.STRING),
                    builder=lambda args: FunctionCallExpr(
                        "str_indexof", list(args), theory
                    ),
                ),
            ]
        )

        for name, op in (
            ("eq", BinaryOp.EQ),
            ("neq", BinaryOp.NEQ),
            ("lt", BinaryOp.LT),
            ("le", BinaryOp.LE),
            ("gt", BinaryOp.GT),
            ("ge", BinaryOp.GE),
        ):
            productions.append(
                _operator_production(
                    name=f"int-{name}",
                    kind="binary",
                    theory=theory,
                    result_type=ValueType.BOOL,
                    arg_types=(ValueType.INT, ValueType.INT),
                    builder=lambda args, op=op: BinaryExpr(args[0], op, args[1]),
                )
            )

    if theory == Theory.BV:
        productions.extend(
            [
                _operator_production(
                    name="bv-not",
                    kind="unary",
                    theory=theory,
                    result_type=ValueType.BV,
                    arg_types=(ValueType.BV,),
                    builder=lambda args: UnaryExpr(UnaryOp.BVNOT, args[0]),
                ),
            ]
        )
        for name, op in (
            ("and", BinaryOp.BVAND),
            ("or", BinaryOp.BVOR),
            ("xor", BinaryOp.BVXOR),
            ("sll", BinaryOp.BVSLL),
            ("slr", BinaryOp.BVSLR),
            ("sra", BinaryOp.BVSRA),
        ):
            productions.append(
                _operator_production(
                    name=f"bv-{name}",
                    kind="binary",
                    theory=theory,
                    result_type=ValueType.BV,
                    arg_types=(ValueType.BV, ValueType.BV),
                    builder=lambda args, op=op: BinaryExpr(args[0], op, args[1]),
                )
            )
        for name, op in (("eq", BinaryOp.EQ), ("neq", BinaryOp.NEQ)):
            productions.append(
                _operator_production(
                    name=f"bv-{name}",
                    kind="binary",
                    theory=theory,
                    result_type=ValueType.BOOL,
                    arg_types=(ValueType.BV, ValueType.BV),
                    builder=lambda args, op=op: BinaryExpr(args[0], op, args[1]),
                )
            )

    return TypedGrammar(theory=theory, productions=tuple(productions))


def default_grammar_for_task(task: PBETask) -> TypedGrammar:
    """Return the default internal grammar for a typed PBE task."""
    return _build_default_grammar(
        task.theory,
        task.inputs,
        bitwidth=task.bitwidth,
        examples=task.as_examples(),
    )


def default_grammar_for_signature(
    theory: Theory,
    variable_names: List[str],
    variable_types: Dict[str, ValueType],
    *,
    bitwidth: int = 32,
    examples: Optional[List[Dict[str, Any]]] = None,
) -> TypedGrammar:
    """Return the default grammar for a raw typed signature."""
    inputs = _default_inputs_for_signature(variable_names, variable_types, bitwidth)
    return _build_default_grammar(
        theory,
        inputs,
        bitwidth=bitwidth,
        examples=examples,
    )
