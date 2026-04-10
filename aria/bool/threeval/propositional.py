"""Three-valued propositional logic utilities.

This module provides a small AST, multiple three-valued semantics, exhaustive
valuation utilities, semantic checks, truth-table generation, and lightweight
formula simplification under partial valuations.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product
from typing import AbstractSet, Iterable, Iterator, Mapping, Optional, Sequence, Union


class TruthValue(Enum):
    """Truth values for three-valued propositional logics."""

    FALSE = 0
    UNKNOWN = 1
    TRUE = 2

    @classmethod
    def coerce(cls, value: TruthLike) -> "TruthValue":
        if isinstance(value, cls):
            return value
        if value is True:
            return cls.TRUE
        if value is False:
            return cls.FALSE
        if value is None:
            return cls.UNKNOWN
        raise TypeError(f"Unsupported truth value: {value!r}")

    def __str__(self) -> str:
        if self is TruthValue.TRUE:
            return "T"
        if self is TruthValue.FALSE:
            return "F"
        return "U"


TruthLike = Union[TruthValue, bool, None]
_BINARY_TABLE = Mapping[tuple[TruthValue, TruthValue], TruthValue]


def _coerce_table(
    table: Mapping[tuple[TruthValue, TruthValue], TruthLike]
) -> dict[tuple[TruthValue, TruthValue], TruthValue]:
    result = {}
    for key, value in table.items():
        left, right = key
        result[(TruthValue.coerce(left), TruthValue.coerce(right))] = (
            TruthValue.coerce(value)
        )
    return result


def _table_from_callable(func) -> dict[tuple[TruthValue, TruthValue], TruthValue]:
    values = (TruthValue.FALSE, TruthValue.UNKNOWN, TruthValue.TRUE)
    return {
        (left, right): TruthValue.coerce(func(left, right))
        for left, right in product(values, repeat=2)
    }


@dataclass(frozen=True)
class Semantics:
    """Truth-functional semantics over ``FALSE``, ``UNKNOWN``, and ``TRUE``."""

    name: str
    designated_values: frozenset[TruthValue]
    negation_table: Mapping[TruthValue, TruthValue]
    conjunction_table: Mapping[tuple[TruthValue, TruthValue], TruthValue]
    disjunction_table: Mapping[tuple[TruthValue, TruthValue], TruthValue]
    implication_table: Optional[Mapping[tuple[TruthValue, TruthValue], TruthValue]] = (
        None
    )

    def __post_init__(self) -> None:
        values = (TruthValue.FALSE, TruthValue.UNKNOWN, TruthValue.TRUE)
        object.__setattr__(
            self,
            "designated_values",
            frozenset(TruthValue.coerce(value) for value in self.designated_values),
        )
        object.__setattr__(
            self,
            "negation_table",
            {TruthValue.coerce(key): TruthValue.coerce(value)
             for key, value in self.negation_table.items()},
        )
        object.__setattr__(self, "conjunction_table", _coerce_table(self.conjunction_table))
        object.__setattr__(self, "disjunction_table", _coerce_table(self.disjunction_table))
        if self.implication_table is not None:
            object.__setattr__(
                self,
                "implication_table",
                _coerce_table(self.implication_table),
            )

        for value in values:
            if value not in self.negation_table:
                raise ValueError(
                    f"Missing negation entry for {value!r} in semantics {self.name!r}"
                )
        for table_name in ("conjunction_table", "disjunction_table"):
            table = getattr(self, table_name)
            for left, right in product(values, repeat=2):
                if (left, right) not in table:
                    raise ValueError(
                        f"Missing {table_name} entry {(left, right)!r} in {self.name!r}"
                    )
        if self.implication_table is not None:
            for left, right in product(values, repeat=2):
                if (left, right) not in self.implication_table:
                    raise ValueError(
                        f"Missing implication entry {(left, right)!r} in {self.name!r}"
                    )

    def with_designated_values(self, values: Iterable[TruthLike]) -> "Semantics":
        return Semantics(
            name=self.name,
            designated_values=frozenset(TruthValue.coerce(value) for value in values),
            negation_table=self.negation_table,
            conjunction_table=self.conjunction_table,
            disjunction_table=self.disjunction_table,
            implication_table=self.implication_table,
        )

    def negate(self, value: TruthLike) -> TruthValue:
        return self.negation_table[TruthValue.coerce(value)]

    def and_(self, left: TruthLike, right: TruthLike) -> TruthValue:
        return self.conjunction_table[
            (TruthValue.coerce(left), TruthValue.coerce(right))
        ]

    def or_(self, left: TruthLike, right: TruthLike) -> TruthValue:
        return self.disjunction_table[(TruthValue.coerce(left), TruthValue.coerce(right))]

    def implies(self, left: TruthLike, right: TruthLike) -> TruthValue:
        left_value = TruthValue.coerce(left)
        right_value = TruthValue.coerce(right)
        if self.implication_table is not None:
            return self.implication_table[(left_value, right_value)]
        return self.or_(self.negate(left_value), right_value)

    def iff(self, left: TruthLike, right: TruthLike) -> TruthValue:
        left_value = TruthValue.coerce(left)
        right_value = TruthValue.coerce(right)
        return self.and_(
            self.implies(left_value, right_value),
            self.implies(right_value, left_value),
        )

    def xor(self, left: TruthLike, right: TruthLike) -> TruthValue:
        return self.negate(self.iff(left, right))

    def nand(self, left: TruthLike, right: TruthLike) -> TruthValue:
        return self.negate(self.and_(left, right))

    def nor(self, left: TruthLike, right: TruthLike) -> TruthValue:
        return self.negate(self.or_(left, right))

    def fold_and(self, values: Sequence[TruthLike]) -> TruthValue:
        result = TruthValue.TRUE
        for value in values:
            result = self.and_(result, value)
        return result

    def fold_or(self, values: Sequence[TruthLike]) -> TruthValue:
        result = TruthValue.FALSE
        for value in values:
            result = self.or_(result, value)
        return result

    def is_designated(
        self,
        value: TruthLike,
        designated_values: Optional[AbstractSet[TruthValue]] = None,
    ) -> bool:
        active_designated_values = self.designated_values
        if designated_values is not None:
            active_designated_values = frozenset(
                TruthValue.coerce(candidate) for candidate in designated_values
            )
        return TruthValue.coerce(value) in active_designated_values


def _strong_kleene_negation_table() -> dict[TruthValue, TruthValue]:
    return {
        TruthValue.FALSE: TruthValue.TRUE,
        TruthValue.UNKNOWN: TruthValue.UNKNOWN,
        TruthValue.TRUE: TruthValue.FALSE,
    }


def _strong_kleene_conjunction(left: TruthValue, right: TruthValue) -> TruthValue:
    if TruthValue.FALSE in (left, right):
        return TruthValue.FALSE
    if TruthValue.UNKNOWN in (left, right):
        return TruthValue.UNKNOWN
    return TruthValue.TRUE


def _strong_kleene_disjunction(left: TruthValue, right: TruthValue) -> TruthValue:
    if TruthValue.TRUE in (left, right):
        return TruthValue.TRUE
    if TruthValue.UNKNOWN in (left, right):
        return TruthValue.UNKNOWN
    return TruthValue.FALSE


def _weak_kleene_binary(left: TruthValue, right: TruthValue, operator: str) -> TruthValue:
    if TruthValue.UNKNOWN in (left, right):
        return TruthValue.UNKNOWN
    if operator == "and":
        return TruthValue.TRUE if left is TruthValue.TRUE and right is TruthValue.TRUE else TruthValue.FALSE
    if operator == "or":
        return TruthValue.TRUE if TruthValue.TRUE in (left, right) else TruthValue.FALSE
    raise ValueError(f"Unknown weak Kleene operator: {operator!r}")


def _lukasiewicz_implication(left: TruthValue, right: TruthValue) -> TruthValue:
    score = min(2, 2 - left.value + right.value)
    return TruthValue(score)


def _godel_implication(left: TruthValue, right: TruthValue) -> TruthValue:
    if left.value <= right.value:
        return TruthValue.TRUE
    return right


STRONG_KLEENE = Semantics(
    name="strong_kleene",
    designated_values=frozenset({TruthValue.TRUE}),
    negation_table=_strong_kleene_negation_table(),
    conjunction_table=_table_from_callable(_strong_kleene_conjunction),
    disjunction_table=_table_from_callable(_strong_kleene_disjunction),
)

WEAK_KLEENE = Semantics(
    name="weak_kleene",
    designated_values=frozenset({TruthValue.TRUE}),
    negation_table=_strong_kleene_negation_table(),
    conjunction_table=_table_from_callable(
        lambda left, right: _weak_kleene_binary(left, right, "and")
    ),
    disjunction_table=_table_from_callable(
        lambda left, right: _weak_kleene_binary(left, right, "or")
    ),
)

LUKASIEWICZ_K3 = Semantics(
    name="lukasiewicz_k3",
    designated_values=frozenset({TruthValue.TRUE}),
    negation_table=_strong_kleene_negation_table(),
    conjunction_table=_table_from_callable(lambda left, right: min(left, right, key=lambda value: value.value)),
    disjunction_table=_table_from_callable(lambda left, right: max(left, right, key=lambda value: value.value)),
    implication_table=_table_from_callable(_lukasiewicz_implication),
)

GODEL_G3 = Semantics(
    name="godel_g3",
    designated_values=frozenset({TruthValue.TRUE}),
    negation_table=_strong_kleene_negation_table(),
    conjunction_table=_table_from_callable(lambda left, right: min(left, right, key=lambda value: value.value)),
    disjunction_table=_table_from_callable(lambda left, right: max(left, right, key=lambda value: value.value)),
    implication_table=_table_from_callable(_godel_implication),
)


class Formula:
    """Base class for three-valued propositional formulas."""

    def evaluate(
        self,
        valuation: Mapping[str, TruthLike],
        semantics: Semantics = STRONG_KLEENE,
    ) -> TruthValue:
        raise NotImplementedError

    def variables(self) -> set[str]:
        raise NotImplementedError

    def atoms(self) -> set[str]:
        return self.variables()

    def subformulas(self) -> tuple["Formula", ...]:
        return (self,)

    def size(self) -> int:
        return len(self.subformulas())

    def depth(self) -> int:
        return 1

    def substitute(self, substitution: Mapping[str, "Formula"]) -> "Formula":
        del substitution
        return self

    def to_string(self, unicode: bool = False) -> str:
        return format_formula(self, unicode=unicode)

    def __str__(self) -> str:
        return self.to_string()


@dataclass(frozen=True)
class Constant(Formula):
    value: TruthValue

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", TruthValue.coerce(self.value))

    def evaluate(
        self,
        valuation: Mapping[str, TruthLike],
        semantics: Semantics = STRONG_KLEENE,
    ) -> TruthValue:
        del valuation
        del semantics
        return self.value

    def variables(self) -> set[str]:
        return set()


@dataclass(frozen=True)
class Variable(Formula):
    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError(f"Variable names must be strings, got {self.name!r}")

    def evaluate(
        self,
        valuation: Mapping[str, TruthLike],
        semantics: Semantics = STRONG_KLEENE,
    ) -> TruthValue:
        del semantics
        return TruthValue.coerce(valuation.get(self.name, TruthValue.UNKNOWN))

    def variables(self) -> set[str]:
        return {self.name}

    def substitute(self, substitution: Mapping[str, Formula]) -> Formula:
        return substitution.get(self.name, self)


@dataclass(frozen=True)
class UnaryFormula(Formula):
    operand: Formula

    def __post_init__(self) -> None:
        if not isinstance(self.operand, Formula):
            raise TypeError(
                f"Unary operand must be a Formula, got {self.operand!r}"
            )

    def variables(self) -> set[str]:
        return self.operand.variables()

    def subformulas(self) -> tuple[Formula, ...]:
        return (self,) + self.operand.subformulas()

    def depth(self) -> int:
        return 1 + self.operand.depth()


@dataclass(frozen=True)
class Not(UnaryFormula):
    def evaluate(
        self,
        valuation: Mapping[str, TruthLike],
        semantics: Semantics = STRONG_KLEENE,
    ) -> TruthValue:
        return semantics.negate(self.operand.evaluate(valuation, semantics))

    def substitute(self, substitution: Mapping[str, Formula]) -> Formula:
        return Not(self.operand.substitute(substitution))


@dataclass(frozen=True)
class BinaryFormula(Formula):
    left: Formula
    right: Formula

    def __post_init__(self) -> None:
        if not isinstance(self.left, Formula):
            raise TypeError(
                f"Binary left operand must be a Formula, got {self.left!r}"
            )
        if not isinstance(self.right, Formula):
            raise TypeError(
                f"Binary right operand must be a Formula, got {self.right!r}"
            )

    def variables(self) -> set[str]:
        return self.left.variables() | self.right.variables()

    def subformulas(self) -> tuple[Formula, ...]:
        return (self,) + self.left.subformulas() + self.right.subformulas()

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())


@dataclass(frozen=True)
class And(BinaryFormula):
    def evaluate(
        self,
        valuation: Mapping[str, TruthLike],
        semantics: Semantics = STRONG_KLEENE,
    ) -> TruthValue:
        return semantics.and_(
            self.left.evaluate(valuation, semantics),
            self.right.evaluate(valuation, semantics),
        )

    def substitute(self, substitution: Mapping[str, Formula]) -> Formula:
        return And(
            self.left.substitute(substitution),
            self.right.substitute(substitution),
        )


@dataclass(frozen=True)
class Or(BinaryFormula):
    def evaluate(
        self,
        valuation: Mapping[str, TruthLike],
        semantics: Semantics = STRONG_KLEENE,
    ) -> TruthValue:
        return semantics.or_(
            self.left.evaluate(valuation, semantics),
            self.right.evaluate(valuation, semantics),
        )

    def substitute(self, substitution: Mapping[str, Formula]) -> Formula:
        return Or(
            self.left.substitute(substitution),
            self.right.substitute(substitution),
        )


@dataclass(frozen=True)
class Implies(BinaryFormula):
    def evaluate(
        self,
        valuation: Mapping[str, TruthLike],
        semantics: Semantics = STRONG_KLEENE,
    ) -> TruthValue:
        return semantics.implies(
            self.left.evaluate(valuation, semantics),
            self.right.evaluate(valuation, semantics),
        )

    def substitute(self, substitution: Mapping[str, Formula]) -> Formula:
        return Implies(
            self.left.substitute(substitution),
            self.right.substitute(substitution),
        )


@dataclass(frozen=True)
class Iff(BinaryFormula):
    def evaluate(
        self,
        valuation: Mapping[str, TruthLike],
        semantics: Semantics = STRONG_KLEENE,
    ) -> TruthValue:
        return semantics.iff(
            self.left.evaluate(valuation, semantics),
            self.right.evaluate(valuation, semantics),
        )

    def substitute(self, substitution: Mapping[str, Formula]) -> Formula:
        return Iff(
            self.left.substitute(substitution),
            self.right.substitute(substitution),
        )


@dataclass(frozen=True)
class Xor(BinaryFormula):
    def evaluate(
        self,
        valuation: Mapping[str, TruthLike],
        semantics: Semantics = STRONG_KLEENE,
    ) -> TruthValue:
        return semantics.xor(
            self.left.evaluate(valuation, semantics),
            self.right.evaluate(valuation, semantics),
        )

    def substitute(self, substitution: Mapping[str, Formula]) -> Formula:
        return Xor(
            self.left.substitute(substitution),
            self.right.substitute(substitution),
        )


@dataclass(frozen=True)
class Nand(BinaryFormula):
    def evaluate(
        self,
        valuation: Mapping[str, TruthLike],
        semantics: Semantics = STRONG_KLEENE,
    ) -> TruthValue:
        return semantics.nand(
            self.left.evaluate(valuation, semantics),
            self.right.evaluate(valuation, semantics),
        )

    def substitute(self, substitution: Mapping[str, Formula]) -> Formula:
        return Nand(
            self.left.substitute(substitution),
            self.right.substitute(substitution),
        )


@dataclass(frozen=True)
class Nor(BinaryFormula):
    def evaluate(
        self,
        valuation: Mapping[str, TruthLike],
        semantics: Semantics = STRONG_KLEENE,
    ) -> TruthValue:
        return semantics.nor(
            self.left.evaluate(valuation, semantics),
            self.right.evaluate(valuation, semantics),
        )

    def substitute(self, substitution: Mapping[str, Formula]) -> Formula:
        return Nor(
            self.left.substitute(substitution),
            self.right.substitute(substitution),
        )


@dataclass(frozen=True)
class NAryFormula(Formula):
    operands: tuple[Formula, ...]

    def __post_init__(self) -> None:
        operands = tuple(self.operands)
        if not operands:
            raise ValueError(f"{type(self).__name__} requires at least one operand")
        for operand in operands:
            if not isinstance(operand, Formula):
                raise TypeError(
                    f"N-ary operands must be Formula instances, got {operand!r}"
                )
        object.__setattr__(self, "operands", operands)

    def variables(self) -> set[str]:
        variables = set()
        for operand in self.operands:
            variables |= operand.variables()
        return variables

    def subformulas(self) -> tuple[Formula, ...]:
        nested = []
        for operand in self.operands:
            nested.extend(operand.subformulas())
        return (self, *nested)

    def depth(self) -> int:
        return 1 + max(operand.depth() for operand in self.operands)


@dataclass(frozen=True)
class NaryAnd(NAryFormula):
    def evaluate(
        self,
        valuation: Mapping[str, TruthLike],
        semantics: Semantics = STRONG_KLEENE,
    ) -> TruthValue:
        return semantics.fold_and(
            [operand.evaluate(valuation, semantics) for operand in self.operands]
        )

    def substitute(self, substitution: Mapping[str, Formula]) -> Formula:
        return NaryAnd(tuple(operand.substitute(substitution) for operand in self.operands))


@dataclass(frozen=True)
class NaryOr(NAryFormula):
    def evaluate(
        self,
        valuation: Mapping[str, TruthLike],
        semantics: Semantics = STRONG_KLEENE,
    ) -> TruthValue:
        return semantics.fold_or(
            [operand.evaluate(valuation, semantics) for operand in self.operands]
        )

    def substitute(self, substitution: Mapping[str, Formula]) -> Formula:
        return NaryOr(tuple(operand.substitute(substitution) for operand in self.operands))


def conjoin(*operands: Formula) -> Formula:
    """Build a conjunction from zero or more operands."""

    if not operands:
        return Constant(TruthValue.TRUE)
    if len(operands) == 1:
        return operands[0]
    return NaryAnd(tuple(operands))


def disjoin(*operands: Formula) -> Formula:
    """Build a disjunction from zero or more operands."""

    if not operands:
        return Constant(TruthValue.FALSE)
    if len(operands) == 1:
        return operands[0]
    return NaryOr(tuple(operands))


_ASCII_SYMBOLS = {
    Constant(TruthValue.TRUE): "true",
    Constant(TruthValue.FALSE): "false",
    Constant(TruthValue.UNKNOWN): "unknown",
}
_UNICODE_SYMBOLS = {
    Constant(TruthValue.TRUE): "⊤",
    Constant(TruthValue.FALSE): "⊥",
    Constant(TruthValue.UNKNOWN): "?",
}

_PRECEDENCE = {
    Constant: 8,
    Variable: 8,
    Not: 7,
    NaryAnd: 6,
    And: 6,
    Nand: 6,
    NaryOr: 5,
    Or: 5,
    Nor: 5,
    Xor: 4,
    Implies: 3,
    Iff: 2,
}


def format_formula(formula: Formula, unicode: bool = False) -> str:
    """Render a formula in readable infix notation."""

    symbols = _UNICODE_SYMBOLS if unicode else _ASCII_SYMBOLS
    unary_symbol = "¬" if unicode else "!"
    binary_symbols = {
        And: "∧" if unicode else "&",
        Or: "∨" if unicode else "|",
        Implies: "→" if unicode else "->",
        Iff: "↔" if unicode else "<->",
        Xor: "⊕" if unicode else "xor",
        Nand: "⊼" if unicode else "nand",
        Nor: "⊽" if unicode else "nor",
        NaryAnd: "∧" if unicode else "&",
        NaryOr: "∨" if unicode else "|",
    }

    def render(node: Formula, parent_precedence: int) -> str:
        node_precedence = _PRECEDENCE[type(node)]
        if isinstance(node, Constant):
            text = symbols[node]
        elif isinstance(node, Variable):
            text = node.name
        elif isinstance(node, Not):
            rendered_operand = render(node.operand, node_precedence)
            text = f"{unary_symbol}{rendered_operand}"
        elif isinstance(node, BinaryFormula):
            symbol = binary_symbols[type(node)]
            left = render(node.left, node_precedence)
            right = render(node.right, node_precedence + 1)
            text = f"{left} {symbol} {right}"
        elif isinstance(node, NAryFormula):
            symbol = binary_symbols[type(node)]
            text = f" {symbol} ".join(
                render(operand, node_precedence) for operand in node.operands
            )
        else:
            raise TypeError(f"Unsupported formula node: {node!r}")

        if node_precedence < parent_precedence:
            return f"({text})"
        return text

    return render(formula, 0)


def all_valuations(
    variables: Iterable[str],
    truth_values: Optional[Sequence[TruthLike]] = None,
) -> Iterator[dict[str, TruthValue]]:
    """Enumerate all assignments for the given variables."""

    names = sorted(set(variables))
    active_truth_values = (
        tuple(TruthValue.coerce(value) for value in truth_values)
        if truth_values is not None
        else (TruthValue.FALSE, TruthValue.UNKNOWN, TruthValue.TRUE)
    )
    for assignment in product(active_truth_values, repeat=len(names)):
        yield dict(zip(names, assignment))


def classical_valuations(variables: Iterable[str]) -> Iterator[dict[str, TruthValue]]:
    """Enumerate only classical two-valued assignments."""

    return all_valuations(variables, truth_values=(TruthValue.FALSE, TruthValue.TRUE))


def evaluate(
    formula: Formula,
    valuation: Mapping[str, TruthLike],
    semantics: Semantics = STRONG_KLEENE,
) -> TruthValue:
    """Evaluate a formula under a valuation and semantics."""

    return formula.evaluate(valuation, semantics)


def truth_table(
    formula: Formula,
    semantics: Semantics = STRONG_KLEENE,
    valuation_space: Optional[Iterable[Mapping[str, TruthLike]]] = None,
) -> list[tuple[dict[str, TruthValue], TruthValue]]:
    """Return a full truth table for a formula."""

    active_valuation_space = valuation_space
    if active_valuation_space is None:
        active_valuation_space = all_valuations(formula.variables())
    return [
        (
            {name: TruthValue.coerce(value) for name, value in valuation.items()},
            formula.evaluate(valuation, semantics),
        )
        for valuation in active_valuation_space
    ]


def entails(
    premises: Iterable[Formula],
    conclusion: Formula,
    valuation_space: Optional[Iterable[Mapping[str, TruthLike]]] = None,
    semantics: Semantics = STRONG_KLEENE,
    designated_values: Optional[AbstractSet[TruthValue]] = None,
) -> bool:
    """Return whether the premises entail the conclusion."""

    premise_list = tuple(premises)
    active_valuation_space = valuation_space
    if active_valuation_space is None:
        variables = conclusion.variables()
        for premise in premise_list:
            variables |= premise.variables()
        active_valuation_space = all_valuations(variables)

    for valuation in active_valuation_space:
        if all(
            semantics.is_designated(
                premise.evaluate(valuation, semantics), designated_values
            )
            for premise in premise_list
        ):
            if not semantics.is_designated(
                conclusion.evaluate(valuation, semantics), designated_values
            ):
                return False
    return True


def is_valid(
    formula: Formula,
    valuation_space: Optional[Iterable[Mapping[str, TruthLike]]] = None,
    semantics: Semantics = STRONG_KLEENE,
    designated_values: Optional[AbstractSet[TruthValue]] = None,
) -> bool:
    """Return whether the formula is designated under all valuations."""

    active_valuation_space = valuation_space
    if active_valuation_space is None:
        active_valuation_space = all_valuations(formula.variables())
    return all(
        semantics.is_designated(formula.evaluate(valuation, semantics), designated_values)
        for valuation in active_valuation_space
    )


def satisfying_valuations(
    formula: Formula,
    semantics: Semantics = STRONG_KLEENE,
    valuation_space: Optional[Iterable[Mapping[str, TruthLike]]] = None,
    designated_values: Optional[AbstractSet[TruthValue]] = None,
) -> list[dict[str, TruthValue]]:
    """Return all valuations under which the formula is designated."""

    active_valuation_space = valuation_space
    if active_valuation_space is None:
        active_valuation_space = all_valuations(formula.variables())
    models = []
    for valuation in active_valuation_space:
        if semantics.is_designated(formula.evaluate(valuation, semantics), designated_values):
            models.append({name: TruthValue.coerce(value) for name, value in valuation.items()})
    return models


def counterexample_valuations(
    formula: Formula,
    semantics: Semantics = STRONG_KLEENE,
    valuation_space: Optional[Iterable[Mapping[str, TruthLike]]] = None,
    designated_values: Optional[AbstractSet[TruthValue]] = None,
) -> list[dict[str, TruthValue]]:
    """Return all valuations under which the formula is not designated."""

    active_valuation_space = valuation_space
    if active_valuation_space is None:
        active_valuation_space = all_valuations(formula.variables())
    countermodels = []
    for valuation in active_valuation_space:
        if not semantics.is_designated(
            formula.evaluate(valuation, semantics), designated_values
        ):
            countermodels.append(
                {name: TruthValue.coerce(value) for name, value in valuation.items()}
            )
    return countermodels


def find_model(
    formula: Formula,
    semantics: Semantics = STRONG_KLEENE,
    valuation_space: Optional[Iterable[Mapping[str, TruthLike]]] = None,
    designated_values: Optional[AbstractSet[TruthValue]] = None,
) -> Optional[dict[str, TruthValue]]:
    """Return the first designated valuation for the formula, if any."""

    models = satisfying_valuations(formula, semantics, valuation_space, designated_values)
    if not models:
        return None
    return models[0]


def find_counterexample(
    formula: Formula,
    semantics: Semantics = STRONG_KLEENE,
    valuation_space: Optional[Iterable[Mapping[str, TruthLike]]] = None,
    designated_values: Optional[AbstractSet[TruthValue]] = None,
) -> Optional[dict[str, TruthValue]]:
    """Return the first non-designated valuation for the formula, if any."""

    countermodels = counterexample_valuations(
        formula, semantics, valuation_space, designated_values
    )
    if not countermodels:
        return None
    return countermodels[0]


def is_satisfiable(
    formula: Formula,
    semantics: Semantics = STRONG_KLEENE,
    valuation_space: Optional[Iterable[Mapping[str, TruthLike]]] = None,
    designated_values: Optional[AbstractSet[TruthValue]] = None,
) -> bool:
    """Return whether the formula is designated under some valuation."""

    return find_model(formula, semantics, valuation_space, designated_values) is not None


def is_unsatisfiable(
    formula: Formula,
    semantics: Semantics = STRONG_KLEENE,
    valuation_space: Optional[Iterable[Mapping[str, TruthLike]]] = None,
    designated_values: Optional[AbstractSet[TruthValue]] = None,
) -> bool:
    """Return whether the formula has no designated valuation."""

    return not is_satisfiable(formula, semantics, valuation_space, designated_values)


def is_consistent(
    formulas: Iterable[Formula],
    semantics: Semantics = STRONG_KLEENE,
    valuation_space: Optional[Iterable[Mapping[str, TruthLike]]] = None,
    designated_values: Optional[AbstractSet[TruthValue]] = None,
) -> bool:
    """Return whether a set of formulas is jointly satisfiable."""

    formula_list = tuple(formulas)
    if not formula_list:
        return True

    active_valuation_space = valuation_space
    if active_valuation_space is None:
        variables = set()
        for formula in formula_list:
            variables |= formula.variables()
        active_valuation_space = all_valuations(variables)

    for valuation in active_valuation_space:
        if all(
            semantics.is_designated(formula.evaluate(valuation, semantics), designated_values)
            for formula in formula_list
        ):
            return True
    return False


def is_equivalent(
    left: Formula,
    right: Formula,
    semantics: Semantics = STRONG_KLEENE,
    valuation_space: Optional[Iterable[Mapping[str, TruthLike]]] = None,
) -> bool:
    """Return whether two formulas agree on all valuations."""

    active_valuation_space = valuation_space
    if active_valuation_space is None:
        active_valuation_space = all_valuations(left.variables() | right.variables())
    return all(
        left.evaluate(valuation, semantics) == right.evaluate(valuation, semantics)
        for valuation in active_valuation_space
    )


def is_contingent(
    formula: Formula,
    semantics: Semantics = STRONG_KLEENE,
    valuation_space: Optional[Iterable[Mapping[str, TruthLike]]] = None,
    designated_values: Optional[AbstractSet[TruthValue]] = None,
) -> bool:
    """Return whether the formula is satisfiable but not valid."""

    return is_satisfiable(
        formula, semantics, valuation_space, designated_values
    ) and not is_valid(formula, semantics=semantics, valuation_space=valuation_space,
                       designated_values=designated_values)


def is_classically_valid(
    formula: Formula,
    semantics: Semantics = STRONG_KLEENE,
) -> bool:
    """Return whether a formula is designated on all classical completions."""

    return is_valid(
        formula,
        valuation_space=classical_valuations(formula.variables()),
        semantics=semantics,
    )


def is_classically_valid_under_completions(
    formula: Formula,
    partial_valuation: Optional[Mapping[str, TruthLike]] = None,
    semantics: Semantics = STRONG_KLEENE,
) -> bool:
    """Return whether all Boolean completions of a partial valuation designate a formula."""

    partial = {}
    free_variables = set(formula.variables())
    for name, value in (partial_valuation or {}).items():
        coerced = TruthValue.coerce(value)
        if coerced is TruthValue.UNKNOWN:
            continue
        partial[name] = coerced
        free_variables.discard(name)

    for completion in classical_valuations(free_variables):
        valuation = dict(partial)
        valuation.update(completion)
        if not semantics.is_designated(formula.evaluate(valuation, semantics)):
            return False
    return True


def _constant_if_semantically_constant(
    formula: Formula,
    semantics: Semantics,
) -> Optional[Constant]:
    values = {
        formula.evaluate(valuation, semantics)
        for valuation in all_valuations(formula.variables())
    }
    if len(values) == 1:
        return Constant(next(iter(values)))
    return None


def _equivalent_candidates(formula: Formula) -> tuple[Formula, ...]:
    if isinstance(formula, UnaryFormula):
        return (formula.operand, Not(formula.operand))
    if isinstance(formula, BinaryFormula):
        return (
            formula.left,
            formula.right,
            Not(formula.left),
            Not(formula.right),
        )
    if isinstance(formula, NAryFormula):
        candidates = list(formula.operands)
        candidates.extend(Not(operand) for operand in formula.operands)
        return tuple(candidates)
    return tuple()


def simplify(
    formula: Formula,
    valuation: Optional[Mapping[str, TruthLike]] = None,
    semantics: Semantics = STRONG_KLEENE,
) -> Formula:
    """Partially evaluate and simplify a formula under a valuation."""

    substitution = {
        name: Constant(TruthValue.coerce(value))
        for name, value in (valuation or {}).items()
    }
    substituted = formula.substitute(substitution)

    def simplify_node(node: Formula) -> Formula:
        if isinstance(node, (Constant, Variable)):
            return node

        if isinstance(node, Not):
            operand = simplify_node(node.operand)
            rewritten = Not(operand)
            if isinstance(operand, Constant):
                return Constant(rewritten.evaluate({}, semantics))
            constant = _constant_if_semantically_constant(rewritten, semantics)
            if constant is not None:
                return constant
            return rewritten

        if isinstance(node, BinaryFormula):
            left = simplify_node(node.left)
            right = simplify_node(node.right)
            rewritten = type(node)(left, right)
            if isinstance(left, Constant) and isinstance(right, Constant):
                return Constant(rewritten.evaluate({}, semantics))

            constant = _constant_if_semantically_constant(rewritten, semantics)
            if constant is not None:
                return constant

            for candidate in _equivalent_candidates(rewritten):
                if is_equivalent(rewritten, candidate, semantics):
                    return candidate
            return rewritten

        if isinstance(node, NaryAnd):
            operands = tuple(simplify_node(operand) for operand in node.operands)
            if any(
                isinstance(operand, Constant)
                and operand.value is TruthValue.FALSE
                for operand in operands
            ):
                return Constant(TruthValue.FALSE)
            filtered = tuple(
                operand
                for operand in operands
                if not (
                    isinstance(operand, Constant)
                    and operand.value is TruthValue.TRUE
                )
            )
            if not filtered:
                return Constant(TruthValue.TRUE)
            if len(filtered) == 1:
                return filtered[0]
            rewritten = NaryAnd(filtered)
            constant = _constant_if_semantically_constant(rewritten, semantics)
            if constant is not None:
                return constant
            return rewritten

        if isinstance(node, NaryOr):
            operands = tuple(simplify_node(operand) for operand in node.operands)
            if any(
                isinstance(operand, Constant)
                and operand.value is TruthValue.TRUE
                for operand in operands
            ):
                return Constant(TruthValue.TRUE)
            filtered = tuple(
                operand
                for operand in operands
                if not (
                    isinstance(operand, Constant)
                    and operand.value is TruthValue.FALSE
                )
            )
            if not filtered:
                return Constant(TruthValue.FALSE)
            if len(filtered) == 1:
                return filtered[0]
            rewritten = NaryOr(filtered)
            constant = _constant_if_semantically_constant(rewritten, semantics)
            if constant is not None:
                return constant
            return rewritten

        raise TypeError(f"Unsupported formula node: {node!r}")

    return simplify_node(substituted)


__all__ = [
    "TruthValue",
    "TruthLike",
    "Semantics",
    "STRONG_KLEENE",
    "WEAK_KLEENE",
    "LUKASIEWICZ_K3",
    "GODEL_G3",
    "Formula",
    "Constant",
    "Variable",
    "UnaryFormula",
    "Not",
    "BinaryFormula",
    "And",
    "Or",
    "Implies",
    "Iff",
    "Xor",
    "Nand",
    "Nor",
    "NAryFormula",
    "NaryAnd",
    "NaryOr",
    "conjoin",
    "disjoin",
    "format_formula",
    "all_valuations",
    "classical_valuations",
    "evaluate",
    "truth_table",
    "entails",
    "is_valid",
    "satisfying_valuations",
    "counterexample_valuations",
    "find_model",
    "find_counterexample",
    "is_satisfiable",
    "is_unsatisfiable",
    "is_consistent",
    "is_equivalent",
    "is_contingent",
    "is_classically_valid",
    "is_classically_valid_under_completions",
    "simplify",
]
