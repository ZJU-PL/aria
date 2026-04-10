"""Three-valued propositional logic with strong Kleene semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from itertools import product
from collections.abc import Iterable, Iterator, Mapping
from typing import Optional, Union


class TruthValue(Enum):
    """Truth values for strong Kleene three-valued logic."""

    FALSE = 0
    UNKNOWN = 1
    TRUE = 2

    @classmethod
    def coerce(cls, value: TruthLike) -> TruthValue:
        if isinstance(value, cls):
            return value
        if value is True:
            return cls.TRUE
        if value is False:
            return cls.FALSE
        if value is None:
            return cls.UNKNOWN
        raise TypeError(f"Unsupported truth value: {value!r}")


TruthLike = Union[TruthValue, bool, None]


class Formula:
    """Base class for three-valued propositional formulas."""

    def evaluate(self, valuation: Mapping[str, TruthLike]) -> TruthValue:
        raise NotImplementedError

    def variables(self) -> set[str]:
        raise NotImplementedError


@dataclass(frozen=True)
class Constant(Formula):
    value: TruthValue

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", TruthValue.coerce(self.value))

    def evaluate(self, valuation: Mapping[str, TruthLike]) -> TruthValue:
        del valuation
        return self.value

    def variables(self) -> set[str]:
        return set()


@dataclass(frozen=True)
class Variable(Formula):
    name: str

    def evaluate(self, valuation: Mapping[str, TruthLike]) -> TruthValue:
        return TruthValue.coerce(valuation.get(self.name, TruthValue.UNKNOWN))

    def variables(self) -> set[str]:
        return {self.name}


@dataclass(frozen=True)
class Not(Formula):
    operand: Formula

    def evaluate(self, valuation: Mapping[str, TruthLike]) -> TruthValue:
        operand_value = self.operand.evaluate(valuation)
        if operand_value == TruthValue.TRUE:
            return TruthValue.FALSE
        if operand_value == TruthValue.FALSE:
            return TruthValue.TRUE
        return TruthValue.UNKNOWN

    def variables(self) -> set[str]:
        return self.operand.variables()


@dataclass(frozen=True)
class BinaryFormula(Formula):
    left: Formula
    right: Formula

    def variables(self) -> set[str]:
        return self.left.variables() | self.right.variables()


@dataclass(frozen=True)
class And(BinaryFormula):
    def evaluate(self, valuation: Mapping[str, TruthLike]) -> TruthValue:
        left_value = self.left.evaluate(valuation)
        right_value = self.right.evaluate(valuation)
        if TruthValue.FALSE in (left_value, right_value):
            return TruthValue.FALSE
        if TruthValue.UNKNOWN in (left_value, right_value):
            return TruthValue.UNKNOWN
        return TruthValue.TRUE


@dataclass(frozen=True)
class Or(BinaryFormula):
    def evaluate(self, valuation: Mapping[str, TruthLike]) -> TruthValue:
        left_value = self.left.evaluate(valuation)
        right_value = self.right.evaluate(valuation)
        if TruthValue.TRUE in (left_value, right_value):
            return TruthValue.TRUE
        if TruthValue.UNKNOWN in (left_value, right_value):
            return TruthValue.UNKNOWN
        return TruthValue.FALSE


@dataclass(frozen=True)
class Implies(BinaryFormula):
    def evaluate(self, valuation: Mapping[str, TruthLike]) -> TruthValue:
        return Or(Not(self.left), self.right).evaluate(valuation)


def all_valuations(variables: Iterable[str]) -> Iterator[dict[str, TruthValue]]:
    """Enumerate all three-valued assignments for the given variables."""

    names = sorted(set(variables))
    values = (TruthValue.FALSE, TruthValue.UNKNOWN, TruthValue.TRUE)
    for assignment in product(values, repeat=len(names)):
        yield dict(zip(names, assignment))


def evaluate(formula: Formula, valuation: Mapping[str, TruthLike]) -> TruthValue:
    """Evaluate a formula under a possibly partial valuation."""

    return formula.evaluate(valuation)


def entails(
    premises: Iterable[Formula],
    conclusion: Formula,
    valuation_space: Optional[Iterable[Mapping[str, TruthLike]]] = None,
) -> bool:
    """Return whether the premises three-valuedly entail the conclusion."""

    premise_list = tuple(premises)
    if valuation_space is None:
        variables = conclusion.variables()
        for premise in premise_list:
            variables |= premise.variables()
        valuation_space = all_valuations(variables)

    for valuation in valuation_space:
        if all(premise.evaluate(valuation) == TruthValue.TRUE for premise in premise_list):
            if conclusion.evaluate(valuation) != TruthValue.TRUE:
                return False
    return True


def is_valid(
    formula: Formula,
    valuation_space: Optional[Iterable[Mapping[str, TruthLike]]] = None,
) -> bool:
    """Return whether the formula is true under all three-valued assignments."""

    if valuation_space is None:
        valuation_space = all_valuations(formula.variables())
    return all(formula.evaluate(valuation) == TruthValue.TRUE for valuation in valuation_space)
