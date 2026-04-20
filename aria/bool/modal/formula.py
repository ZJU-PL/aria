"""Modal formula AST definitions."""

from __future__ import annotations

from dataclasses import dataclass


class Formula:
    """Base class for modal formulas."""

    def atoms(self) -> set[str]:
        raise NotImplementedError


@dataclass(frozen=True)
class Constant(Formula):
    value: bool

    def __post_init__(self) -> None:
        if not isinstance(self.value, bool):
            raise TypeError(f"Modal constants must be bools, got {self.value!r}")

    def atoms(self) -> set[str]:
        return set()


@dataclass(frozen=True)
class Atom(Formula):
    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError(f"Atom names must be strings, got {self.name!r}")
        if self.name in {"true", "false"}:
            raise ValueError(
                "Atom names 'true' and 'false' are reserved modal constants"
            )

    def atoms(self) -> set[str]:
        return {self.name}


@dataclass(frozen=True)
class UnaryFormula(Formula):
    operand: Formula

    def __post_init__(self) -> None:
        if not isinstance(self.operand, Formula):
            raise TypeError(
                f"Unary modal operands must be Formula instances, got {self.operand!r}"
            )

    def atoms(self) -> set[str]:
        return self.operand.atoms()


@dataclass(frozen=True)
class Not(UnaryFormula):
    pass


@dataclass(frozen=True)
class BinaryFormula(Formula):
    left: Formula
    right: Formula

    def __post_init__(self) -> None:
        if not isinstance(self.left, Formula):
            raise TypeError(
                f"Binary modal left operand must be a Formula, got {self.left!r}"
            )
        if not isinstance(self.right, Formula):
            raise TypeError(
                f"Binary modal right operand must be a Formula, got {self.right!r}"
            )

    def atoms(self) -> set[str]:
        return self.left.atoms() | self.right.atoms()


@dataclass(frozen=True)
class And(BinaryFormula):
    pass


@dataclass(frozen=True)
class Or(BinaryFormula):
    pass


@dataclass(frozen=True)
class Implies(BinaryFormula):
    pass


@dataclass(frozen=True)
class Iff(BinaryFormula):
    pass


@dataclass(frozen=True)
class Box(UnaryFormula):
    pass


@dataclass(frozen=True)
class Diamond(UnaryFormula):
    pass


__all__ = [
    "Formula",
    "Constant",
    "Atom",
    "UnaryFormula",
    "Not",
    "BinaryFormula",
    "And",
    "Or",
    "Implies",
    "Iff",
    "Box",
    "Diamond",
]
