"""Structural utilities and formatting for modal formulas."""

from __future__ import annotations

from typing import FrozenSet, Set

from .formula import (
    And,
    Atom,
    BinaryFormula,
    Box,
    Constant,
    Diamond,
    Formula,
    Iff,
    Implies,
    Not,
    Or,
    UnaryFormula,
)


def formula_size(formula: Formula) -> int:
    if isinstance(formula, (Constant, Atom)):
        return 1
    if isinstance(formula, UnaryFormula):
        return 1 + formula_size(formula.operand)
    if isinstance(formula, BinaryFormula):
        return 1 + formula_size(formula.left) + formula_size(formula.right)
    raise TypeError(f"Unsupported modal formula: {formula!r}")


def modal_depth(formula: Formula) -> int:
    if isinstance(formula, (Constant, Atom)):
        return 0
    if isinstance(formula, Not):
        return modal_depth(formula.operand)
    if isinstance(formula, (Box, Diamond)):
        return 1 + modal_depth(formula.operand)
    if isinstance(formula, BinaryFormula):
        return max(modal_depth(formula.left), modal_depth(formula.right))
    raise TypeError(f"Unsupported modal formula: {formula!r}")


def subformulas(formula: Formula) -> FrozenSet[Formula]:
    seen: Set[Formula] = set()

    def _collect(current: Formula) -> None:
        if current in seen:
            return
        seen.add(current)
        if isinstance(current, UnaryFormula):
            _collect(current.operand)
            return
        if isinstance(current, BinaryFormula):
            _collect(current.left)
            _collect(current.right)

    _collect(formula)
    return frozenset(seen)


def format_formula(formula: Formula, unicode: bool = False) -> str:
    if unicode:
        unary_symbols = {Not: "¬", Box: "□", Diamond: "◇"}
        binary_symbols = {And: " ∧ ", Or: " ∨ ", Implies: " → ", Iff: " ↔ "}
        constants = {True: "⊤", False: "⊥"}
    else:
        unary_symbols = {Not: "!", Box: "[]", Diamond: "<>"}
        binary_symbols = {And: " & ", Or: " | ", Implies: " -> ", Iff: " <-> "}
        constants = {True: "true", False: "false"}

    precedence = {
        Constant: 4,
        Atom: 4,
        Not: 3,
        Box: 3,
        Diamond: 3,
        And: 2,
        Or: 1,
        Implies: 0,
        Iff: -1,
    }

    def _format(current: Formula, parent_prec: int, is_right: bool = False) -> str:
        current_prec = precedence[type(current)]
        if isinstance(current, Constant):
            rendered = constants[current.value]
        elif isinstance(current, Atom):
            rendered = current.name
        elif isinstance(current, UnaryFormula):
            rendered = unary_symbols[type(current)] + _format(
                current.operand, current_prec
            )
        elif isinstance(current, BinaryFormula):
            left = _format(current.left, current_prec)
            right_parent_prec = current_prec
            if isinstance(current, Implies):
                right_parent_prec = current_prec - 1
            right = _format(current.right, right_parent_prec, is_right=True)
            rendered = left + binary_symbols[type(current)] + right
        else:
            raise TypeError(f"Unsupported modal formula: {current!r}")

        needs_parens = current_prec < parent_prec
        if isinstance(current, Implies) and is_right and current_prec == parent_prec:
            needs_parens = False
        elif (
            isinstance(current, BinaryFormula)
            and is_right
            and current_prec <= parent_prec
        ):
            needs_parens = current_prec < parent_prec
        if needs_parens:
            return f"({rendered})"
        return rendered

    return _format(formula, -1)


__all__ = ["formula_size", "modal_depth", "subformulas", "format_formula"]
