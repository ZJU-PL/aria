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

    def _needs_child_parens(child: Formula, parent: Formula, is_right: bool) -> bool:
        parent_prec = precedence[type(parent)]
        child_prec = precedence[type(child)]
        if child_prec < parent_prec:
            return True
        if child_prec > parent_prec or not isinstance(child, BinaryFormula):
            return False

        if isinstance(parent, Implies):
            return not is_right
        return is_right

    def _format(current: Formula, parent_prec: int, is_right: bool = False) -> str:
        current_prec = precedence[type(current)]
        if isinstance(current, Constant):
            rendered = constants[current.value]
        elif isinstance(current, Atom):
            rendered = current.name
        elif isinstance(current, UnaryFormula):
            operand = _format(current.operand, current_prec)
            if isinstance(current.operand, BinaryFormula):
                operand = f"({operand})"
            rendered = unary_symbols[type(current)] + operand
        elif isinstance(current, BinaryFormula):
            left = _format(current.left, current_prec)
            if _needs_child_parens(current.left, current, is_right=False):
                left = f"({left})"

            right = _format(current.right, current_prec, is_right=True)
            if _needs_child_parens(current.right, current, is_right=True):
                right = f"({right})"
            rendered = left + binary_symbols[type(current)] + right
        else:
            raise TypeError(f"Unsupported modal formula: {current!r}")

        return rendered

    return _format(formula, -1)


__all__ = ["formula_size", "modal_depth", "subformulas", "format_formula"]
