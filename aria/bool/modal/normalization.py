"""Normalization helpers for modal formulas."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import Formula

def eliminate_implications(formula: "Formula") -> "Formula":
    """Return an implication-free formula equivalent to ``formula``."""

    from . import And, Box, Constant, Diamond, Implies, Not, Or

    if isinstance(formula, Constant):
        return formula
    if isinstance(formula, Not):
        return Not(eliminate_implications(formula.operand))
    if isinstance(formula, And):
        return And(
            eliminate_implications(formula.left),
            eliminate_implications(formula.right),
        )
    if isinstance(formula, Or):
        return Or(
            eliminate_implications(formula.left),
            eliminate_implications(formula.right),
        )
    if isinstance(formula, Implies):
        return Or(
            Not(eliminate_implications(formula.left)),
            eliminate_implications(formula.right),
        )
    if isinstance(formula, Box):
        return Box(eliminate_implications(formula.operand))
    if isinstance(formula, Diamond):
        return Diamond(eliminate_implications(formula.operand))
    return formula


def to_nnf(formula: "Formula") -> "Formula":
    """Convert ``formula`` to implication-free modal negation normal form."""

    return _nnf(eliminate_implications(formula))


def _nnf(formula: "Formula") -> "Formula":
    from . import And, Box, Constant, Diamond, Implies, Not, Or

    if isinstance(formula, Constant):
        return formula
    if isinstance(formula, Not):
        operand = formula.operand
        if isinstance(operand, Constant):
            return Constant(not operand.value)
        if isinstance(operand, Not):
            return _nnf(operand.operand)
        if isinstance(operand, And):
            return Or(_nnf(Not(operand.left)), _nnf(Not(operand.right)))
        if isinstance(operand, Or):
            return And(_nnf(Not(operand.left)), _nnf(Not(operand.right)))
        if isinstance(operand, Box):
            return Diamond(_nnf(Not(operand.operand)))
        if isinstance(operand, Diamond):
            return Box(_nnf(Not(operand.operand)))
        return Not(operand)
    if isinstance(formula, And):
        return And(_nnf(formula.left), _nnf(formula.right))
    if isinstance(formula, Or):
        return Or(_nnf(formula.left), _nnf(formula.right))
    if isinstance(formula, Box):
        return Box(_nnf(formula.operand))
    if isinstance(formula, Diamond):
        return Diamond(_nnf(formula.operand))
    if isinstance(formula, Implies):
        return _nnf(eliminate_implications(formula))
    return formula
