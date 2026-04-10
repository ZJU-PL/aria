"""Normalization helpers for modal formulas."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .formula import Formula

def eliminate_implications(formula: "Formula") -> "Formula":
    """Return an implication-free formula equivalent to ``formula``."""

    from .formula import And, Box, Constant, Diamond, Iff, Implies, Not, Or

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
    if isinstance(formula, Iff):
        left = eliminate_implications(formula.left)
        right = eliminate_implications(formula.right)
        return And(Or(Not(left), right), Or(Not(right), left))
    if isinstance(formula, Box):
        return Box(eliminate_implications(formula.operand))
    if isinstance(formula, Diamond):
        return Diamond(eliminate_implications(formula.operand))
    return formula


def simplify(formula: "Formula") -> "Formula":
    """Simplify ``formula`` using lightweight modal-preserving rewrites."""

    from .formula import And, Atom, Box, Constant, Diamond, Iff, Implies, Not, Or

    if isinstance(formula, (Constant, Atom)):
        return formula
    if isinstance(formula, Not):
        operand = simplify(formula.operand)
        if isinstance(operand, Constant):
            return Constant(not operand.value)
        if isinstance(operand, Not):
            return simplify(operand.operand)
        return Not(operand)
    if isinstance(formula, And):
        left = simplify(formula.left)
        right = simplify(formula.right)
        if left == Constant(False) or right == Constant(False):
            return Constant(False)
        if left == Constant(True):
            return right
        if right == Constant(True):
            return left
        if left == right:
            return left
        return And(left, right)
    if isinstance(formula, Or):
        left = simplify(formula.left)
        right = simplify(formula.right)
        if left == Constant(True) or right == Constant(True):
            return Constant(True)
        if left == Constant(False):
            return right
        if right == Constant(False):
            return left
        if left == right:
            return left
        return Or(left, right)
    if isinstance(formula, Implies):
        left = simplify(formula.left)
        right = simplify(formula.right)
        if left == Constant(False) or right == Constant(True):
            return Constant(True)
        if left == Constant(True):
            return right
        if right == Constant(False):
            return simplify(Not(left))
        if left == right:
            return Constant(True)
        return Implies(left, right)
    if isinstance(formula, Iff):
        left = simplify(formula.left)
        right = simplify(formula.right)
        if left == right:
            return Constant(True)
        if left == Constant(True):
            return right
        if right == Constant(True):
            return left
        if left == Constant(False):
            return simplify(Not(right))
        if right == Constant(False):
            return simplify(Not(left))
        return Iff(left, right)
    if isinstance(formula, Box):
        operand = simplify(formula.operand)
        if operand == Constant(True):
            return Constant(True)
        return Box(operand)
    if isinstance(formula, Diamond):
        operand = simplify(formula.operand)
        if operand == Constant(False):
            return Constant(False)
        return Diamond(operand)
    return formula


def to_nnf(formula: "Formula") -> "Formula":
    """Convert ``formula`` to implication-free modal negation normal form."""

    return _nnf(eliminate_implications(formula))


def _nnf(formula: "Formula") -> "Formula":
    from .formula import And, Box, Constant, Diamond, Iff, Implies, Not, Or

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
    if isinstance(formula, (Implies, Iff)):
        return _nnf(eliminate_implications(formula))
    return formula
