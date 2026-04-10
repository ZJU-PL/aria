"""Interoperability helpers for three-valued formulas and Boolean NNF."""

from __future__ import annotations

from typing import Literal

import aria.bool.nnf as nnf
from aria.bool.nnf import operators as nnf_operators

from .propositional import (
    And,
    Constant,
    Formula,
    Iff,
    Implies,
    Nand,
    NaryAnd,
    NaryOr,
    Nor,
    Not,
    Or,
    TruthValue,
    Variable,
    Xor,
    conjoin,
    disjoin,
)


UnknownPolicy = Literal["error", "false", "true"]


def from_nnf(sentence: nnf.NNF) -> Formula:
    """Convert a Boolean NNF sentence into a three-valued formula AST."""

    if sentence == nnf.true:
        return Constant(TruthValue.TRUE)
    if sentence == nnf.false:
        return Constant(TruthValue.FALSE)
    if isinstance(sentence, nnf.Var):
        atom = Variable(str(sentence.name))
        if sentence.true:
            return atom
        return Not(atom)
    if isinstance(sentence, nnf.And):
        return conjoin(*(from_nnf(child) for child in sentence.children))
    if isinstance(sentence, nnf.Or):
        return disjoin(*(from_nnf(child) for child in sentence.children))
    raise TypeError(f"Unsupported NNF node: {sentence!r}")


def to_nnf(formula: Formula, unknown_policy: UnknownPolicy = "error") -> nnf.NNF:
    """Convert a three-valued formula into Boolean NNF.

    The conversion is a classical embedding: it preserves behavior on Boolean
    valuations only. If the formula contains `UNKNOWN`, choose how that constant
    should be mapped into a Boolean sentence with `unknown_policy`.
    """

    if unknown_policy not in {"error", "false", "true"}:
        raise ValueError(f"Unsupported unknown policy: {unknown_policy!r}")

    def convert(node: Formula) -> nnf.NNF:
        if isinstance(node, Constant):
            if node.value is TruthValue.TRUE:
                return nnf.true
            if node.value is TruthValue.FALSE:
                return nnf.false
            if unknown_policy == "error":
                raise ValueError(
                    "Cannot convert formulas containing UNKNOWN to Boolean NNF "
                    "with unknown_policy='error'"
                )
            return nnf.true if unknown_policy == "true" else nnf.false
        if isinstance(node, Variable):
            return nnf.Var(node.name)
        if isinstance(node, Not):
            operand = node.operand
            if isinstance(operand, Variable):
                return nnf.Var(operand.name, False)
            return convert(operand).negate()
        if isinstance(node, And):
            return convert(node.left) & convert(node.right)
        if isinstance(node, Or):
            return convert(node.left) | convert(node.right)
        if isinstance(node, Implies):
            return nnf_operators.implies(convert(node.left), convert(node.right))
        if isinstance(node, Iff):
            return nnf_operators.iff(convert(node.left), convert(node.right))
        if isinstance(node, Xor):
            return nnf_operators.xor(convert(node.left), convert(node.right))
        if isinstance(node, Nand):
            return nnf_operators.nand(convert(node.left), convert(node.right))
        if isinstance(node, Nor):
            return nnf_operators.nor(convert(node.left), convert(node.right))
        if isinstance(node, NaryAnd):
            return nnf.And(convert(operand) for operand in node.operands)
        if isinstance(node, NaryOr):
            return nnf.Or(convert(operand) for operand in node.operands)
        raise TypeError(f"Unsupported three-valued formula node: {node!r}")

    return convert(formula).simplify()


__all__ = ["UnknownPolicy", "from_nnf", "to_nnf"]
