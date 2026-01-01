#!/usr/bin/env python3
"""
ff_ast.py  –  AST classes for finite-field expressions
"""
from __future__ import annotations
from typing import Dict, List


class FieldExpr:
    """Base class for all finite-field expressions."""

    # Intentionally empty - serves as base class marker


class FieldAdd(FieldExpr):
    """AST node for finite field addition."""

    def __init__(self, *args):
        self.args = list(args)


class FieldMul(FieldExpr):
    """AST node for finite field multiplication."""

    def __init__(self, *args):
        self.args = list(args)


class FieldEq(FieldExpr):
    """AST node for finite field equality."""

    def __init__(self, l, r):
        self.left, self.right = l, r


class FieldVar(FieldExpr):
    """AST node for finite field variable."""

    def __init__(self, name):
        self.name = name


class FieldConst(FieldExpr):
    """AST node for finite field constant."""

    def __init__(self, val):
        self.value = val


class FieldSub(FieldExpr):
    """AST node for finite field subtraction."""

    def __init__(self, *args):
        if len(args) < 2:
            raise ValueError("FieldSub needs at least two operands")
        self.args = list(args)


class FieldNeg(FieldExpr):
    """AST node for finite field negation."""

    def __init__(self, arg):
        self.arg = arg


class FieldPow(FieldExpr):
    """AST node for finite field exponentiation."""

    def __init__(self, base, exponent: int):
        if exponent < 0:
            raise ValueError("Exponent must be non-negative")
        self.base = base
        self.exponent = exponent


class FieldDiv(FieldExpr):
    """AST node for finite field division."""

    def __init__(self, num, denom):
        self.num, self.denom = num, denom


class BoolOr(FieldExpr):
    """AST node for Boolean OR."""

    def __init__(self, *args):
        self.args = list(args)


class BoolAnd(FieldExpr):
    """AST node for Boolean AND."""

    def __init__(self, *args):
        self.args = list(args)


class BoolNot(FieldExpr):
    """AST node for Boolean NOT."""

    def __init__(self, arg):
        self.arg = arg


class BoolImplies(FieldExpr):
    """AST node for Boolean implication."""

    def __init__(self, antecedent, consequent):
        self.antecedent = antecedent
        self.consequent = consequent


class BoolIte(FieldExpr):
    """AST node for Boolean if-then-else."""

    def __init__(self, cond, then_expr, else_expr):
        self.cond = cond
        self.then_expr = then_expr
        self.else_expr = else_expr


class BoolVar(FieldExpr):
    """AST node for Boolean variable."""

    def __init__(self, name):
        self.name = name


class ParsedFormula:
    """Represents a parsed finite-field formula."""

    def __init__(
        self,
        field_size: int,
        variables: Dict[str, str],
        assertions: List[FieldExpr],
        expected_status: str | None = None,
    ):
        self.field_size = field_size
        self.variables = variables  # name → sort id (unused here)
        self.assertions = assertions
        self.expected_status = expected_status  # 'sat', 'unsat', or None
