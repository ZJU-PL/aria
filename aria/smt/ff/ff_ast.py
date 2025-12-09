#!/usr/bin/env python3
"""
ff_ast.py  –  AST classes for finite-field expressions
"""
from __future__ import annotations
from typing import Dict, List

class FieldExpr:
    """Base class for all finite-field expressions."""
    pass

class FieldAdd(FieldExpr):
    def __init__(self, *args):
        self.args = list(args)

class FieldMul(FieldExpr):
    def __init__(self, *args):
        self.args = list(args)

class FieldEq(FieldExpr):
    def __init__(self, l, r):
        self.left, self.right = l, r

class FieldVar(FieldExpr):
    def __init__(self, name):
        self.name = name

class FieldConst(FieldExpr):
    def __init__(self, val):
        self.value = val

class FieldSub(FieldExpr):
    def __init__(self, *args):
        if len(args) < 2:
            raise ValueError("FieldSub needs at least two operands")
        self.args = list(args)

class FieldNeg(FieldExpr):
    def __init__(self, arg):
        self.arg = arg

class FieldPow(FieldExpr):
    def __init__(self, base, exponent: int):
        if exponent < 0:
            raise ValueError("Exponent must be non-negative")
        self.base = base
        self.exponent = exponent

class FieldDiv(FieldExpr):
    def __init__(self, num, denom):
        self.num, self.denom = num, denom

class BoolOr(FieldExpr):
    def __init__(self, *args):
        self.args = list(args)

class BoolAnd(FieldExpr):
    def __init__(self, *args):
        self.args = list(args)

class BoolNot(FieldExpr):
    def __init__(self, arg):
        self.arg = arg

class BoolImplies(FieldExpr):
    def __init__(self, antecedent, consequent):
        self.antecedent = antecedent
        self.consequent = consequent

class BoolIte(FieldExpr):
    def __init__(self, cond, then_expr, else_expr):
        self.cond = cond
        self.then_expr = then_expr
        self.else_expr = else_expr

class BoolVar(FieldExpr):
    def __init__(self, name):
        self.name = name

class ParsedFormula:
    def __init__(self, field_size: int,
                 variables: Dict[str, str],
                 assertions: List[FieldExpr],
                 expected_status: str | None = None):
        self.field_size = field_size
        self.variables = variables    # name → sort id (unused here)
        self.assertions = assertions
        self.expected_status = expected_status  # 'sat', 'unsat', or None
