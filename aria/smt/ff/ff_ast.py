#!/usr/bin/env python3
"""
ff_ast.py  –  AST classes for finite-field expressions
"""
from __future__ import annotations
from typing import Dict, List, Optional, Set


BOOL_SORT = "bool"


def ff_sort_id(modulus: int) -> str:
    """Return the canonical sort id for a finite-field sort."""
    return "ff:%d" % modulus


def is_bool_sort(sort_id: str) -> bool:
    """Return whether the sort id denotes Boolean sort."""
    return sort_id == BOOL_SORT


def is_ff_sort(sort_id: str) -> bool:
    """Return whether the sort id denotes a finite-field sort."""
    return sort_id.startswith("ff:")


def field_modulus_from_sort(sort_id: str) -> Optional[int]:
    """Extract the field modulus from a finite-field sort id."""
    if not is_ff_sort(sort_id):
        return None
    return int(sort_id.split(":", 1)[1])


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

    def __init__(self, val, modulus: Optional[int] = None):
        self.value = val
        self.modulus = modulus


class FieldSub(FieldExpr):
    """AST node for n-ary finite field subtraction."""

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


class BoolConst(FieldExpr):
    """AST node for a Boolean constant."""

    def __init__(self, value: bool):
        self.value = value


class BoolXor(FieldExpr):
    """AST node for Boolean XOR."""

    def __init__(self, *args):
        self.args = list(args)


class ParsedFormula:
    """Represents a parsed finite-field formula."""

    def __init__(
        self,
        field_size: Optional[int],
        variables: Dict[str, str],
        assertions: List[FieldExpr],
        expected_status: Optional[str] = None,
        field_sizes: Optional[List[int]] = None,
    ):
        """
        Initializes a new ParsedFormula.

        Args:
            field_size: The size of the finite field.
            variables: A dictionary mapping variable names to their sort ID.
            assertions: A list of assertions in the formula.
            expected_status: The expected status of the formula ('sat',
                             'unsat', or None).
        """
        self.variables = variables  # name → sort id
        self.assertions = assertions
        self.expected_status = expected_status  # 'sat', 'unsat', or None
        moduli: Set[int] = set(field_sizes or [])
        for sort_id in variables.values():
            modulus = field_modulus_from_sort(sort_id)
            if modulus is not None:
                moduli.add(modulus)
        self.field_sizes = sorted(moduli)
        if field_size is not None:
            self.field_size = field_size
        elif len(self.field_sizes) == 1:
            self.field_size = self.field_sizes[0]
        else:
            self.field_size = None

    def is_single_field(self) -> bool:
        """Return whether the formula references exactly one finite field."""
        return len(self.field_sizes) == 1

    def require_single_field(self) -> int:
        """Return the sole field size, raising when the formula is mixed-sort."""
        if not self.is_single_field():
            raise ValueError(
                "Expected a single finite-field sort, got %s" % self.field_sizes
            )
        return self.field_sizes[0]


def infer_field_modulus(expr: FieldExpr, variables: Dict[str, str]) -> Optional[int]:
    """Infer the modulus of a finite-field expression."""
    sort_id = infer_expr_sort(expr, variables)
    return field_modulus_from_sort(sort_id) if sort_id is not None else None


def infer_expr_sort(expr: FieldExpr, variables: Dict[str, str]) -> Optional[str]:
    """Infer the sort of an expression."""
    if isinstance(expr, FieldConst):
        return ff_sort_id(expr.modulus) if expr.modulus is not None else None
    if isinstance(expr, FieldVar):
        return variables[expr.name]
    if isinstance(expr, FieldEq):
        left = infer_expr_sort(expr.left, variables)
        right = infer_expr_sort(expr.right, variables)
        if left is not None and right is not None and left != right:
            raise ValueError("Mismatched sorts: %s vs %s" % (left, right))
        return BOOL_SORT
    if isinstance(expr, FieldAdd) or isinstance(expr, FieldMul) or isinstance(
        expr, FieldSub
    ):
        sort_id = None
        for arg in expr.args:
            arg_sort = infer_expr_sort(arg, variables)
            if arg_sort is None:
                continue
            if sort_id is None:
                sort_id = arg_sort
            elif sort_id != arg_sort:
                raise ValueError("Mismatched sorts: %s vs %s" % (sort_id, arg_sort))
        return sort_id
    if isinstance(expr, FieldNeg):
        return infer_expr_sort(expr.arg, variables)
    if isinstance(expr, FieldPow):
        return infer_expr_sort(expr.base, variables)
    if isinstance(expr, FieldDiv):
        num_sort = infer_expr_sort(expr.num, variables)
        den_sort = infer_expr_sort(expr.denom, variables)
        if num_sort is None:
            return den_sort
        if den_sort is None:
            return num_sort
        if num_sort != den_sort:
            raise ValueError("Mismatched sorts: %s vs %s" % (num_sort, den_sort))
        return num_sort
    if isinstance(expr, BoolIte):
        then_sort = infer_expr_sort(expr.then_expr, variables)
        else_sort = infer_expr_sort(expr.else_expr, variables)
        if then_sort is None:
            return else_sort
        if else_sort is None:
            return then_sort
        if then_sort != else_sort:
            raise ValueError("Mismatched sorts: %s vs %s" % (then_sort, else_sort))
        return then_sort
    if isinstance(expr, BoolVar) or isinstance(expr, BoolConst):
        return BOOL_SORT
    if (
        isinstance(expr, BoolOr)
        or isinstance(expr, BoolAnd)
        or isinstance(expr, BoolNot)
        or isinstance(expr, BoolImplies)
        or isinstance(expr, BoolXor)
    ):
        return BOOL_SORT
    return None
