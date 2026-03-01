#!/usr/bin/env python3
"""
ff_int_solver.py  –  Finite-field formulas via integer translation.
"""
from __future__ import annotations

import functools
from typing import Dict, Optional

import z3

from .ff_ast import (
    BoolAnd,
    BoolConst,
    BoolIte,
    BoolImplies,
    BoolNot,
    BoolOr,
    BoolVar,
    BoolXor,
    FieldAdd,
    FieldConst,
    FieldDiv,
    FieldEq,
    FieldExpr,
    FieldMul,
    FieldNeg,
    FieldPow,
    FieldSub,
    FieldVar,
    ParsedFormula,
    field_modulus_from_sort,
    infer_field_modulus,
    is_bool_sort,
)
from .ff_numbertheory import is_probable_prime
from .ff_preprocess import preprocess_formula


class FFIntSolver:
    """Prime-field solver via a direct translation to non-linear integers."""

    def __init__(self):
        self.solver = z3.SolverFor("QF_NIA")
        self.vars: Dict[str, z3.ExprRef] = {}
        self.var_sorts: Dict[str, str] = {}

    def check(self, formula: ParsedFormula) -> z3.CheckSatResult:
        """Check satisfiability of a finite-field formula."""
        normalized = preprocess_formula(formula)
        self._reset()
        self._setup_fields(normalized.field_sizes)
        self.var_sorts = dict(normalized.variables)
        self._declare_vars(normalized.variables)
        for assertion in normalized.assertions:
            self.solver.add(self._tr(assertion))
        return self.solver.check()

    def model(self) -> Optional[z3.ModelRef]:
        """Get the model if the formula is satisfiable."""
        if self.solver.check() == z3.sat:
            return self.solver.model()
        return None

    def _reset(self) -> None:
        self.solver = z3.SolverFor("QF_NIA")
        self.vars = {}
        self.var_sorts = {}

    def _setup_fields(self, fields) -> None:
        for modulus in fields:
            if not is_probable_prime(modulus):
                raise ValueError(
                    "Finite-field sort requires prime p, got %d" % modulus
                )

    def _declare_vars(self, varmap: Dict[str, str]) -> None:
        for name, sort_id in varmap.items():
            if is_bool_sort(sort_id):
                self.vars[name] = z3.Bool(name)
                continue
            modulus = field_modulus_from_sort(sort_id)
            if modulus is None:
                raise ValueError("unsupported sort %s" % sort_id)
            iv = z3.Int(name)
            self.vars[name] = iv
            self.solver.add(z3.And(iv >= 0, iv < modulus))

    def _mod(self, term: z3.ArithRef, modulus: int) -> z3.ArithRef:
        return term % modulus

    def _pow_mod(self, base: z3.ArithRef, modulus: int, exp: int) -> z3.ArithRef:
        result = z3.IntVal(1)
        running_base = base
        exponent = exp
        while exponent > 0:
            if exponent & 1:
                result = self._mod(result * running_base, modulus)
            exponent >>= 1
            if exponent:
                running_base = self._mod(running_base * running_base, modulus)
        return result

    def _field_modulus(self, expr: FieldExpr) -> int:
        modulus = infer_field_modulus(expr, self.var_sorts)
        if modulus is None:
            raise ValueError("expected a finite-field expression")
        return modulus

    def _tr(
        self, expr: FieldExpr
    ) -> z3.ExprRef:  # pylint: disable=too-many-return-statements,too-many-branches
        if isinstance(expr, FieldAdd):
            modulus = self._field_modulus(expr)
            result = z3.IntVal(0)
            for arg in expr.args:
                result = self._mod(result + self._tr(arg), modulus)
            return result

        if isinstance(expr, FieldMul):
            modulus = self._field_modulus(expr)
            result = z3.IntVal(1)
            for arg in expr.args:
                result = self._mod(result * self._tr(arg), modulus)
            return result

        if isinstance(expr, FieldEq):
            return self._tr(expr.left) == self._tr(expr.right)

        if isinstance(expr, FieldVar):
            return self.vars[expr.name]

        if isinstance(expr, FieldConst):
            if expr.modulus is None:
                raise ValueError("field constants must carry a modulus")
            if not 0 <= expr.value < expr.modulus:
                raise ValueError("constant outside field range")
            return z3.IntVal(expr.value)

        if isinstance(expr, FieldSub):
            modulus = self._field_modulus(expr)
            result = self._tr(expr.args[0])
            for arg in expr.args[1:]:
                result = self._mod(result - self._tr(arg), modulus)
            return result

        if isinstance(expr, FieldNeg):
            modulus = self._field_modulus(expr)
            return self._mod(-self._tr(expr.arg), modulus)

        if isinstance(expr, FieldPow):
            modulus = self._field_modulus(expr)
            return self._pow_mod(self._tr(expr.base), modulus, expr.exponent)

        if isinstance(expr, FieldDiv):
            raise ValueError(
                "Finite-field division is unsupported without an explicit nonzero side condition"
            )

        if isinstance(expr, BoolOr):
            return z3.Or(*[self._tr(arg) for arg in expr.args])

        if isinstance(expr, BoolAnd):
            return z3.And(*[self._tr(arg) for arg in expr.args])

        if isinstance(expr, BoolXor):
            args = [self._tr(arg) for arg in expr.args]
            return functools.reduce(z3.Xor, args)

        if isinstance(expr, BoolNot):
            return z3.Not(self._tr(expr.arg))

        if isinstance(expr, BoolImplies):
            return z3.Implies(self._tr(expr.antecedent), self._tr(expr.consequent))

        if isinstance(expr, BoolIte):
            return z3.If(
                self._tr(expr.cond),
                self._tr(expr.then_expr),
                self._tr(expr.else_expr),
            )

        if isinstance(expr, BoolVar):
            return self.vars[expr.name]

        if isinstance(expr, BoolConst):
            return z3.BoolVal(expr.value)

        raise TypeError("unknown AST node %s" % type(expr).__name__)
