#!/usr/bin/env python3
"""
ff_bv_solver.py  –  Finite-field solver via a faithful bit-vector encoding.
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


class FFBVSolver:
    """Faithful translation of finite-field constraints to QF_BV."""

    def __init__(self, theory: str = "QF_BV"):  # pylint: disable=unused-argument
        self.solver = z3.Solver()
        self.vars: Dict[str, z3.ExprRef] = {}
        self.var_sorts: Dict[str, str] = {}
        self.field_infos: Dict[int, Dict[str, object]] = {}

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
        if self.solver.reason_unknown():
            return None
        if self.solver.check() == z3.sat:
            return self.solver.model()
        return None

    def _reset(self) -> None:
        self.solver = z3.Solver()
        self.vars = {}
        self.var_sorts = {}
        self.field_infos = {}

    def _setup_fields(self, fields) -> None:
        for modulus in fields:
            if not is_probable_prime(modulus):
                raise ValueError("Field size must be prime >= 2, got %d" % modulus)
            width = (modulus - 1).bit_length()
            wide_width = width * 2
            self.field_infos[modulus] = {
                "k": width,
                "kw": wide_width,
                "p_bv": z3.BitVecVal(modulus, width),
                "p_wide_bv": z3.BitVecVal(modulus, wide_width),
            }

    def _declare_vars(self, varmap: Dict[str, str]) -> None:
        for name, sort_id in varmap.items():
            if is_bool_sort(sort_id):
                self.vars[name] = z3.Bool(name)
                continue
            modulus = field_modulus_from_sort(sort_id)
            if modulus is None:
                raise ValueError("unsupported sort %s" % sort_id)
            info = self.field_infos[modulus]
            width = info["k"]
            bv = z3.BitVec(name, width)
            self.vars[name] = bv
            bound = z3.BitVecVal(modulus, width + 1)
            self.solver.add(z3.ULT(z3.ZeroExt(1, bv), bound))

    def _info(self, modulus: int) -> Dict[str, object]:
        return self.field_infos[modulus]

    def _field_modulus(self, expr: FieldExpr) -> int:
        modulus = infer_field_modulus(expr, self.var_sorts)
        if modulus is None:
            raise ValueError("expected a finite-field expression")
        return modulus

    def _to_wide(self, expr: z3.BitVecRef, modulus: int) -> z3.BitVecRef:
        info = self._info(modulus)
        return z3.ZeroExt(info["kw"] - expr.size(), expr)

    def _mod_p(self, wide: z3.BitVecRef, modulus: int) -> z3.BitVecRef:
        info = self._info(modulus)
        reduced = z3.URem(wide, info["p_wide_bv"])
        return z3.Extract(info["k"] - 1, 0, reduced)

    def _pow_mod_p(
        self, base_bv: z3.BitVecRef, modulus: int, exponent: int
    ) -> z3.BitVecRef:
        info = self._info(modulus)
        result_wide = z3.BitVecVal(1, info["kw"])
        base_wide = self._to_wide(base_bv, modulus)
        running_exponent = exponent
        while running_exponent > 0:
            if running_exponent & 1:
                result_wide = self._to_wide(
                    self._mod_p(result_wide * base_wide, modulus), modulus
                )
            running_exponent >>= 1
            if running_exponent:
                base_wide = self._to_wide(
                    self._mod_p(base_wide * base_wide, modulus), modulus
                )
        return self._mod_p(result_wide, modulus)

    def _tr(
        self, expr: FieldExpr
    ) -> z3.ExprRef:  # pylint: disable=too-many-return-statements,too-many-branches
        if isinstance(expr, FieldAdd):
            modulus = self._field_modulus(expr)
            info = self._info(modulus)
            wide = z3.BitVecVal(0, info["kw"])
            for arg in expr.args:
                wide = self._to_wide(
                    self._mod_p(wide + self._to_wide(self._tr(arg), modulus), modulus),
                    modulus,
                )
            return self._mod_p(wide, modulus)

        if isinstance(expr, FieldMul):
            modulus = self._field_modulus(expr)
            info = self._info(modulus)
            wide = z3.BitVecVal(1, info["kw"])
            for arg in expr.args:
                wide = self._to_wide(
                    self._mod_p(wide * self._to_wide(self._tr(arg), modulus), modulus),
                    modulus,
                )
            return self._mod_p(wide, modulus)

        if isinstance(expr, FieldEq):
            return self._tr(expr.left) == self._tr(expr.right)

        if isinstance(expr, FieldVar):
            return self.vars[expr.name]

        if isinstance(expr, FieldConst):
            if expr.modulus is None:
                raise ValueError("field constants must carry a modulus")
            info = self._info(expr.modulus)
            return z3.BitVecVal(expr.value, info["k"])

        if isinstance(expr, FieldSub):
            modulus = self._field_modulus(expr)
            wide = self._to_wide(self._tr(expr.args[0]), modulus)
            for arg in expr.args[1:]:
                sub = self._to_wide(self._tr(arg), modulus)
                wide = self._to_wide(
                    self._mod_p(wide + (self._info(modulus)["p_wide_bv"] - sub), modulus),
                    modulus,
                )
            return self._mod_p(wide, modulus)

        if isinstance(expr, FieldNeg):
            modulus = self._field_modulus(expr)
            sub = self._to_wide(self._tr(expr.arg), modulus)
            wide = self._info(modulus)["p_wide_bv"] - sub
            return self._mod_p(wide, modulus)

        if isinstance(expr, FieldPow):
            modulus = self._field_modulus(expr)
            return self._pow_mod_p(self._tr(expr.base), modulus, expr.exponent)

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

        raise TypeError("Unexpected AST node %s" % type(expr).__name__)
