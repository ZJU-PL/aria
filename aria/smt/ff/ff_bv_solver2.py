"""
ff_bv_solver2.py  –  Alternative finite-field solver via BV/Int bridging.
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

__all__ = ["FFBVBridgeSolver"]
class FFBVBridgeSolver:
    """Finite-field solver using BV2Int/Int2BV for modulo arithmetic."""

    def __init__(self, theory: str = "QF_BV"):  # pylint: disable=unused-argument
        self.solver = z3.Solver()
        self.vars: Dict[str, z3.ExprRef] = {}
        self.var_sorts: Dict[str, str] = {}
        self.field_widths: Dict[int, int] = {}

    def check(self, formula: ParsedFormula) -> z3.CheckSatResult:
        """Translate *formula* and call the underlying Z3 solver."""
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
        self.field_widths = {}

    def _setup_fields(self, fields) -> None:
        for modulus in fields:
            if not is_probable_prime(modulus):
                raise ValueError("Field size must be prime >= 2, got %d" % modulus)
            self.field_widths[modulus] = (modulus - 1).bit_length()

    def _declare_vars(self, varmap: Dict[str, str]) -> None:
        for name, sort_id in varmap.items():
            if is_bool_sort(sort_id):
                self.vars[name] = z3.Bool(name)
                continue
            modulus = field_modulus_from_sort(sort_id)
            if modulus is None:
                raise ValueError("unsupported sort %s" % sort_id)
            width = self.field_widths[modulus]
            bv = z3.BitVec(name, width)
            self.vars[name] = bv
            self.solver.add(self._as_int(bv) < modulus)

    def _field_modulus(self, expr: FieldExpr) -> int:
        modulus = infer_field_modulus(expr, self.var_sorts)
        if modulus is None:
            raise ValueError("expected a finite-field expression")
        return modulus

    def _as_int(self, bv: z3.BitVecRef) -> z3.ArithRef:
        return z3.BV2Int(bv, False)

    def _as_bv(self, int_expr: z3.ArithRef, modulus: int) -> z3.BitVecRef:
        return z3.Int2BV(int_expr, self.field_widths[modulus])

    def _mod_p_int(self, int_expr: z3.ArithRef, modulus: int) -> z3.ArithRef:
        return int_expr % modulus

    def _tr(
        self, expr: FieldExpr
    ) -> z3.ExprRef:  # pylint: disable=too-many-return-statements,too-many-branches
        if isinstance(expr, FieldAdd):
            modulus = self._field_modulus(expr)
            total = z3.IntVal(0)
            for arg in expr.args:
                total = total + self._as_int(self._tr(arg))
            return self._as_bv(self._mod_p_int(total, modulus), modulus)

        if isinstance(expr, FieldSub):
            modulus = self._field_modulus(expr)
            total = self._as_int(self._tr(expr.args[0]))
            for arg in expr.args[1:]:
                total = total - self._as_int(self._tr(arg))
            return self._as_bv(self._mod_p_int(total, modulus), modulus)

        if isinstance(expr, FieldNeg):
            modulus = self._field_modulus(expr)
            total = -self._as_int(self._tr(expr.arg))
            return self._as_bv(self._mod_p_int(total, modulus), modulus)

        if isinstance(expr, FieldMul):
            modulus = self._field_modulus(expr)
            total = z3.IntVal(1)
            for arg in expr.args:
                total = total * self._as_int(self._tr(arg))
            return self._as_bv(self._mod_p_int(total, modulus), modulus)

        if isinstance(expr, FieldPow):
            modulus = self._field_modulus(expr)
            base = self._as_int(self._tr(expr.base))
            result = z3.IntVal(1)
            exponent = expr.exponent
            running_base = base
            while exponent > 0:
                if exponent & 1:
                    result = self._mod_p_int(result * running_base, modulus)
                exponent >>= 1
                if exponent:
                    running_base = self._mod_p_int(
                        running_base * running_base, modulus
                    )
            return self._as_bv(result, modulus)

        if isinstance(expr, FieldDiv):
            raise ValueError(
                "Finite-field division is unsupported without an explicit nonzero side condition"
            )

        if isinstance(expr, FieldEq):
            return self._tr(expr.left) == self._tr(expr.right)

        if isinstance(expr, FieldVar):
            return self.vars[expr.name]

        if isinstance(expr, FieldConst):
            if expr.modulus is None:
                raise ValueError("field constants must carry a modulus")
            return z3.BitVecVal(expr.value, self.field_widths[expr.modulus])

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
