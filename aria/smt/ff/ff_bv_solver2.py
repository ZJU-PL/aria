"""
ff_bv_solver2.py  –  Alternative finite-field solver via BV2Int / Int2BV bridge.

Strategy
--------
Represent every field element as a k-bit bit-vector (k = ⌈log₂ p⌉).
Operations are carried out in the *integer* domain using `BV2Int` to
convert operands, then reduced modulo *p*, and finally converted back to
k-bit bit-vectors with `Int2BV`.  This cleanly captures the algebraic
semantics without requiring an explicit wider (2·k) bit-vector for
intermediate results as in `ff_bv_solver.py`.

Compared to the faithful wide-BV encoding, this version:
• avoids manual modulo reduction logic – relies on Z3’s integer modulo;
• may be slower but demonstrates an orthogonal encoding style;
• serves as a reference for experimentation and cross-checking.

Public API mirrors `FFBVSolver` so that the regression driver can easily
switch between implementations.
"""
from __future__ import annotations
from typing import Dict, Optional, List
import z3

from .ff_ast import (
    FieldExpr, FieldAdd, FieldMul, FieldNeg, FieldEq, FieldVar, FieldConst,
    FieldSub, FieldPow, FieldDiv,
    BoolOr, BoolAnd, BoolNot, BoolImplies, BoolIte, BoolVar, ParsedFormula,
)

__all__ = ["FFBVBridgeSolver"]


class FFBVBridgeSolver:
    """Finite-field solver using Int/BV bridge for modulo arithmetic."""

    def __init__(self, theory: str = "QF_BV"):
        self.solver: z3.Solver = z3.Solver()
        self.vars: Dict[str, z3.BitVecRef] = {}
        self.p: Optional[int] = None  # prime field size
        self.k: Optional[int] = None  # bit-width of field elements (⌈log₂ p⌉)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def check(self, formula: ParsedFormula) -> z3.CheckSatResult:
        """Translate *formula* and call the underlying Z3 solver."""
        self._setup_field(formula.field_size)
        self._declare_vars(formula.variables)

        for a in formula.assertions:
            self.solver.add(self._tr(a))

        return self.solver.check()

    def model(self) -> Optional[z3.ModelRef]:
        if self.solver.reason_unknown():
            return None
        if self.solver.check() == z3.sat:
            return self.solver.model()
        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _setup_field(self, p: int) -> None:
        if p <= 1:
            raise ValueError("Field size must be prime ≥ 2")
        self.p = p
        self.k = (p - 1).bit_length()

    def _declare_vars(self, varmap: Dict[str, str]) -> None:
        for v, sort_type in varmap.items():
            if v in self.vars:
                continue
            if sort_type == "bool":
                self.vars[v] = z3.Bool(v)
            else:  # finite-field element
                bv = z3.BitVec(v, self.k)
                self.vars[v] = bv
                # Enforce 0 ≤ v_int < p
                self.solver.add(self._as_int(bv) < self.p)

    # ------------  Int/BV bridge helpers  -----------------------------
    def _as_int(self, bv: z3.BitVecRef) -> z3.ArithRef:
        return z3.BV2Int(bv, False)

    def _as_bv(self, int_expr: z3.ArithRef) -> z3.BitVecRef:
        return z3.Int2BV(int_expr, self.k)

    def _mod_p_int(self, int_expr: z3.ArithRef) -> z3.ArithRef:
        """Return *int_expr* mod p as an `Int` expression."""
        return int_expr % self.p if self.p is not None else int_expr

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------
    def _tr(self, e: FieldExpr) -> z3.ExprRef:
        """Translate a finite-field / Boolean AST node to a Z3 expression."""
        # Field algebra --------------------------------------------------
        if isinstance(e, FieldAdd):
            total_int = z3.IntVal(0)
            for arg in e.args:
                total_int = total_int + self._as_int(self._tr(arg))
            return self._as_bv(self._mod_p_int(total_int))

        if isinstance(e, FieldSub):
            if not e.args:
                raise ValueError("FieldSub expects at least one argument")
            total_int = self._as_int(self._tr(e.args[0]))
            for arg in e.args[1:]:
                total_int = total_int - self._as_int(self._tr(arg))
            return self._as_bv(self._mod_p_int(total_int))

        if isinstance(e, FieldNeg):
            neg_int = -self._as_int(self._tr(e.arg))
            return self._as_bv(self._mod_p_int(neg_int))

        if isinstance(e, FieldMul):
            prod_int = z3.IntVal(1)
            for arg in e.args:
                prod_int = prod_int * self._as_int(self._tr(arg))
            return self._as_bv(self._mod_p_int(prod_int))

        if isinstance(e, FieldPow):
            base_int = self._as_int(self._tr(e.base))
            # Fast exponentiation in the integer domain using Python int ops
            # (Z3 lifts them to Int expressions automatically)
            res_int = z3.IntVal(1)
            exp = e.exponent
            b = base_int
            while exp > 0:
                if exp & 1:
                    res_int = res_int * b
                exp >>= 1
                if exp:
                    b = b * b
            return self._as_bv(self._mod_p_int(res_int))

        if isinstance(e, FieldDiv):
            # a / b ≡ a * b^{p-2} (Fermat) since p is prime
            inv_pow = FieldPow(e.denom, self.p - 2 if self.p else 0)
            return self._tr(FieldMul(e.num, inv_pow))

        if isinstance(e, FieldEq):
            return self._tr(e.left) == self._tr(e.right)

        if isinstance(e, FieldVar):
            return self.vars[e.name]

        if isinstance(e, FieldConst):
            if self.p is None or not (0 <= e.value < self.p):
                raise ValueError("Constant out of field range")
            return z3.BitVecVal(e.value, self.k)

        # Boolean layer ---------------------------------------------------
        if isinstance(e, BoolOr):
            return z3.Or(*[self._tr(a) for a in e.args])

        if isinstance(e, BoolAnd):
            return z3.And(*[self._tr(a) for a in e.args])

        if isinstance(e, BoolNot):
            return z3.Not(self._tr(e.arg))

        if isinstance(e, BoolImplies):
            return z3.Implies(self._tr(e.antecedent), self._tr(e.consequent))

        if isinstance(e, BoolIte):
            return z3.If(self._tr(e.cond), self._tr(e.then_expr), self._tr(e.else_expr))

        if isinstance(e, BoolVar):
            return self.vars[e.name]

        raise TypeError(f"Unexpected AST node {type(e).__name__}")
