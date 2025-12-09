#!/usr/bin/env python3
"""
ffbv_solver.py  –  Finite-field solver via faithful BV encoding

Usage example
-------------
python ffbv_solver.py demo
python ffbv_solver.py regress  path/to/ff/benchmarks
"""
from __future__ import annotations
from typing import Dict, Optional, List
import sys, os, pathlib
import z3
from .ff_ast import (
    FieldExpr, FieldAdd, FieldMul, FieldNeg, FieldEq, FieldVar, FieldConst,
    FieldSub, FieldPow, FieldDiv, BoolOr, BoolAnd, BoolNot, BoolImplies, BoolIte, BoolVar, ParsedFormula
)
from .ff_parser import parse_ff_file

# -----------------------------------------------------------------------

class FFBVSolver:
    """
    Faithful translation of finite-field (prime field) constraints to QF_BV.
    All temporaries are evaluated in 2·k bits, with k = ⌈log₂ p⌉.
    """
    def __init__(self, theory:str="QF_BV"):
        self.solver         = z3.Solver()
        self.vars : Dict[str,z3.BitVecRef] = {}
        self.p              : Optional[int]        = None
        self.k              : Optional[int]        = None   # base width
        self.kw             : Optional[int]        = None   # 2*k width
        self.p_wide_bv      : Optional[z3.BitVecVal] = None

    # ----------------------  public API  -------------------------------
    def check(self, formula:ParsedFormula)->z3.CheckSatResult:
        self._setup_field(formula.field_size)
        self._declare_vars(formula.variables)

        for a in formula.assertions:
            self.solver.add(self._tr(a))

        return self.solver.check()

    def model(self)->Optional[z3.ModelRef]:
        if self.solver.reason_unknown():
            return None
        if self.solver.check() == z3.sat:
            return self.solver.model()
        return None

    # -----------------------  helpers  ---------------------------------
    def _setup_field(self, p:int)->None:
        if p <= 1:
            raise ValueError("Field size must be prime ≥ 2")
        self.p  = p
        self.k  = (p-1).bit_length()
        self.kw = self.k * 2
        self.p_wide_bv = z3.BitVecVal(p, self.kw)

    def _declare_vars(self, varmap:Dict[str,str])->None:
        for v, sort_type in varmap.items():
            if v in self.vars: continue
            if sort_type == 'bool':
                self.vars[v] = z3.Bool(v)
            else:  # 'ff' or finite field
                self.vars[v] = z3.BitVec(v, self.k)
                # 0 ≤ v < p
                self.solver.add(z3.ULT(self.vars[v], z3.BitVecVal(self.p, self.k)))

    # ----------  BV arithmetic with exact mod-p reduction  --------------
    def _to_wide(self, e:z3.BitVecRef)->z3.BitVecRef:
        return z3.ZeroExt(self.kw - e.size(), e)

    def _mod_p(self, wide:z3.BitVecRef)->z3.BitVecRef:
        """Reduce a kw-bit term modulo p and return a k-bit value."""
        tmp   = z3.URem(wide, self.p_wide_bv)     # kw bits
        return z3.Extract(self.k-1, 0, tmp)       # back to k bits

    def _pow_mod_p(self, base_bv:z3.BitVecRef, exponent:int)->z3.BitVecRef:
        """Return (base_bv ** exponent) mod p as a k-bit bit-vector."""
        result_wide = z3.BitVecVal(1, self.kw)
        base_wide   = self._to_wide(base_bv)
        e = exponent
        while e > 0:
            if e & 1:
                result_wide = self._mod_p(result_wide * base_wide)
                result_wide = self._to_wide(result_wide)
            e >>= 1
            if e:
                base_wide = self._mod_p(base_wide * base_wide)
                base_wide = self._to_wide(base_wide)
        return self._mod_p(result_wide)

    # ----------------  recursive translation  --------------------------
    def _tr(self, e:FieldExpr)->z3.ExprRef:
        if isinstance(e, FieldAdd):
            wide = z3.BitVecVal(0, self.kw)
            for arg in e.args:
                wide = self._mod_p(wide + self._to_wide(self._tr(arg)))
                # prepare for next iteration in wide format
                wide = self._to_wide(wide)
            return self._mod_p(wide)

        if isinstance(e, FieldMul):
            wide = z3.BitVecVal(1, self.kw)
            for arg in e.args:
                wide = self._mod_p(wide * self._to_wide(self._tr(arg)))
                wide = self._to_wide(wide)
            return self._mod_p(wide)

        if isinstance(e, FieldEq):
            return self._tr(e.left) == self._tr(e.right)

        if isinstance(e, FieldVar):
            return self.vars[e.name]

        if isinstance(e, FieldConst):
            if not (0 <= e.value < self.p):
                raise ValueError("Constant out of field range")
            return z3.BitVecVal(e.value, self.k)

        if isinstance(e, FieldSub):
            # a − b − c ≡ a + (p−b) + (p−c)  mod p
            wide = self._to_wide(self._tr(e.args[0]))
            for arg in e.args[1:]:
                sub = self._to_wide(self._tr(arg))
                wide = self._mod_p(wide + (self.p_wide_bv - sub))
                wide = self._to_wide(wide)
            return self._mod_p(wide)

        if isinstance(e, FieldNeg):
            sub = self._to_wide(self._tr(e.arg))
            wide = (self.p_wide_bv - sub)
            return self._mod_p(wide)

        if isinstance(e, FieldPow):
            base_bv = self._tr(e.base)
            return self._pow_mod_p(base_bv, e.exponent)

        if isinstance(e, FieldDiv):
            # num * denom^{p-2}  mod p  (Fermat’s little theorem, p prime)
            inv = FieldPow(e.denom, self.p - 2)
            return self._tr(FieldMul(e.num, inv))

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
