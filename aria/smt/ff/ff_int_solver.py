#!/usr/bin/env python3
"""
ff_int_solver.py  –  Finite-field formulas via integer translation
Sound for every prime field GF(p).
"""
from __future__ import annotations
from typing import Dict, List, Optional
import math, sys, pathlib
import z3
from .ff_ast import (
    FieldExpr, FieldAdd, FieldMul, FieldEq, FieldVar, FieldConst,
    FieldSub, FieldNeg, FieldPow, FieldDiv, BoolOr, BoolAnd, BoolNot, BoolImplies, BoolIte, BoolVar, ParsedFormula
)
# ---------------------------------------------------------------------

def _is_prime(n:int)->bool:
    return n >= 2 and all(n % k for k in range(2, int(math.isqrt(n))+1))

class FFIntSolver:
    """
    Prime-field solver via a direct translation to non-linear integers.
    """
    def __init__(self):
        self.solver   = z3.SolverFor("QF_NIA")
        self.vars : Dict[str,z3.IntRef] = {}
        self.p    : Optional[int] = None

    # ---------------------------------------------------------------
    def check(self, formula:ParsedFormula)->z3.CheckSatResult:
        self._setup_field(formula.field_size)
        self._declare_vars(formula.variables)
        for a in formula.assertions:
            self.solver.add(self._tr(a))
        return self.solver.check()

    def model(self)->Optional[z3.ModelRef]:
        if self.solver.check() == z3.sat:
            return self.solver.model()
        return None

    # ---------------------------------------------------------------
    def _setup_field(self, p:int)->None:
        if not _is_prime(p):
            raise ValueError(f"Finite-field sort requires prime p, got {p}")
        self.p = p

    def _declare_vars(self, varmap:Dict[str,str])->None:
        for v, sort_type in varmap.items():
            if v in self.vars: continue
            if sort_type == 'bool':
                self.vars[v] = z3.Bool(v)
            else:  # 'ff' or finite field
                iv = z3.Int(v)
                self.vars[v] = iv
                self.solver.add(z3.And(iv >= 0, iv < self.p))

    # ---------------- translation helpers --------------------------
    def _mod(self, term:z3.IntRef)->z3.IntRef:
        return term % self.p             # ← portable remainder

    def _pow_mod(self, base:z3.IntRef, exp:int)->z3.IntRef:
        """Return (base ** exp) mod p using square-and-multiply."""
        result = z3.IntVal(1)
        b = base
        e = exp
        while e > 0:
            if e & 1:
                result = self._mod(result * b)
            e >>= 1
            if e:
                b = self._mod(b * b)
        return result

    def _tr(self, e:FieldExpr)->z3.ExprRef:
        if isinstance(e, FieldAdd):
            res = z3.IntVal(0)
            for arg in e.args:
                res = self._mod(res + self._tr(arg))
            return res

        if isinstance(e, FieldMul):
            res = z3.IntVal(1)
            for arg in e.args:
                res = self._mod(res * self._tr(arg))
            return res

        if isinstance(e, FieldEq):
            return self._tr(e.left) == self._tr(e.right)

        if isinstance(e, FieldVar):
            return self.vars[e.name]

        if isinstance(e, FieldConst):
            if not (0 <= e.value < self.p):
                raise ValueError("constant outside field range")
            return z3.IntVal(e.value)

        if isinstance(e, FieldSub):
            res = self._tr(e.args[0])
            for arg in e.args[1:]:
                res = self._mod(res - self._tr(arg))
            return res

        if isinstance(e, FieldNeg):
            return self._mod(- self._tr(e.arg))

        if isinstance(e, FieldPow):
            base_int = self._tr(e.base)
            return self._pow_mod(base_int, e.exponent)

        if isinstance(e, FieldDiv):
            # num * denom^{p-2}  mod p   (for prime p)
            inv = self._pow_mod(self._tr(e.denom), self.p - 2)
            return self._mod(self._tr(e.num) * inv)

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

        raise TypeError(f"unknown AST node {type(e).__name__}")
