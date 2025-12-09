#!/usr/bin/env python3
import z3
from aria.bool.interpolant.core_based_itp import BooleanInterpolant
from aria.bool.interpolant.pysmt_itp import pysmt_binary_itp
from aria.utils.pysmt_solver import PySMTSolver

def bool_itp():
    x, y = z3.Bools("x y")
    a = z3.And(x, y); b = z3.And(z3.Not(x), z3.Not(y))
    itp = BooleanInterpolant.compute_itp(a, b, [x, y])
    print(f"Bool: {list(itp)}")

def pysmt_itp():
    try:
        x, y = z3.Bools("x y")
        a = z3.And(x, y); b = z3.And(z3.Not(x), z3.Not(y))
        itp = pysmt_binary_itp(a, b)
        print(f"PySMT: {itp}")
    except Exception as e:
        print(f"PySMT failed: {e}")

def arith_itp():
    try:
        x, y = z3.Ints("x y")
        a = z3.And(x > 0, y > 0); b = z3.And(x < 0, y < 0)
        solver = PySMTSolver()
        itp = solver.binary_interpolant(a, b, logic="LIA")
        print(f"Arith: {itp}")
    except Exception as e:
        print(f"Arith failed: {e}")

if __name__ == "__main__":
    bool_itp(); pysmt_itp(); arith_itp() 