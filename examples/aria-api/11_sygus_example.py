#!/usr/bin/env python3
import z3
from aria.synthesis.cvc5.sygus_pbe import StringSyGuSPBE
from aria.utils.z3_plus_smtlib_solver import Z3SolverPlus


def string_syn():
    try:
        s = StringSyGuSPBE(debug=False)
        ex = [("hello", "world", "helloworld"), ("syn", "thesis", "synthesis")]
        r = s.synthesize_concat_function(ex, "concat_func")
        if r:
            print(f"Concat: {r}")
        ex2 = [("hello", "hel"), ("synthesis", "syn")]
        r2 = s.synthesize_string_transformer(ex2, "first_three")
        if r2:
            print(f"Transform: {r2}")
    except Exception as e:
        print(f"String failed: {e}")


def func_syn():
    try:
        s = Z3SolverPlus(debug=False)
        maxf = z3.Function("max", z3.IntSort(), z3.IntSort(), z3.IntSort())
        x, y = z3.Ints("x y")
        cons = [
            maxf(x, y) >= x,
            maxf(x, y) >= y,
            z3.Or(maxf(x, y) == x, maxf(x, y) == y),
        ]
        r = s.sygus([maxf], cons, [x, y], logic="LIA")
        if r and "define-fun" in r:
            print(f"Max: {r}")
    except Exception as e:
        print(f"Func failed: {e}")


def bv_syn():
    try:
        s = Z3SolverPlus(debug=False)
        bvf = z3.Function("bv_func", z3.BitVecSort(4), z3.BitVecSort(4))
        x = z3.BitVec("x", 4)
        cons = [
            bvf(z3.BitVecVal(1, 4)) == z3.BitVecVal(8, 4),
            bvf(z3.BitVecVal(2, 4)) == z3.BitVecVal(1, 4),
            bvf(z3.BitVecVal(4, 4)) == z3.BitVecVal(2, 4),
        ]
        r = s.sygus([bvf], cons, [x], logic="BV")
        if r and "define-fun" in r:
            print(f"BV: {r}")
    except Exception as e:
        print(f"BV failed: {e}")


if __name__ == "__main__":
    string_syn()
    func_syn()
    bv_syn()
