#!/usr/bin/env python3
import z3
from aria.itp.utils import unify, pmatch, antipattern
from aria.unification.core import unify as core_unify, reify
from aria.unification.variable import var


def z3_unify():
    x, y = z3.Ints("x y")
    f = z3.Function("f", z3.IntSort(), z3.IntSort())
    t1, t2 = f(x), f(z3.IntVal(5))
    s = unify([x], t1, t2)
    if s:
        print(f"Unify {t1} with {t2}: {s}")
    t1, t2 = f(x), f(y)
    s = unify([x, y], t1, t2)
    if s:
        print(f"Unify {t1} with {t2}: {s}")


def pattern_match():
    x, y = z3.Ints("x y")
    f = z3.Function("f", z3.IntSort(), z3.IntSort())
    pat, t = f(x), f(z3.IntVal(42))
    m = pmatch([x], pat, t)
    if m:
        print(f"Match {pat} against {t}: {m}")


def anti_unify():
    try:
        a, b, c, d = z3.Ints("a b c d")
        f = z3.Function("f", z3.IntSort(), z3.IntSort(), z3.IntSort())
        t1, t2 = f(a, b), f(c, d)
        pvars, pat = antipattern([t1, t2])
        print(f"Generalize {t1}, {t2}: {pat}")
    except Exception as e:
        print(f"Anti-unify failed: {e}")


def core_unify_ex():
    x = var("x")
    y = var("y")
    t1, t2 = (1, x, (3, y)), (1, 2, (3, 4))
    s = core_unify(t1, t2)
    if s:
        print(f"Unify {t1} with {t2}: {s}")
        print(f"Result: {reify(t1, s)}")


if __name__ == "__main__":
    z3_unify()
    pattern_match()
    anti_unify()
    core_unify_ex()
