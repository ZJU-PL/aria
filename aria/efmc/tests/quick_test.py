# coding: utf-8
from typing import List

import z3


def is_equiv(fml_a: z3.ExprRef, fml_b: z3.ExprRef):
    s = z3.Solver()
    s.add(fml_a != fml_b)
    return s.check() == z3.unsat


def optimize(formula: z3.ExprRef, obj: z3.ExprRef, minimize=False, timeout: int = 0):
    s = z3.Optimize()
    s.add(formula)
    if timeout > 0:
        s.set("timeout", timeout)
    if minimize:
        obj = s.minimize(obj)
    else:
        obj = s.maximize(obj)
    if s.check() == z3.sat:
        return obj.value()


def box_optimize(formula: z3.ExprRef, minimize: List, maximize: List, timeout: int = 0):
    s = z3.Optimize()
    s.set("opt.priority", "box")
    s.add(formula)
    if timeout > 0:
        s.set("timeout", timeout)
    min_objectives = [s.minimize(exp) for exp in minimize]
    max_objectives = [s.maximize(exp) for exp in maximize]
    if s.check() == z3.sat:
        # print(s.sexpr())
        min_res = [obj.value() for obj in min_objectives]
        max_res = [obj.value() for obj in max_objectives]

        return min_res, max_res


x, y, xp, yp = z3.Reals("x y x! y!")

fml = z3.And(
    z3.Or(
        z3.And(x <= -2, x >= -3),
        z3.And(-1 <= x, 2 >= x),
        z3.And(-1 <= x, 0 >= x),
        z3.And(-1 <= x, 3 >= x),
        z3.And(-1 <= x, 4 >= x),
        z3.And(-1 <= x, 5 >= x),
        z3.And(-1 <= x, 6 >= x),
    ),
    z3.Or(z3.And(xp == x + 2, x < 1), z3.And(xp == x + 1, x >= 1)),
)

"""
Seems to be a bug??
FIXME: maybe choose the self-compiled python packages for Z3 (then we can use newer versions...)
"""

# for i in range(3):
# print(box_optimize(fml, [xp], [xp]))
# print(optimize(fml, xp, minimize=True))
# print(optimize(fml, xp, minimize=False))

if __name__ == "__main__":
    a, b = z3.Bools("a b")
    fml_m = z3.Implies(z3.Not(a), b)
    fml_n = z3.Or(a, b)
    print(is_equiv(fml_m, fml_n))
