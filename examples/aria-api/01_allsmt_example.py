#!/usr/bin/env python3
import z3
from aria.allsmt import create_allsmt_solver

def bool_example():
    a, b, c = z3.Bools('a b c')
    f = z3.And(z3.Or(a, b), z3.Or(z3.Not(a), c), z3.Or(b, z3.Not(c)))
    s = create_allsmt_solver("z3")
    s.solve(f, [a, b, c], model_limit=10)
    print(f"Boolean: {s.get_model_count()} models"); s.print_models(False)

def int_example():
    x, y = z3.Ints('x y')
    f = z3.And(x + y == 5, x >= 0, y >= 0, x <= 3, y <= 3)
    s = create_allsmt_solver("z3")
    s.solve(f, [x, y], model_limit=10)
    print(f"Int: {s.get_model_count()} models"); s.print_models(False)

def mixed_example():
    x, y = z3.Ints('x y'); p, q = z3.Bools('p q')
    f = z3.And(z3.Implies(p, x > 0), z3.Implies(q, y > 0), z3.Or(p, q), x + y <= 3, x >= 0, y >= 0)
    s = create_allsmt_solver("z3")
    s.solve(f, [x, y, p, q], model_limit=15)
    print(f"Mixed: {s.get_model_count()} models"); s.print_models(False)

def compare_solvers():
    a, b = z3.Bools('a b')
    f = z3.And(z3.Or(a, b), z3.Or(z3.Not(a), z3.Not(b)))
    z3s = create_allsmt_solver("z3"); z3s.solve(f, [a, b], model_limit=5)
    print(f"Z3: {z3s.get_model_count()} models")
    try:
        ps = create_allsmt_solver("pysmt"); ps.solve(f, [a, b], model_limit=5)
        print(f"PySMT: {ps.get_model_count()} models")
    except: print("PySMT not available")

def real_example():
    x, y = z3.Reals('x y')
    f = z3.And(x*x + y*y <= 1, x >= 0, y >= 0, x <= 1, y <= 1)
    s = create_allsmt_solver("z3")
    s.solve(f, [x, y], model_limit=3)
    print(f"Real: {s.get_model_count()} sample models"); s.print_models(False)

def main():
    print("AllSMT Examples\n" + "="*20)
    bool_example(); int_example(); mixed_example(); compare_solvers(); real_example()
    print("Done!")

if __name__ == "__main__":
    main()