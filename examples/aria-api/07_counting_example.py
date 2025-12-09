#!/usr/bin/env python3
import z3
from aria.counting.bool.z3py_expr_counting import count_z3_solutions, count_z3_models_by_enumeration
from aria.counting.arith_counting_latte import count_lia_models
from aria.allsmt.bool_enumeration import count_models

def bool_count():
    a, b, c = z3.Bools('a b c')
    f = z3.And(z3.Or(a, b), z3.Or(z3.Not(a), c))
    try:
        print(f"Bool count: {count_z3_solutions(f)}, Enum: {count_z3_models_by_enumeration(f)}")
    except: print("Boolean counting failed")

def compare_methods():
    p, q = z3.Bools('p q')
    f = z3.Or(z3.And(p, q), z3.And(z3.Not(p), z3.Not(q)))
    for m in ['solver', 'enum1', 'enum2']:
        try:
            print(f"{m}: {count_models(f, [p, q], method=m)}")
        except: print(f"{m}: failed")

def arith_count():
    x, y = z3.Ints('x y')
    f = z3.And(x + y == 5, x >= 0, y >= 0, x <= 3, y <= 3)
    try:
        print(f"Arith count: {count_lia_models(f)}")
    except: print("Arith counter failed")

def main():
    print("Counting Examples\n" + "="*20)
    bool_count(); compare_methods(); arith_count()
    print("Done!")

if __name__ == "__main__":
    main() 