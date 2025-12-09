#!/usr/bin/env python3
import z3
from aria.backbone.smt_backbone_literals import get_backbone_literals

def basic_backbone():
    a, b, c = z3.Bools('a b c')
    f = z3.And(z3.Or(a, b), z3.Implies(a, c))
    lits = [a, z3.Not(a), b, z3.Not(b), c, z3.Not(c)]
    try:
        bb = get_backbone_literals(f, lits, 'model_enumeration')
        print(f"Backbone: {len(bb)}"); [print(l) for l in bb]
    except: print("Backbone failed")

def compare_algorithms():
    x, y = z3.Ints('x y')
    f = z3.And(x + y >= 5, x >= 2, y >= 1, x <= 10, y <= 10)
    lits = [x >= 3, x <= 8, y >= 2, y <= 7]
    for alg in ['model_enumeration', 'sequence_checking']:
        try:
            bb = get_backbone_literals(f, lits, alg)
            print(f"{alg}: {len(bb)}")
        except: print(f"{alg}: failed")

def main():
    print("Backbone Examples\n" + "="*20)
    basic_backbone(); compare_algorithms()
    print("Done!")

if __name__ == "__main__":
    main() 