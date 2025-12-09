#!/usr/bin/env python3
import z3
from aria.unsat_core.marco import SubsetSolver, MapSolver, enumerate_sets

def basic_core():
    x, y, z = z3.Ints('x y z')
    cs = [x + y <= 5, x >= 3, y >= 4, x + y >= 10, z >= 0]
    s = z3.Solver(); [s.add(c) for c in cs]
    r = s.check(); print(f"Basic: {r}")
    if r == z3.unsat:
        s2 = z3.Solver(); a = [z3.Bool(f"c{i}") for i in range(len(cs))]
        [s2.add(z3.Implies(a[i], cs[i])) for i in range(len(cs))]
        if s2.check(a) == z3.unsat:
            print(f"Core: {s2.unsat_core()}")

def named_core():
    a, b, c = z3.Bools('a b c')
    c1, c2, c3, c4, c5 = z3.Bools('c1 c2 c3 c4 c5')
    s = z3.Solver()
    s.add(z3.Implies(c1, a), z3.Implies(c2, z3.Not(a)), z3.Implies(c3, z3.Or(a, b)), z3.Implies(c4, z3.Not(b)), z3.Implies(c5, c))
    r = s.check([c1, c2, c3, c4, c5]); print(f"Named: {r}")
    if r == z3.unsat:
        print(f"Core: {s.unsat_core()}")

def marco_mus():
    x, y = z3.Reals('x y')
    cs = [x > 2, x < 1, y >= 0, y < 0]
    try:
        csolver = SubsetSolver(cs); msolver = MapSolver(n=csolver.n)
        mus, mss = 0, 0
        for orig, lits in enumerate_sets(csolver, msolver):
            if orig == "MUS": mus += 1; print(f"MUS {mus}: {len(lits)}")
            elif orig == "MSS": mss += 1; print(f"MSS {mss}: {len(lits)}")
        print(f"Total: {mus} MUS, {mss} MSS")
    except Exception as e:
        print(f"Marco failed: {e}")

def simple_core():
    x, y, z = z3.Ints('x y z')
    cs = [("lx", x >= 10), ("ux", x <= 5), ("ly", y >= 8), ("uy", y <= 3), ("sum", x + y == 15)]
    s = z3.Solver(); a = []
    for name, c in cs:
        aa = z3.Bool(name); s.add(z3.Implies(aa, c)); a.append(aa)
    r = s.check(a); print(f"Simple: {r}")
    if r == z3.unsat:
        print(f"Conflicting: {list(s.unsat_core())}")

def main():
    print("Unsat Core Examples\n" + "="*20)
    basic_core(); named_core(); marco_mus(); simple_core()
    print("Done!")

if __name__ == "__main__":
    main() 