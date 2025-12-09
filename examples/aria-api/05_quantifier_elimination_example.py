#!/usr/bin/env python3
import z3
from aria.quant.qe.qe_lme import qelim_exists_lme

def basic_qe():
    x, y = z3.Reals('x y')
    f = z3.And(x + y > 0, x < 5)
    try:
        r = qelim_exists_lme(f, [x])
        print(f"QE: {r}")
    except: print("QE failed")

def lra_qe():
    x, y, z = z3.Reals('x y z')
    f = z3.And(2*x + y <= 10, x >= 0, x + z >= 5)
    try:
        r = qelim_exists_lme(f, [x])
        print(f"LRA QE: {r}")
    except: print("LRA QE failed")

def z3_qe():
    x, y = z3.Reals('x y')
    f = z3.Exists([x], z3.And(x + y > 0, x < 5))
    qe = z3.Tactic('qe')(f)
    print(f"Z3 QE: {qe}")

def main():
    print("QE Examples\n" + "="*20)
    basic_qe(); lra_qe(); z3_qe()
    print("Done!")

if __name__ == "__main__":
    main() 