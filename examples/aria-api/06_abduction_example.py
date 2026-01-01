#!/usr/bin/env python3
import z3
from aria.abduction.qe_abduct import qe_abduce
from aria.abduction.dillig_abduct import dillig_abduce


def basic_abduction():
    x, y = z3.Reals("x y")
    pre = z3.And(x <= 0, y > 1)
    post = x + y <= 5
    try:
        r = qe_abduce(pre, post)
        print(f"QE: {r}")
    except:
        print("QE abduction failed")


def dillig_abduction():
    x, y = z3.Ints("x y")
    pre = z3.And(x >= 0, y >= 0)
    post = x + y >= 5
    try:
        r = dillig_abduce(pre, post)
        print(f"Dillig: {r if r is not None else 'No abduction'}")
    except:
        print("Dillig abduction failed")


def compare_methods():
    x, y = z3.Reals("x y")
    pre = z3.And(x >= 0, y >= 0)
    post = x + y >= 5
    for name, method in [("QE", qe_abduce), ("Dillig", dillig_abduce)]:
        try:
            r = method(pre, post)
            print(f"{name}: {r if r is not None else 'No result'}")
        except:
            print(f"{name}: Failed")


def main():
    print("Abduction Examples\n" + "=" * 20)
    basic_abduction()
    dillig_abduction()
    compare_methods()
    print("Done!")


if __name__ == "__main__":
    main()
