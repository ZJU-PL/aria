#!/usr/bin/env python3
"""Unification and pattern-matching examples using aria.smt.unification."""

from aria.smt.unification import unify, reify, var
from aria.smt.unification.match import Dispatcher


def basic_unify():
    x, y = var("x"), var("y")
    t1, t2 = (1, x, (3, y)), (1, 2, (3, 4))
    s = unify(t1, t2)
    if s:
        print(f"Unify {t1} with {t2}: {s}")
        print(f"Result: {reify(t1, s)}")


def list_unify():
    x = var("x")
    s = unify([1, x, 3], [1, 2, 3])
    if s:
        print(f"Unify list: {s}")
        print(f"Reified: {reify([1, x, 3], s)}")


def pattern_dispatch():
    x = var("x")
    d = Dispatcher("eval_op")
    d.add(("add", 0, x), lambda op, a, b: b)
    d.add(("add", x, 0), lambda op, a, b: a)
    d.add(("mul", 1, x), lambda op, a, b: b)

    print("Pattern dispatch (add 0 5):", d("add", 0, 5))
    print("Pattern dispatch (mul 1 42):", d("mul", 1, 42))


def main():
    print("Unification Examples\n" + "=" * 20)
    basic_unify()
    list_unify()
    pattern_dispatch()
    print("Done!")


if __name__ == "__main__":
    main()
