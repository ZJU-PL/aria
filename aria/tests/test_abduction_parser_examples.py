"""Executable parser examples for manual smoke testing."""

import z3

from aria.abduction.abductor_parser import parse_abduction_problem


def example_int() -> None:
    """Integer variables example."""
    smt2_str = """
    (declare-fun x () Int)
    (declare-fun y () Int)
    (declare-fun z () Int)
    (declare-fun w () Int)
    (declare-fun u () Int)
    (declare-fun v () Int)
    (assert (>= x 0))
    (assert (or (>= x 0) (< u v)))
    (get-abduct A (and (>= (+ x y z w u v) 2) (<= (+ x y z w) 3)))
    """

    try:
        precond, goal, variables = parse_abduction_problem(smt2_str)
        print("== Integer Example ==")
        print("Variables:", list(variables.keys()))
        print("Precondition:", precond)
        print("Goal:", goal)
    except (ValueError, z3.Z3Exception) as exc:
        print(f"Error: {exc}")


def example_bv() -> None:
    """Bit-vector variables example."""
    smt2_str = """
    (declare-fun x () (_ BitVec 32))
    (declare-fun y () (_ BitVec 32))
    (declare-fun z () (_ BitVec 32))
    (assert (bvuge x #x00000000))
    (assert (bvult y #x00000064))
    (get-abduct A (bvuge (bvadd x y z) #x00000002))
    """

    try:
        precond, goal, variables = parse_abduction_problem(smt2_str)
        print("\n== Bit-Vector Example ==")
        print("Variables:", list(variables.keys()))
        print("Precondition:", precond)
        print("Goal:", goal)
    except (ValueError, z3.Z3Exception) as exc:
        print(f"Error: {exc}")


def example_mixed() -> None:
    """Mixed types example."""
    smt2_str = """
    (declare-fun x () Int)
    (declare-fun y () (_ BitVec 32))
    (declare-fun arr () (Array Int Int))
    (declare-fun f (Int Int) Bool)
    (assert (>= x 0))
    (assert (bvult y #x00000064))
    (assert (= (select arr 5) 10))
    (get-abduct A (> x 5))
    """

    try:
        precond, goal, variables = parse_abduction_problem(smt2_str)
        print("\n== Mixed Types Example ==")
        print("Variables:", list(variables.keys()))
        for name, var in variables.items():
            if isinstance(var, z3.FuncDeclRef):
                args = [str(var.domain(idx)) for idx in range(var.arity())]
                print(f"  {name}: Function({', '.join(args)}) -> {var.range()}")
            else:
                print(f"  {name}: {var.sort()}")
        print("Precondition:", precond)
        print("Goal:", goal)
    except (ValueError, z3.Z3Exception) as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    example_int()
    example_bv()
    example_mixed()
