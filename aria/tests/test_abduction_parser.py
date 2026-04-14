"""Tests for SMT-LIB abduction parsing."""

import z3

from aria.abduction.abductor_parser import (
    extract_abduction_goal,
    parse_abduction_problem,
    parse_smt2_expr,
)


def test_parse_abduction_problem_supports_let_distinct_and_mod() -> None:
    smt2_str = """
    (declare-const x Int)
    (declare-const y Int)
    (assert (let ((sum (+ x y))) (and (> sum 3) (= (mod sum 2) 0))))
    (get-abduct abd (distinct x y))
    """

    precond, goal, variables = parse_abduction_problem(smt2_str)

    assert set(variables) == {"x", "y"}
    solver = z3.Solver()
    solver.add(precond, x := variables["x"] == 1, variables["y"] == 3)
    assert solver.check() == z3.sat
    assert z3.is_distinct(goal)


def test_parse_abduction_problem_supports_uninterpreted_functions_and_arrays() -> None:
    smt2_str = """
    (declare-fun f (Int Bool) Int)
    (declare-fun arr () (Array Int Int))
    (declare-const flag Bool)
    (assert (= (select (store arr 5 (f 1 flag)) 5) (f 1 flag)))
    (get-abduct abd (= (f 2 false) 9))
    """

    precond, goal, variables = parse_abduction_problem(smt2_str)

    assert "f" in variables
    assert "arr" in variables
    assert z3.is_eq(goal)
    solver = z3.Solver()
    solver.add(precond)
    assert solver.check() == z3.sat


def test_parse_smt2_expr_supports_bitvectors_and_extract() -> None:
    smt2_str = """
    (declare-const x (_ BitVec 8))
    (declare-const y (_ BitVec 8))
    """
    _, _, variables = parse_abduction_problem(f"{smt2_str}(get-abduct abd true)")

    expr = parse_smt2_expr("(= ((_ extract 7 4) (bvadd x y)) #b1111)", variables)

    assert z3.is_eq(expr)
    assert expr.sort() == z3.BoolSort()


def test_extract_abduction_goal_ignores_comments_and_keeps_strings() -> None:
    smt2_str = """
    ; ignore this comment
    (declare-const s String)
    (assert (= s "abc"))
    (get-abduct abd (= s "a b c"))
    """

    assert extract_abduction_goal(smt2_str) == '(= s "a b c")'


def test_parse_abduction_problem_supports_sort_aliases() -> None:
    smt2_str = """
    (define-sort Byte () (_ BitVec 8))
    (declare-const x Byte)
    (declare-const y Byte)
    (assert (= x y))
    (get-abduct abd (bvule x #x0f))
    """

    precond, goal, variables = parse_abduction_problem(smt2_str)

    assert "x" in variables
    assert variables["x"].sort().size() == 8
    assert z3.is_eq(precond)
    assert goal.sort() == z3.BoolSort()


def test_parse_smt2_expr_reuses_problem_prelude_for_alias_sorts() -> None:
    smt2_str = """
    (define-sort Byte () (_ BitVec 8))
    (declare-fun arr () (Array Byte Byte))
    (declare-const idx Byte)
    (get-abduct abd true)
    """

    _, _, variables = parse_abduction_problem(smt2_str)

    expr = parse_smt2_expr("(= (select arr idx) #x2a)", variables)

    assert z3.is_eq(expr)
