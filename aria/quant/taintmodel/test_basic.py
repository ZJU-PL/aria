# tests/test_basic.py --------------------------------------------------------
import pytest, os, textwrap
from aria.quant.taintmodel.solver import QuantSolver
from aria.quant.taintmodel.taint import infer_sic, infer_sic_with_taints
from z3 import *


@pytest.fixture
def solver():
    return QuantSolver()


def run(slv, smt):
    f = parse_smt2_string(smt)
    if isinstance(f, list):
        f = And(*f)
    return slv.solve(f)[0]  # "sat"/"unsat"/"unknown"


def test_motivating_example(solver):
    smt = """
    (declare-const a Int)
    (declare-const b Int)
    (assert (forall ((x Int)) (> (+ (* a x) b) 0)))
    (check-sat)
    """
    assert run(solver, smt) == "sat"


def test_unsat_simple(solver):
    smt = """
    (declare-const a Int)
    (assert (forall ((x Int)) (< x a)))
    (assert (forall ((x Int)) (> x a)))
    (check-sat)
    """
    assert run(solver, smt) == "unsat"


def test_array_select_store_rewrite():
    a = Array("a", IntSort(), IntSort())
    i, j, v = Ints("i j v")
    expr = Select(Store(a, i, v), j)
    sic = infer_sic(expr, {i, j})
    ite_expr = If(i == j, v, Select(a, j))
    sic_expected = infer_sic(ite_expr, {i, j})
    s = Solver()
    s.add(sic != sic_expected)
    assert s.check() == unsat


def _assert_equiv(lhs, rhs):
    s = Solver()
    s.add(lhs != rhs)
    assert s.check() == unsat


def test_motivation_ax_b_sic():
    a, b, x = Ints("a b x")
    expr = a * x + b > 0
    sic = infer_sic(expr, {x})
    _assert_equiv(sic, a == 0)


def test_bv_or_sic():
    a, b = BitVecs("a b", 8)
    expr = a | b
    sic = infer_sic(expr, {a})
    _assert_equiv(sic, b == BitVecVal(255, 8))


def test_taint_variable_path_matches_direct():
    a, b, x = Ints("a b x")
    expr = a * x + b > 0
    sic_direct = infer_sic(expr, {x})
    sic_taint, _ = infer_sic_with_taints(expr, {x})
    _assert_equiv(sic_direct, sic_taint)
