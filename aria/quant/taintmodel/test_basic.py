# tests/test_basic.py --------------------------------------------------------
import pytest, os, textwrap
from aria.quant.taintmodel.solver import QuantSolver
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
