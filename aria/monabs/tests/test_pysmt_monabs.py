"""Tests for PySMT-based monadic predicate abstraction checking functions."""
import pytest
from pysmt.shortcuts import And, Bool, Not, Or, Symbol
from pysmt.typing import BOOL

from aria.monabs.cores.unary_check_pysmt import (
    unary_check,
    unary_check_cached,
    unary_check_incremental,
    unary_check_incremental_cached,
)
from aria.monabs.cores.dis_check_pysmt import (
    disjunctive_check_cached,
    disjunctive_check_incremental_cached,
)
from aria.monabs.cores.con_check_pysmt import (
    conjunctive_check,
    conjunctive_check_incremental,
)


def _vars():
    """Create test variables."""
    return Symbol("x", BOOL), Symbol("y", BOOL)


@pytest.mark.parametrize(
    "func",
    [
        unary_check,
        unary_check_incremental,
        unary_check_cached,
        unary_check_incremental_cached,
    ],
)
def test_unary_variants(func):
    """Test unary check variants."""
    x, _ = _vars()
    precond = x  # forces x true, so Not(x) becomes unsat
    cnts = [x, Not(x), Or(x, Bool(False))]
    assert func(precond, cnts) == [1, 0, 1]


@pytest.mark.parametrize(
    "func",
    [disjunctive_check_cached, disjunctive_check_incremental_cached],
)
def test_disjunctive_variants(func):
    """Test disjunctive check variants."""
    x, y = _vars()
    precond = And(x, y)
    cnts = [x, y, Not(x)]
    # Under precond, x and y are satisfiable; Not(x) is contradictory.
    assert func(precond, cnts) == [1, 1, 0]


@pytest.mark.parametrize("algo", [0, 1, 2])
@pytest.mark.parametrize(
    "func",
    [conjunctive_check, conjunctive_check_incremental],
)
def test_conjunctive_variants(func, algo):
    """Test conjunctive check variants with different algorithms."""
    x, y = _vars()
    precond = And(x, y)
    cnts = [x, y, Not(x)]
    # Expect the conflicting constraint to be marked unsat.
    res = func(precond, cnts, alogorithm=algo)
    assert res == [1, 1, 0]
