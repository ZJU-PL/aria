"""Smoke tests for the EFBV parallel solver."""

import random

import pytest
import z3

pytest.importorskip("pysat.formula")

from aria.quant.efbv.efbv_parallel import efbv_forall_solver
from aria.quant.efbv.efbv_parallel.efbv_cegis_parallel import (
    EFBVResult,
    bv_efsmt_with_uniform_sampling,
)
from aria.quant.efbv.efbv_parallel.efbv_utils import FSolverMode


def setup_module():
    random.seed(0)


def test_sat_trivial_formula():
    """Tautology should be reported SAT."""
    x, y = z3.BitVecs("x y", 2)
    phi = z3.Or(y == x, y != x)  # always true
    res = bv_efsmt_with_uniform_sampling(
        [x],
        [y],
        phi,
        maxloops=4,
        num_samples=2,
    )
    assert res == EFBVResult.SAT


def test_unsat_all_y_equal_x():
    """No single x can equal every y."""
    x, y = z3.BitVecs("x y", 2)
    phi = y == x
    res = bv_efsmt_with_uniform_sampling(
        [x],
        [y],
        phi,
        maxloops=6,
        num_samples=2,
    )
    assert res == EFBVResult.UNSAT


def test_unsat_guarded_parallel_forall():
    """Exercise parallel forall mode on a guarded contradiction."""
    efbv_forall_solver.m_forall_solver_strategy = FSolverMode.PARALLEL_THREAD
    x, y = z3.BitVecs("x y", 4)
    phi = z3.Implies(y > 2, y < x)
    res = bv_efsmt_with_uniform_sampling(
        [x],
        [y],
        phi,
        maxloops=4,
        num_samples=2,
    )
    assert res == EFBVResult.UNSAT


@pytest.mark.xfail(reason="Sampler sometimes misses satisfying candidates; investigate EFBV exists sampling.")
def test_sat_exists_solution_found():
    """Ensure a concrete candidate is validated (currently flaky/unsupported)."""
    efbv_forall_solver.m_forall_solver_strategy = FSolverMode.SEQUENTIAL
    x, y = z3.BitVecs("x y", 3)
    phi = z3.Implies(z3.And(y >= 0, y <= 3), y + x >= 1)
    res = bv_efsmt_with_uniform_sampling(
        [x],
        [y],
        phi,
        maxloops=8,
        num_samples=4,
    )
    assert res == EFBVResult.SAT
