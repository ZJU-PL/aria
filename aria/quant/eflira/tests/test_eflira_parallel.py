"""Smoke tests for the EFLIRA parallel solver."""

import z3

from aria.quant.eflira.eflira_parallel import (
    EFLIRAResult,
    FSolverMode,
    lira_efsmt_with_parallel_cegis,
)


def test_sat_bounded_implication():
    """Bounded forall should be satisfiable."""
    x, y = z3.Ints("x y")
    phi = z3.Implies(z3.And(y >= 0, y <= 5), y - 2 * x < 10)
    res = lira_efsmt_with_parallel_cegis(
        [x],
        [y],
        phi,
        maxloops=6,
        num_samples=2,
        forall_mode=FSolverMode.PARALLEL_THREAD,
        num_workers=3,
    )
    assert res == EFLIRAResult.SAT


def test_sat_trivial_true_sequential():
    """Trivial true constraint should be SAT in sequential mode."""
    x, y = z3.Ints("x y")
    phi = z3.BoolVal(True)
    res = lira_efsmt_with_parallel_cegis(
        [x],
        [y],
        phi,
        maxloops=2,
        num_samples=1,
        forall_mode=FSolverMode.SEQUENTIAL,
        num_workers=1,
    )
    assert res == EFLIRAResult.SAT


def test_unsat_contradiction():
    """forall y. (y < x and y >= x) is unsatisfiable."""
    x, y = z3.Ints("x y")
    phi = z3.And(y < x, y >= x)
    res = lira_efsmt_with_parallel_cegis(
        [x],
        [y],
        phi,
        maxloops=2,
        num_samples=1,
        forall_mode=FSolverMode.SEQUENTIAL,
        num_workers=1,
    )
    assert res == EFLIRAResult.UNSAT
