"""Tests for the unified optimization result API."""

import z3

from aria.optimization import OptimizationStatus
from aria.optimization.maxsmt import solve_maxsmt, solve_maxsmt_result


def test_solve_maxsmt_result_exposes_normalized_result():
    x = z3.Int("x")
    result = solve_maxsmt_result(
        [x >= 0, x <= 2],
        [x == 0, x == 2],
        [1.0, 2.0],
        algorithm="z3-opt",
    )

    assert result.status == OptimizationStatus.OPTIMAL
    assert result.cost == 1.0
    assert result.model is not None
    assert result.engine == "z3-opt"


def test_solve_maxsmt_compatibility_wrapper_still_returns_tuple():
    x = z3.Int("x")
    sat, model, cost = solve_maxsmt(
        [x >= 0, x <= 2],
        [x == 0, x == 2],
        [1.0, 2.0],
        algorithm="z3-opt",
    )

    assert sat
    assert model is not None
    assert cost == 1.0


def test_maxsmt_solver_exposes_normalized_result_method():
    from aria.optimization.maxsmt import MaxSMTSolver

    x = z3.Int("x_result")
    solver = MaxSMTSolver(algorithm="z3-opt")
    solver.add_hard_constraints([x >= 0, x <= 2])
    solver.add_soft_constraints([x == 0, x == 2], [1.0, 2.0])

    result = solver.solve_result()

    assert result.status == OptimizationStatus.OPTIMAL
    assert result.cost == 1.0
    assert result.engine == "z3-opt"
    assert result.model is not None
