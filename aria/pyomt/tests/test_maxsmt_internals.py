"""Regression tests for MaxSMT solver internals."""

import z3

from aria.pyomt.maxsmt import solve_maxsmt
from aria.pyomt.maxsmt.base import MaxSMTSolverBase


class _DummySolver(MaxSMTSolverBase):
    def solve(self):
        raise NotImplementedError


def test_variable_collection_keeps_boolean_symbols_and_skips_literals():
    solver = _DummySolver()
    x = z3.Int("x")
    b = z3.Bool("b")

    variables = solver._get_variables(z3.And(x >= 0, b))

    assert x in variables
    assert b in variables
    assert z3.IntVal(0) not in variables


def test_empty_soft_constraints_return_zero_cost_for_all_algorithms():
    x = z3.Int("x")
    hard = [x == 0]

    for algorithm in ["core-guided", "ihs", "local-search", "z3-opt"]:
        sat, model, cost = solve_maxsmt(hard, [], [], algorithm=algorithm)
        assert sat
        assert model is not None
        assert cost == 0.0
        assert model.eval(x, model_completion=True).as_long() == 0


def test_local_search_handles_boolean_soft_constraints():
    x = z3.Bool("x")

    sat, model, cost = solve_maxsmt([], [x], [1.0], algorithm="local-search")

    assert sat
    assert model is not None
    assert cost == 0.0
    assert z3.is_true(model.eval(x, model_completion=True))


def test_local_search_prefers_lower_weight_violation():
    x = z3.Int("x")
    hard = [x >= 0, x <= 2]
    soft = [x == 0, x == 2]
    weights = [1.0, 2.0]

    sat, model, cost = solve_maxsmt(
        hard, soft, weights, algorithm="local-search"
    )

    assert sat
    assert model is not None
    assert cost == 1.0
    assert model.eval(x, model_completion=True).as_long() == 2


def test_all_algorithms_agree_on_simple_weighted_instance():
    a, b = z3.Bools("a b")
    hard = [z3.Not(z3.And(a, b))]
    soft = [a, b]
    weights = [1.0, 2.0]

    for algorithm in ["core-guided", "ihs", "local-search", "z3-opt"]:
        sat, model, cost = solve_maxsmt(
            hard, soft, weights, algorithm=algorithm
        )
        assert sat
        assert model is not None
        assert cost == 1.0
        assert z3.is_false(model.eval(a, model_completion=True))
        assert z3.is_true(model.eval(b, model_completion=True))
