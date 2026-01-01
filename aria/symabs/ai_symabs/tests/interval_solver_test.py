"""Tests for interval domain solver."""

from aria.symabs.ai_symabs.domains.interval import Interval, IntervalAbstractState
from aria.symabs.ai_symabs.domains.interval import IntervalDomain


def test_solver_constrained_satisfiable():
    """Test solver with constrained satisfiable state."""
    domain = IntervalDomain(["a", "b", "c"])
    state = IntervalAbstractState(
        {
            "a": Interval(0, 100),
            "b": Interval(-50, -50),
            "c": Interval(1, 11),
        }
    )
    solution = domain.model(domain.gamma_hat(state))

    assert solution is not None

    assert 0 <= solution.value_of("a") <= 100
    assert solution.value_of("b") == -50
    assert 1 <= solution.value_of("c") <= 11


def test_solver_constrained_unsatisfiable():
    """Test solver with constrained unsatisfiable state."""
    domain = IntervalDomain(["a", "b", "c"])
    state = IntervalAbstractState(
        {
            "a": Interval(0, 100),
            "b": Interval(float("inf"), float("-inf")),
            "c": Interval(1, 11),
        }
    )
    solution = domain.model(domain.gamma_hat(state))

    assert solution is None


def test_solver_one_unconstrained_satisfiable():
    """Test solver with one unconstrained variable."""
    domain = IntervalDomain(["a", "b", "c"])
    state = IntervalAbstractState(
        {
            "a": Interval(0, 100),
            "b": Interval(-50, -50),
            "c": Interval(float("-inf"), float("inf")),
        }
    )
    solution = domain.model(domain.gamma_hat(state))

    assert solution is not None

    assert 0 <= solution.value_of("a") <= 100
    assert solution.value_of("b") == -50
    assert isinstance(solution.value_of("c"), int)
