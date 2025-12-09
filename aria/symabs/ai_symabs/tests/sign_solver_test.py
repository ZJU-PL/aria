from aria.symabs.ai_symabs.domains.sign import Sign, SignAbstractState
from aria.symabs.ai_symabs.domains.sign import SignDomain


def test_solver_constrained_satisfiable():
    domain = SignDomain(["a", "b", "c"])
    state = SignAbstractState({"a": Sign.Positive, "b": Sign.Negative, "c": Sign.Positive})
    solution = domain.model(domain.gamma_hat(state))

    assert solution is not None

    assert solution.value_of("a") > 0
    assert solution.value_of("b") < 0
    assert solution.value_of("c") > 0


def test_solver_constrained_unsatisfiable():
    domain = SignDomain(["a", "b", "c"])
    state = SignAbstractState({"a": Sign.Positive, "b": Sign.Negative, "c": Sign.Bottom})
    solution = domain.model(domain.gamma_hat(state))

    assert solution is None


def test_solver_one_unconstrained_satisfiable():
    domain = SignDomain(["a", "b", "c"])
    state = SignAbstractState({"a": Sign.Positive, "b": Sign.Negative, "c": Sign.Top})
    solution = domain.model(domain.gamma_hat(state))

    assert solution is not None

    assert solution.value_of("a") > 0
    assert solution.value_of("b") < 0
    assert isinstance(solution.value_of("c"), int)
