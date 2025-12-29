"""Tests for RSY algorithm."""
import z3

from aria.symabs.ai_symabs.domains.algorithms import RSY
from aria.symabs.ai_symabs.domains.sign import Sign, SignAbstractState
from aria.symabs.ai_symabs.domains.sign import SignDomain


def test_rsy_alpha_hat_add_subtract():
    """Attempts to analyze computation of the form:

    x' := x - 5
    x'' := x + 5
    """

    domain = SignDomain(["x", "x'", "x''"])

    x = domain.z3_variable("x")
    xp = domain.z3_variable("x'")
    xpp = domain.z3_variable("x''")

    phi = z3.And(xp == x - 5, xpp == xp + 5)

    # Just from the statements themselves, we can't say anything about the sign
    # of x/x'/x''
    alpha_hat = RSY(domain, phi)
    assert alpha_hat.sign_of("x") == Sign.Top
    assert alpha_hat.sign_of("x'") == Sign.Top
    assert alpha_hat.sign_of("x''") == Sign.Top


def test_rsy_post_hat_add_subtract():
    """Here, we add the knowledge of the input state
    """
    domain = SignDomain(["x", "x'", "x''"])

    x = domain.z3_variable("x")
    xp = domain.z3_variable("x'")
    xpp = domain.z3_variable("x''")

    phi = z3.And(xp == x - 5, xpp == xp + 5)

    # This is where we add our supposition about the input
    state = SignAbstractState({"x": Sign.Positive, "x'": Sign.Top, "x''": Sign.Top})
    phi = z3.And(domain.gamma_hat(state), phi)

    post_hat = RSY(domain, phi)

    assert post_hat.sign_of("x") == Sign.Positive
    assert post_hat.sign_of("x'") == Sign.Top
    assert post_hat.sign_of("x''") == Sign.Positive


def test_rsy_alpha_hat_useful():
    """Attempts to analyze computation of the form:

    x' := ((x * x) + 1) * ((x * x) + 1)
    """

    domain = SignDomain(["x", "x'"])

    x = domain.z3_variable("x")
    xp = domain.z3_variable("x'")

    phi = z3.And(xp == ((x * x) + 1) * ((x * x) + 1))

    # Just from the statements themselves, we can say that x' is positive
    # (regardless of the value of x).
    alpha_hat = RSY(domain, phi)
    assert alpha_hat.sign_of("x") == Sign.Top
    assert alpha_hat.sign_of("x'") == Sign.Positive
