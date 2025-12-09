# TODO: This file is just a copy/paste/substitute from test_rsy.py, we should
# have a better way of dealing with the duplication here.

import z3

from aria.symabs.ai_symabs.domains.algorithms import bilateral
from aria.symabs.ai_symabs.domains.sign import Sign, SignAbstractState
from aria.symabs.ai_symabs.domains.sign import SignDomain


def test_bilateral_alpha_hat_add_subtract():
    """Attempts to analyze computation of the form:

    x' := x - 5
    x'' := x' + 5
    """

    domain = SignDomain(["x", "x'", "x''"])

    x = domain.z3_variable("x")
    xp = domain.z3_variable("x'")
    xpp = domain.z3_variable("x''")

    phi = z3.And(xp == x - 5, xpp == xp + 5)

    # Just from the statements themselves, we can't say anything about the sign
    # of x/x'/x''
    alpha_hat = bilateral(domain, phi)
    assert alpha_hat.sign_of("x") == Sign.Top
    assert alpha_hat.sign_of("x'") == Sign.Top
    assert alpha_hat.sign_of("x''") == Sign.Top


def test_bilateral_post_hat_add_subtract():
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

    post_hat = bilateral(domain, phi)

    assert post_hat.sign_of("x") == Sign.Positive
    assert post_hat.sign_of("x'") == Sign.Top
    assert post_hat.sign_of("x''") == Sign.Positive


def test_bilateral_alpha_hat_useful():
    """Attempts to analyze computation of the form:

    x' := ((x * x) + 1) * ((x * x) + 1)
    """

    domain = SignDomain(["x", "x'"])

    x = domain.z3_variable("x")
    xp = domain.z3_variable("x'")

    phi = z3.And(xp == ((x * x) + 1) * ((x * x) + 1))

    # Just from the statements themselves, we can say that x' is positive
    # (regardless of the value of x).
    alpha_hat = bilateral(domain, phi)
    assert alpha_hat.sign_of("x") == Sign.Top
    assert alpha_hat.sign_of("x'") == Sign.Positive


def test_bilateral_alphahat_bottom():
    """Attempts to analyze computation where the resulting alpha-hat is bottom

    phi := y = x*x && y < 0
    """

    domain = SignDomain(["x", "y"])

    x = domain.z3_variable("x")
    y = domain.z3_variable("y")

    phi = z3.And(y == x * x, y < 0)

    alpha_hat = bilateral(domain, phi)
    assert alpha_hat == domain.bottom


def test_bilateral_disjunction():
    """Attempts to analyze computation of the form:

    x' := x * x
    With the assumption that x > 0 or x < 0
    """

    domain = SignDomain(["x", "x'"])

    x = domain.z3_variable("x")
    xp = domain.z3_variable("x'")

    phi = z3.And(z3.Or(x > 0, x < 0), xp == x * x)

    alpha_hat = bilateral(domain, phi)
    assert alpha_hat.sign_of("x") == Sign.Top
    assert alpha_hat.sign_of("x'") == Sign.Positive
