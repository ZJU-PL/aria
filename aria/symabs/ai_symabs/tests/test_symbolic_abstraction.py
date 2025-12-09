import z3

from aria.symabs.ai_symabs.domains.algorithms import bilateral, RSY
from aria.symabs.ai_symabs.domains.interval import Interval
from aria.symabs.ai_symabs.domains.interval import IntervalDomain


def test_bilateral_alpha_hat_add_subtract():
    """Attempts to analyze computation of the form:

    x' := x - 5
    x'' := x' + 5
    """
    domain = IntervalDomain(["x", "x'", "x''"])

    x = domain.z3_variable("x")
    xp = domain.z3_variable("x'")
    xpp = domain.z3_variable("x''")

    # Bounds needed to avoid infinite ascending chains (in practice we should
    # use, eg., widening).
    phi = z3.And(xp == x - 5, xpp == xp + 5, x <= 5, x >= -5)

    # Just from the statements themselves, we can't say anything about the sign
    # of x/x'/x''
    alpha_hat = bilateral(domain, phi)
    assert alpha_hat == RSY(domain, phi)
    assert alpha_hat.interval_of("x") == Interval(-5, 5)
    assert alpha_hat.interval_of("x'") == Interval(-10, 0)
    assert alpha_hat.interval_of("x''") == Interval(-5, 5)
