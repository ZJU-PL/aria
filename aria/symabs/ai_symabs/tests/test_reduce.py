from aria.symabs.ai_symabs.domains.interval import (
    Interval,
    IntervalAbstractState,
    IntervalDomain,
)
from aria.symabs.ai_symabs.domains.reduced_product import (
    ReducedProductAbstractState,
    ReducedProductDomain,
)
from aria.symabs.ai_symabs.domains.sign import Sign, SignAbstractState, SignDomain


def test_reduce_sign_interval():
    domain_A = SignDomain(["x", "y", "z"])
    input_state_A = SignAbstractState(
        {
            "x": Sign.Negative,
            "y": Sign.Positive,
            "z": Sign.Top,
        }
    )
    domain_B = IntervalDomain(["x", "y", "z"])
    input_state_B = IntervalAbstractState(
        {
            "x": Interval(-2, 3),
            "y": Interval(-5, 5),
            "z": Interval(1, 15),
        }
    )

    domain = ReducedProductDomain(["x", "y", "z"], domain_A, domain_B)
    input_state = ReducedProductAbstractState(input_state_A, input_state_B)
    reduced = domain.reduce(input_state)

    assert reduced.state_A.sign_of("x") == Sign.Negative
    assert reduced.state_A.sign_of("y") == Sign.Positive
    assert reduced.state_A.sign_of("z") == Sign.Positive

    assert reduced.state_B.interval_of("x") == Interval(-2, -1)
    assert reduced.state_B.interval_of("y") == Interval(1, 5)
    assert reduced.state_B.interval_of("z") == Interval(1, 15)
