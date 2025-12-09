from aria.symabs.ai_symabs.domains.interval import Interval, IntervalAbstractState
from aria.symabs.ai_symabs.domains.interval import IntervalDomain


def test_join_bottom_top():
    domain = IntervalDomain(["a", "b", "c"])
    joined = domain.join([domain.bottom, domain.top])

    assert joined == domain.top


def test_join_meet_three_states():
    domain = IntervalDomain(["a", "b", "c"])
    state1 = IntervalAbstractState({
        "a": Interval(-5, 5),
        "b": Interval(4, 6),
        "c": Interval(float("inf"), float("-inf")),
    })
    state2 = IntervalAbstractState({
        "a": Interval(-4, 6),
        "b": Interval(float("-inf"), -2),
        "c": Interval(5, 5),
    })
    state3 = IntervalAbstractState({
        "a": Interval(-5, 5),
        "b": Interval(-4, float("inf")),
        "c": Interval(5, 5),
    })

    joined = domain.join([state1, state2, state3])

    assert joined.interval_of("a") == Interval(-5, 6)
    assert joined.interval_of("b") == Interval(float("-inf"), float("inf"))
    assert joined.interval_of("c") == Interval(5, 5)

    met = domain.meet([state1, state2, state3])

    assert met.interval_of("a") == Interval(-4, 5)
    assert met.interval_of("b") == Interval(float("inf"), float("-inf"))
    assert met.interval_of("c") == Interval(float("inf"), float("-inf"))


def test_abstract_consequence_low_best():
    domain = IntervalDomain(["a", "b", "c"])
    lower = IntervalAbstractState({
        "a": Interval(0, float("inf")),
        "b": Interval(float("-inf"), float("inf")),
        "c": Interval(float("-inf"), float("inf")),
    })
    upper = lower.copy()

    consequence = domain.abstract_consequence(lower, upper)

    assert consequence == lower
