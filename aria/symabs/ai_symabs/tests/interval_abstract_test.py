import random

from aria.symabs.ai_symabs.domains.interval import Interval, IntervalAbstractState


def test_single_interval_comparisons():
    random_intervals = []
    for _ in range(100):
        # These bounds are chosen arbitrarily.
        lower = random.randint(-100000, +100000)
        upper = random.randint(lower, +100000)
        random_intervals.append(Interval(lower, upper))
        random_intervals.append(Interval(lower, float("inf")))
        random_intervals.append(Interval(float("-inf"), upper))

    # First, we test that Top is greater than everything else and Bottom is
    # less than everything else.
    top = Interval(float("-inf"), float("inf"))
    bottom = Interval(float("inf"), float("-inf"))
    assert bottom <= top
    for interval in random_intervals:
        assert bottom <= interval <= top

    # Next, we test that nothing else is greater than Top or less than Bottom
    for interval in random_intervals:
        assert not interval >= top
        assert not interval <= bottom

    # Non-containing intervals should be incomparable.
    assert not (Interval(5, 100) <= Interval(6, 101))
    assert not (Interval(5, 100) >= Interval(6, 101))


def test_interval_state_creation_query():
    state1 = IntervalAbstractState({
        "a": Interval(-100, 50),
        "b": Interval(float("-inf"), 5),
        "c": Interval(100, 200),
        "d": Interval(6, float("inf")),
    })

    assert state1.interval_of("a") == Interval(-100, 50)
    assert state1.interval_of("b") == Interval(float("-inf"), 5)
    assert state1.interval_of("c") == Interval(100, 200)
    assert state1.interval_of("d") == Interval(6, float("inf"))


def test_interval_state_creation_change_query():
    state1 = IntervalAbstractState({
        "a": Interval(-100, 50),
        "b": Interval(float("-inf"), 5),
        "c": Interval(100, 200),
        "d": Interval(6, float("inf")),
    })

    state1.set_interval("a", Interval(-99, 50))

    assert state1.interval_of("a") == Interval(-99, 50)
    assert state1.interval_of("b") == Interval(float("-inf"), 5)
    assert state1.interval_of("c") == Interval(100, 200)
    assert state1.interval_of("d") == Interval(6, float("inf"))


def test_interval_state_equality():
    state1 = IntervalAbstractState({
        "a": Interval(-100, 50),
        "b": Interval(float("-inf"), 5),
        "c": Interval(100, 200),
        "d": Interval(6, float("inf")),
    })
    state2 = IntervalAbstractState({
        "a": Interval(-100, 50),
        "b": Interval(float("-inf"), 5),
        "c": Interval(100, 200),
        "d": Interval(6, float("inf")),
    })
    assert state1 == state2

    state2.set_interval("a", Interval(-99, 50))
    assert state1 != state2


def test_interval_state_ineq():
    state1 = IntervalAbstractState({
        "a": Interval(-100, 50),
        "b": Interval(float("-inf"), 5),
        "c": Interval(100, 200),
        "d": Interval(float("inf"), float("-inf")),
    })
    state2 = IntervalAbstractState({
        "a": Interval(-100, 50),
        "b": Interval(float("-inf"), float("inf")),
        "c": Interval(100, 201),
        "d": Interval(6, float("inf")),
    })
    state3 = IntervalAbstractState({
        "a": Interval(-100, 50),
        "b": Interval(float("-inf"), 4),
        "c": Interval(100, 201),
        "d": Interval(7, float("inf")),
    })

    assert state1 <= state2
    assert not (state2 <= state1)
    assert not (state1 <= state3)
    assert not (state3 <= state1)
    assert not (state2 <= state3)
    assert state3 <= state2

    assert state2 >= state1
    assert not (state1 >= state2)
    assert not (state3 >= state1)
    assert not (state1 >= state3)
    assert not (state3 >= state2)
    assert state2 >= state3
