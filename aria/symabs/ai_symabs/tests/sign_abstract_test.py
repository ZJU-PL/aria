"""Tests for sign abstract domain."""

from aria.symabs.ai_symabs.domains.sign import Sign, SignAbstractState


def test_sign_enum_comparisons():
    """Test sign enum comparison operations."""
    # First, we test that Top is greater than everything and Bottom is less
    # than everything
    for sign in Sign:
        assert Sign.Bottom <= sign <= Sign.Top

    # Next, we test that nothing else is greater than Top or less than Bottom
    for sign in Sign:
        if sign != Sign.Top:
            assert sign < Sign.Top
        if sign != Sign.Bottom:
            assert sign > Sign.Bottom

    # Positive and Negative should be uncomparable
    assert Sign.Negative > Sign.Positive
    assert Sign.Positive > Sign.Negative
    assert Sign.Negative < Sign.Positive
    assert Sign.Positive < Sign.Negative


def test_sign_state_creation_query():
    state1 = SignAbstractState(
        {"a": Sign.Positive, "b": Sign.Negative, "c": Sign.Top, "d": Sign.Bottom}
    )

    assert state1.sign_of("a") == Sign.Positive
    assert state1.sign_of("b") == Sign.Negative
    assert state1.sign_of("c") == Sign.Top
    assert state1.sign_of("d") == Sign.Bottom


def test_sign_state_creation_change_query():
    state1 = SignAbstractState(
        {"a": Sign.Positive, "b": Sign.Negative, "c": Sign.Top, "d": Sign.Bottom}
    )

    state1.set_sign("a", Sign.Bottom)

    assert state1.sign_of("a") == Sign.Bottom
    assert state1.sign_of("b") == Sign.Negative
    assert state1.sign_of("c") == Sign.Top
    assert state1.sign_of("d") == Sign.Bottom


def test_sign_state_equality():
    state1 = SignAbstractState(
        {"a": Sign.Positive, "b": Sign.Negative, "c": Sign.Top, "d": Sign.Bottom}
    )
    state2 = SignAbstractState(
        {"a": Sign.Positive, "b": Sign.Negative, "c": Sign.Top, "d": Sign.Bottom}
    )

    assert state1 == state2


def test_sign_state_leq():
    state1 = SignAbstractState(
        {"a": Sign.Positive, "b": Sign.Negative, "c": Sign.Top, "d": Sign.Bottom}
    )
    state2 = SignAbstractState(
        {"a": Sign.Positive, "b": Sign.Top, "c": Sign.Top, "d": Sign.Positive}
    )
    state3 = SignAbstractState(
        {"a": Sign.Positive, "b": Sign.Positive, "c": Sign.Top, "d": Sign.Bottom}
    )

    assert state1 <= state2
    assert state2 > state1
    assert state1 > state3
    assert state3 > state1
    assert state2 > state3
    assert state3 <= state2


def test_sign_state_geq():
    state1 = SignAbstractState(
        {"a": Sign.Positive, "b": Sign.Negative, "c": Sign.Top, "d": Sign.Bottom}
    )
    state2 = SignAbstractState(
        {"a": Sign.Positive, "b": Sign.Top, "c": Sign.Top, "d": Sign.Positive}
    )
    state3 = SignAbstractState(
        {"a": Sign.Positive, "b": Sign.Positive, "c": Sign.Top, "d": Sign.Bottom}
    )

    assert state2 >= state1
    assert state1 < state2
    assert state3 < state1
    assert state1 < state3
    assert state3 < state2
    assert state2 >= state3
