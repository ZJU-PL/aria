from aria.symabs.ai_symabs.domains.sign import Sign, SignAbstractState
from aria.symabs.ai_symabs.domains.sign import SignDomain


def test_join_bottom_top():
    domain = SignDomain(["a", "b", "c"])
    joined = domain.join([domain.bottom, domain.top])

    assert joined == domain.top


def test_join_three_states():
    domain = SignDomain(["a", "b", "c"])
    state1 = SignAbstractState({"a": Sign.Positive, "b": Sign.Negative, "c": Sign.Bottom})
    state2 = SignAbstractState({"a": Sign.Positive, "b": Sign.Positive, "c": Sign.Negative})
    state3 = SignAbstractState({"a": Sign.Positive, "b": Sign.Positive, "c": Sign.Negative})

    joined = domain.join([state1, state2, state3])

    assert joined.sign_of("a") == Sign.Positive
    assert joined.sign_of("b") == Sign.Top
    assert joined.sign_of("c") == Sign.Negative


def test_meet_three_states():
    domain = SignDomain(["a", "b", "c"])
    state1 = SignAbstractState({"a": Sign.Positive, "b": Sign.Negative, "c": Sign.Top})
    state2 = SignAbstractState({"a": Sign.Positive, "b": Sign.Positive, "c": Sign.Top})
    state3 = SignAbstractState({"a": Sign.Positive, "b": Sign.Positive, "c": Sign.Top})

    met = domain.meet([state1, state2, state3])

    assert met.sign_of("a") == Sign.Positive
    assert met.sign_of("b") == Sign.Bottom
    assert met.sign_of("c") == Sign.Top


def test_abstract_consequence_low_best():
    domain = SignDomain(["a", "b", "c"])
    lower = SignAbstractState({"a": Sign.Positive, "b": Sign.Top, "c": Sign.Top})
    upper = SignAbstractState({"a": Sign.Positive, "b": Sign.Top, "c": Sign.Top})

    consequence = domain.abstract_consequence(lower, upper)

    assert consequence == lower
