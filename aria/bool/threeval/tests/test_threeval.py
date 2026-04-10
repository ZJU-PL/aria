from aria.bool.threeval import (
    And,
    Constant,
    Implies,
    Not,
    Or,
    TruthValue,
    Variable,
    entails,
    evaluate,
    is_valid,
    all_valuations,
)


ALL_VALUES = [TruthValue.FALSE, TruthValue.UNKNOWN, TruthValue.TRUE]


def test_not_matches_strong_kleene_truth_table():
    p = Variable("p")
    expected = {
        TruthValue.FALSE: TruthValue.TRUE,
        TruthValue.UNKNOWN: TruthValue.UNKNOWN,
        TruthValue.TRUE: TruthValue.FALSE,
    }

    for value, result in expected.items():
        assert evaluate(Not(p), {"p": value}) == result


def test_and_matches_strong_kleene_truth_table():
    p = Variable("p")
    q = Variable("q")
    expected = {
        (TruthValue.FALSE, TruthValue.FALSE): TruthValue.FALSE,
        (TruthValue.FALSE, TruthValue.UNKNOWN): TruthValue.FALSE,
        (TruthValue.FALSE, TruthValue.TRUE): TruthValue.FALSE,
        (TruthValue.UNKNOWN, TruthValue.FALSE): TruthValue.FALSE,
        (TruthValue.UNKNOWN, TruthValue.UNKNOWN): TruthValue.UNKNOWN,
        (TruthValue.UNKNOWN, TruthValue.TRUE): TruthValue.UNKNOWN,
        (TruthValue.TRUE, TruthValue.FALSE): TruthValue.FALSE,
        (TruthValue.TRUE, TruthValue.UNKNOWN): TruthValue.UNKNOWN,
        (TruthValue.TRUE, TruthValue.TRUE): TruthValue.TRUE,
    }

    for inputs, result in expected.items():
        left, right = inputs
        assert evaluate(And(p, q), {"p": left, "q": right}) == result


def test_or_matches_strong_kleene_truth_table():
    p = Variable("p")
    q = Variable("q")
    expected = {
        (TruthValue.FALSE, TruthValue.FALSE): TruthValue.FALSE,
        (TruthValue.FALSE, TruthValue.UNKNOWN): TruthValue.UNKNOWN,
        (TruthValue.FALSE, TruthValue.TRUE): TruthValue.TRUE,
        (TruthValue.UNKNOWN, TruthValue.FALSE): TruthValue.UNKNOWN,
        (TruthValue.UNKNOWN, TruthValue.UNKNOWN): TruthValue.UNKNOWN,
        (TruthValue.UNKNOWN, TruthValue.TRUE): TruthValue.TRUE,
        (TruthValue.TRUE, TruthValue.FALSE): TruthValue.TRUE,
        (TruthValue.TRUE, TruthValue.UNKNOWN): TruthValue.TRUE,
        (TruthValue.TRUE, TruthValue.TRUE): TruthValue.TRUE,
    }

    for inputs, result in expected.items():
        left, right = inputs
        assert evaluate(Or(p, q), {"p": left, "q": right}) == result


def test_implies_matches_strong_kleene_truth_table():
    p = Variable("p")
    q = Variable("q")
    expected = {
        (TruthValue.FALSE, TruthValue.FALSE): TruthValue.TRUE,
        (TruthValue.FALSE, TruthValue.UNKNOWN): TruthValue.TRUE,
        (TruthValue.FALSE, TruthValue.TRUE): TruthValue.TRUE,
        (TruthValue.UNKNOWN, TruthValue.FALSE): TruthValue.UNKNOWN,
        (TruthValue.UNKNOWN, TruthValue.UNKNOWN): TruthValue.UNKNOWN,
        (TruthValue.UNKNOWN, TruthValue.TRUE): TruthValue.TRUE,
        (TruthValue.TRUE, TruthValue.FALSE): TruthValue.FALSE,
        (TruthValue.TRUE, TruthValue.UNKNOWN): TruthValue.UNKNOWN,
        (TruthValue.TRUE, TruthValue.TRUE): TruthValue.TRUE,
    }

    for inputs, result in expected.items():
        left, right = inputs
        assert evaluate(Implies(p, q), {"p": left, "q": right}) == result


def test_variable_evaluation_defaults_to_unknown_under_partial_valuation():
    p = Variable("p")

    assert evaluate(p, {}) == TruthValue.UNKNOWN
    assert evaluate(p, {"p": True}) == TruthValue.TRUE
    assert evaluate(p, {"p": False}) == TruthValue.FALSE
    assert evaluate(p, {"p": None}) == TruthValue.UNKNOWN


def test_kleene_connectives_handle_unknown_values():
    p = Variable("p")
    q = Variable("q")

    assert evaluate(Not(p), {"p": None}) == TruthValue.UNKNOWN
    assert evaluate(And(p, q), {"p": True, "q": None}) == TruthValue.UNKNOWN
    assert evaluate(And(p, q), {"p": False, "q": None}) == TruthValue.FALSE
    assert evaluate(Or(p, q), {"p": False, "q": None}) == TruthValue.UNKNOWN
    assert evaluate(Or(p, q), {"p": True, "q": None}) == TruthValue.TRUE


def test_all_valuations_enumerates_every_three_valued_assignment():
    valuations = list(all_valuations(["q", "p"]))

    assert len(valuations) == 9
    assert valuations[0] == {"p": TruthValue.FALSE, "q": TruthValue.FALSE}
    assert valuations[-1] == {"p": TruthValue.TRUE, "q": TruthValue.TRUE}


def test_unknown_sensitive_entailment_and_validity_behavior():
    p = Variable("p")

    assert not entails([], p)
    assert is_valid(Constant(TruthValue.TRUE))
    assert not is_valid(Constant(TruthValue.UNKNOWN))


def test_implication_uses_three_valued_semantics():
    p = Variable("p")
    q = Variable("q")

    assert evaluate(Implies(p, q), {"p": True, "q": None}) == TruthValue.UNKNOWN
    assert evaluate(Implies(p, q), {"p": False, "q": None}) == TruthValue.TRUE
    assert evaluate(Implies(p, q), {"p": None, "q": False}) == TruthValue.UNKNOWN


def test_entailment_requires_true_conclusion_when_premises_are_true():
    p = Variable("p")
    q = Variable("q")

    assert entails([And(p, q)], p)
    assert not entails([Or(p, q)], p)


def test_validity_checks_all_three_valued_assignments():
    p = Variable("p")

    assert is_valid(Or(p, Not(p))) is False
    assert is_valid(Or(p, Constant(TruthValue.TRUE))) is True
