import pytest

from aria.bool.modal import (
    And,
    Atom,
    Box,
    Constant,
    Diamond,
    format_formula,
    Formula,
    Iff,
    Implies,
    KripkeModel,
    ModalSyntaxError,
    Not,
    Or,
    eliminate_implications,
    parse_formula,
    satisfies,
    simplify,
    to_nnf,
)


def make_semantics_model() -> KripkeModel:
    return KripkeModel(
        worlds=frozenset({"root", "left", "right", "leaf"}),
        relation=frozenset(
            {
            ("root", "left"),
            ("root", "right"),
            ("left", "leaf"),
            ("right", "right"),
            }
        ),
        valuation={
            "p": frozenset({"root", "leaf"}),
            "q": frozenset({"left", "right"}),
        },
    )


def test_parse_formula_respects_operator_precedence() -> None:
    assert parse_formula("p & q | r -> s") == Implies(
        Or(And(Atom("p"), Atom("q")), Atom("r")),
        Atom("s"),
    )


def test_parse_formula_uses_parentheses_and_modal_prefixes() -> None:
    assert parse_formula("[] (p -> <> !q)") == Box(
        Implies(Atom("p"), Diamond(Not(Atom("q"))))
    )


def test_parse_formula_supports_constants_and_unicode_aliases() -> None:
    assert parse_formula("□(p → ◇⊤)") == Box(
        Implies(Atom("p"), Diamond(Constant(True)))
    )
    assert parse_formula("⊥") == Constant(False)


def test_parse_formula_supports_biconditionals() -> None:
    assert parse_formula("p <-> q -> r") == Iff(
        Atom("p"), Implies(Atom("q"), Atom("r"))
    )
    assert parse_formula("p ↔ q") == Iff(Atom("p"), Atom("q"))


def test_format_formula_round_trips_in_ascii_and_unicode() -> None:
    formula = Box(Implies(Atom("p"), Diamond(Not(Atom("q")))))

    assert format_formula(formula) == "[](p -> <>!q)"
    assert parse_formula(format_formula(formula)) == formula
    assert format_formula(formula, unicode=True) == "□(p → ◇¬q)"
    assert parse_formula(format_formula(formula, unicode=True)) == formula


def test_format_formula_round_trips_biconditionals() -> None:
    formula = Iff(Atom("p"), Implies(Atom("q"), Diamond(Atom("r"))))

    assert format_formula(formula) == "p <-> q -> <>r"
    assert parse_formula(format_formula(formula)) == formula


def test_format_formula_preserves_right_nested_conjunctions() -> None:
    formula = And(Atom("p"), And(Atom("q"), Atom("r")))

    assert format_formula(formula) == "p & (q & r)"
    assert parse_formula(format_formula(formula)) == formula


def test_format_formula_preserves_right_nested_disjunctions() -> None:
    formula = Or(Atom("p"), Or(Atom("q"), Atom("r")))

    assert format_formula(formula) == "p | (q | r)"
    assert parse_formula(format_formula(formula)) == formula


def test_format_formula_preserves_left_nested_implications() -> None:
    formula = Implies(Implies(Atom("p"), Atom("q")), Atom("r"))

    assert format_formula(formula) == "(p -> q) -> r"
    assert parse_formula(format_formula(formula)) == formula


def test_format_formula_preserves_implication_over_biconditional() -> None:
    formula = Implies(Atom("p"), Iff(Atom("q"), Atom("r")))

    assert format_formula(formula) == "p -> (q <-> r)"
    assert parse_formula(format_formula(formula)) == formula


def test_format_formula_preserves_right_nested_biconditionals() -> None:
    formula = Iff(Atom("p"), Iff(Atom("q"), Atom("r")))

    assert format_formula(formula) == "p <-> (q <-> r)"
    assert parse_formula(format_formula(formula)) == formula


@pytest.mark.parametrize(
    "text",
    [
        "",
        "p &",
        "(p | q",
        "p q",
        "[] )",
        "p $ q",
    ],
)
def test_parse_formula_rejects_malformed_input(text: str) -> None:
    with pytest.raises(ModalSyntaxError):
        _ = parse_formula(text)


def test_atom_names_reject_reserved_constant_keywords() -> None:
    with pytest.raises(ValueError):
        Atom("true")
    with pytest.raises(ValueError):
        Atom("false")


def test_eliminate_implications_removes_implication_nodes() -> None:
    formula = parse_formula("p -> ([]q -> <>r)")

    assert eliminate_implications(formula) == Or(
        Not(Atom("p")),
        Or(Not(Box(Atom("q"))), Diamond(Atom("r"))),
    )


def test_eliminate_implications_removes_biconditional_nodes() -> None:
    formula = parse_formula("p <-> q")

    assert eliminate_implications(formula) == And(
        Or(Not(Atom("p")), Atom("q")),
        Or(Not(Atom("q")), Atom("p")),
    )


def test_simplify_applies_boolean_and_modal_identities() -> None:
    assert simplify(parse_formula("(p & true) <-> !!p")) == Constant(True)
    assert simplify(parse_formula("[]true")) == Constant(True)
    assert simplify(parse_formula("<>false")) == Constant(False)


def test_to_nnf_pushes_negations_through_modal_operators() -> None:
    formula = parse_formula("!([](p | !q) -> <>!p)")

    assert to_nnf(formula) == And(
        Box(Or(Atom("p"), Not(Atom("q")))),
        Box(Atom("p")),
    )


def test_to_nnf_preserves_semantics_on_existing_evaluator() -> None:
    model = make_semantics_model()
    formula = parse_formula("!(<>(p -> q) | []!p)")
    normalized = to_nnf(formula)

    for world in model.worlds:
        assert satisfies(model, world, normalized) == satisfies(model, world, formula)


def test_to_nnf_eliminates_implications_and_internal_negations() -> None:
    normalized = to_nnf(parse_formula("!(<>(p -> q) -> []!(r | s))"))

    def assert_modal_nnf(formula: Formula) -> None:
        if isinstance(formula, (Atom, Constant)):
            return
        if isinstance(formula, Not):
            assert isinstance(formula.operand, (Atom, Constant))
            return
        if isinstance(formula, (And, Or)):
            assert_modal_nnf(formula.left)
            assert_modal_nnf(formula.right)
            return
        if isinstance(formula, (Box, Diamond)):
            assert_modal_nnf(formula.operand)
            return
        raise AssertionError(f"Expected implication-free NNF, got {formula!r}")

    assert_modal_nnf(normalized)
