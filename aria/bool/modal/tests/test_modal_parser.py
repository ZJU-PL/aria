import pytest

from aria.bool.modal import (
    And,
    Atom,
    Box,
    Constant,
    Diamond,
    Formula,
    Implies,
    KripkeModel,
    ModalSyntaxError,
    Not,
    Or,
    eliminate_implications,
    parse_formula,
    satisfies,
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


def test_eliminate_implications_removes_implication_nodes() -> None:
    formula = parse_formula("p -> ([]q -> <>r)")

    assert eliminate_implications(formula) == Or(
        Not(Atom("p")),
        Or(Not(Box(Atom("q"))), Diamond(Atom("r"))),
    )


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
