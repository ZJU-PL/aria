import aria.bool.nnf as nnf

from aria.bool.threeval import (
    And,
    Constant,
    GODEL_G3,
    Iff,
    Implies,
    LUKASIEWICZ_K3,
    Nand,
    NaryAnd,
    NaryOr,
    Nor,
    Not,
    Or,
    STRONG_KLEENE,
    TruthValue,
    Variable,
    WEAK_KLEENE,
    Xor,
    all_valuations,
    build_obdd_from_boolean_function,
    classical_valuations,
    conjoin,
    constructive_semantic_minimize,
    counterexample_valuations,
    direct_supervaluation_obdd_pair,
    disjoin,
    enumerate_prime_implicants,
    entails,
    evaluate,
    find_counterexample,
    find_model,
    from_nnf,
    format_formula,
    is_classically_valid,
    is_classically_valid_under_completions,
    is_consistent,
    is_contingent,
    is_equivalent,
    is_satisfiable,
    is_valid,
    is_semantically_minimal_variant,
    parse_formula,
    prime_implicant_semantic_minimize,
    satisfying_valuations,
    semantically_minimize,
    simplify,
    supervaluation,
    supervaluation_boolean_pair,
    supervaluation_obdd_pair,
    to_nnf,
    truth_table,
)


def test_not_matches_strong_kleene_truth_table():
    p = Variable("p")
    expected = {
        TruthValue.FALSE: TruthValue.TRUE,
        TruthValue.UNKNOWN: TruthValue.UNKNOWN,
        TruthValue.TRUE: TruthValue.FALSE,
    }

    for value, result in expected.items():
        assert evaluate(Not(p), {"p": value}) == result


def test_binary_connectives_match_strong_kleene_truth_tables():
    p = Variable("p")
    q = Variable("q")
    cases = {
        And(p, q): {
            (TruthValue.FALSE, TruthValue.UNKNOWN): TruthValue.FALSE,
            (TruthValue.UNKNOWN, TruthValue.TRUE): TruthValue.UNKNOWN,
            (TruthValue.TRUE, TruthValue.TRUE): TruthValue.TRUE,
        },
        Or(p, q): {
            (TruthValue.FALSE, TruthValue.UNKNOWN): TruthValue.UNKNOWN,
            (TruthValue.UNKNOWN, TruthValue.TRUE): TruthValue.TRUE,
            (TruthValue.FALSE, TruthValue.FALSE): TruthValue.FALSE,
        },
        Implies(p, q): {
            (TruthValue.TRUE, TruthValue.FALSE): TruthValue.FALSE,
            (TruthValue.UNKNOWN, TruthValue.FALSE): TruthValue.UNKNOWN,
            (TruthValue.FALSE, TruthValue.UNKNOWN): TruthValue.TRUE,
        },
        Iff(p, q): {
            (TruthValue.TRUE, TruthValue.TRUE): TruthValue.TRUE,
            (TruthValue.FALSE, TruthValue.TRUE): TruthValue.FALSE,
            (TruthValue.UNKNOWN, TruthValue.TRUE): TruthValue.UNKNOWN,
        },
        Xor(p, q): {
            (TruthValue.TRUE, TruthValue.FALSE): TruthValue.TRUE,
            (TruthValue.TRUE, TruthValue.TRUE): TruthValue.FALSE,
            (TruthValue.UNKNOWN, TruthValue.TRUE): TruthValue.UNKNOWN,
        },
        Nand(p, q): {
            (TruthValue.TRUE, TruthValue.TRUE): TruthValue.FALSE,
            (TruthValue.FALSE, TruthValue.TRUE): TruthValue.TRUE,
            (TruthValue.UNKNOWN, TruthValue.TRUE): TruthValue.UNKNOWN,
        },
        Nor(p, q): {
            (TruthValue.FALSE, TruthValue.FALSE): TruthValue.TRUE,
            (TruthValue.TRUE, TruthValue.FALSE): TruthValue.FALSE,
            (TruthValue.UNKNOWN, TruthValue.FALSE): TruthValue.UNKNOWN,
        },
    }

    for formula, expected in cases.items():
        for inputs, result in expected.items():
            left, right = inputs
            assert evaluate(formula, {"p": left, "q": right}) == result


def test_weak_kleene_differs_from_strong_kleene_on_infectious_unknowns():
    p = Variable("p")
    q = Variable("q")
    valuation = {"p": TruthValue.FALSE, "q": TruthValue.UNKNOWN}

    assert evaluate(And(p, q), valuation, STRONG_KLEENE) == TruthValue.FALSE
    assert evaluate(And(p, q), valuation, WEAK_KLEENE) == TruthValue.UNKNOWN


def test_lukasiewicz_and_godel_implications_are_available():
    p = Variable("p")
    q = Variable("q")
    valuation = {"p": TruthValue.UNKNOWN, "q": TruthValue.FALSE}

    assert evaluate(Implies(p, q), valuation, LUKASIEWICZ_K3) == TruthValue.UNKNOWN
    assert evaluate(Implies(p, q), valuation, GODEL_G3) == TruthValue.FALSE


def test_all_valuations_and_classical_valuations_are_enumerated_in_sorted_order():
    three_valued = list(all_valuations(["q", "p"]))
    classical = list(classical_valuations(["q", "p"]))

    assert len(three_valued) == 9
    assert three_valued[0] == {"p": TruthValue.FALSE, "q": TruthValue.FALSE}
    assert three_valued[-1] == {"p": TruthValue.TRUE, "q": TruthValue.TRUE}
    assert classical == [
        {"p": TruthValue.FALSE, "q": TruthValue.FALSE},
        {"p": TruthValue.FALSE, "q": TruthValue.TRUE},
        {"p": TruthValue.TRUE, "q": TruthValue.FALSE},
        {"p": TruthValue.TRUE, "q": TruthValue.TRUE},
    ]


def test_truth_table_and_model_search_utilities_expose_witnesses():
    p = Variable("p")
    table = truth_table(Or(p, Not(p)))

    assert len(table) == 3
    assert table[1][0] == {"p": TruthValue.UNKNOWN}
    assert table[1][1] == TruthValue.UNKNOWN
    assert find_model(p) == {"p": TruthValue.TRUE}
    assert find_counterexample(p) == {"p": TruthValue.FALSE}


def test_satisfiability_counterexamples_and_consistency_checks_work():
    p = Variable("p")
    satisfiable = And(p, Constant(TruthValue.TRUE))

    assert is_satisfiable(satisfiable)
    assert is_contingent(satisfiable)
    assert counterexample_valuations(Or(p, Not(p)))[0] == {"p": TruthValue.UNKNOWN}
    assert satisfying_valuations(And(p, Constant(TruthValue.TRUE))) == [
        {"p": TruthValue.TRUE}
    ]
    assert not is_consistent([p, Not(p)])


def test_validity_and_entailment_support_custom_designated_values():
    p = Variable("p")
    tolerant_kleene = STRONG_KLEENE.with_designated_values(
        {TruthValue.TRUE, TruthValue.UNKNOWN}
    )

    assert not is_valid(Or(p, Not(p)))
    assert is_valid(Or(p, Not(p)), semantics=tolerant_kleene)
    assert entails([And(p, Variable("q"))], p)


def test_classical_validity_helpers_use_boolean_completions_only():
    p = Variable("p")

    assert not is_valid(Or(p, Not(p)))
    assert is_classically_valid(Or(p, Not(p)))
    assert is_classically_valid_under_completions(Or(p, Not(p)), {"p": None})


def test_formula_structure_substitution_and_simplification_helpers_work():
    p = Variable("p")
    q = Variable("q")
    formula = Implies(And(p, Constant(TruthValue.TRUE)), Or(q, Constant(TruthValue.FALSE)))
    substituted = formula.substitute({"q": p})
    simplified = simplify(formula, {"q": False})

    assert substituted.variables() == {"p"}
    assert formula.size() == 7
    assert formula.depth() == 3
    assert simplify(And(Constant(TruthValue.TRUE), p)) == p
    assert simplified == Not(p)
    assert format_formula(simplified) == "!p"


def test_nary_helpers_preserve_neutral_elements_and_simplify():
    p = Variable("p")
    q = Variable("q")

    assert conjoin() == Constant(TruthValue.TRUE)
    assert disjoin() == Constant(TruthValue.FALSE)
    assert isinstance(conjoin(p, q), NaryAnd)
    assert isinstance(disjoin(p, q), NaryOr)
    assert simplify(conjoin(Constant(TruthValue.TRUE), p, q)) == NaryAnd((p, q))
    assert simplify(disjoin(Constant(TruthValue.FALSE), p, q)) == NaryOr((p, q))


def test_parser_and_pretty_printer_handle_extended_connectives():
    formula = parse_formula("!(p & q) -> (r xor unknown)")

    assert format_formula(formula) == "!(p & q) -> r xor unknown"
    assert format_formula(formula, unicode=True) == "¬(p ∧ q) → r ⊕ ?"
    assert evaluate(
        formula,
        {"p": True, "q": False, "r": False},
    ) == TruthValue.UNKNOWN


def test_equivalence_tracks_semantic_not_just_designated_agreement():
    p = Variable("p")

    assert is_equivalent(Implies(p, p), Constant(TruthValue.TRUE)) is False
    assert is_equivalent(Iff(p, p), Constant(TruthValue.TRUE)) is False
    assert is_equivalent(And(Constant(TruthValue.TRUE), p), p)


def test_nnf_adapter_round_trips_boolean_structure():
    sentence = nnf.And(
        {
            nnf.Var("p"),
            nnf.Or({nnf.Var("q", False), nnf.Var("r")}),
        }
    )

    converted = from_nnf(sentence)

    assert is_equivalent(converted, And(Variable("p"), Or(Not(Variable("q")), Variable("r"))))
    round_tripped = to_nnf(converted)
    assert round_tripped.equivalent(sentence)


def test_to_nnf_handles_unknown_policy_explicitly():
    formula = Or(Variable("p"), Constant(TruthValue.UNKNOWN))

    false_mapped = to_nnf(formula, unknown_policy="false")
    true_mapped = to_nnf(formula, unknown_policy="true")

    assert false_mapped.equivalent(nnf.Var("p"))
    assert true_mapped == nnf.true


def test_supervaluation_improves_excluded_middle_under_unknown():
    p = Variable("p")
    formula = Or(p, Not(p))

    assert evaluate(formula, {"p": TruthValue.UNKNOWN}) == TruthValue.UNKNOWN
    assert supervaluation(formula, {"p": TruthValue.UNKNOWN}) == TruthValue.TRUE


def test_supervaluation_boolean_pair_tracks_true_and_nonfalse_regions():
    p = Variable("p")
    formula = Constant(TruthValue.UNKNOWN)
    sv_true, sv_nonfalse = supervaluation_boolean_pair(formula)

    assert all(value is False for value in sv_true.values())
    assert all(value is True for value in sv_nonfalse.values())

    sv_true_em, sv_nonfalse_em = supervaluation_boolean_pair(Or(p, Not(p)))
    assert all(value is True for value in sv_true_em.values())
    assert all(value is True for value in sv_nonfalse_em.values())


def test_bdd_prime_implicant_pipeline_reduces_paths_to_primes():
    obdd = build_obdd_from_boolean_function(
        ["x", "y"],
        lambda valuation: valuation["x"] or valuation["y"],
    )

    primes = enumerate_prime_implicants(obdd)

    assert primes == [(("x", True), ("y", None)), (("x", None), ("y", True))]


def test_supervaluation_obdd_pair_matches_mapping_variant():
    p = Variable("p")
    mapping_true, mapping_nonfalse = supervaluation_boolean_pair(Or(p, Not(p)))
    obdd_true, obdd_nonfalse = supervaluation_obdd_pair(Or(p, Not(p)))

    for valuation in classical_valuations(["p"]):
        bit = {name: value is TruthValue.TRUE for name, value in valuation.items()}
        key = tuple(sorted(bit.items()))
        assert mapping_true[key] is (obdd_true.restrict("p", bit["p"]) == 1)
        assert mapping_nonfalse[key] is (obdd_nonfalse.restrict("p", bit["p"]) == 1)


def test_direct_supervaluation_obdd_translation_handles_unknown_constants():
    p = Variable("p")
    formula = And(Or(p, Constant(TruthValue.UNKNOWN)), Not(Constant(TruthValue.FALSE)))

    direct_pair = direct_supervaluation_obdd_pair(formula)
    public_true, public_nonfalse = supervaluation_obdd_pair(formula)

    for valuation in classical_valuations(["p"]):
        bit = valuation["p"] is TruthValue.TRUE
        assert (direct_pair.is_true.restrict("p", bit) == 1) is (
            public_true.restrict("p", bit) == 1
        )
        assert (direct_pair.is_nonfalse.restrict("p", bit) == 1) is (
            public_nonfalse.restrict("p", bit) == 1
        )


def test_semantic_minimization_algorithms_match_supervaluation_definition():
    p = Variable("p")
    formula = Or(p, Not(p))

    constructive = constructive_semantic_minimize(formula)
    primes = prime_implicant_semantic_minimize(formula)

    assert is_semantically_minimal_variant(formula, constructive)
    assert is_semantically_minimal_variant(formula, primes)
    assert is_equivalent(
        semantically_minimize(formula, method="constructive"), constructive
    )
    assert is_equivalent(semantically_minimize(formula), primes)


def test_semantic_minimization_recovers_precision_gain_on_excluded_middle():
    p = Variable("p")
    formula = Or(p, Not(p))
    minimized = semantically_minimize(formula)

    valuation = {"p": None}

    assert evaluate(formula, valuation) == TruthValue.UNKNOWN
    assert supervaluation(formula, valuation) == TruthValue.TRUE
    assert evaluate(minimized, valuation) == TruthValue.TRUE


def test_semantic_minimization_handles_explicit_unknown_constant():
    p = Variable("p")
    formula = Or(p, Constant(TruthValue.UNKNOWN))
    minimized = semantically_minimize(formula)

    assert is_semantically_minimal_variant(formula, minimized)
    assert evaluate(minimized, {"p": False}) == TruthValue.UNKNOWN
    assert evaluate(minimized, {"p": True}) == TruthValue.TRUE
