import importlib
from typing import Any, Dict

import pytest


EXPECTED_API = (
    "Formula",
    "Constant",
    "Atom",
    "Not",
    "And",
    "Or",
    "Implies",
    "Iff",
    "Box",
    "Diamond",
    "FrameLogic",
    "KripkeModel",
    "CountermodelWitness",
    "ModelWitness",
    "formula_size",
    "modal_depth",
    "subformulas",
    "format_formula",
    "simplify",
    "satisfies",
    "is_valid",
    "entails",
    "validate_frame",
    "find_model",
    "find_countermodel",
    "find_entailment_countermodel",
)


@pytest.fixture(scope="module")
def modal_api() -> Dict[str, Any]:
    modal = importlib.import_module("aria.bool.modal")
    missing = [name for name in EXPECTED_API if not hasattr(modal, name)]
    if missing:
        pytest.fail(f"aria.bool.modal is missing expected API names: {missing}")
    return {name: getattr(modal, name) for name in EXPECTED_API}


def make_dead_end_model(modal_api: Dict[str, Any]) -> Any:
    return modal_api["KripkeModel"](
        worlds={"w"},
        relation=set(),
        valuation={"p": {"w"}},
    )


def make_branching_model(modal_api: Dict[str, Any]) -> Any:
    return modal_api["KripkeModel"](
        worlds={"root", "left", "right"},
        relation={("root", "left"), ("root", "right")},
        valuation={"p": {"left"}, "q": {"left", "right"}},
    )


def make_nested_model(modal_api: Dict[str, Any]) -> Any:
    return modal_api["KripkeModel"](
        worlds={"root", "mid", "leaf_good", "leaf_bad"},
        relation={
            ("root", "mid"),
            ("mid", "leaf_good"),
            ("mid", "leaf_bad"),
        },
        valuation={"p": {"leaf_good"}},
    )


def make_uniform_successor_model(modal_api: Dict[str, Any]) -> Any:
    return modal_api["KripkeModel"](
        worlds={"root", "left", "right"},
        relation={("root", "left"), ("root", "right")},
        valuation={"p": {"left", "right"}, "q": {"left", "right"}},
    )


def make_t_model(modal_api: Dict[str, Any]) -> Any:
    return modal_api["KripkeModel"](
        worlds={"w"},
        relation={("w", "w")},
        valuation={"p": set()},
    )


def make_serial_not_reflexive_model(modal_api: Dict[str, Any]) -> Any:
    return modal_api["KripkeModel"](
        worlds={"a", "b"},
        relation={("a", "b"), ("b", "a")},
        valuation={},
    )


def make_b_model(modal_api: Dict[str, Any]) -> Any:
    return modal_api["KripkeModel"](
        worlds={"a", "b"},
        relation={("a", "a"), ("b", "b"), ("a", "b"), ("b", "a")},
        valuation={},
    )


def make_transitive_non_reflexive_model(modal_api: Dict[str, Any]) -> Any:
    return modal_api["KripkeModel"](
        worlds={"a", "b"},
        relation={("a", "b")},
        valuation={},
    )


def make_reflexive_non_transitive_model(modal_api: Dict[str, Any]) -> Any:
    return modal_api["KripkeModel"](
        worlds={"a", "b", "c"},
        relation={
            ("a", "a"),
            ("b", "b"),
            ("c", "c"),
            ("a", "b"),
            ("b", "c"),
        },
        valuation={},
    )


def make_s4_not_s5_model(modal_api: Dict[str, Any]) -> Any:
    return modal_api["KripkeModel"](
        worlds={"a", "b"},
        relation={("a", "a"), ("b", "b"), ("a", "b")},
        valuation={},
    )


def make_s5_model(modal_api: Dict[str, Any]) -> Any:
    return modal_api["KripkeModel"](
        worlds={"a", "b"},
        relation={("a", "a"), ("b", "b"), ("a", "b"), ("b", "a")},
        valuation={},
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "worlds": set(),
            "relation": set(),
            "valuation": {},
        },
        {
            "worlds": {"w"},
            "relation": {("w", "ghost")},
            "valuation": {},
        },
        {
            "worlds": {"w"},
            "relation": set(),
            "valuation": {"p": {"ghost"}},
        },
    ],
)
def test_malformed_models_raise_value_error(
    modal_api: Dict[str, Any], kwargs: Dict[str, Any]
):
    with pytest.raises(ValueError):
        modal_api["KripkeModel"](**kwargs)


def test_formula_nodes_share_base_type_and_support_local_boolean_semantics(
    modal_api: Dict[str, Any],
):
    model = make_dead_end_model(modal_api)
    formula = modal_api["Formula"]
    constant = modal_api["Constant"]
    atom = modal_api["Atom"]
    not_ = modal_api["Not"]
    and_ = modal_api["And"]
    or_ = modal_api["Or"]
    implies = modal_api["Implies"]
    iff = modal_api["Iff"]
    box = modal_api["Box"]
    diamond = modal_api["Diamond"]
    satisfies = modal_api["satisfies"]
    p = atom("p")
    q = atom("q")

    assert isinstance(constant(True), formula)
    assert isinstance(box(p), formula)
    assert isinstance(diamond(p), formula)

    assert satisfies(model, "w", p)
    assert not satisfies(model, "w", q)
    assert not satisfies(model, "w", not_(p))
    assert satisfies(model, "w", and_(p, constant(True)))
    assert satisfies(model, "w", or_(q, p))
    assert not satisfies(model, "w", implies(p, q))
    assert satisfies(model, "w", implies(q, p))
    assert satisfies(model, "w", iff(p, constant(True)))
    assert not satisfies(model, "w", iff(p, q))


def test_bad_formula_operands_raise_type_error(modal_api: Dict[str, Any]):
    not_ = modal_api["Not"]
    and_ = modal_api["And"]
    atom = modal_api["Atom"]

    with pytest.raises(TypeError):
        not_("p")
    with pytest.raises(TypeError):
        and_(atom("p"), "q")


def test_box_is_vacuously_true_and_diamond_false_at_dead_ends(
    modal_api: Dict[str, Any],
):
    model = make_dead_end_model(modal_api)
    atom = modal_api["Atom"]
    box = modal_api["Box"]
    diamond = modal_api["Diamond"]
    satisfies = modal_api["satisfies"]
    p = atom("p")

    assert satisfies(model, "w", box(p))
    assert not satisfies(model, "w", diamond(p))


def test_branching_successors_distinguish_box_and_diamond(
    modal_api: Dict[str, Any],
):
    model = make_branching_model(modal_api)
    atom = modal_api["Atom"]
    box = modal_api["Box"]
    diamond = modal_api["Diamond"]
    satisfies = modal_api["satisfies"]
    p = atom("p")
    q = atom("q")

    assert satisfies(model, "root", diamond(p))
    assert not satisfies(model, "root", box(p))
    assert satisfies(model, "root", box(q))


def test_nested_modalities_follow_successor_structure_recursively(
    modal_api: Dict[str, Any],
):
    model = make_nested_model(modal_api)
    atom = modal_api["Atom"]
    box = modal_api["Box"]
    diamond = modal_api["Diamond"]
    satisfies = modal_api["satisfies"]
    p = atom("p")

    assert satisfies(model, "root", box(diamond(p)))
    assert satisfies(model, "root", diamond(diamond(p)))
    assert not satisfies(model, "root", diamond(box(p)))


def test_is_valid_checks_all_worlds_in_the_model(modal_api: Dict[str, Any]):
    model = make_branching_model(modal_api)
    atom = modal_api["Atom"]
    not_ = modal_api["Not"]
    or_ = modal_api["Or"]
    diamond = modal_api["Diamond"]
    is_valid = modal_api["is_valid"]
    p = atom("p")
    q = atom("q")

    assert not is_valid(model, p)
    assert not is_valid(model, diamond(q))
    assert is_valid(model, or_(p, not_(p)))


def test_entails_succeeds_when_every_world_satisfying_premises_satisfies_conclusion(
    modal_api: Dict[str, Any],
):
    model = make_uniform_successor_model(modal_api)
    atom = modal_api["Atom"]
    box = modal_api["Box"]
    implies = modal_api["Implies"]
    entails = modal_api["entails"]
    p = atom("p")
    q = atom("q")

    assert entails(model, [box(implies(p, q)), box(p)], box(q))


def test_entails_fails_with_a_finite_model_counterexample(
    modal_api: Dict[str, Any],
):
    model = make_branching_model(modal_api)
    atom = modal_api["Atom"]
    box = modal_api["Box"]
    diamond = modal_api["Diamond"]
    entails = modal_api["entails"]
    p = atom("p")

    assert not entails(model, [diamond(p)], box(p))


def test_satisfies_rejects_unknown_worlds(modal_api: Dict[str, Any]):
    model = make_dead_end_model(modal_api)
    atom = modal_api["Atom"]
    satisfies = modal_api["satisfies"]

    with pytest.raises(ValueError):
        satisfies(model, "ghost", atom("p"))


def test_kripke_model_exposes_relation_property_checks(modal_api: Dict[str, Any]):
    t_model = make_t_model(modal_api)
    d_model = make_serial_not_reflexive_model(modal_api)
    s4_model = make_reflexive_non_transitive_model(modal_api)
    s5_model = make_s5_model(modal_api)

    assert t_model.is_reflexive()
    assert t_model.is_transitive()
    assert t_model.is_symmetric()
    assert t_model.is_serial()

    assert d_model.is_serial()
    assert not d_model.is_reflexive()

    assert s4_model.is_reflexive()
    assert not s4_model.is_transitive()
    assert not s4_model.is_symmetric()

    assert s5_model.is_reflexive()
    assert s5_model.is_transitive()
    assert s5_model.is_symmetric()


def test_frame_validation_distinguishes_k_t_s4_and_s5(modal_api: Dict[str, Any]):
    frame_logic = modal_api["FrameLogic"]
    validate_frame = modal_api["validate_frame"]
    k_model = make_dead_end_model(modal_api)
    d_model = make_serial_not_reflexive_model(modal_api)
    t_model = make_t_model(modal_api)
    b_model = make_b_model(modal_api)
    k4_model = make_transitive_non_reflexive_model(modal_api)
    non_s4_model = make_reflexive_non_transitive_model(modal_api)
    s4_model = make_s4_not_s5_model(modal_api)
    s5_model = make_s5_model(modal_api)

    assert validate_frame(k_model, frame_logic.K)
    assert not validate_frame(k_model, frame_logic.D)
    assert not validate_frame(k_model, frame_logic.T)

    assert validate_frame(d_model, frame_logic.D)
    assert validate_frame(t_model, frame_logic.T)
    assert validate_frame(b_model, frame_logic.B)

    assert validate_frame(k4_model, frame_logic.K4)
    assert not validate_frame(k4_model, frame_logic.T)
    assert not validate_frame(non_s4_model, frame_logic.S4)

    assert validate_frame(s4_model, frame_logic.S4)
    assert not validate_frame(s4_model, frame_logic.S5)

    assert validate_frame(s5_model, frame_logic.S5)


def test_kripke_model_reachability_and_generated_submodels(modal_api: Dict[str, Any]):
    model = modal_api["KripkeModel"](
        worlds={"root", "left", "right", "leaf", "isolated"},
        relation={
            ("root", "left"),
            ("left", "leaf"),
            ("root", "right"),
            ("isolated", "isolated"),
        },
        valuation={"p": {"root", "leaf", "isolated"}},
    )

    assert model.predecessors("leaf") == frozenset({"left"})
    assert model.reachable_worlds("root") == frozenset(
        {"root", "left", "right", "leaf"}
    )

    submodel = model.generated_submodel("root")

    assert submodel.worlds == frozenset({"root", "left", "right", "leaf"})
    assert ("isolated", "isolated") not in submodel.relation
    assert submodel.truth_set("p") == frozenset({"root", "leaf"})


def test_restrict_to_worlds_validates_input(modal_api: Dict[str, Any]):
    model = make_branching_model(modal_api)

    with pytest.raises(ValueError):
        model.restrict_to_worlds(set())

    with pytest.raises(ValueError):
        model.restrict_to_worlds({"ghost"})


def test_frame_aware_validity_checks_reject_incompatible_models(
    modal_api: Dict[str, Any],
):
    atom = modal_api["Atom"]
    box = modal_api["Box"]
    implies = modal_api["Implies"]
    frame_logic = modal_api["FrameLogic"]
    is_valid = modal_api["is_valid"]
    formula = implies(box(atom("p")), atom("p"))

    assert is_valid(make_t_model(modal_api), formula, logic=frame_logic.T)

    with pytest.raises(ValueError):
        is_valid(make_dead_end_model(modal_api), formula, logic=frame_logic.T)


def test_find_countermodel_returns_k_witness_for_t_axiom(modal_api: Dict[str, Any]):
    atom = modal_api["Atom"]
    box = modal_api["Box"]
    implies = modal_api["Implies"]
    find_countermodel = modal_api["find_countermodel"]
    satisfies = modal_api["satisfies"]
    formula = implies(box(atom("p")), atom("p"))

    witness = find_countermodel(formula, logic="K", max_worlds=1)

    assert witness is not None
    assert witness.world in witness.model.worlds
    assert not satisfies(witness.model, witness.world, formula)
    assert witness.model.relation == frozenset()
    assert witness.model.valuation["p"] == frozenset()


def test_find_countermodel_respects_t_frames(modal_api: Dict[str, Any]):
    atom = modal_api["Atom"]
    box = modal_api["Box"]
    implies = modal_api["Implies"]
    find_countermodel = modal_api["find_countermodel"]
    formula = implies(box(atom("p")), atom("p"))

    assert find_countermodel(formula, logic="T", max_worlds=2) is None


def test_find_model_returns_bounded_witness(modal_api: Dict[str, Any]):
    atom = modal_api["Atom"]
    diamond = modal_api["Diamond"]
    find_model = modal_api["find_model"]
    satisfies = modal_api["satisfies"]
    witness = find_model(diamond(atom("p")), logic="D", max_worlds=2)

    assert witness is not None
    assert witness.world in witness.model.worlds
    assert satisfies(witness.model, witness.world, diamond(atom("p")))
    assert all(witness.model.successors(world) for world in witness.model.worlds)


def test_z3_backend_matches_exhaustive_model_search(modal_api: Dict[str, Any]):
    atom = modal_api["Atom"]
    box = modal_api["Box"]
    diamond = modal_api["Diamond"]
    and_ = modal_api["And"]
    find_model = modal_api["find_model"]
    formula = and_(diamond(atom("p")), box(diamond(atom("p"))))

    exhaustive = find_model(formula, logic="K", max_worlds=2, backend="exhaustive")
    solver_witness = find_model(formula, logic="K", max_worlds=2, backend="z3")

    assert exhaustive is not None
    assert solver_witness is not None
    assert solver_witness.model.worlds == exhaustive.model.worlds


def test_z3_backend_matches_exhaustive_countermodel_search(modal_api: Dict[str, Any]):
    atom = modal_api["Atom"]
    box = modal_api["Box"]
    implies = modal_api["Implies"]
    find_countermodel = modal_api["find_countermodel"]
    formula = implies(box(atom("p")), atom("p"))

    exhaustive = find_countermodel(formula, logic="K", max_worlds=1, backend="exhaustive")
    solver_witness = find_countermodel(formula, logic="K", max_worlds=1, backend="z3")

    assert exhaustive is not None
    assert solver_witness is not None
    assert solver_witness.model.relation == exhaustive.model.relation


def test_find_model_returns_none_when_bound_is_too_small(modal_api: Dict[str, Any]):
    atom = modal_api["Atom"]
    diamond = modal_api["Diamond"]
    box = modal_api["Box"]
    and_ = modal_api["And"]
    find_model = modal_api["find_model"]
    formula = and_(diamond(atom("p")), box(modal_api["Not"](atom("p"))))

    assert find_model(formula, logic="K", max_worlds=1) is None


def test_formula_utility_helpers_measure_and_collect_structure(
    modal_api: Dict[str, Any],
):
    atom = modal_api["Atom"]
    and_ = modal_api["And"]
    box = modal_api["Box"]
    diamond = modal_api["Diamond"]
    formula_size = modal_api["formula_size"]
    modal_depth = modal_api["modal_depth"]
    subformulas = modal_api["subformulas"]
    formula = box(and_(atom("p"), diamond(atom("q"))))

    assert formula_size(formula) == 5
    assert modal_depth(formula) == 2
    assert subformulas(formula) == frozenset(
        {
            formula,
            and_(atom("p"), diamond(atom("q"))),
            atom("p"),
            diamond(atom("q")),
            atom("q"),
        }
    )


def test_find_countermodel_separates_t_from_s4(modal_api: Dict[str, Any]):
    atom = modal_api["Atom"]
    box = modal_api["Box"]
    implies = modal_api["Implies"]
    find_countermodel = modal_api["find_countermodel"]
    satisfies = modal_api["satisfies"]
    formula = implies(box(atom("p")), box(box(atom("p"))))

    witness = find_countermodel(formula, logic="T", max_worlds=3)

    assert witness is not None
    assert not satisfies(witness.model, witness.world, formula)
    assert all((world, world) in witness.model.relation for world in witness.model.worlds)
    assert find_countermodel(formula, logic="S4", max_worlds=3) is None


def test_find_countermodel_separates_s4_from_s5(modal_api: Dict[str, Any]):
    atom = modal_api["Atom"]
    box = modal_api["Box"]
    diamond = modal_api["Diamond"]
    implies = modal_api["Implies"]
    find_countermodel = modal_api["find_countermodel"]
    satisfies = modal_api["satisfies"]
    formula = implies(diamond(atom("p")), box(diamond(atom("p"))))

    witness = find_countermodel(formula, logic="S4", max_worlds=2)

    assert witness is not None
    assert not satisfies(witness.model, witness.world, formula)
    assert find_countermodel(formula, logic="S5", max_worlds=2) is None


def test_find_entailment_countermodel_returns_world_witness(modal_api: Dict[str, Any]):
    atom = modal_api["Atom"]
    box = modal_api["Box"]
    find_entailment_countermodel = modal_api["find_entailment_countermodel"]
    satisfies = modal_api["satisfies"]
    premise = box(atom("p"))
    conclusion = atom("p")

    witness = find_entailment_countermodel([premise], conclusion, logic="K", max_worlds=1)

    assert witness is not None
    assert satisfies(witness.model, witness.world, premise)
    assert not satisfies(witness.model, witness.world, conclusion)
    assert find_entailment_countermodel([premise], conclusion, logic="T", max_worlds=2) is None


def test_find_entailment_countermodel_separates_t_and_s4(modal_api: Dict[str, Any]):
    atom = modal_api["Atom"]
    box = modal_api["Box"]
    find_entailment_countermodel = modal_api["find_entailment_countermodel"]
    satisfies = modal_api["satisfies"]
    premise = box(atom("p"))
    conclusion = box(box(atom("p")))

    witness = find_entailment_countermodel([premise], conclusion, logic="T", max_worlds=3)

    assert witness is not None
    assert satisfies(witness.model, witness.world, premise)
    assert not satisfies(witness.model, witness.world, conclusion)
    assert find_entailment_countermodel([premise], conclusion, logic="S4", max_worlds=3) is None


def test_find_entailment_countermodel_separates_s4_and_s5(modal_api: Dict[str, Any]):
    atom = modal_api["Atom"]
    box = modal_api["Box"]
    diamond = modal_api["Diamond"]
    find_entailment_countermodel = modal_api["find_entailment_countermodel"]
    satisfies = modal_api["satisfies"]
    premise = diamond(atom("p"))
    conclusion = box(diamond(atom("p")))

    witness = find_entailment_countermodel(
        [premise], conclusion, logic="S4", max_worlds=2
    )

    assert witness is not None
    assert satisfies(witness.model, witness.world, premise)
    assert not satisfies(witness.model, witness.world, conclusion)
    assert find_entailment_countermodel([premise], conclusion, logic="S5", max_worlds=2) is None
