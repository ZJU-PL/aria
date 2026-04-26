"""
Unit tests for the ADT + LIA mixed sampler.
"""

import pytest
import z3

from aria.sampling.base import Logic, SamplingMethod, SamplingOptions
from aria.sampling.factory import create_sampler, sample_models_from_formula
from aria.sampling.dtlia import ADTLIASampler


def _make_maybe_int(name: str) -> z3.DatatypeSortRef:
    maybe = z3.Datatype(name)
    maybe.declare("none")
    maybe.declare("some", ("value", z3.IntSort()))
    return maybe.create()


def _make_flag_box(name: str) -> z3.DatatypeSortRef:
    flag_box = z3.Datatype(name)
    flag_box.declare("off")
    flag_box.declare("on", ("flag", z3.BoolSort()))
    return flag_box.create()


class TestADTLIASampler:
    """Test cases for ADTLIASampler."""

    def test_supports_correct_logic(self):
        sampler = ADTLIASampler()
        assert sampler.supports_logic(Logic.QF_DTLIA) is True
        assert sampler.supports_logic(Logic.QF_LIA) is False

    def test_factory_registration(self):
        sampler = create_sampler(Logic.QF_DTLIA)
        assert isinstance(sampler, ADTLIASampler)

    def test_sample_models_from_formula(self):
        maybe = _make_maybe_int("MaybeIntFactory")
        x = z3.Int("x")
        box = z3.Const("box", maybe)
        formula = z3.And(x >= 0, x <= 1, box == maybe.some(x))

        result = sample_models_from_formula(
            formula,
            Logic.QF_DTLIA,
            SamplingOptions(num_samples=10, include_selector_closure=True),
        )

        assert len(result) == 2
        assert {sample["x"] for sample in result} == {0, 1}

    def test_samples_mixed_integer_and_datatype_observables(self):
        maybe = _make_maybe_int("MaybeIntMixed")
        x = z3.Int("x")
        box = z3.Const("box", maybe)
        formula = z3.And(x >= 0, x <= 1, box == maybe.some(x), maybe.value(box) == x)

        sampler = ADTLIASampler()
        sampler.init_from_formula(formula)
        result = sampler.sample(
            SamplingOptions(num_samples=10, tracked_terms=[x, "value(box)"])
        )

        assert len(result) == 2
        assert {sample["x"] for sample in result} == {0, 1}
        assert {sample["value(box)"] for sample in result} == {0, 1}

    def test_projected_uniqueness_with_full_model_output(self):
        maybe = _make_maybe_int("MaybeIntFullModel")
        x = z3.Int("x")
        box = z3.Const("box", maybe)
        formula = z3.And(x >= 0, x <= 1, box == maybe.some(x))

        sampler = ADTLIASampler()
        sampler.init_from_formula(formula)
        result = sampler.sample(
            SamplingOptions(
                num_samples=10,
                projection_terms=[x],
                return_full_model=True,
                include_selector_closure=True,
            )
        )

        assert len(result) == 2
        assert {sample["x"] for sample in result} == {0, 1}
        assert all("box" in sample for sample in result)
        assert all("value(box)" in sample for sample in result)

    def test_projection_by_selector_term(self):
        maybe = _make_maybe_int("MaybeIntSelectorProjection")
        x = z3.Int("x")
        box = z3.Const("box", maybe)
        formula = z3.And(x >= 0, x <= 2, box == maybe.some(x))

        sampler = ADTLIASampler()
        sampler.init_from_formula(formula)
        result = sampler.sample(
            SamplingOptions(
                num_samples=10,
                projection_terms=["value(box)"],
                include_selector_closure=True,
            )
        )

        assert len(result) == 3
        assert {sample["value(box)"] for sample in result} == {0, 1, 2}

    def test_selector_closure_alias_propagation(self):
        maybe = _make_maybe_int("MaybeIntAlias")
        x = z3.Int("x")
        box = z3.Const("box", maybe)
        alias = z3.Const("alias", maybe)
        formula = z3.And(x >= 0, x <= 1, alias == box, box == maybe.some(x))

        sampler = ADTLIASampler()
        sampler.init_from_formula(formula)
        result = sampler.sample(
            SamplingOptions(
                num_samples=10,
                projection_terms=["value(alias)"],
                include_selector_closure=True,
            )
        )

        assert len(result) == 2
        assert {sample["value(alias)"] for sample in result} == {0, 1}

    def test_selector_closure_rejects_disjunctive_constructor_evidence(self):
        maybe = _make_maybe_int("MaybeIntDisjunctive")
        x = z3.Int("x")
        box = z3.Const("box", maybe)
        formula = z3.And(x >= 0, x <= 1, z3.Or(box == maybe.none, box == maybe.some(x)))

        sampler = ADTLIASampler()
        sampler.init_from_formula(formula)

        with pytest.raises(ValueError, match="Unknown projection term"):
            sampler.sample(
                SamplingOptions(
                    num_samples=10,
                    projection_terms=["value(box)"],
                    include_selector_closure=True,
                )
            )

    def test_selector_closure_rejects_negated_recognizer(self):
        maybe = _make_maybe_int("MaybeIntNegated")
        x = z3.Int("x")
        box = z3.Const("box", maybe)
        formula = z3.And(x >= 0, z3.Not(maybe.is_some(box)))

        sampler = ADTLIASampler()
        sampler.init_from_formula(formula)

        with pytest.raises(ValueError, match="Unknown projection term"):
            sampler.sample(
                SamplingOptions(
                    num_samples=10,
                    projection_terms=["value(box)"],
                    include_selector_closure=True,
                )
            )

    def test_unsat_mixed_formula_returns_no_samples(self):
        maybe = _make_maybe_int("MaybeIntUnsat")
        x = z3.Int("x")
        box = z3.Const("box", maybe)
        formula = z3.And(box == maybe.some(x), x < 0, maybe.value(box) >= 0)

        sampler = ADTLIASampler()
        sampler.init_from_formula(formula)
        result = sampler.sample(SamplingOptions(num_samples=10))

        assert len(result) == 0
        assert result.success is False

    def test_shape_mode_enumerates_constructor_shapes(self):
        maybe = _make_maybe_int("MaybeIntShapes")
        x = z3.Int("x")
        box = z3.Const("box", maybe)
        formula = z3.And(x >= 0, x <= 1, z3.Or(box == maybe.none, box == maybe.some(x)))

        sampler = ADTLIASampler()
        sampler.init_from_formula(formula)
        result = sampler.sample(
            SamplingOptions(
                method=SamplingMethod.SEARCH_TREE,
                num_samples=4,
                max_shapes=4,
                candidates_per_shape=2,
                include_selector_closure=True,
                return_full_model=True,
            )
        )

        constructors = {
            sample["box"]["constructor"] if isinstance(sample["box"], dict) else sample["box"]
            for sample in result
        }
        assert constructors == {"none", "some"}
        assert result.stats["shape_count"] >= 2

    def test_shape_mode_shares_budget_across_shapes(self):
        maybe = _make_maybe_int("MaybeIntShapeBudget")
        x = z3.Int("x")
        box = z3.Const("box", maybe)
        formula = z3.And(x >= 0, x <= 2, z3.Or(box == maybe.none, box == maybe.some(x)))

        sampler = ADTLIASampler()
        sampler.init_from_formula(formula)
        result = sampler.sample(
            SamplingOptions(
                method=SamplingMethod.SEARCH_TREE,
                num_samples=2,
                max_shapes=4,
                candidates_per_shape=3,
                include_selector_closure=True,
                return_full_model=True,
            )
        )

        constructors = [
            sample["box"]["constructor"] if isinstance(sample["box"], dict) else sample["box"]
            for sample in result
        ]
        assert set(constructors) == {"none", "some"}

    def test_shape_mode_payload_sampling_respects_bounds(self):
        maybe = _make_maybe_int("MaybeIntShapePayload")
        x = z3.Int("x")
        box = z3.Const("box", maybe)
        formula = z3.And(x >= 0, x <= 2, box == maybe.some(x))

        sampler = ADTLIASampler()
        sampler.init_from_formula(formula)
        result = sampler.sample(
            SamplingOptions(
                method=SamplingMethod.SEARCH_TREE,
                num_samples=3,
                max_shapes=2,
                candidates_per_shape=5,
                include_selector_closure=True,
                projection_terms=["value(box)"],
            )
        )

        assert {sample["value(box)"] for sample in result} == {0, 1, 2}
        assert result.stats["shape_count"] == 1
        assert result.stats["residual_projection_mode"] == "explicit"

    def test_shape_mode_max_distance_spreads_integer_payloads(self):
        maybe = _make_maybe_int("MaybeIntDiversity")
        x = z3.Int("x")
        box = z3.Const("box", maybe)
        formula = z3.And(x >= 0, x <= 4, box == maybe.some(x))

        sampler = ADTLIASampler()
        sampler.init_from_formula(formula)
        result = sampler.sample(
            SamplingOptions(
                method=SamplingMethod.SEARCH_TREE,
                num_samples=2,
                max_shapes=1,
                candidates_per_shape=5,
                include_selector_closure=True,
                projection_terms=["value(box)"],
                diversity_mode="max_distance",
            )
        )

        values = {sample["value(box)"] for sample in result}
        assert values == {0, 4}
        assert result.stats["diversity_mode"] == "max_distance"
        assert result.stats["candidate_count"] >= 5

    def test_shape_mode_defaults_to_payload_projection_with_full_output(self):
        maybe = _make_maybe_int("MaybeIntImplicitPayload")
        x = z3.Int("x")
        box = z3.Const("box", maybe)
        formula = z3.And(x >= 0, x <= 2, box == maybe.some(x))

        sampler = ADTLIASampler()
        sampler.init_from_formula(formula)
        result = sampler.sample(
            SamplingOptions(
                method=SamplingMethod.SEARCH_TREE,
                num_samples=3,
                max_shapes=2,
                candidates_per_shape=5,
                include_selector_closure=True,
            )
        )

        assert {sample["x"] for sample in result} == {0, 1, 2}
        assert all("box" in sample for sample in result)
        assert result.stats["residual_projection_mode"] == "payload_terms"

    def test_shape_mode_default_payload_projection_keeps_non_int_scalars(self):
        flag_box = _make_flag_box("FlagBoxImplicitPayload")
        box = z3.Const("box", flag_box)
        formula = z3.Or(
            box == flag_box.off,
            box == flag_box.on(True),
            box == flag_box.on(False),
        )

        sampler = ADTLIASampler()
        sampler.init_from_formula(formula)
        result = sampler.sample(
            SamplingOptions(
                method=SamplingMethod.SEARCH_TREE,
                num_samples=3,
                max_shapes=3,
                candidates_per_shape=3,
                include_selector_closure=True,
                return_full_model=True,
            )
        )

        assert len(result) == 3
        assert {
            sample["box"]["constructor"]
            if isinstance(sample["box"], dict)
            else sample["box"]
            for sample in result
        } == {"off", "on"}
        assert {sample["flag(box)"] for sample in result if "flag(box)" in sample} == {
            False,
            True,
        }

    def test_shape_mode_coverage_guided_covers_multiple_shapes(self):
        maybe = _make_maybe_int("MaybeIntCoverageShapes")
        x = z3.Int("x")
        box = z3.Const("box", maybe)
        formula = z3.And(
            x >= 0,
            x <= 1,
            z3.Or(box == maybe.none, box == maybe.some(x)),
        )

        sampler = ADTLIASampler()
        sampler.init_from_formula(formula)
        result = sampler.sample(
            SamplingOptions(
                method=SamplingMethod.SEARCH_TREE,
                num_samples=2,
                max_shapes=4,
                candidates_per_shape=3,
                include_selector_closure=True,
                return_full_model=True,
                diversity_mode="coverage_guided",
            )
        )

        constructors = [
            sample["box"]["constructor"] if isinstance(sample["box"], dict) else sample["box"]
            for sample in result
        ]
        assert set(constructors) == {"none", "some"}
        assert result.stats["diversity_mode"] == "coverage_guided"
        assert result.stats["coverage_feature_count"] >= 2
        assert result.stats["coverage_selected_feature_count"] >= 2
        assert result.stats["coverage_ratio"] > 0.0

    def test_shape_mode_coverage_guided_prefers_integer_boundaries(self):
        maybe = _make_maybe_int("MaybeIntCoverageBounds")
        x = z3.Int("x")
        box = z3.Const("box", maybe)
        formula = z3.And(x >= 0, x <= 4, box == maybe.some(x))

        sampler = ADTLIASampler()
        sampler.init_from_formula(formula)
        result = sampler.sample(
            SamplingOptions(
                method=SamplingMethod.SEARCH_TREE,
                num_samples=2,
                max_shapes=1,
                candidates_per_shape=5,
                include_selector_closure=True,
                projection_terms=["value(box)"],
                diversity_mode="coverage_guided",
            )
        )

        values = {sample["value(box)"] for sample in result}
        assert values == {0, 4}
        assert result.stats["coverage_selection"] == "weighted_set_cover"
        assert result.stats["coverage_selected_feature_count"] > 0

    def test_shape_mode_partial_exploration_prunes_infeasible_branches(self):
        tree = z3.Datatype("IntTree")
        tree.declare("leaf", ("leaf_value", z3.IntSort()))
        tree.declare("node", ("left", tree), ("right", tree))
        tree = tree.create()

        root = z3.Const("root", tree)
        formula = z3.And(
            tree.is_node(root),
            tree.is_leaf(tree.left(root)),
            tree.leaf_value(tree.left(root)) >= 0,
            tree.is_node(tree.right(root)),
            tree.is_leaf(tree.left(tree.right(root))),
            tree.leaf_value(tree.left(tree.right(root))) == 1,
            tree.is_leaf(tree.right(tree.right(root))),
            tree.leaf_value(tree.right(tree.right(root))) == 2,
        )

        sampler = ADTLIASampler()
        sampler.init_from_formula(formula)
        result = sampler.sample(
            SamplingOptions(
                method=SamplingMethod.SEARCH_TREE,
                num_samples=4,
                max_shapes=8,
                candidates_per_shape=4,
                include_selector_closure=True,
                return_full_model=True,
            )
        )

        assert len(result) >= 1
        assert result.stats["shape_enumerator"] == "partial_feasibility"
        assert result.stats["shape_count"] == 1
        assert result.stats["shape_pruned_branches"] > 0
        assert result.stats["shape_solver_checks"] >= result.stats["shape_pruned_branches"]
        assert result.stats["shape_exploration_complete"] is True
        assert result.stats["shape_exploration_termination_reason"] == "exhausted"

    def test_shape_mode_partial_exploration_keeps_recursive_payload_terms(self):
        tree = z3.Datatype("MaybeTree")
        tree.declare("empty")
        tree.declare("branch", ("value", z3.IntSort()), ("next", tree))
        tree = tree.create()

        root = z3.Const("root", tree)
        formula = z3.And(
            tree.is_branch(root),
            tree.value(root) >= 0,
            tree.value(root) <= 1,
            tree.is_branch(tree.next(root)),
            tree.value(tree.next(root)) == tree.value(root) + 10,
            tree.next(tree.next(root)) == tree.empty,
        )

        sampler = ADTLIASampler()
        sampler.init_from_formula(formula)
        result = sampler.sample(
            SamplingOptions(
                method=SamplingMethod.SEARCH_TREE,
                num_samples=4,
                max_shapes=8,
                candidates_per_shape=4,
                include_selector_closure=True,
                projection_terms=["value(root)", "value(next(root))"],
            )
        )

        assert {sample["value(root)"] for sample in result} == {0, 1}
        assert {sample["value(next(root))"] for sample in result} == {10, 11}
        assert any(
            "value(next(root))" in payload_terms
            for payload_terms in result.stats["shape_payload_terms"]
        )

    def test_shape_mode_reports_truncation_when_max_shapes_hits_limit(self):
        maybe = _make_maybe_int("MaybeIntShapeTruncation")
        x = z3.Int("x")
        left = z3.Const("left", maybe)
        right = z3.Const("right", maybe)
        formula = z3.And(
            x >= 0,
            x <= 1,
            z3.Or(left == maybe.none, left == maybe.some(x)),
            z3.Or(right == maybe.none, right == maybe.some(x)),
        )

        sampler = ADTLIASampler()
        sampler.init_from_formula(formula)
        result = sampler.sample(
            SamplingOptions(
                method=SamplingMethod.SEARCH_TREE,
                num_samples=2,
                max_shapes=1,
                candidates_per_shape=2,
                include_selector_closure=True,
                return_full_model=True,
            )
        )

        assert len(result) >= 1
        assert result.stats["shape_count"] == 1
        assert result.stats["shape_exploration_complete"] is False
        assert result.stats["shape_exploration_termination_reason"] == "max_shapes"
        assert result.stats["time_ms"] >= result.stats["shape_enumeration_time_ms"]

    def test_shape_mode_rejects_unknown_diversity_mode(self):
        maybe = _make_maybe_int("MaybeIntBadDiversity")
        x = z3.Int("x")
        box = z3.Const("box", maybe)
        formula = z3.And(x >= 0, x <= 1, box == maybe.some(x))

        sampler = ADTLIASampler()
        sampler.init_from_formula(formula)

        with pytest.raises(ValueError, match="Unknown diversity_mode"):
            sampler.sample(
                SamplingOptions(
                    method=SamplingMethod.SEARCH_TREE,
                    num_samples=2,
                    diversity_mode="furthest_first",
                )
            )

    def test_shape_mode_rejects_unknown_search_tree_option(self):
        maybe = _make_maybe_int("MaybeIntBadOption")
        x = z3.Int("x")
        box = z3.Const("box", maybe)
        formula = z3.And(x >= 0, x <= 1, box == maybe.some(x))

        sampler = ADTLIASampler()
        sampler.init_from_formula(formula)

        with pytest.raises(ValueError, match="Unknown DTLIA search-tree options"):
            sampler.sample(
                SamplingOptions(
                    method=SamplingMethod.SEARCH_TREE,
                    num_samples=2,
                    branch_budget=4,
                )
            )

    def test_shape_mode_supports_multiple_recursive_roots(self):
        tree = z3.Datatype("ForestNode")
        tree.declare("leaf", ("value", z3.IntSort()))
        tree.declare("node", ("left", tree), ("right", tree))
        tree = tree.create()

        left_root = z3.Const("left_root", tree)
        right_root = z3.Const("right_root", tree)
        formula = z3.And(
            tree.is_leaf(left_root),
            tree.value(left_root) >= 0,
            tree.value(left_root) <= 1,
            tree.is_node(right_root),
            tree.left(right_root) == left_root,
            tree.is_leaf(tree.right(right_root)),
            tree.value(tree.right(right_root)) == tree.value(left_root) + 10,
        )

        sampler = ADTLIASampler()
        sampler.init_from_formula(formula)
        result = sampler.sample(
            SamplingOptions(
                method=SamplingMethod.SEARCH_TREE,
                num_samples=4,
                max_shapes=4,
                candidates_per_shape=4,
                include_selector_closure=True,
                projection_terms=["value(left_root)", "value(right(right_root))"],
            )
        )

        assert {sample["value(left_root)"] for sample in result} == {0, 1}
        assert {sample["value(right(right_root))"] for sample in result} == {10, 11}
        assert result.stats["shape_count"] == 1
