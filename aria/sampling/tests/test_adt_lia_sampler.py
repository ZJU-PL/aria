"""
Unit tests for the ADT + LIA mixed sampler.
"""

import pytest
import z3

from aria.sampling.base import Logic, SamplingMethod, SamplingOptions
from aria.sampling.factory import create_sampler, sample_models_from_formula
from aria.sampling.linear_ira.adt_lia_sampler import ADTLIASampler


def _make_maybe_int(name: str) -> z3.DatatypeSortRef:
    maybe = z3.Datatype(name)
    maybe.declare("none")
    maybe.declare("some", ("value", z3.IntSort()))
    return maybe.create()


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
