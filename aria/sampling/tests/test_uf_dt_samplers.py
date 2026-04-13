"""
Unit tests for UF and datatype finite-domain samplers.
"""

from typing import cast

import pytest
import z3

from aria.sampling.base import Logic, SamplingOptions
from aria.sampling.factory import create_sampler, sample_models_from_formula
from aria.sampling.finite_domain.dt.base import DatatypeSampler
from aria.sampling.finite_domain.uf.base import UninterpretedFunctionSampler
from aria.sampling.finite_domain.ufdt.base import MixedUFDatatypeSampler


class TestUninterpretedFunctionSampler:
    """Test cases for UninterpretedFunctionSampler."""

    def test_supports_correct_logic(self):
        sampler = UninterpretedFunctionSampler()
        assert sampler.supports_logic(Logic.QF_UF) is True
        assert sampler.supports_logic(Logic.QF_UFLIA) is True
        assert sampler.supports_logic(Logic.QF_DT) is False

    def test_sample_ground_uf_formula(self):
        color, (red, green, blue) = z3.EnumSort("ColorUF", ["red", "green", "blue"])
        x = z3.Const("x", color)
        f = z3.Function("f", color, color)
        formula = cast(z3.ExprRef, z3.And(x != red, f(x) == green))

        sampler = UninterpretedFunctionSampler()
        sampler.init_from_formula(formula)
        result = sampler.sample(SamplingOptions(num_samples=5))

        assert len(result) == 2
        for sample in result:
            assert sample["x"] in {"green", "blue"}
            assert sample["f(x)"] == "green"

    def test_sample_formula_without_tracked_terms(self):
        sampler = UninterpretedFunctionSampler()
        sampler.init_from_formula(z3.BoolVal(True))

        result = sampler.sample(SamplingOptions(num_samples=3))

        assert len(result) == 1
        assert result[0] == {}

    def test_factory_registration(self):
        sampler = create_sampler(Logic.QF_UF)
        assert isinstance(sampler, UninterpretedFunctionSampler)

    def test_factory_registration_for_uflia(self):
        sampler = create_sampler(Logic.QF_UFLIA)
        assert isinstance(sampler, UninterpretedFunctionSampler)

    def test_sample_ground_uf_lia_formula(self):
        x = z3.Int("x_uflia")
        y = z3.Int("y_uflia")
        f = z3.Function("f_uflia", z3.IntSort(), z3.IntSort())
        formula = cast(
            z3.ExprRef,
            z3.And(x >= 0, x <= 1, y == f(x), y <= 3),
        )

        sampler = UninterpretedFunctionSampler()
        sampler.init_from_formula(formula)
        result = sampler.sample(
            SamplingOptions(num_samples=10, projection_terms=[x, y, f(x)])
        )

        assert len(result) == 2
        assert {sample["x_uflia"] for sample in result} == {0, 1}
        for sample in result:
            assert sample["y_uflia"] == sample["f_uflia(x_uflia)"]
            assert sample["y_uflia"] <= 3

    def test_projected_sampling_by_term(self):
        color, (red, green, blue) = z3.EnumSort(
            "ColorUFProjection", ["red", "green", "blue"]
        )
        x = z3.Const("x", color)
        y = z3.Const("y", color)
        f = z3.Function("f", color, color)
        formula = z3.And(z3.Or(x == red, x == green), z3.Or(y == red, y == blue), f(x) == blue)
        formula = cast(z3.ExprRef, formula)

        sampler = UninterpretedFunctionSampler()
        sampler.init_from_formula(formula)
        result = sampler.sample(
            SamplingOptions(num_samples=10, projection_terms=[x, "f(x)"])
        )

        assert len(result) == 2
        assert all(set(sample.keys()) == {"x", "f(x)"} for sample in result)
        assert {sample["x"] for sample in result} == {"red", "green"}
        assert {sample["f(x)"] for sample in result} == {"blue"}

    def test_projected_uniqueness_with_full_model_output(self):
        color, (red, green) = z3.EnumSort("ColorUFFullModel", ["red", "green"])
        x = z3.Const("x", color)
        y = z3.Const("y", color)
        formula = cast(
            z3.ExprRef,
            z3.And(z3.Or(x == red, x == green), z3.Or(y == red, y == green)),
        )

        sampler = UninterpretedFunctionSampler()
        sampler.init_from_formula(formula)
        result = sampler.sample(
            SamplingOptions(num_samples=10, projection_terms=[x], return_full_model=True)
        )

        assert len(result) == 2
        assert all(set(sample.keys()) == {"x", "y"} for sample in result)
        assert {sample["x"] for sample in result} == {"red", "green"}

    def test_projected_uniqueness_with_custom_tracked_terms(self):
        color, (red, green) = z3.EnumSort("ColorUFTracked", ["red", "green"])
        x = z3.Const("x", color)
        y = z3.Const("y", color)
        formula = cast(
            z3.ExprRef,
            z3.And(z3.Or(x == red, x == green), z3.Or(y == red, y == green)),
        )

        sampler = UninterpretedFunctionSampler()
        sampler.init_from_formula(formula)
        result = sampler.sample(
            SamplingOptions(num_samples=10, projection_terms=[x], tracked_terms=[y])
        )

        assert len(result) == 2
        assert all(set(sample.keys()) == {"y"} for sample in result)
        assert {sample["y"] for sample in result} == {"red", "green"}

    def test_equal_arguments_preserve_uf_congruence(self):
        color, (red, green) = z3.EnumSort("ColorUFCongruence", ["red", "green"])
        x = z3.Const("x", color)
        y = z3.Const("y", color)
        f = z3.Function("f", color, color)
        formula = cast(
            z3.ExprRef,
            z3.And(x == y, z3.Or(x == red, x == green), f(x) == red),
        )

        sampler = UninterpretedFunctionSampler()
        sampler.init_from_formula(formula)
        result = sampler.sample(
            SamplingOptions(num_samples=5, tracked_terms=[f(x), f(y)])
        )

        assert len(result) == 2
        for sample in result:
            assert sample["f(x)"] == "red"
            assert sample["f(y)"] == "red"

    def test_nested_ground_uf_terms_are_tracked(self):
        color, (red, green) = z3.EnumSort("ColorUFNested", ["red", "green"])
        x = z3.Const("x", color)
        g = z3.Function("g", color, color)
        f = z3.Function("f", color, color)
        formula = cast(
            z3.ExprRef,
            z3.And(z3.Or(x == red, x == green), g(x) == red, f(g(x)) == green),
        )

        sampler = UninterpretedFunctionSampler()
        sampler.init_from_formula(formula)
        result = sampler.sample(
            SamplingOptions(num_samples=10, tracked_terms=["g(x)", "f(g(x))"])
        )

        assert len(result) == 2
        for sample in result:
            assert sample["g(x)"] == "red"
            assert sample["f(g(x))"] == "green"


class TestDatatypeSampler:
    """Test cases for DatatypeSampler."""

    def test_supports_correct_logic(self):
        sampler = DatatypeSampler()
        assert sampler.supports_logic(Logic.QF_DT) is True
        assert sampler.supports_logic(Logic.QF_UF) is False

    def test_sample_enum_datatype(self):
        color, (red, green, blue) = z3.EnumSort("ColorDT", ["red", "green", "blue"])
        x = z3.Const("x", color)
        formula = cast(z3.ExprRef, z3.Or(x == red, x == blue))

        sampler = DatatypeSampler()
        sampler.init_from_formula(formula)
        result = sampler.sample(SamplingOptions(num_samples=5))

        assert len(result) == 2
        assert {sample["x"] for sample in result} == {"red", "blue"}

    def test_sample_finite_algebraic_datatype(self):
        color, (red, green) = z3.EnumSort("MaybeColorBase", ["red", "green"])
        maybe = z3.Datatype("MaybeColor")
        maybe.declare("none")
        maybe.declare("some", ("value", color))
        maybe = maybe.create()

        x = z3.Const("x", maybe)
        formula = cast(z3.ExprRef, x != maybe.none)

        sampler = DatatypeSampler()
        sampler.init_from_formula(formula)
        result = sampler.sample(SamplingOptions(num_samples=5))

        assert len(result) == 2
        values = {sample["x"]["constructor"] for sample in result}
        assert values == {"some"}
        payloads = {sample["x"]["fields"][0] for sample in result}
        assert payloads == {"red", "green"}

    def test_sample_models_from_formula(self):
        color, (red, green) = z3.EnumSort("ColorFactory", ["red", "green"])
        x = z3.Const("x", color)
        formula = cast(z3.ExprRef, z3.Or(x == red, x == green))

        result = sample_models_from_formula(
            formula, Logic.QF_DT, SamplingOptions(num_samples=2)
        )

        assert len(result) == 2

    def test_projected_sampling_by_variable(self):
        color, (red, green) = z3.EnumSort("ProjectedColorDT", ["red", "green"])
        x = z3.Const("x", color)
        y = z3.Const("y", color)
        formula = cast(
            z3.ExprRef,
            z3.And(z3.Or(x == red, x == green), z3.Or(y == red, y == green)),
        )

        sampler = DatatypeSampler()
        sampler.init_from_formula(formula)
        result = sampler.sample(SamplingOptions(num_samples=10, projection_terms=[x]))

        assert len(result) == 2
        assert all(set(sample.keys()) == {"x"} for sample in result)
        assert {sample["x"] for sample in result} == {"red", "green"}

    def test_projected_uniqueness_with_full_model_output(self):
        color, (red, green) = z3.EnumSort("DatatypeFullModelColor", ["red", "green"])
        x = z3.Const("x", color)
        y = z3.Const("y", color)
        formula = cast(
            z3.ExprRef,
            z3.And(z3.Or(x == red, x == green), z3.Or(y == red, y == green)),
        )

        sampler = DatatypeSampler()
        sampler.init_from_formula(formula)
        result = sampler.sample(
            SamplingOptions(num_samples=10, projection_terms=[x], return_full_model=True)
        )

        assert len(result) == 2
        assert all(set(sample.keys()) == {"x", "y"} for sample in result)
        assert {sample["x"] for sample in result} == {"red", "green"}

    def test_selector_terms_can_be_projected(self):
        color, (red, green) = z3.EnumSort("DatatypeSelectorColor", ["red", "green"])
        maybe = z3.Datatype("DatatypeSelectorMaybe")
        maybe.declare("none")
        maybe.declare("some", ("value", color))
        maybe = maybe.create()

        box = z3.Const("box", maybe)
        formula = cast(z3.ExprRef, z3.And(box != maybe.none, maybe.value(box) == red))

        sampler = DatatypeSampler()
        sampler.init_from_formula(formula)
        result = sampler.sample(
            SamplingOptions(num_samples=10, projection_terms=["value(box)"])
        )

        assert len(result) == 1
        assert result[0] == {"value(box)": "red"}

    def test_selector_closure_adds_payload_observation(self):
        color, (red, green) = z3.EnumSort("DatatypeClosureColor", ["red", "green"])
        maybe = z3.Datatype("DatatypeClosureMaybe")
        maybe.declare("none")
        maybe.declare("some", ("value", color))
        maybe = maybe.create()

        payload = z3.Const("payload", color)
        box = z3.Const("box", maybe)
        formula = cast(
            z3.ExprRef,
            z3.And(
                z3.Or(payload == red, payload == green),
                box == maybe.some(payload),
            ),
        )

        sampler = DatatypeSampler()
        sampler.init_from_formula(formula)
        result = sampler.sample(
            SamplingOptions(
                num_samples=10,
                projection_terms=["value(box)"],
                include_selector_closure=True,
            )
        )

        assert len(result) == 2
        assert {sample["value(box)"] for sample in result} == {"red", "green"}

    def test_selector_closure_does_not_use_disjunctive_constructor_hints(self):
        color, (red, green) = z3.EnumSort("DatatypeDisjunctiveColor", ["red", "green"])
        maybe = z3.Datatype("DatatypeDisjunctiveMaybe")
        maybe.declare("none")
        maybe.declare("some", ("value", color))
        maybe = maybe.create()

        box = z3.Const("box", maybe)
        formula = cast(
            z3.ExprRef,
            z3.Or(box == maybe.none, box == maybe.some(red)),
        )

        sampler = DatatypeSampler()
        sampler.init_from_formula(formula)

        with pytest.raises(ValueError, match="Unknown projection term"):
            sampler.sample(
                SamplingOptions(
                    num_samples=10,
                    projection_terms=["value(box)"],
                    include_selector_closure=True,
                )
            )

    def test_selector_closure_does_not_use_negated_recognizers(self):
        color, (red, green) = z3.EnumSort("DatatypeNegatedColor", ["red", "green"])
        maybe = z3.Datatype("DatatypeNegatedMaybe")
        maybe.declare("none")
        maybe.declare("some", ("value", color))
        maybe = maybe.create()

        box = z3.Const("box", maybe)
        formula = cast(z3.ExprRef, z3.Not(maybe.is_some(box)))

        sampler = DatatypeSampler()
        sampler.init_from_formula(formula)

        with pytest.raises(ValueError, match="Unknown projection term"):
            sampler.sample(
                SamplingOptions(
                    num_samples=10,
                    projection_terms=["value(box)"],
                    include_selector_closure=True,
                )
            )


class TestMixedUFDatatypeSampler:
    """Test cases for MixedUFDatatypeSampler."""

    def test_supports_correct_logic(self):
        sampler = MixedUFDatatypeSampler()
        assert sampler.supports_logic(Logic.QF_UFDT) is True
        assert sampler.supports_logic(Logic.QF_UF) is False

    def test_sample_mixed_formula(self):
        color, (red, green) = z3.EnumSort("MixedColor", ["red", "green"])
        maybe = z3.Datatype("MixedMaybe")
        maybe.declare("none")
        maybe.declare("some", ("value", color))
        maybe = maybe.create()

        x = z3.Const("x", color)
        box = z3.Const("box", maybe)
        tag = z3.Function("tag", color, maybe)
        formula = cast(
            z3.ExprRef,
            z3.And(z3.Or(x == red, x == green), box == tag(x), box == maybe.some(x)),
        )

        sampler = MixedUFDatatypeSampler()
        sampler.init_from_formula(formula)
        result = sampler.sample(SamplingOptions(num_samples=10))

        assert len(result) == 2
        assert {sample["x"] for sample in result} == {"red", "green"}
        assert all(sample["box"]["constructor"] == "some" for sample in result)
        assert all(sample["tag(x)"]["constructor"] == "some" for sample in result)

    def test_projected_sampling_mixed_logic(self):
        color, (red, green) = z3.EnumSort("ProjectedMixedColor", ["red", "green"])
        maybe = z3.Datatype("ProjectedMixedMaybe")
        maybe.declare("none")
        maybe.declare("some", ("value", color))
        maybe = maybe.create()

        x = z3.Const("x", color)
        y = z3.Const("y", color)
        tag = z3.Function("tag", color, maybe)
        formula = cast(
            z3.ExprRef,
            z3.And(
                z3.Or(x == red, x == green),
                z3.Or(y == red, y == green),
                tag(x) != maybe.none,
            ),
        )

        result = sample_models_from_formula(
            formula,
            Logic.QF_UFDT,
            SamplingOptions(num_samples=10, projection_terms=["tag(x)"]),
        )

        assert len(result) == 2
        assert all(set(sample.keys()) == {"tag(x)"} for sample in result)

    def test_projected_uniqueness_with_full_model_output(self):
        color, (red, green) = z3.EnumSort("MixedFullModelColor", ["red", "green"])
        maybe = z3.Datatype("MixedFullModelMaybe")
        maybe.declare("none")
        maybe.declare("some", ("value", color))
        maybe = maybe.create()

        x = z3.Const("x", color)
        y = z3.Const("y", color)
        box = z3.Const("box", maybe)
        tag = z3.Function("tag", color, maybe)
        formula = cast(
            z3.ExprRef,
            z3.And(
                z3.Or(x == red, x == green),
                z3.Or(y == red, y == green),
                box == maybe.some(x),
                tag(x) == maybe.some(x),
            ),
        )

        sampler = MixedUFDatatypeSampler()
        sampler.init_from_formula(formula)
        result = sampler.sample(
            SamplingOptions(
                num_samples=10,
                projection_terms=[x],
                return_full_model=True,
            )
        )

        assert len(result) == 2
        assert all(
            set(sample.keys()) == {"box", "tag(x)", "x", "y"} for sample in result
        )
        assert {sample["x"] for sample in result} == {"red", "green"}

    def test_factory_registration(self):
        sampler = create_sampler(Logic.QF_UFDT)
        assert isinstance(sampler, MixedUFDatatypeSampler)

    def test_mixed_sampler_projects_selector_over_uf_result(self):
        color, (red, green) = z3.EnumSort("MixedSelectorColor", ["red", "green"])
        maybe = z3.Datatype("MixedSelectorMaybe")
        maybe.declare("none")
        maybe.declare("some", ("value", color))
        maybe = maybe.create()

        x = z3.Const("x", color)
        tag = z3.Function("tag", color, maybe)
        formula = cast(
            z3.ExprRef,
            z3.And(
                z3.Or(x == red, x == green),
                tag(x) == maybe.some(x),
                maybe.value(tag(x)) == x,
            ),
        )

        sampler = MixedUFDatatypeSampler()
        sampler.init_from_formula(formula)
        result = sampler.sample(
            SamplingOptions(num_samples=10, projection_terms=["value(tag(x))"])
        )

        assert len(result) == 2
        assert {sample["value(tag(x))"] for sample in result} == {"red", "green"}

    def test_mixed_selector_closure_propagates_across_aliases(self):
        color, (red, green) = z3.EnumSort("MixedAliasColor", ["red", "green"])
        maybe = z3.Datatype("MixedAliasMaybe")
        maybe.declare("none")
        maybe.declare("some", ("value", color))
        maybe = maybe.create()

        x = z3.Const("x", color)
        box = z3.Const("box", maybe)
        tag = z3.Function("tag", color, maybe)
        formula = cast(
            z3.ExprRef,
            z3.And(
                z3.Or(x == red, x == green),
                box == tag(x),
                box == maybe.some(x),
            ),
        )

        sampler = MixedUFDatatypeSampler()
        sampler.init_from_formula(formula)
        result = sampler.sample(
            SamplingOptions(
                num_samples=10,
                projection_terms=["value(tag(x))"],
                include_selector_closure=True,
            )
        )

        assert len(result) == 2
        assert {sample["value(tag(x))"] for sample in result} == {"red", "green"}

    def test_mixed_selector_closure_requires_positive_constructor_evidence(self):
        color, (red, green) = z3.EnumSort("MixedNegativeAliasColor", ["red", "green"])
        maybe = z3.Datatype("MixedNegativeAliasMaybe")
        maybe.declare("none")
        maybe.declare("some", ("value", color))
        maybe = maybe.create()

        x = z3.Const("x", color)
        box = z3.Const("box", maybe)
        tag = z3.Function("tag", color, maybe)
        formula = cast(
            z3.ExprRef,
            z3.And(z3.Or(x == red, x == green), box == tag(x), z3.Not(maybe.is_some(box))),
        )

        sampler = MixedUFDatatypeSampler()
        sampler.init_from_formula(formula)

        with pytest.raises(ValueError, match="Unknown projection term"):
            sampler.sample(
                SamplingOptions(
                    num_samples=10,
                    projection_terms=["value(tag(x))"],
                    include_selector_closure=True,
                )
            )

    def test_unknown_exprref_projection_is_rejected(self):
        color, (red, green) = z3.EnumSort("MixedUnknownExprColor", ["red", "green"])
        maybe = z3.Datatype("MixedUnknownExprMaybe")
        maybe.declare("none")
        maybe.declare("some", ("value", color))
        maybe = maybe.create()

        box = z3.Const("box", maybe)
        formula = cast(z3.ExprRef, z3.Or(box == maybe.none, box == maybe.some(red)))

        sampler = DatatypeSampler()
        sampler.init_from_formula(formula)

        with pytest.raises(ValueError, match="Unknown projection term"):
            sampler.sample(
                SamplingOptions(
                    num_samples=10,
                    tracked_terms=[maybe.value(box)],
                )
            )

    def test_stats_include_real_counts(self):
        color, (red, green) = z3.EnumSort("MixedStatsColor", ["red", "green"])
        x = z3.Const("x", color)
        formula = cast(z3.ExprRef, z3.Or(x == red, x == green))

        sampler = MixedUFDatatypeSampler()
        sampler.init_from_formula(formula)
        result = sampler.sample(SamplingOptions(num_samples=2))

        assert result.stats["iterations"] == 2
        assert result.stats["solver_checks"] >= 2
        assert result.stats["tracked_term_count"] >= 1
        assert result.stats["projection_term_count"] >= 1
        assert result.stats["output_term_count"] >= 1
        assert result.stats["time_ms"] >= 0


if __name__ == "__main__":
    pytest.main()
