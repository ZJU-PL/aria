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


if __name__ == "__main__":
    pytest.main()
