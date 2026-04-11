"""Unit tests for floating-point samplers."""

from typing import cast

import pytest
import z3

from aria.sampling.base import Logic, SamplingMethod, SamplingOptions
from aria.sampling.finite_domain.fp.base import FloatingPointSampler
from aria.sampling.finite_domain.fp.hash_sampler import HashBasedFPSampler
from aria.sampling.finite_domain.fp.total_order_sampler import TotalOrderFPSampler


class TestFloatingPointSampler:
    """Test cases for FloatingPointSampler."""

    def test_supports_correct_logic(self):
        sampler = FloatingPointSampler()

        assert sampler.supports_logic(Logic.QF_FP) is True
        assert sampler.supports_logic(Logic.QF_BV) is False

    def test_sample_simple_formula(self):
        sampler = FloatingPointSampler()
        x = z3.FP("x", z3.Float32())
        formula = cast(z3.ExprRef, x == z3.FPVal(1.5, z3.Float32()))

        sampler.init_from_formula(formula)
        result = sampler.sample(SamplingOptions(num_samples=2))

        assert len(result) == 1
        assert result[0]["x"] == "1.5 [bits=0x3fc00000]"

    def test_render_mode_pretty(self):
        sampler = FloatingPointSampler()
        x = z3.FP("x", z3.Float32())
        formula = cast(z3.ExprRef, x == z3.FPVal(1.5, z3.Float32()))

        sampler.init_from_formula(formula)
        result = sampler.sample(SamplingOptions(num_samples=1, render_mode="pretty"))

        assert result[0]["x"] == "1.5"

    def test_render_mode_bits(self):
        sampler = FloatingPointSampler()
        x = z3.FP("x", z3.Float32())
        formula = cast(z3.ExprRef, x == z3.FPVal(1.5, z3.Float32()))

        sampler.init_from_formula(formula)
        result = sampler.sample(SamplingOptions(num_samples=1, render_mode="bits"))

        assert result[0]["x"] == "0x3fc00000"

    def test_invalid_render_mode_raises_error(self):
        sampler = FloatingPointSampler()
        x = z3.FP("x", z3.Float32())
        formula = cast(z3.ExprRef, x == z3.FPVal(1.5, z3.Float32()))

        sampler.init_from_formula(formula)
        with pytest.raises(ValueError, match="Unsupported FP render_mode"):
            sampler.sample(SamplingOptions(num_samples=1, render_mode="bogus"))

    def test_distinguishes_signed_zeroes(self):
        sampler = FloatingPointSampler()
        x = z3.FP("x", z3.Float32())
        pos_zero = z3.FPVal(0.0, z3.Float32())
        neg_zero = z3.fpNeg(pos_zero)
        formula = cast(z3.ExprRef, z3.Or(x == pos_zero, x == neg_zero))

        sampler.init_from_formula(formula)
        result = sampler.sample(SamplingOptions(num_samples=3))

        assert len(result) == 2
        assert {sample["x"] for sample in result} == {
            "+0.0 [bits=0x00000000]",
            "-0.0 [bits=0x80000000]",
        }

    def test_sample_unsatisfiable_formula(self):
        sampler = FloatingPointSampler()
        x = z3.FP("x", z3.Float32())
        formula = cast(
            z3.ExprRef,
            z3.And(
                x == z3.FPVal(1.0, z3.Float32()),
                x == z3.FPVal(2.0, z3.Float32()),
            ),
        )

        sampler.init_from_formula(formula)
        result = sampler.sample(SamplingOptions(num_samples=1))

        assert len(result) == 0
        assert result.success is False

    def test_reports_enumeration_method(self):
        sampler = FloatingPointSampler()

        assert sampler.get_supported_methods() == {SamplingMethod.ENUMERATION}


class TestHashBasedFPSampler:
    """Test cases for HashBasedFPSampler."""

    def test_supports_hash_method(self):
        sampler = HashBasedFPSampler()

        assert sampler.supports_logic(Logic.QF_FP) is True
        assert SamplingMethod.HASH_BASED in sampler.get_supported_methods()

    def test_sample_without_variables_raises_error(self):
        sampler = HashBasedFPSampler()
        formula = z3.BoolVal(True)
        sampler.init_from_formula(formula)

        with pytest.raises(ValueError, match="No floating-point variables"):
            sampler.sample(SamplingOptions(num_samples=1))

    def test_sample_simple_formula(self):
        sampler = HashBasedFPSampler()
        x = z3.FP("x", z3.Float32())
        formula = cast(z3.ExprRef, x == z3.FPVal(1.5, z3.Float32()))

        sampler.init_from_formula(formula)
        result = sampler.sample(SamplingOptions(num_samples=3, random_seed=3))

        assert len(result) == 1
        assert result[0]["x"] == "1.5 [bits=0x3fc00000]"


class TestTotalOrderFPSampler:
    """Test cases for TotalOrderFPSampler."""

    def test_supports_total_order_method(self):
        sampler = TotalOrderFPSampler()

        assert sampler.supports_logic(Logic.QF_FP) is True
        assert SamplingMethod.TOTAL_ORDER in sampler.get_supported_methods()

    def test_spreads_signed_zeroes_in_total_order(self):
        sampler = TotalOrderFPSampler()
        x = z3.FP("x", z3.Float32())
        pos_zero = z3.FPVal(0.0, z3.Float32())
        neg_zero = z3.fpNeg(pos_zero)
        pos_one = z3.FPVal(1.0, z3.Float32())
        neg_one = z3.fpNeg(pos_one)
        formula = cast(
            z3.ExprRef,
            z3.Or(x == neg_one, x == neg_zero, x == pos_zero, x == pos_one),
        )

        sampler.init_from_formula(formula)
        result = sampler.sample(
            SamplingOptions(num_samples=3, render_mode="bits", candidate_pool_size=8)
        )

        assert len(result) == 3
        assert result[0]["x"] == "0xbf800000"
        assert result[1]["x"] == "0x00000000"
        assert result[2]["x"] == "0x3f800000"


if __name__ == "__main__":
    pytest.main()
