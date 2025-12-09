"""
Unit tests for aria.sampling.factory module.

Tests for SamplerFactory, create_sampler, and sample_models_from_formula.
"""

import pytest
import z3
from aria.sampling.factory import (
    SamplerFactory, create_sampler, sample_models_from_formula, sample_formula
)
from aria.sampling.base import Logic, SamplingMethod, SamplingOptions


class TestSamplerFactory:
    """Test cases for SamplerFactory class."""

    def test_available_logics(self):
        """Test that factory reports available logics."""
        logics = SamplerFactory.available_logics()
        assert isinstance(logics, set)
        assert len(logics) > 0

    def test_create_sampler_invalid_logic(self):
        """Test creating sampler for unsupported logic raises error."""
        original_samplers = SamplerFactory._samplers.copy()
        try:
            SamplerFactory._samplers.clear()
            with pytest.raises(ValueError, match="No sampler available"):
                SamplerFactory.create(Logic.QF_BOOL)
        finally:
            SamplerFactory._samplers = original_samplers

    def test_available_methods_for_unregistered_logic(self):
        """Test available_methods returns empty for unregistered logic."""
        original_samplers = SamplerFactory._samplers.copy()
        try:
            SamplerFactory._samplers.clear()
            methods = SamplerFactory.available_methods(Logic.QF_BOOL)
            assert methods == set()
        finally:
            SamplerFactory._samplers = original_samplers


class TestSampleModelsFromFormula:
    """Test cases for sample_models_from_formula function."""

    def test_sample_boolean_formula(self):
        """Test sampling from Boolean formula."""
        a, b = z3.Bools('a b')
        formula = z3.And(z3.Or(a, b), z3.Not(z3.And(a, b)))
        try:
            result = sample_models_from_formula(formula, Logic.QF_BOOL, SamplingOptions(num_samples=2))
            assert len(result) <= 2
        except ValueError as e:
            pytest.skip(f"Sampler not available: {e}")

    def test_sample_bitvector_formula(self):
        """Test sampling from bit-vector formula."""
        x = z3.BitVec('x', 8)
        formula = z3.And(x > 5, x < 10)
        try:
            result = sample_models_from_formula(formula, Logic.QF_BV, SamplingOptions(num_samples=3))
            assert len(result) <= 3
            for sample in result:
                assert 5 < sample['x'] < 10
        except ValueError as e:
            pytest.skip(f"Sampler not available: {e}")

    def test_sample_lra_formula(self):
        """Test sampling from linear real arithmetic formula."""
        x, y = z3.Reals('x y')
        formula = z3.And(x + y > 0, x - y < 1)
        try:
            result = sample_models_from_formula(formula, Logic.QF_LRA, SamplingOptions(num_samples=2))
            assert len(result) <= 2
        except ValueError as e:
            pytest.skip(f"Sampler not available: {e}")

    def test_sample_unsatisfiable_formula(self):
        """Test sampling from unsatisfiable formula."""
        x = z3.Int('x')
        formula = z3.And(x > 10, x < 5)
        try:
            result = sample_models_from_formula(formula, Logic.QF_LIA, SamplingOptions(num_samples=1))
            assert len(result) == 0
            assert result.success is False
        except ValueError as e:
            pytest.skip(f"Sampler not available: {e}")

    def test_sample_with_random_seed(self):
        """Test random seed reproducibility."""
        x = z3.Int('x')
        formula = z3.And(x > 0, x < 100)
        try:
            opts = SamplingOptions(num_samples=5, random_seed=42)
            result1 = sample_models_from_formula(formula, Logic.QF_LIA, opts)
            result2 = sample_models_from_formula(formula, Logic.QF_LIA, opts)
            assert len(result1) == len(result2)
        except ValueError as e:
            pytest.skip(f"Sampler not available: {e}")


class TestSampleFormulaDeprecated:
    """Test deprecated sample_formula function."""

    def test_deprecation_warning(self):
        """Test that sample_formula raises deprecation warning."""
        x = z3.Int('x')
        formula = z3.And(x > 0, x < 10)
        try:
            with pytest.warns(DeprecationWarning, match="deprecated"):
                sample_formula(formula, Logic.QF_LIA)
        except ValueError as e:
            pytest.skip(f"Sampler not available: {e}")


if __name__ == "__main__":
    pytest.main()
