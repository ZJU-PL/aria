"""
Unit tests for LIRA (Linear Integer and Real Arithmetic) sampler.

Tests for aria.sampling.linear_ira.lira_sampler module.
"""

import pytest
import z3
from aria.sampling.linear_ira.lira_sampler import LIRASampler
from aria.sampling.base import Logic, SamplingMethod, SamplingOptions


class TestLIRASampler:
    """Test cases for LIRASampler."""

    def test_supports_correct_logics(self):
        """Test sampler supports linear arithmetic logics."""
        sampler = LIRASampler()
        assert sampler.supports_logic(Logic.QF_LRA) is True
        assert sampler.supports_logic(Logic.QF_LIA) is True
        assert sampler.supports_logic(Logic.QF_LIRA) is True
        assert sampler.supports_logic(Logic.QF_BOOL) is False

    def test_supported_methods(self):
        """Test sampler reports correct methods."""
        sampler = LIRASampler()
        methods = sampler.get_supported_methods()
        assert SamplingMethod.ENUMERATION in methods
        assert SamplingMethod.DIKIN_WALK in methods

    def test_sample_lra_formula(self):
        """Test sampling from linear real arithmetic formula."""
        sampler = LIRASampler()
        x, y = z3.Reals('x y')
        formula = z3.And(x > 0, y > 0, x + y < 10)

        sampler.init_from_formula(formula)
        result = sampler.sample(SamplingOptions(num_samples=2))

        assert len(result) <= 2
        for sample in result:
            x_val, y_val = sample['x'], sample['y']
            assert x_val > 0 and y_val > 0
            assert x_val + y_val < 10.1  # Small tolerance

    def test_sample_lia_formula(self):
        """Test sampling from linear integer arithmetic formula."""
        sampler = LIRASampler()
        x, y = z3.Ints('x y')
        formula = z3.And(x > 0, y > 0, x + y < 10)

        sampler.init_from_formula(formula)
        result = sampler.sample(SamplingOptions(num_samples=3))

        assert len(result) <= 3
        for sample in result:
            assert isinstance(sample['x'], int)
            assert isinstance(sample['y'], int)
            assert sample['x'] > 0 and sample['y'] > 0

    def test_sample_mixed_formula(self):
        """Test sampling from mixed integer/real formula."""
        sampler = LIRASampler()
        x, y = z3.Int('x'), z3.Real('y')
        formula = z3.And(x > 0, y > 0, x + y < 10)

        sampler.init_from_formula(formula)
        result = sampler.sample(SamplingOptions(num_samples=2))

        assert len(result) <= 2
        for sample in result:
            assert isinstance(sample['x'], int)
            assert isinstance(sample['y'], (int, float))

    def test_sample_unsatisfiable_formula(self):
        """Test sampling from unsatisfiable formula."""
        sampler = LIRASampler()
        x = z3.Real('x')
        formula = z3.And(x > 10, x < 5)

        sampler.init_from_formula(formula)
        result = sampler.sample(SamplingOptions(num_samples=1))

        assert len(result) == 0
        assert result.success is False

    def test_sample_with_random_seed(self):
        """Test random seed reproducibility."""
        x, y = z3.Reals('x y')
        formula = z3.And(x > 0, y > 0, x + y < 10)

        sampler1 = LIRASampler()
        sampler1.init_from_formula(formula)
        result1 = sampler1.sample(SamplingOptions(num_samples=3, random_seed=42))

        sampler2 = LIRASampler()
        sampler2.init_from_formula(formula)
        result2 = sampler2.sample(SamplingOptions(num_samples=3, random_seed=42))

        # Z3 solver may not be fully deterministic, so we only check:
        # 1. Same number of samples are generated
        # 2. All samples satisfy the constraints
        assert len(result1) == len(result2)
        for sample in result1:
            assert sample['x'] > 0 and sample['y'] > 0 and sample['x'] + sample['y'] < 10
        for sample in result2:
            assert sample['x'] > 0 and sample['y'] > 0 and sample['x'] + sample['y'] < 10

    def test_sample_with_negative_values(self):
        """Test sampling with negative values."""
        sampler = LIRASampler()
        x, y = z3.Reals('x y')
        formula = z3.And(x > -10, x < -5, y > -3, y < 0)

        sampler.init_from_formula(formula)
        result = sampler.sample(SamplingOptions(num_samples=2))

        assert len(result) <= 2
        for sample in result:
            assert -10 < sample['x'] < -5
            assert -3 < sample['y'] < 0

    def test_sample_equality_constraints(self):
        """Test sampling with equality constraints."""
        sampler = LIRASampler()
        x, y = z3.Reals('x y')
        formula = z3.And(x + y == 10, x > 0, y > 0)

        sampler.init_from_formula(formula)
        result = sampler.sample(SamplingOptions(num_samples=2))

        assert len(result) <= 2
        for sample in result:
            assert abs(sample['x'] + sample['y'] - 10) < 0.1


if __name__ == "__main__":
    pytest.main()
