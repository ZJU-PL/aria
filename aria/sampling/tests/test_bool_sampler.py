"""
Unit tests for Boolean sampler.

Tests for aria.sampling.finite_domain.bool.base module.
"""

import pytest
import z3
from aria.sampling.finite_domain.bool.base import BooleanSampler
from aria.sampling.base import Logic, SamplingMethod, SamplingOptions


class TestBooleanSampler:
    """Test cases for BooleanSampler class."""

    def test_supports_correct_logic(self):
        """Test sampler supports QF_BOOL logic."""
        sampler = BooleanSampler()
        assert sampler.supports_logic(Logic.QF_BOOL) is True
        assert sampler.supports_logic(Logic.QF_BV) is False

    def test_sample_without_init_raises_error(self):
        """Test sampling without initialization raises error."""
        sampler = BooleanSampler()
        with pytest.raises(ValueError, match="not initialized"):
            sampler.sample(SamplingOptions(num_samples=1))

    def test_sample_simple_formula(self):
        """Test sampling from simple Boolean formula."""
        sampler = BooleanSampler()
        a, b = z3.Bools('a b')
        formula = z3.And(a, z3.Not(b))

        sampler.init_from_formula(formula)
        result = sampler.sample(SamplingOptions(num_samples=1))

        assert len(result) == 1
        assert result[0]['a'] is True
        assert result[0]['b'] is False

    def test_sample_multiple_models(self):
        """Test sampling multiple models."""
        sampler = BooleanSampler()
        a, b = z3.Bools('a b')
        formula = z3.Or(a, b)

        sampler.init_from_formula(formula)
        result = sampler.sample(SamplingOptions(num_samples=3))

        assert len(result) <= 3
        assert result.success is True

    def test_sample_with_random_seed(self):
        """Test random seed reproducibility."""
        a, b, c = z3.Bools('a b c')
        formula = z3.Or(a, b, c)

        sampler1 = BooleanSampler()
        sampler1.init_from_formula(formula)
        result1 = sampler1.sample(SamplingOptions(num_samples=3, random_seed=42))

        sampler2 = BooleanSampler()
        sampler2.init_from_formula(formula)
        result2 = sampler2.sample(SamplingOptions(num_samples=3, random_seed=42))

        assert len(result1) == len(result2)
        for s1, s2 in zip(result1, result2):
            assert s1 == s2

    def test_sample_unsatisfiable_formula(self):
        """Test sampling from unsatisfiable formula."""
        sampler = BooleanSampler()
        a = z3.Bool('a')
        formula = z3.And(a, z3.Not(a))

        sampler.init_from_formula(formula)
        result = sampler.sample(SamplingOptions(num_samples=1))

        assert len(result) == 0
        assert result.success is False

    def test_sample_blocking_prevents_duplicates(self):
        """Test blocking clauses prevent duplicate samples."""
        sampler = BooleanSampler()
        a, b = z3.Bools('a b')
        formula = z3.Or(a, b)

        sampler.init_from_formula(formula)
        result = sampler.sample(SamplingOptions(num_samples=5))

        # Should get at most 3 unique solutions
        assert len(result) <= 3

        # All samples should be unique
        samples_as_tuples = [tuple(sorted(sample.items())) for sample in result]
        assert len(samples_as_tuples) == len(set(samples_as_tuples))

    def test_sample_complex_formula(self):
        """Test sampling from complex Boolean formula."""
        sampler = BooleanSampler()
        a, b, c, d = z3.Bools('a b c d')
        formula = z3.And(z3.Or(a, b), z3.Or(c, d), z3.Or(z3.Not(a), z3.Not(c)))

        sampler.init_from_formula(formula)
        result = sampler.sample(SamplingOptions(num_samples=5))

        assert len(result) > 0
        # Verify samples satisfy formula
        solver = z3.Solver()
        solver.add(formula)
        for sample in result:
            solver.push()
            for var_name, value in sample.items():
                var = z3.Bool(var_name)
                solver.add(var if value else z3.Not(var))
            assert solver.check() == z3.sat
            solver.pop()


if __name__ == "__main__":
    pytest.main()
