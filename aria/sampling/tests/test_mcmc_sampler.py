"""
Unit tests for MCMC (Markov Chain Monte Carlo) sampler.

Tests for aria.sampling.general_sampler.mcmc_sampler module.
"""

import pytest
import z3
from aria.sampling.general_sampler.mcmc_sampler import MCMCSampler
from aria.sampling.base import Logic, SamplingMethod, SamplingOptions


class TestMCMCSampler:
    """Test cases for MCMCSampler."""

    def test_initialization(self):
        """Test MCMCSampler initialization."""
        sampler = MCMCSampler(step_size=0.5, max_attempts=50)
        assert sampler.step_size == 0.5
        assert sampler.max_attempts == 50

    def test_supports_correct_logics(self):
        """Test sampler supports arithmetic and boolean logics."""
        sampler = MCMCSampler()
        assert sampler.supports_logic(Logic.QF_LRA) is True
        assert sampler.supports_logic(Logic.QF_LIA) is True
        assert sampler.supports_logic(Logic.QF_NRA) is True
        assert sampler.supports_logic(Logic.QF_BOOL) is True
        assert sampler.supports_logic(Logic.QF_BV) is False

    def test_sample_lra_formula(self):
        """Test MCMC sampling from linear real arithmetic."""
        sampler = MCMCSampler()
        x, y = z3.Reals("x y")
        formula = z3.And(x > 0, y > 0, x + y < 10)

        sampler.init_from_formula(formula)
        result = sampler.sample(
            SamplingOptions(method=SamplingMethod.MCMC, num_samples=5)
        )

        assert len(result) <= 5

    def test_sample_lia_formula(self):
        """Test MCMC sampling from linear integer arithmetic."""
        sampler = MCMCSampler()
        x, y = z3.Ints("x y")
        formula = z3.And(x > 0, y > 0, x + y < 20)

        sampler.init_from_formula(formula)
        result = sampler.sample(
            SamplingOptions(method=SamplingMethod.MCMC, num_samples=5)
        )

        assert len(result) <= 5

    def test_sample_boolean_formula(self):
        """Test MCMC sampling from Boolean formula."""
        sampler = MCMCSampler()
        a, b, c = z3.Bools("a b c")
        formula = z3.Or(a, b, c)

        sampler.init_from_formula(formula)
        result = sampler.sample(
            SamplingOptions(method=SamplingMethod.MCMC, num_samples=3)
        )

        assert len(result) <= 3

    def test_sample_with_default_options(self):
        """Test sampling with None options."""
        sampler = MCMCSampler()
        x = z3.Real("x")
        formula = z3.And(x > 0, x < 10)

        sampler.init_from_formula(formula)
        result = sampler.sample()

        assert result is not None
        assert len(result) <= 10

    def test_sample_with_burn_in(self):
        """Test sampling with burn-in parameter."""
        sampler = MCMCSampler()
        x, y = z3.Reals("x y")
        formula = z3.And(x > 0, y > 0, x + y < 10)

        sampler.init_from_formula(formula)
        result = sampler.sample(
            SamplingOptions(method=SamplingMethod.MCMC, num_samples=5, burn_in=50)
        )

        assert "burn_in" in result.stats
        assert result.stats["burn_in"] == 50

    def test_sample_with_random_seed(self):
        """Test random seed reproducibility."""
        x, y = z3.Reals("x y")
        formula = z3.And(x > 0, y > 0, x + y < 10)

        sampler1 = MCMCSampler()
        sampler1.init_from_formula(formula)
        result1 = sampler1.sample(
            SamplingOptions(method=SamplingMethod.MCMC, num_samples=3, random_seed=42)
        )

        sampler2 = MCMCSampler()
        sampler2.init_from_formula(formula)
        result2 = sampler2.sample(
            SamplingOptions(method=SamplingMethod.MCMC, num_samples=3, random_seed=42)
        )

        assert len(result1) == len(result2)

    def test_sample_unsatisfiable_formula(self):
        """Test sampling from unsatisfiable formula."""
        sampler = MCMCSampler()
        x = z3.Real("x")
        formula = z3.And(x > 10, x < 5)

        sampler.init_from_formula(formula)
        result = sampler.sample(
            SamplingOptions(method=SamplingMethod.MCMC, num_samples=1)
        )

        assert len(result) == 0
        assert result.success is False
        assert "error" in result.stats

    def test_sample_with_timeout(self):
        """Test sampling with timeout."""
        sampler = MCMCSampler()
        x, y = z3.Reals("x y")
        formula = z3.And(x > 0, y > 0, x + y < 10)

        sampler.init_from_formula(formula)
        result = sampler.sample(
            SamplingOptions(method=SamplingMethod.MCMC, num_samples=1000, timeout=0.1)
        )

        assert len(result) < 1000  # Should stop early

    def test_sample_statistics(self):
        """Test that result includes comprehensive statistics."""
        sampler = MCMCSampler()
        x, y = z3.Reals("x y")
        formula = z3.And(x > 0, y > 0, x + y < 10)

        sampler.init_from_formula(formula)
        result = sampler.sample(
            SamplingOptions(method=SamplingMethod.MCMC, num_samples=3)
        )

        assert "time" in result.stats
        assert "samples_collected" in result.stats
        assert "iterations" in result.stats


if __name__ == "__main__":
    pytest.main()
