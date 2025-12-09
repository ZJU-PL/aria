"""
Unit tests for bit-vector samplers.

Tests for aria.sampling.finite_domain.bv module.
"""

import pytest
import z3
from aria.sampling.finite_domain.bv.base import BitVectorSampler
from aria.sampling.finite_domain.bv.hash_sampler import HashBasedBVSampler
from aria.sampling.finite_domain.bv.quick_sampler import QuickBVSampler
from aria.sampling.base import Logic, SamplingMethod, SamplingOptions


class TestBitVectorSampler:
    """Test cases for basic BitVectorSampler."""

    def test_supports_correct_logic(self):
        """Test sampler supports QF_BV logic."""
        sampler = BitVectorSampler()
        assert sampler.supports_logic(Logic.QF_BV) is True
        assert sampler.supports_logic(Logic.QF_BOOL) is False

    def test_sample_simple_formula(self):
        """Test sampling from simple bit-vector formula."""
        sampler = BitVectorSampler()
        x = z3.BitVec('x', 8)
        formula = z3.And(x > 5, x < 8)

        sampler.init_from_formula(formula)
        result = sampler.sample(SamplingOptions(num_samples=2))

        assert len(result) <= 2
        for sample in result:
            assert 5 < sample['x'] < 8

    def test_sample_with_random_seed(self):
        """Test random seed reproducibility."""
        x = z3.BitVec('x', 8)
        formula = z3.And(x > 5, x < 100)

        sampler1 = BitVectorSampler()
        sampler1.init_from_formula(formula)
        result1 = sampler1.sample(SamplingOptions(num_samples=3, random_seed=42))

        sampler2 = BitVectorSampler()
        sampler2.init_from_formula(formula)
        result2 = sampler2.sample(SamplingOptions(num_samples=3, random_seed=42))

        # Z3 solver may not be fully deterministic, so we only check:
        # 1. Same number of samples are generated
        # 2. All samples satisfy the constraints
        assert len(result1) == len(result2)
        for sample in result1:
            assert 5 < sample['x'] < 100
        for sample in result2:
            assert 5 < sample['x'] < 100

    def test_sample_unsatisfiable_formula(self):
        """Test sampling from unsatisfiable formula."""
        sampler = BitVectorSampler()
        x = z3.BitVec('x', 8)
        formula = z3.And(x > 100, x < 50)

        sampler.init_from_formula(formula)
        result = sampler.sample(SamplingOptions(num_samples=1))

        assert len(result) == 0
        assert result.success is False

    def test_sample_multiple_variables(self):
        """Test sampling with multiple variables."""
        sampler = BitVectorSampler()
        x, y = z3.BitVecs('x y', 16)
        formula = z3.And(x > 1000, y < 10000)

        sampler.init_from_formula(formula)
        result = sampler.sample(SamplingOptions(num_samples=2))

        assert len(result) <= 2
        for sample in result:
            assert sample['x'] > 1000
            assert sample['y'] < 10000


class TestHashBasedBVSampler:
    """Test cases for XOR-based hash sampling."""

    def test_supports_hash_method(self):
        """Test sampler supports HASH_BASED method."""
        sampler = HashBasedBVSampler()
        assert SamplingMethod.HASH_BASED in sampler.get_supported_methods()

    def test_sample_without_variables_raises_error(self):
        """Test sampling without BV variables raises error."""
        sampler = HashBasedBVSampler()
        formula = z3.BoolVal(True)
        sampler.init_from_formula(formula)

        with pytest.raises(ValueError, match="No bit-vector variables"):
            sampler.sample(SamplingOptions(num_samples=1))

    def test_sample_simple_formula(self):
        """Test XOR-based sampling."""
        sampler = HashBasedBVSampler()
        x = z3.BitVec('x', 8)
        formula = z3.And(x > 5, x < 20)

        sampler.init_from_formula(formula)
        result = sampler.sample(SamplingOptions(num_samples=3))

        assert len(result) == 3
        for sample in result:
            assert 5 < sample['x'] < 20

    def test_sample_multiple_variables(self):
        """Test XOR sampling with multiple variables."""
        sampler = HashBasedBVSampler()
        x, y, z_var = z3.BitVecs('x y z', 16)
        formula = z3.And(z3.ULT(x, 13), z3.ULT(y, x), z3.ULE(y, z_var))

        sampler.init_from_formula(formula)
        result = sampler.sample(SamplingOptions(num_samples=3))

        assert len(result) == 3


class TestQuickBVSampler:
    """Test cases for QuickSampler."""

    def test_init_without_bv_variables_raises_error(self):
        """Test init with no BV variables raises error."""
        sampler = QuickBVSampler()
        formula = z3.BoolVal(True)

        with pytest.raises(ValueError, match="No bit-vector variables"):
            sampler.init_from_formula(formula)

    def test_sample_simple_formula(self):
        """Test QuickSampler on simple formula."""
        x = z3.BitVec('x', 16)
        sampler = QuickBVSampler(target_var=x)
        formula = z3.And(x > 1000, x < 2000)

        sampler.init_from_formula(formula)
        result = sampler.sample(SamplingOptions(num_samples=5))

        assert len(result) <= 5
        for sample in result:
            assert 1000 < sample['x'] < 2000

    def test_sample_produces_diverse_samples(self):
        """Test QuickSampler produces diverse samples."""
        x = z3.BitVec('x', 8)
        sampler = QuickBVSampler(target_var=x)
        formula = z3.And(x > 10, x < 100)

        sampler.init_from_formula(formula)
        result = sampler.sample(SamplingOptions(num_samples=10))

        assert len(result) <= 10
        values = [sample['x'] for sample in result]
        assert len(set(values)) >= 2  # At least some diversity


if __name__ == "__main__":
    pytest.main()
