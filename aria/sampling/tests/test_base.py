"""
Unit tests for aria.sampling.base module.

Tests for base classes including Logic, SamplingMethod, SamplingOptions, and SamplingResult.
"""

import pytest
from aria.sampling.base import Logic, SamplingMethod, SamplingOptions, SamplingResult


class TestLogic:
    """Test cases for Logic enum."""

    def test_logic_enum_values(self):
        """Test that all logic enum values exist."""
        assert Logic.QF_BOOL.value == "QF_BOOL"
        assert Logic.QF_BV.value == "QF_BV"
        assert Logic.QF_LRA.value == "QF_LRA"
        assert Logic.QF_LIRA.value == "QF_LIRA"

    def test_logic_enum_membership(self):
        """Test that logic enums work in sets."""
        supported = {Logic.QF_BOOL, Logic.QF_BV}
        assert Logic.QF_BOOL in supported
        assert Logic.QF_LRA not in supported


class TestSamplingMethod:
    """Test cases for SamplingMethod enum."""

    def test_sampling_method_values(self):
        """Test sampling method enum values."""
        assert SamplingMethod.ENUMERATION.value == "enumeration"
        assert SamplingMethod.MCMC.value == "mcmc"
        assert SamplingMethod.HASH_BASED.value == "hash_based"


class TestSamplingOptions:
    """Test cases for SamplingOptions class."""

    def test_default_values(self):
        """Test default values."""
        options = SamplingOptions()
        assert options.method == SamplingMethod.ENUMERATION
        assert options.num_samples == 1
        assert options.timeout is None

    def test_with_parameters(self):
        """Test with custom parameters."""
        options = SamplingOptions(
            method=SamplingMethod.MCMC,
            num_samples=10,
            timeout=30.0,
            random_seed=42
        )
        assert options.method == SamplingMethod.MCMC
        assert options.num_samples == 10
        assert options.timeout == 30.0

    def test_with_kwargs(self):
        """Test additional kwargs."""
        options = SamplingOptions(num_samples=5, burn_in=100)
        assert options.additional_options["burn_in"] == 100


class TestSamplingResult:
    """Test cases for SamplingResult class."""

    def test_empty_result(self):
        """Test empty sampling result."""
        result = SamplingResult(samples=[])
        assert len(result) == 0
        assert result.success is False

    def test_with_samples(self):
        """Test result with samples."""
        samples = [{"x": 1, "y": 2}, {"x": 3, "y": 4}]
        result = SamplingResult(samples=samples)
        assert len(result) == 2
        assert result.success is True

    def test_indexing_and_iteration(self):
        """Test indexing and iteration."""
        samples = [{"x": 1}, {"x": 2}]
        result = SamplingResult(samples=samples)
        assert result[0] == {"x": 1}
        assert list(result) == samples


if __name__ == "__main__":
    pytest.main()
