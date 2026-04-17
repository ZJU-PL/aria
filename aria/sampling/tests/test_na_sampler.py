"""Unit tests for nonlinear arithmetic sampling."""

import z3

from aria.sampling.base import Logic, SamplingMethod, SamplingOptions
from aria.sampling.factory import create_sampler, sample_models_from_formula
from aria.sampling.nonlinear_ira import NASampler


class TestNASampler:
    """Tests for QF_NRA/QF_NIA sampling support."""

    def test_supports_correct_logics(self):
        sampler = NASampler()
        assert sampler.supports_logic(Logic.QF_NRA) is True
        assert sampler.supports_logic(Logic.QF_NIA) is True
        assert sampler.supports_logic(Logic.QF_LRA) is False

    def test_supported_methods(self):
        sampler = NASampler()
        methods = sampler.get_supported_methods()
        assert SamplingMethod.ENUMERATION in methods
        assert SamplingMethod.SEARCH_TREE in methods

    def test_enumerates_nia_models(self):
        x, y = z3.Ints("x y")
        formula = z3.And(x * x == 1, y == x + 1)

        result = sample_models_from_formula(
            formula,
            Logic.QF_NIA,
            SamplingOptions(num_samples=5),
        )

        assert len(result) == 2
        seen = {(sample["x"], sample["y"]) for sample in result}
        assert seen == {(-1, 0), (1, 2)}

    def test_enumeration_handles_unsat_formula(self):
        x = z3.Int("x")
        formula = z3.And(x * x == 2, x == 0)

        result = sample_models_from_formula(
            formula,
            Logic.QF_NIA,
            SamplingOptions(num_samples=2),
        )

        assert len(result) == 0
        assert result.success is False

    def test_search_tree_sampling_returns_feasible_nia_model(self):
        x = z3.Int("x")
        y = z3.Int("y")
        formula = z3.And(x * y == 6, x > 0, y > 0, x <= 3, y <= 6)

        result = sample_models_from_formula(
            formula,
            Logic.QF_NIA,
            SamplingOptions(
                method=SamplingMethod.SEARCH_TREE,
                num_samples=1,
                search_tree_width=3,
            ),
        )

        assert len(result) == 1
        sample = result[0]
        assert sample["x"] * sample["y"] == 6
        assert sample["x"] > 0
        assert sample["y"] > 0

    def test_factory_creates_na_sampler_for_nra_search_tree(self):
        sampler = create_sampler(Logic.QF_NRA, SamplingMethod.SEARCH_TREE)
        assert isinstance(sampler, NASampler)

    def test_enumerates_nra_model_with_real_output(self):
        x = z3.Real("x")
        formula = z3.And(x * x == 2, x > 0)

        result = sample_models_from_formula(
            formula,
            Logic.QF_NRA,
            SamplingOptions(num_samples=1),
        )

        assert len(result) == 1
        assert isinstance(result[0]["x"], str)
        assert "1.414" in result[0]["x"]
