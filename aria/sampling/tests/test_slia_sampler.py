"""Unit tests for the QF_SLIA sampler."""

import z3

from aria.sampling.base import Logic, SamplingMethod, SamplingOptions
from aria.sampling.factory import create_sampler, sample_models_from_formula
from aria.sampling.slia import SLIASampler


class TestSLIASampler:
    """Test cases for SLIASampler."""

    def test_supports_correct_logic(self):
        sampler = SLIASampler()
        assert sampler.supports_logic(Logic.QF_SLIA) is True
        assert sampler.supports_logic(Logic.QF_LIA) is False

    def test_supported_methods(self):
        sampler = SLIASampler()
        assert sampler.get_supported_methods() == {
            SamplingMethod.ENUMERATION,
            SamplingMethod.SEARCH_TREE,
        }

    def test_factory_registration(self):
        sampler = create_sampler(Logic.QF_SLIA)
        assert isinstance(sampler, SLIASampler)

    def test_sample_models_from_formula(self):
        s = z3.String("s")
        x = z3.Int("x")
        formula = z3.And(
            z3.Or(
                s == z3.StringVal(""),
                s == z3.StringVal("a"),
                s == z3.StringVal("bb"),
            ),
            x == z3.Length(s),
        )

        result = sample_models_from_formula(
            formula, Logic.QF_SLIA, SamplingOptions(num_samples=10)
        )

        assert len(result) == 3
        assert {(sample["s"], sample["x"]) for sample in result} == {
            ("", 0),
            ("a", 1),
            ("bb", 2),
        }

    def test_projection_by_length_term_name(self):
        s = z3.String("s")
        formula = z3.Or(
            s == z3.StringVal(""),
            s == z3.StringVal("a"),
            s == z3.StringVal("bb"),
        )
        length_term = z3.Length(s)

        sampler = SLIASampler()
        sampler.init_from_formula(formula)
        result = sampler.sample(
            SamplingOptions(
                num_samples=10,
                projection_terms=[str(length_term)],
                tracked_terms=[length_term],
            )
        )

        assert len(result) == 3
        assert {sample[str(length_term)] for sample in result} == {0, 1, 2}

    def test_return_full_model_includes_observed_string_terms(self):
        s = z3.String("s")
        x = z3.Int("x")
        formula = z3.And(s == z3.StringVal("ab"), x == z3.Length(s))
        length_term = z3.Length(s)

        sampler = SLIASampler()
        sampler.init_from_formula(formula)
        result = sampler.sample(
            SamplingOptions(
                num_samples=2,
                projection_terms=[s],
                return_full_model=True,
            )
        )

        assert len(result) == 1
        assert result[0]["s"] == "ab"
        assert result[0]["x"] == 2
        assert result[0][str(length_term)] == 2

    def test_unsat_formula_returns_no_samples(self):
        s = z3.String("s")
        x = z3.Int("x")
        formula = z3.And(s == z3.StringVal("ab"), x == z3.Length(s), x == 3)

        sampler = SLIASampler()
        sampler.init_from_formula(formula)
        result = sampler.sample(SamplingOptions(num_samples=5))

        assert len(result) == 0
        assert result.success is False

    def test_search_tree_length_first_sampling(self):
        s = z3.String("s")
        x = z3.Int("x")
        formula = z3.And(
            x == z3.Length(s),
            z3.Or(
                s == z3.StringVal(""),
                z3.And(
                    x == 2,
                    z3.PrefixOf(z3.StringVal("a"), s),
                    z3.Contains(s, z3.StringVal("b")),
                ),
                z3.And(
                    x == 3,
                    z3.PrefixOf(z3.StringVal("a"), s),
                    z3.Contains(s, z3.StringVal("b")),
                ),
            ),
        )

        sampler = SLIASampler()
        sampler.init_from_formula(formula)
        result = sampler.sample(
            SamplingOptions(
                method=SamplingMethod.SEARCH_TREE,
                num_samples=3,
                max_length_shapes=4,
                candidates_per_length_shape=3,
                diversity_mode="coverage_guided",
            )
        )

        assert len(result) == 3
        assert {sample["x"] for sample in result} == {0, 2, 3}
        assert result.stats["length_shape_count"] == 3
        assert result.stats["candidate_count"] >= 3
        assert result.stats["shape_signature_count"] >= 2
        assert result.stats["coverage_ratio"] > 0.0

    def test_search_tree_returns_full_observed_model(self):
        s = z3.String("s")
        x = z3.Int("x")
        formula = z3.And(x == z3.Length(s), x == 2, z3.PrefixOf(z3.StringVal("a"), s))
        length_term = z3.Length(s)

        sampler = SLIASampler()
        sampler.init_from_formula(formula)
        result = sampler.sample(
            SamplingOptions(
                method=SamplingMethod.SEARCH_TREE,
                num_samples=2,
                max_length_shapes=2,
                candidates_per_length_shape=2,
                return_full_model=True,
            )
        )

        assert len(result) >= 1
        assert all("s" in sample for sample in result)
        assert all("x" in sample for sample in result)
        assert all(str(length_term) in sample for sample in result)

    def test_shrinking_finds_small_string_witness(self):
        s = z3.String("s")
        formula = z3.And(
            z3.Length(s) == 3,
            z3.Contains(s, z3.StringVal("b")),
            z3.Contains(s, z3.StringVal("c")),
            s != z3.StringVal("abc"),
        )

        sampler = SLIASampler()
        sampler.init_from_formula(formula)
        result = sampler.sample(
            SamplingOptions(
                num_samples=1,
                shrink_samples=True,
            )
        )

        assert len(result) == 1
        assert result[0]["s"] == "acb"
        assert result.stats["shrink_passes"] == 1
        assert result.stats["shrink_solver_checks"] == 1

    def test_search_tree_tracks_regex_branch_coverage(self):
        s = z3.String("s")
        x = z3.Int("x")
        regex = z3.Union(
            z3.Concat(z3.Re("a"), z3.Star(z3.Range("0", "9"))),
            z3.Concat(z3.Re("z"), z3.Star(z3.Range("0", "9"))),
        )
        formula = z3.And(
            x == z3.Length(s),
            x == 2,
            z3.InRe(s, regex),
        )

        sampler = SLIASampler()
        sampler.init_from_formula(formula)
        result = sampler.sample(
            SamplingOptions(
                method=SamplingMethod.SEARCH_TREE,
                num_samples=2,
                max_length_shapes=1,
                candidates_per_length_shape=12,
                diversity_mode="coverage_guided",
            )
        )

        assert len(result) == 2
        assert {sample["s"][0] for sample in result} == {"a", "z"}
        assert result.stats["regex_atom_count"] == 1
        assert result.stats["regex_branch_coverage_ratio"] > 0.0

    def test_search_tree_boundary_focus_prefers_near_unsat_lengths(self):
        s = z3.String("s")
        x = z3.Int("x")
        formula = z3.And(
            x == z3.Length(s),
            x >= 1,
            x <= 4,
            z3.PrefixOf(z3.StringVal("a"), s),
        )

        sampler = SLIASampler()
        sampler.init_from_formula(formula)
        result = sampler.sample(
            SamplingOptions(
                method=SamplingMethod.SEARCH_TREE,
                num_samples=2,
                max_length_shapes=4,
                candidates_per_length_shape=2,
                diversity_mode="enumeration",
                boundary_focus=True,
                boundary_neighbor_radius=1,
            )
        )

        assert len(result) == 2
        assert {sample["x"] for sample in result} == {1, 4}
        assert result.stats["boundary_focus"] is True
        assert result.stats["selected_boundary_candidate_count"] == 2
