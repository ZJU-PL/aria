import pytest
import z3
from pysat.formula import CNF

from aria.prob import (
    CompiledWMC,
    UniformDensity,
    WMIOptions,
    compile_wmc,
    covariance,
    discrete_density,
    expectation,
    probability,
    variance,
    wmi_integrate,
)


def test_exact_discrete_factorized_density_mass_and_sampling():
    x = z3.Int("x")
    density = discrete_density({"x": {0: 0.1, 1: 0.3, 2: 0.6}})

    result = wmi_integrate(x >= 1, density)

    assert float(result) == pytest.approx(0.9, rel=1e-9)
    assert result.backend == "wmi-exact-discrete"


def test_discrete_factorized_density_validation():
    with pytest.raises(ValueError, match="sum to 1.0"):
        discrete_density({"x": {0: 0.2, 1: 0.2}})
    with pytest.raises(ValueError, match="non-negative"):
        discrete_density({"x": {0: -0.1, 1: 1.1}})
    with pytest.raises(ValueError, match="integer support"):
        discrete_density({"x": {0.5: 1.0}})


def test_exact_discrete_factorized_moments():
    x = z3.Int("x")
    density = discrete_density({"x": {0: 0.1, 1: 0.3, 2: 0.6}})
    support = z3.BoolVal(True)

    cov = covariance(x, x, support, density)
    var = variance(x, support, density)

    assert float(cov) == pytest.approx(0.45, rel=1e-9)
    assert float(var) == pytest.approx(0.45, rel=1e-9)
    assert cov.error_bound == pytest.approx(0.0, rel=1e-9)
    assert cov.stats["conditioning_probability"] == pytest.approx(1.0, rel=1e-9)


def test_exact_discrete_factorized_multivar_composite_expectation():
    x = z3.Int("x")
    y = z3.Int("y")
    density = discrete_density(
        {
            "x": {0: 0.25, 2: 0.75},
            "y": {1: 0.4, 3: 0.6},
        }
    )

    result = expectation(x + y, z3.And(x >= 0, y >= 1), density)

    assert float(result) == pytest.approx(3.7, rel=1e-9)
    assert result.stats["satisfying_assignment_count"] == 4
    assert result.stats["conditioning_mass"] == pytest.approx(1.0, rel=1e-9)


def test_sampling_moments_expose_shared_stats():
    x, y = z3.Reals("x y")
    density = UniformDensity({"x": (0, 1), "y": (0, 1)})
    formula = z3.And(x >= 0, x <= 1, y >= 0, y <= 1)
    options = WMIOptions(num_samples=4000, random_seed=9)

    cov = covariance(x, y, formula, density, options)

    assert cov.stats["sample_count"] == 4000
    assert "effective_conditioning_weight" in cov.stats
    assert "estimator_error_bounds" in cov.stats
    assert "moment_confidence_half_widths" in cov.stats
    assert "conditioning_mass_estimate" in cov.stats
    assert cov.error_bound is not None


def test_compiled_wmc_supports_cached_cnf_queries():
    base = CNF(from_clauses=[[1, 2], [-1, 2]])
    query = CNF(from_clauses=[[2]])
    evidence = CNF(from_clauses=[[1]])
    weights = {1: 0.4, -1: 0.6, 2: 0.7, -2: 0.3}

    compiled = compile_wmc(base, weights)
    result_one = compiled.probability_cnf(query, evidence)
    result_two = compiled.probability_cnf(query, evidence)
    expected = probability(query, weights, evidence=evidence)

    assert isinstance(compiled, CompiledWMC)
    assert float(result_one) == pytest.approx(float(expected), rel=1e-9)
    assert float(result_two) == pytest.approx(float(expected), rel=1e-9)
    assert not result_one.stats["numerator_cache_hit"]
    assert result_two.stats["numerator_cache_hit"]
    assert result_two.stats["cache_entry_count"] >= 2


def test_compiled_wmc_handles_empty_evidence_and_distinct_cnf_queries():
    base = CNF(from_clauses=[[1, 2]])
    query_one = CNF(from_clauses=[[1]])
    query_two = CNF(from_clauses=[[2]])
    weights = {1: 0.4, -1: 0.6, 2: 0.7, -2: 0.3}

    compiled = compile_wmc(base, weights)
    result_one = compiled.probability_cnf(query_one, CNF())
    result_two = compiled.probability_cnf(query_two)

    assert float(result_one) == pytest.approx(float(probability(query_one, weights)), rel=1e-9)
    assert float(result_two) == pytest.approx(float(probability(query_two, weights)), rel=1e-9)
    assert result_one.stats["evidence_num_clauses"] == 0
    assert result_two.stats["cache_entry_count"] >= 2


def test_compiled_wmc_rejects_unknown_cnf_variables():
    base = CNF(from_clauses=[[1]])
    query = CNF(from_clauses=[[2]])
    weights = {1: 0.4, -1: 0.6}

    compiled = compile_wmc(base, weights)
    with pytest.raises(ValueError, match="not present in the compiled CNF"):
        compiled.probability_cnf(query)


def test_top_level_probability_routes_compiled_cnf_queries():
    base = CNF(from_clauses=[[1, 2]])
    query = CNF(from_clauses=[[1]])
    weights = {1: 0.4, -1: 0.6, 2: 0.7, -2: 0.3}

    compiled = compile_wmc(base, weights)
    result = probability(query, compiled)

    assert float(result) == pytest.approx(float(probability(query, weights)), rel=1e-9)
