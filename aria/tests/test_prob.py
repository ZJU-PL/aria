import math

import pytest
import z3
from pysat.formula import CNF

from aria.prob import (
    BetaDensity,
    CompiledWMC,
    ExponentialDensity,
    GaussianDensity,
    InferenceResult,
    UniformDensity,
    WMCOptions,
    WMIMethod,
    WMIOptions,
    compile_wmc,
    conditional_probability,
    expectation,
    probability,
    wmc_count,
    wmi_integrate,
)


def test_public_imports_and_result_formatting():
    result = InferenceResult(value=0.25, exact=True, backend="test")
    assert format(result, ".2f") == "0.25"
    assert float(result) == pytest.approx(0.25)

    assert UniformDensity
    assert GaussianDensity
    assert ExponentialDensity
    assert BetaDensity
    assert CompiledWMC


def test_wmc_exact_count_and_compiled_queries():
    cnf = CNF(from_clauses=[[1, 2]])
    weights = {1: 0.2, -1: 0.8, 2: 0.3, -2: 0.7}

    total = wmc_count(cnf, weights)
    assert total == pytest.approx(0.44, rel=1e-9)

    compiled = compile_wmc(cnf, weights, WMCOptions(strict_complements=True))
    assert compiled.backend == "wmc-dnnf"
    marginal_x = compiled.probability(query=[1])
    assert marginal_x.exact
    assert float(marginal_x) == pytest.approx(0.2 / 0.44, rel=1e-9)

    conditioned_y = compiled.probability(query=[2], evidence=[-1])
    assert conditioned_y.exact
    assert float(conditioned_y) == pytest.approx(1.0, rel=1e-9)


def test_wmc_unit_clause_exact_backend_and_evidence_queries():
    cnf = CNF(from_clauses=[[1]])
    weights = {1: 0.4, -1: 0.6}

    compiled = compile_wmc(cnf, weights, WMCOptions(strict_complements=True))
    assert compiled.backend == "wmc-dnnf"
    assert compiled.count() == pytest.approx(0.4, rel=1e-9)
    assert compiled.count([1]) == pytest.approx(0.4, rel=1e-9)
    assert compiled.count([-1]) == pytest.approx(0.0, rel=1e-9)
    assert float(compiled.probability(query=[1])) == pytest.approx(1.0, rel=1e-9)


def test_top_level_cnf_probability_is_strict_about_complements():
    cnf = CNF(from_clauses=[[1, 2]])
    weights = {1: 0.2, -1: 0.8, 2: 0.3, -2: 0.7}
    result = probability(cnf, weights)
    assert result.exact
    assert float(result) == pytest.approx(0.44, rel=1e-9)

    with pytest.raises(ValueError, match="must sum to 1.0"):
        probability(cnf, {1: 0.2, -1: 0.2, 2: 0.3, -2: 0.7})


def test_exact_discrete_uniform_wmi_and_expectation():
    x = z3.Int("x")
    density = UniformDensity({"x": (0, 2)}, discrete=True)

    mass = wmi_integrate(x < 2, density)
    assert mass.exact
    assert mass.backend == "wmi-exact-discrete-uniform"
    assert float(mass) == pytest.approx(2.0 / 3.0, rel=1e-9)

    exp = expectation(x, z3.And(x >= 0, x <= 2), density)
    assert exp.exact
    assert float(exp) == pytest.approx(1.0, rel=1e-9)


def test_bounded_support_probability_and_expectation():
    x, y = z3.Reals("x y")
    triangle = z3.And(x >= 0, y >= 0, x <= 1, y <= 1, x + y <= 1)
    density = UniformDensity({"x": (0, 1), "y": (0, 1)})
    options = WMIOptions(
        method=WMIMethod.BOUNDED_SUPPORT_MONTE_CARLO,
        num_samples=6000,
        random_seed=7,
    )

    mass = wmi_integrate(triangle, density, options)
    assert not mass.exact
    assert mass.backend == "wmi-bounded-support-monte-carlo"
    assert float(mass) == pytest.approx(0.5, abs=0.05)
    assert mass.error_bound is not None

    cond = conditional_probability(triangle, x <= 0.5, density, options)
    assert float(cond) == pytest.approx(0.75, abs=0.07)

    exp = expectation(x, triangle, density, options)
    assert float(exp) == pytest.approx(1.0 / 3.0, abs=0.07)


def test_legacy_method_aliases_still_work():
    x, y = z3.Reals("x y")
    density = UniformDensity({"x": (0, 1), "y": (0, 1)})
    formula = z3.And(x >= 0, y >= 0, x + y <= 1)

    region = wmi_integrate(
        formula, density, WMIOptions(method="region", num_samples=4000, random_seed=1)
    )
    sampling = wmi_integrate(
        formula,
        density,
        WMIOptions(method="sampling", num_samples=4000, random_seed=1),
    )

    assert float(region) == pytest.approx(0.5, abs=0.06)
    assert float(sampling) == pytest.approx(0.5, abs=0.06)


def test_gaussian_requires_diagonal_covariance():
    with pytest.raises(ValueError, match="diagonal covariance"):
        GaussianDensity({"x": 0.0, "y": 0.0}, {"x": {"x": 1.0, "y": 0.1}, "y": {"y": 1.0}})
