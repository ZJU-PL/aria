import pytest
import z3

from aria.prob.core.density import GaussianDensity, UniformDensity

from ._config import WMIMethod, WMIOptions
from ._dispatch import WMI_BACKENDS
from ._exact_backend import _exact_discrete_expectation, _exact_discrete_mass, _exact_discrete_solver
from .moments import covariance, moment
from ._sampling_backends import _bounded_support_monte_carlo, _importance_sampling
from ._sampling_utils import (
    _running_error_bound,
    _uniform_sample_from_support,
    _uniform_support_measure,
)
from ._selection import _effective_method, _validate_wmi_inputs
from .factories import (
    beta_density,
    exponential_density,
    gaussian_density,
    uniform_density,
)


def test_effective_method_selection():
    x = z3.Real("x")
    y = z3.Int("y")

    assert (
        _effective_method(
            UniformDensity({"y": (0, 2)}, discrete=True),
            WMIOptions(),
            [y],
        )
        == WMIMethod.EXACT_DISCRETE
    )
    assert (
        _effective_method(
            UniformDensity({"x": (0, 1)}),
            WMIOptions(),
            [x],
        )
        == WMIMethod.BOUNDED_SUPPORT_MONTE_CARLO
    )
    assert set(WMI_BACKENDS.keys()) == {
        WMIMethod.EXACT_DISCRETE,
        WMIMethod.BOUNDED_SUPPORT_MONTE_CARLO,
        WMIMethod.IMPORTANCE_SAMPLING,
    }
    assert (
        _effective_method(
            GaussianDensity({"x": 0.0}, {"x": {"x": 1.0}}),
            WMIOptions(),
            [x],
        )
        == WMIMethod.IMPORTANCE_SAMPLING
    )
    assert (
        _effective_method(
            UniformDensity({"x": (0, 1)}),
            WMIOptions(method="region"),
            [x],
        )
        == WMIMethod.BOUNDED_SUPPORT_MONTE_CARLO
    )
    assert (
        _effective_method(
            UniformDensity({"x": (0, 1)}),
            WMIOptions(method="sampling"),
            [x],
        )
        == WMIMethod.BOUNDED_SUPPORT_MONTE_CARLO
    )


def test_validate_wmi_inputs_rejects_unsupported_sorts():
    b = z3.Bool("b")
    with pytest.raises(ValueError, match="supports only Int/Real variables"):
        _validate_wmi_inputs(b, UniformDensity({}))


def test_exact_discrete_backend():
    x = z3.Int("x")
    density = UniformDensity({"x": (0, 2)}, discrete=True)

    result = _exact_discrete_mass(x < 2, density, [x])
    assert result.exact
    assert result.backend == "wmi-exact-discrete-uniform"
    assert float(result) == pytest.approx(2.0 / 3.0, rel=1e-9)

    solver = _exact_discrete_solver(x < 2, density, [x])
    assert solver.check() == z3.sat

    expectation = _exact_discrete_expectation(x, z3.And(x >= 0, x <= 2), density, [x])
    assert expectation.exact
    assert float(expectation) == pytest.approx(1.0, rel=1e-9)


def test_bounded_support_backend():
    x = z3.Real("x")
    density = UniformDensity({"x": (0, 1)})

    result = _bounded_support_monte_carlo(
        x <= 0.5,
        density,
        WMIOptions(method=WMIMethod.BOUNDED_SUPPORT_MONTE_CARLO, num_samples=6000, random_seed=5),
        [x],
    )
    assert not result.exact
    assert result.backend == "wmi-bounded-support-monte-carlo"
    assert float(result) == pytest.approx(0.5, abs=0.05)
    assert result.error_bound is not None


def test_importance_sampling_backend():
    x = z3.Real("x")
    density = GaussianDensity({"x": 0.0}, {"x": {"x": 1.0}})

    result = _importance_sampling(
        z3.BoolVal(True),
        density,
        WMIOptions(method=WMIMethod.IMPORTANCE_SAMPLING, num_samples=2000, random_seed=2),
        [x],
    )
    assert not result.exact
    assert result.backend == "wmi-importance-sampling"
    assert float(result) == pytest.approx(1.0, rel=1e-9)
    assert result.error_bound == pytest.approx(0.0, rel=1e-9)


def test_sampling_utilities():
    x = z3.Real("x")
    y = z3.Int("y")
    bounds = {"x": (0.0, 1.0), "y": (1, 3)}

    sample = _uniform_sample_from_support([x, y], bounds, __import__("random").Random(7))
    assert 0.0 <= sample["x"] <= 1.0
    assert 1 <= sample["y"] <= 3
    assert _uniform_support_measure([x, y], bounds) == pytest.approx(3.0, rel=1e-9)
    assert _running_error_bound(3, 2.0, 2.0, 1.0, 1.96) is not None


def test_factories_construct_expected_density_types():
    assert isinstance(uniform_density({"x": (0, 1)}), UniformDensity)
    assert isinstance(
        gaussian_density({"x": 0.0}, {"x": {"x": 1.0}}),
        GaussianDensity,
    )
    assert exponential_density({"x": 1.0})
    assert beta_density({"x": 2.0}, {"x": 3.0})


def test_moment_and_covariance_helpers():
    x = z3.Int("x")
    density = UniformDensity({"x": (0, 2)}, discrete=True)
    support = z3.And(x >= 0, x <= 2)

    first = moment(x, 1, support, density)
    second = moment(x, 2, support, density)
    cov = covariance(x, x, support, density)

    assert float(first) == pytest.approx(1.0, rel=1e-9)
    assert float(second) == pytest.approx(5.0 / 3.0, rel=1e-9)
    assert float(cov) == pytest.approx(2.0 / 3.0, rel=1e-9)


def test_moment_rejects_invalid_orders():
    x = z3.Real("x")
    density = UniformDensity({"x": (0, 1)})
    support = z3.And(x >= 0, x <= 1)

    with pytest.raises(ValueError, match="positive integer"):
        moment(x, 0, support, density)
    with pytest.raises(ValueError, match="positive integer"):
        moment(x, -1, support, density)
    with pytest.raises(ValueError, match="positive integer"):
        moment(x, 1.5, support, density)
