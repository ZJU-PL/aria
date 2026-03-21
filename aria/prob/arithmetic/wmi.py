"""
Weighted model integration with explicit exact and Monte Carlo backends.
"""

from __future__ import annotations

from typing import Optional

import z3

from aria.prob.core.density import (
    BetaDensity,
    Density,
    DiscreteFactorizedDensity,
    ExponentialDensity,
    GaussianDensity,
    ProductDensity,
    UniformDensity,
    product_density,
)
from aria.prob.core.results import InferenceResult
from ._config import WMIMethod, WMIOptions
from ._dispatch import WMI_BACKENDS
from ._selection import _effective_method, _validate_wmi_inputs, _validate_wmi_options
from .factories import (
    beta_density as _beta_density_factory,
    discrete_density as _discrete_density_factory,
    exponential_density as _exponential_density_factory,
    gaussian_density as _gaussian_density_factory,
    uniform_density as _uniform_density_factory,
)


def wmi_integrate(
    formula: z3.ExprRef, density: Density, options: Optional[WMIOptions] = None
) -> InferenceResult:
    """
    Compute the probability mass of a formula under a normalized density.
    """

    opts = options or WMIOptions()
    variables = _validate_wmi_inputs(formula, density)
    _validate_wmi_options(opts, density, variables)
    method = _effective_method(density, opts, variables)

    backend = WMI_BACKENDS.get(method)
    if backend is None:
        raise ValueError("Unsupported WMI method: {}".format(method))
    return backend(formula, density, opts, variables)


def uniform_density(bounds, discrete=False):
    return _uniform_density_factory(bounds, discrete=discrete)


def gaussian_density(means, covariances):
    return _gaussian_density_factory(means, covariances)


def exponential_density(rates):
    return _exponential_density_factory(rates)


def beta_density(alphas, betas):
    return _beta_density_factory(alphas, betas)


def discrete_density(pmfs):
    return _discrete_density_factory(pmfs)


__all__ = [
    "Density",
    "UniformDensity",
    "GaussianDensity",
    "ExponentialDensity",
    "BetaDensity",
    "DiscreteFactorizedDensity",
    "ProductDensity",
    "product_density",
    "WMIMethod",
    "WMIOptions",
    "wmi_integrate",
    "uniform_density",
    "gaussian_density",
    "exponential_density",
    "beta_density",
    "discrete_density",
]
