"""
Probabilistic reasoning utilities.

This package provides:
- exact weighted model counting over Boolean CNF formulas
- explicit Monte Carlo / exact backends for arithmetic probability mass queries
- high-level helpers for probabilities, conditionals, moments, and variance
"""

from .api.query import conditional_probability, probability
from .arithmetic._config import WMIMethod, WMIOptions
from .arithmetic.factories import (
    beta_density,
    discrete_density,
    exponential_density,
    gaussian_density,
    uniform_density,
)
from .arithmetic.moments import covariance, expectation, moment, variance
from .arithmetic.wmi import wmi_integrate
from .boolean.base import WMCBackend, WMCOptions
from .boolean.wmc import CompiledWMC, compile_wmc, wmc_count
from .core.density import (
    Density,
    UniformDensity,
    GaussianDensity,
    ExponentialDensity,
    BetaDensity,
    DiscreteFactorizedDensity,
    ProductDensity,
    product_density,
)
from .core.results import InferenceResult

__all__ = [
    "InferenceResult",
    "Density",
    "UniformDensity",
    "GaussianDensity",
    "ExponentialDensity",
    "BetaDensity",
    "DiscreteFactorizedDensity",
    "ProductDensity",
    "product_density",
    "WMCBackend",
    "WMCOptions",
    "CompiledWMC",
    "compile_wmc",
    "wmc_count",
    "WMIMethod",
    "WMIOptions",
    "wmi_integrate",
    "probability",
    "conditional_probability",
    "moment",
    "expectation",
    "covariance",
    "variance",
    "uniform_density",
    "gaussian_density",
    "exponential_density",
    "beta_density",
    "discrete_density",
]
