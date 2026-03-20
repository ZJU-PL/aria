"""
Probabilistic reasoning utilities.

This package provides:
- exact weighted model counting over Boolean CNF formulas
- explicit Monte Carlo / exact backends for arithmetic probability mass queries
- high-level helpers for probabilities, conditionals, and expectations
"""

from .density import (
    Density,
    UniformDensity,
    GaussianDensity,
    ExponentialDensity,
    BetaDensity,
    ProductDensity,
    product_density,
)
from .query import probability, conditional_probability, expectation
from .results import InferenceResult
from .wmc import (
    WMCBackend,
    WMCOptions,
    CompiledWMC,
    compile_wmc,
    wmc_count,
    WMIMethod,
    WMIOptions,
    wmi_integrate,
    uniform_density,
    gaussian_density,
    exponential_density,
    beta_density,
)

__all__ = [
    "InferenceResult",
    "Density",
    "UniformDensity",
    "GaussianDensity",
    "ExponentialDensity",
    "BetaDensity",
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
    "expectation",
    "uniform_density",
    "gaussian_density",
    "exponential_density",
    "beta_density",
]
