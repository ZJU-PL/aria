"""
Probabilistic reasoning utilities.

This package provides:
- exact weighted model counting over Boolean CNF formulas
- explicit Monte Carlo / exact backends for arithmetic probability mass queries
- high-level helpers for probabilities, conditionals, expectations, and variance
"""

from .core import (
    Density,
    UniformDensity,
    GaussianDensity,
    ExponentialDensity,
    BetaDensity,
    ProductDensity,
    product_density,
)
from .api import probability, conditional_probability, expectation, variance
from .core import InferenceResult
from .boolean import (
    WMCBackend,
    WMCOptions,
    CompiledWMC,
    compile_wmc,
    wmc_count,
)
from .arithmetic import (
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
    "variance",
    "uniform_density",
    "gaussian_density",
    "exponential_density",
    "beta_density",
]
