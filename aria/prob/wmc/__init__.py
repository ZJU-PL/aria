"""
Probabilistic reasoning utilities.

Currently provides:
- Weighted Model Counting (WMC) over propositional CNF formulas with
  DNNF-based exact evaluation and SAT-based enumeration backend.
- Weighted Model Integration (WMI) over continuous LRA/LIA formulas with
  Monte Carlo integration and density function support.

Public API:
- wmc_count, WMCBackend, WMCOptions
- wmi_integrate, WMIMethod, WMIOptions, Density
- UniformDensity, GaussianDensity, ExponentialDensity, BetaDensity
- uniform_density, gaussian_density, exponential_density, beta_density, product_density

Key Features:
- Multiple integration methods: sampling and region-based
- Support for common probability distributions (uniform, Gaussian, exponential, beta)
- Robust error handling and input validation

Example:
    # Weighted Model Counting
    from pysat.formula import CNF
    from aria.prob import wmc_count, WMCBackend, WMCOptions

    cnf = CNF(from_clauses=[[1, 2], [-1, 3]])
    weights = {1: 0.6, -1: 0.4, 2: 0.7, -2: 0.3, 3: 0.5, -3: 0.5}
    result = wmc_count(cnf, weights, WMCOptions(backend=WMCBackend.DNNF))

    # Weighted Model Integration
    import z3
    from aria.prob import wmi_integrate, WMIOptions, UniformDensity

    x, y = z3.Reals('x y')
    formula = z3.And(x + y > 0, x < 1, y < 1, x > 0, y > 0)
    density = UniformDensity({'x': (0, 1), 'y': (0, 1)})

    # WMI computation
    options = WMIOptions(num_samples=10000)
    result = wmi_integrate(formula, density, options)

    # Region-based integration for bounded regions
    options = WMIOptions(method="region", num_samples=1000)
    result = wmi_integrate(formula, density, options)
"""

from .base import WMCBackend, WMCOptions
from .wmc import wmc_count
from .wmi import (
    wmi_integrate,
    WMIMethod,
    WMIOptions,
    Density,
    UniformDensity,
    GaussianDensity,
    ExponentialDensity,
    BetaDensity,
    uniform_density,
    gaussian_density,
    exponential_density,
    beta_density,
    product_density,
)

__all__ = [
    "WMCBackend",
    "WMCOptions",
    "wmc_count",
    "wmi_integrate",
    "WMIMethod",
    "WMIOptions",
    "Density",
    "UniformDensity",
    "GaussianDensity",
    "ExponentialDensity",
    "BetaDensity",
    "uniform_density",
    "gaussian_density",
    "exponential_density",
    "beta_density",
    "product_density",
]
