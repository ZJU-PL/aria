"""
Arithmetic weighted-model-integration APIs.
"""

from ._config import WMIMethod, WMIOptions
from .factories import (
    beta_density,
    discrete_density,
    exponential_density,
    gaussian_density,
    uniform_density,
)
from .moments import covariance, expectation, moment, variance
from .query import conditional_probability, probability
from .wmi import wmi_integrate

__all__ = [
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
