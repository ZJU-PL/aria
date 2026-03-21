"""
Density constructor helpers for arithmetic probabilistic inference.
"""

from __future__ import annotations

from typing import Dict, Tuple

from aria.prob.core.density import (
    BetaDensity,
    ExponentialDensity,
    GaussianDensity,
    UniformDensity,
)


def uniform_density(
    bounds: Dict[str, Tuple[float, float]], discrete: bool = False
) -> UniformDensity:
    return UniformDensity(bounds, discrete=discrete)


def gaussian_density(
    means: Dict[str, float], covariances: Dict[str, Dict[str, float]]
) -> GaussianDensity:
    return GaussianDensity(means, covariances)


def exponential_density(rates: Dict[str, float]) -> ExponentialDensity:
    return ExponentialDensity(rates)


def beta_density(alphas: Dict[str, float], betas: Dict[str, float]) -> BetaDensity:
    return BetaDensity(alphas, betas)


__all__ = [
    "uniform_density",
    "gaussian_density",
    "exponential_density",
    "beta_density",
]
