"""
Density models used by probabilistic inference.
"""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


Bounds = Dict[str, Tuple[float, float]]


class Density(ABC):
    """Abstract base class for normalized density or mass functions."""

    @abstractmethod
    def __call__(self, assignment: Dict[str, Any]) -> float:
        raise NotImplementedError

    def support(self) -> Optional[Bounds]:
        return None

    def is_normalized(self) -> bool:
        return True

    def factorizes(self) -> bool:
        return False

    def sample_assignment(self, rng: random.Random) -> Dict[str, Any]:
        raise NotImplementedError(
            "{} does not support direct sampling".format(self.__class__.__name__)
        )


class UniformDensity(Density):
    """Uniform density over a rectangular box or finite integer grid."""

    def __init__(self, bounds: Bounds, discrete: bool = False):
        self.bounds = dict(bounds)
        self.discrete = discrete
        self._volume = 1.0

        for var_name, (min_val, max_val) in self.bounds.items():
            if discrete:
                if not float(min_val).is_integer() or not float(max_val).is_integer():
                    raise ValueError(
                        "Discrete uniform bounds for '{}' must be integers".format(
                            var_name
                        )
                    )
                if min_val > max_val:
                    raise ValueError(
                        "Discrete uniform bounds for '{}' must satisfy min <= max".format(
                            var_name
                        )
                    )
                self._volume *= int(max_val) - int(min_val) + 1
            else:
                if min_val >= max_val:
                    raise ValueError(
                        "Uniform bounds for '{}' must satisfy min < max".format(
                            var_name
                        )
                    )
                self._volume *= float(max_val) - float(min_val)

    def __call__(self, assignment: Dict[str, Any]) -> float:
        for var_name, value in assignment.items():
            if var_name not in self.bounds:
                continue
            min_val, max_val = self.bounds[var_name]
            if self.discrete:
                if int(value) != value:
                    return 0.0
                if value < min_val or value > max_val:
                    return 0.0
            else:
                if value < min_val or value > max_val:
                    return 0.0
        if self._volume <= 0:
            return 0.0
        return 1.0 / self._volume

    def support(self) -> Bounds:
        return dict(self.bounds)

    def factorizes(self) -> bool:
        return True

    def sample_assignment(self, rng: random.Random) -> Dict[str, Any]:
        sample = {}
        for var_name, (min_val, max_val) in self.bounds.items():
            if self.discrete:
                sample[var_name] = rng.randint(int(min_val), int(max_val))
            else:
                sample[var_name] = rng.uniform(float(min_val), float(max_val))
        return sample


class GaussianDensity(Density):
    """Diagonal multivariate Gaussian density."""

    def __init__(
        self, means: Dict[str, float], covariances: Dict[str, Dict[str, float]]
    ):
        self.means = dict(means)
        self.covariances = dict(covariances)
        self.variables = sorted(self.means.keys())
        self._variances = {}
        self._normalization = 1.0

        for var_name in self.variables:
            row = self.covariances.get(var_name, {})
            if set(row.keys()) - {var_name}:
                raise ValueError(
                    "GaussianDensity only supports diagonal covariance matrices"
                )
            if var_name not in row:
                raise ValueError(
                    "Missing diagonal covariance entry for '{}'".format(var_name)
                )
            variance = float(row[var_name])
            if variance <= 0:
                raise ValueError(
                    "Gaussian variance for '{}' must be positive".format(var_name)
                )
            self._variances[var_name] = variance
            self._normalization *= 1.0 / math.sqrt(2.0 * math.pi * variance)

        for row_name, row in self.covariances.items():
            off_diagonal = [
                col_name for col_name, value in row.items() if col_name != row_name and value
            ]
            if off_diagonal:
                raise ValueError(
                    "GaussianDensity only supports diagonal covariance matrices"
                )

    def __call__(self, assignment: Dict[str, Any]) -> float:
        density = self._normalization
        for var_name in self.variables:
            if var_name not in assignment:
                continue
            diff = float(assignment[var_name]) - float(self.means[var_name])
            variance = self._variances[var_name]
            density *= math.exp(-(diff * diff) / (2.0 * variance))
        return density

    def factorizes(self) -> bool:
        return True

    def sample_assignment(self, rng: random.Random) -> Dict[str, Any]:
        sample = {}
        for var_name in self.variables:
            sample[var_name] = rng.gauss(
                float(self.means[var_name]), math.sqrt(self._variances[var_name])
            )
        return sample


class ExponentialDensity(Density):
    """Product of exponential densities on non-negative variables."""

    def __init__(self, rates: Dict[str, float]):
        self.rates = dict(rates)
        for var_name, rate in self.rates.items():
            if rate <= 0:
                raise ValueError(
                    "Exponential rate for '{}' must be positive".format(var_name)
                )

    def __call__(self, assignment: Dict[str, Any]) -> float:
        density = 1.0
        for var_name, rate in self.rates.items():
            if var_name not in assignment:
                continue
            value = float(assignment[var_name])
            if value < 0:
                return 0.0
            density *= float(rate) * math.exp(-float(rate) * value)
        return density

    def support(self) -> Bounds:
        return {var_name: (0.0, float("inf")) for var_name in self.rates}

    def factorizes(self) -> bool:
        return True

    def sample_assignment(self, rng: random.Random) -> Dict[str, Any]:
        return {
            var_name: rng.expovariate(float(rate))
            for var_name, rate in self.rates.items()
        }


class BetaDensity(Density):
    """Product of beta densities on [0, 1]."""

    def __init__(self, alphas: Dict[str, float], betas: Dict[str, float]):
        self.alphas = dict(alphas)
        self.betas = dict(betas)
        self.variables = sorted(self.alphas.keys())
        self._normalizations = {}

        if set(self.alphas.keys()) != set(self.betas.keys()):
            raise ValueError("BetaDensity alpha/beta variables must match")

        for var_name in self.variables:
            alpha = float(self.alphas[var_name])
            beta = float(self.betas[var_name])
            if alpha <= 0 or beta <= 0:
                raise ValueError(
                    "Beta parameters for '{}' must be positive".format(var_name)
                )
            self._normalizations[var_name] = math.gamma(alpha + beta) / (
                math.gamma(alpha) * math.gamma(beta)
            )

    def __call__(self, assignment: Dict[str, Any]) -> float:
        density = 1.0
        for var_name in self.variables:
            if var_name not in assignment:
                continue
            value = float(assignment[var_name])
            if value < 0.0 or value > 1.0:
                return 0.0
            alpha = float(self.alphas[var_name])
            beta = float(self.betas[var_name])
            density *= (
                self._normalizations[var_name]
                * math.pow(value, alpha - 1.0)
                * math.pow(1.0 - value, beta - 1.0)
            )
        return density

    def support(self) -> Bounds:
        return {var_name: (0.0, 1.0) for var_name in self.variables}

    def factorizes(self) -> bool:
        return True

    def sample_assignment(self, rng: random.Random) -> Dict[str, Any]:
        return {
            var_name: rng.betavariate(
                float(self.alphas[var_name]), float(self.betas[var_name])
            )
            for var_name in self.variables
        }


class ProductDensity(Density):
    """Product of independent densities."""

    def __init__(self, densities: Sequence[Density]):
        self.densities = list(densities)

    def __call__(self, assignment: Dict[str, Any]) -> float:
        result = 1.0
        for density in self.densities:
            result *= density(assignment)
        return result

    def support(self) -> Optional[Bounds]:
        combined = {}
        for density in self.densities:
            bounds = density.support()
            if bounds is None:
                return None
            for var_name, (min_val, max_val) in bounds.items():
                if var_name in combined:
                    old_min, old_max = combined[var_name]
                    combined[var_name] = (max(old_min, min_val), min(old_max, max_val))
                else:
                    combined[var_name] = (min_val, max_val)
        return combined

    def factorizes(self) -> bool:
        return True

    def is_normalized(self) -> bool:
        return all(density.is_normalized() for density in self.densities)

    def sample_assignment(self, rng: random.Random) -> Dict[str, Any]:
        sample = {}
        for density in self.densities:
            part = density.sample_assignment(rng)
            overlap = set(sample.keys()).intersection(part.keys())
            if overlap:
                raise ValueError(
                    "ProductDensity sampling requires disjoint variable sets; got {}".format(
                        sorted(overlap)
                    )
                )
            sample.update(part)
        return sample


def product_density(densities: Iterable[Density]) -> ProductDensity:
    return ProductDensity(list(densities))
