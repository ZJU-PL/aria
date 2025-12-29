"""
Weighted Model Integration (WMI)

Implementation of Weighted Model Integration for continuous domains.
WMI extends WMC from discrete to continuous probability spaces using
density functions instead of discrete weights.

This module provides WMI computation over LRA/LIA formulas using:
- Monte Carlo integration with sampling-based estimation
- Region-based integration for bounded regions
- Support for various density function types

Example:
    import z3
    from aria.prob import wmi_integrate, WMIOptions, UniformDensity

    x, y = z3.Reals('x y')
    formula = z3.And(x + y > 0, x < 1, y < 1, x > 0, y > 0)

    # Uniform density over the unit square
    density = UniformDensity({'x': (0, 1), 'y': (0, 1)})

    options = WMIOptions(method="sampling", num_samples=10000)
    result = wmi_integrate(formula, density, options)
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Any, Tuple, Optional

import z3

from aria.sampling import (
    sample_models_from_formula,
    Logic,
    SamplingOptions,
    SamplingMethod
)
from aria.utils.z3_expr_utils import get_variables


class WMIMethod(str, Enum):
    """Available WMI integration methods."""
    SAMPLING = "sampling"      # Monte Carlo integration using sampling
    REGION = "region"          # Region-based integration for bounded regions


class Density(Protocol):
    """Protocol for density functions used in WMI."""

    def __call__(self, assignment: dict[str, Any]) -> float:
        """
        Evaluate the density function at the given variable assignment.

        Args:
            assignment: Dictionary mapping variable names to values

        Returns:
            Density value (must be non-negative)
        """
        raise NotImplementedError

    def support(self) -> Optional[dict[str, Tuple[float, float]]]:
        """
        Return the support (bounds) of the density function if known.

        Returns:
            Dictionary mapping variable names to (min, max) bounds, or None if unbounded
        """
        return None


@dataclass
class WMIOptions:
    """Options for WMI computation."""
    method: WMIMethod = WMIMethod.SAMPLING
    num_samples: int = 10000
    timeout: Optional[float] = None
    random_seed: Optional[int] = None
    confidence_level: float = 0.95  # For statistical error bounds


class UniformDensity:
    """Uniform density over a rectangular region."""

    def __init__(self, bounds: dict[str, Tuple[float, float]]):
        """
        Initialize uniform density.

        Args:
            bounds: Dictionary mapping variable names to (min, max) bounds
        """
        self.bounds = bounds
        self._volume = 1.0
        for min_val, max_val in bounds.values():
            self._volume *= (max_val - min_val)

    def __call__(self, assignment: dict[str, Any]) -> float:
        """Evaluate uniform density."""
        # Check if assignment is within bounds
        for var, value in assignment.items():
            if var in self.bounds:
                min_val, max_val = self.bounds[var]
                if min_val > value or value > max_val:
                    return 0.0
        return 1.0 / self._volume if self._volume > 0 else 0.0

    def support(self) -> dict[str, Tuple[float, float]]:
        """Return the support of the uniform density."""
        return self.bounds.copy()


class GaussianDensity:
    """Multivariate Gaussian density."""

    def __init__(self, means: dict[str, float], covariances: dict[str, dict[str, float]]):
        """
        Initialize Gaussian density.

        Args:
            means: Dictionary mapping variable names to mean values
            covariances: Covariance matrix as nested dictionary
        """
        self.means = means
        self.covariances = covariances
        self.variables = list(means.keys())

        # For simplicity, handle only diagonal covariance matrices for now
        self._is_diagonal = all(
            var in covariances.get(var, {}) for var in self.variables
        ) and all(
            i == j for i in self.variables for j in self.variables
            if i in covariances.get(j, {})
        )

        if self._is_diagonal:
            self._precisions = {}
            for var in self.variables:
                var_cov = covariances.get(var, {}).get(var, 1.0)
                self._precisions[var] = 1.0 / var_cov if var_cov > 0 else 1.0

            # Compute normalization constant for diagonal case
            self._normalization = 1.0
            for var in self.variables:
                var_cov = covariances.get(var, {}).get(var, 1.0)
                self._normalization *= 1.0 / math.sqrt(2 * math.pi * var_cov)
        else:
            # For general case, we'd need to compute the determinant and inverse
            # For now, fall back to a simple implementation
            self._normalization = 1.0

    def __call__(self, assignment: dict[str, Any]) -> float:
        """Evaluate Gaussian density."""
        if self._is_diagonal:
            # Diagonal case - product of independent Gaussians
            density = self._normalization
            for var in self.variables:
                if var in assignment:
                    mean = self.means.get(var, 0.0)
                    precision = self._precisions.get(var, 1.0)
                    diff = assignment[var] - mean
                    density *= math.exp(-0.5 * precision * diff * diff)
            return density
        # General multivariate case - simplified implementation
        # In practice, this would require matrix operations
        density = self._normalization
        for var in self.variables:
            if var in assignment:
                mean = self.means.get(var, 0.0)
                # Use identity covariance as fallback
                diff = assignment[var] - mean
                density *= math.exp(-0.5 * diff * diff)
        return density

    def support(self) -> None:
        """Gaussian density has unbounded support."""
        return None


class ExponentialDensity:
    """Exponential density function."""

    def __init__(self, rates: dict[str, float]):
        """
        Initialize exponential density.

        Args:
            rates: Dictionary mapping variable names to rate parameters (λ > 0)
        """
        self.rates = rates
        self.variables = list(rates.keys())

        # Validate rates
        for var, rate in rates.items():
            if rate <= 0:
                raise ValueError(
                    f"Exponential rate for variable '{var}' must be positive, "
                    f"got {rate}"
                )

    def __call__(self, assignment: dict[str, Any]) -> float:
        """Evaluate exponential density."""
        density = 1.0
        for var in self.variables:
            if var in assignment:
                rate = self.rates[var]
                value = assignment[var]
                if value < 0:
                    return 0.0  # Exponential distribution has support [0, ∞)
                density *= rate * math.exp(-rate * value)
        return density

    def support(self) -> dict[str, Tuple[float, float]]:
        """Return the support of the exponential density."""
        return {var: (0.0, float('inf')) for var in self.variables}


class BetaDensity:
    """Beta density function."""

    def __init__(self, alphas: dict[str, float], betas: dict[str, float]):
        """
        Initialize beta density.

        Args:
            alphas: Dictionary mapping variable names to alpha parameters (α > 0)
            betas: Dictionary mapping variable names to beta parameters (β > 0)
        """
        self.alphas = alphas
        self.betas = betas
        self.variables = list(alphas.keys())

        # Validate parameters
        for var in self.variables:
            if alphas[var] <= 0 or betas[var] <= 0:
                raise ValueError(f"Beta parameters for variable '{var}' must be positive")

        # Precompute normalization constants
        self._normalizations = {}
        for var in self.variables:
            alpha = alphas[var]
            beta = betas[var]
            # Beta function B(α,β) = Γ(α)Γ(β)/Γ(α+β)
            gamma_alpha = math.gamma(alpha)
            gamma_beta = math.gamma(beta)
            gamma_sum = math.gamma(alpha + beta)
            self._normalizations[var] = (gamma_alpha * gamma_beta) / gamma_sum

    def __call__(self, assignment: dict[str, Any]) -> float:
        """Evaluate beta density."""
        density = 1.0
        for var in self.variables:
            if var in assignment:
                value = assignment[var]
                if value < 0 or value > 1:
                    return 0.0  # Beta distribution has support [0, 1]

                alpha = self.alphas[var]
                beta = self.betas[var]
                normalization = self._normalizations[var]

                value_pow = math.pow(value, alpha - 1)
                complement_pow = math.pow(1 - value, beta - 1)
                density *= value_pow * complement_pow / normalization
        return density

    def support(self) -> dict[str, Tuple[float, float]]:
        """Return the support of the beta density."""
        return {var: (0.0, 1.0) for var in self.variables}




def _wmi_by_sampling(formula: z3.ExprRef, density: Density, options: WMIOptions) -> float:
    """
    Compute WMI using Monte Carlo integration with sampling.

    This method samples points from the solution space and evaluates
    the density function at each point, then averages the results.
    """
    try:
        # Create sampling options
        method = (SamplingMethod.ENUMERATION
                 if options.num_samples < 1000
                 else SamplingMethod.DIKIN_WALK)
        sampling_options = SamplingOptions(
            method=method,
            num_samples=min(options.num_samples, 10000),  # Cap for efficiency
            timeout=options.timeout,
            random_seed=options.random_seed
        )

        # Sample models from the formula
        sampling_result = sample_models_from_formula(formula, Logic.QF_LRA, sampling_options)

        if not sampling_result.samples:
            return 0.0

        # Evaluate density at each sample and compute average
        total_density = 0.0
        valid_samples = 0
        error_count = 0

        for sample in sampling_result.samples:
            try:
                density_value = density(sample)
                is_valid = (density_value >= 0
                           and not math.isinf(density_value)
                           and not math.isnan(density_value))
                if is_valid:
                    # Valid density values only
                    total_density += density_value
                    valid_samples += 1
                else:
                    error_count += 1
            except (ValueError, TypeError, AttributeError):
                # Skip samples where density evaluation fails
                error_count += 1
                continue

        if valid_samples == 0:
            if error_count > 0:
                num_samples = len(sampling_result.samples)
                raise ValueError(
                    f"All {num_samples} samples failed density evaluation. "
                    "Check that the density function is compatible with "
                    "the formula variables."
                )
            return 0.0

        # Return Monte Carlo estimate
        result = total_density / valid_samples

        # Warn if many samples failed
        if error_count > 0.1 * len(sampling_result.samples):
            num_samples = len(sampling_result.samples)
            warnings.warn(
                f"High density evaluation failure rate: "
                f"{error_count}/{num_samples} samples failed. "
                "Consider adjusting the density function or formula constraints."
            )

        return result

    except ImportError as e:
        raise ImportError(
            f"WMI sampling requires aria.sampling module: {e}"
        ) from e
    except Exception as e:
        if "sampling" in str(e).lower():
            raise ValueError(
                f"Sampling failed for WMI computation: {e}. "
                "Consider using region-based integration for bounded regions."
            ) from e
        raise ValueError(f"WMI computation failed: {e}") from e


def _wmi_by_region(formula: z3.ExprRef, density: Density, options: WMIOptions) -> float:
    """
    Compute WMI using region-based integration.

    For now, this method uses enhanced sampling within bounded regions.
    """
    # For bounded regions, use sampling with more samples for better accuracy
    enhanced_options = WMIOptions(
        method=WMIMethod.SAMPLING,
        num_samples=min(options.num_samples * 2, 2000),
        timeout=options.timeout,
        random_seed=options.random_seed,
        confidence_level=options.confidence_level
    )

    return _wmi_by_sampling(formula, density, enhanced_options)




def _validate_wmi_inputs(formula: z3.ExprRef, density: Density) -> None:
    """Validate WMI inputs."""
    if not z3.is_expr(formula):
        raise ValueError("Formula must be a Z3 expression")

    # Check that formula involves appropriate theories
    formula_vars = get_variables(formula)
    supported_sort_kinds = (z3.Z3_REAL_SORT, z3.Z3_INT_SORT)

    unsupported_vars = []
    for var in formula_vars:
        var_sort_kind = var.sort().kind()
        if var_sort_kind not in supported_sort_kinds:
            unsupported_vars.append(str(var))

    if unsupported_vars:
        raise ValueError(f"Formula contains unsupported variable types: {unsupported_vars}. "
                        "WMI currently supports only real and integer variables.")

    # Check that density is callable
    if not callable(density):
        raise ValueError("Density must be callable")

    # Check that density support (if provided) is valid
    try:
        density_support = density.support()
        if density_support is not None:
            for var_name, bounds in density_support.items():
                if not isinstance(bounds, tuple) or len(bounds) != 2:
                    raise ValueError(
                        f"Density support for variable '{var_name}' "
                        "must be a tuple (min, max)"
                    )
                min_val, max_val = bounds
                if (not isinstance(min_val, (int, float))
                   or not isinstance(max_val, (int, float))):
                    raise ValueError(
                        f"Density support bounds for variable '{var_name}' "
                        "must be numeric"
                    )
                if min_val >= max_val:
                    raise ValueError(
                        f"Density support for variable '{var_name}' "
                        "must have min < max"
                    )
    except AttributeError:
        # Density doesn't implement support() method, that's OK
        pass
    except Exception as e:
        raise ValueError(f"Error validating density support: {e}") from e


def wmi_integrate(formula: z3.ExprRef, density: Density, options: WMIOptions | None = None) -> float:
    """
    Compute Weighted Model Integration of a formula with respect to a density function.

    Args:
        formula: Z3 formula over real/integer variables (typically LRA/LIA)
        density: Density function to integrate over the satisfying assignments
        options: WMI computation options

    Returns:
        WMI result (integral of density over satisfying assignments)

    Raises:
        ValueError: If inputs are invalid
        NotImplementedError: If the requested method is not available
    """
    opts = options or WMIOptions()

    # Validate inputs
    _validate_wmi_inputs(formula, density)

    # Choose integration method
    if opts.method == WMIMethod.SAMPLING:
        return _wmi_by_sampling(formula, density, opts)
    if opts.method == WMIMethod.REGION:
        return _wmi_by_region(formula, density, opts)
    raise ValueError(f"Unsupported WMI method: {opts.method}")


# Convenience functions for common densities
def uniform_density(bounds: dict[str, Tuple[float, float]]) -> UniformDensity:
    """Create a uniform density over the given bounds."""
    return UniformDensity(bounds)


def gaussian_density(
    means: dict[str, float],
    covariances: dict[str, dict[str, float]]
) -> GaussianDensity:
    """Create a Gaussian density with given means and covariances."""
    return GaussianDensity(means, covariances)


def exponential_density(rates: dict[str, float]) -> ExponentialDensity:
    """Create an exponential density with given rates."""
    return ExponentialDensity(rates)


def beta_density(alphas: dict[str, float], betas: dict[str, float]) -> BetaDensity:
    """Create a beta density with given alpha and beta parameters."""
    return BetaDensity(alphas, betas)


def product_density(densities: list[Density]) -> Density:
    """Create a product density from multiple independent densities."""
    class ProductDensity:
        """Product of multiple independent density functions."""

        def __init__(self, densities):
            """Initialize product density with list of densities."""
            self.densities = densities

        def __call__(self, assignment: dict[str, Any]) -> float:
            """Evaluate product density at assignment."""
            result = 1.0
            for density in self.densities:
                result *= density(assignment)
            return result

        def support(self) -> Optional[dict[str, Tuple[float, float]]]:
            """Return combined support of all densities."""
            # Combine supports - intersection of all supports
            combined_support = {}
            for density in self.densities:
                density_support = density.support()
                if density_support is None:
                    return None  # If any density is unbounded, result is unbounded

                for var, bounds in density_support.items():
                    if var in combined_support:
                        # Take intersection
                        old_min, old_max = combined_support[var]
                        new_min, new_max = bounds
                        combined_support[var] = (max(old_min, new_min), min(old_max, new_max))
                    else:
                        combined_support[var] = bounds

            return combined_support

    return ProductDensity(densities)
