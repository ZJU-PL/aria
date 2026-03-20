"""
Weighted model integration with explicit exact and Monte Carlo backends.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import z3

from aria.counting.arith.arith_counting_latte import count_lia_models
from aria.prob._helpers import (
    assignment_satisfies,
    evaluate_term,
    finite_support,
)
from aria.prob.density import (
    BetaDensity,
    Density,
    ExponentialDensity,
    GaussianDensity,
    ProductDensity,
    UniformDensity,
    product_density,
)
from aria.prob.results import InferenceResult
from aria.utils.z3_expr_utils import get_variables, z3_value_to_python


class WMIMethod(str, Enum):
    """Available WMI backends."""

    AUTO = "auto"
    BOUNDED_SUPPORT_MONTE_CARLO = "bounded_support_monte_carlo"
    IMPORTANCE_SAMPLING = "importance_sampling"
    EXACT_DISCRETE = "exact_discrete"
    SAMPLING = "sampling"
    REGION = "region"


@dataclass
class WMIOptions:
    """Options for WMI and probability queries over arithmetic formulas."""

    method: WMIMethod = WMIMethod.AUTO
    num_samples: int = 10000
    timeout: Optional[float] = None
    random_seed: Optional[int] = None
    confidence_level: float = 0.95
    proposal: Optional[Density] = None


def _coerce_method(method: Any) -> WMIMethod:
    if isinstance(method, WMIMethod):
        return method
    return WMIMethod(str(method))


def _z_score(confidence_level: float) -> float:
    if confidence_level >= 0.99:
        return 2.576
    if confidence_level >= 0.95:
        return 1.960
    if confidence_level >= 0.90:
        return 1.645
    return 1.0


def _supported_formula_variables(formula: z3.ExprRef) -> List[z3.ExprRef]:
    variables = sorted(get_variables(formula), key=str)
    unsupported = []
    for var in variables:
        if var.sort() not in (z3.IntSort(), z3.RealSort()):
            unsupported.append(str(var))
    if unsupported:
        raise ValueError(
            "WMI currently supports only Int/Real variables, got {}".format(
                unsupported
            )
        )
    return variables


def _validate_density(density: Density) -> None:
    if not callable(density):
        raise ValueError("Density must be callable")

    bounds = density.support()
    if bounds is None:
        return

    for var_name, bound in bounds.items():
        if not isinstance(bound, tuple) or len(bound) != 2:
            raise ValueError(
                "Density support for '{}' must be a (min, max) tuple".format(var_name)
            )
        min_val, max_val = bound
        if not isinstance(min_val, (int, float)) or not isinstance(max_val, (int, float)):
            raise ValueError(
                "Density support for '{}' must be numeric".format(var_name)
            )
        if math.isnan(min_val) or math.isnan(max_val):
            raise ValueError(
                "Density support for '{}' cannot contain NaN".format(var_name)
            )


def _support_formula(variables: List[z3.ExprRef], bounds: Dict[str, Tuple[float, float]]) -> z3.BoolRef:
    constraints = []
    for var in variables:
        var_name = str(var)
        if var_name not in bounds:
            raise ValueError(
                "Density support is missing bounds for variable '{}'".format(var_name)
            )
        min_val, max_val = bounds[var_name]
        if var.sort() == z3.IntSort():
            if not float(min_val).is_integer() or not float(max_val).is_integer():
                raise ValueError(
                    "Exact discrete integration requires integer bounds for '{}'".format(
                        var_name
                    )
                )
            constraints.append(var >= int(min_val))
            constraints.append(var <= int(max_val))
        else:
            constraints.append(var >= z3.RealVal(str(min_val)))
            constraints.append(var <= z3.RealVal(str(max_val)))
    return z3.And(*constraints) if constraints else z3.BoolVal(True)


def _uniform_support_measure(
    variables: List[z3.ExprRef], bounds: Dict[str, Tuple[float, float]]
) -> float:
    measure = 1.0
    for var in variables:
        min_val, max_val = bounds[str(var)]
        if var.sort() == z3.IntSort():
            measure *= int(max_val) - int(min_val) + 1
        else:
            measure *= float(max_val) - float(min_val)
    return measure


def _uniform_sample_from_support(
    variables: List[z3.ExprRef],
    bounds: Dict[str, Tuple[float, float]],
    rng: random.Random,
) -> Dict[str, Any]:
    assignment = {}
    for var in variables:
        min_val, max_val = bounds[str(var)]
        if var.sort() == z3.IntSort():
            assignment[str(var)] = rng.randint(int(min_val), int(max_val))
        else:
            assignment[str(var)] = rng.uniform(float(min_val), float(max_val))
    return assignment


def _running_error_bound(
    sample_count: int, sample_sum: float, sample_sum_squares: float, scale: float, z_score: float
) -> Optional[float]:
    if sample_count <= 1:
        return None
    mean = sample_sum / float(sample_count)
    variance = max(sample_sum_squares / float(sample_count) - mean * mean, 0.0)
    std = math.sqrt(variance)
    return scale * z_score * std / math.sqrt(float(sample_count))


def _exact_discrete_mass(
    formula: z3.ExprRef, density: UniformDensity, variables: List[z3.ExprRef]
) -> InferenceResult:
    if not density.discrete:
        raise ValueError("Exact discrete integration requires UniformDensity(discrete=True)")

    if any(var.sort() != z3.IntSort() for var in variables):
        raise ValueError("Exact discrete integration currently supports Int variables only")

    bounds = density.support()
    support_formula = _support_formula(variables, bounds)
    numerator = count_lia_models(z3.And(formula, support_formula))
    denominator = count_lia_models(support_formula)
    if denominator == 0:
        raise ValueError("Discrete uniform support is empty")

    return InferenceResult(
        value=float(numerator) / float(denominator),
        exact=True,
        backend="wmi-exact-discrete-uniform",
        stats={
            "numerator_count": numerator,
            "denominator_count": denominator,
            "num_variables": len(variables),
        },
        error_bound=0.0,
    )


def _bounded_support_monte_carlo(
    formula: z3.ExprRef, density: Density, options: WMIOptions, variables: List[z3.ExprRef]
) -> InferenceResult:
    bounds = density.support()
    if bounds is None or not finite_support(bounds):
        raise ValueError(
            "Bounded-support Monte Carlo requires a finite rectangular support"
        )

    measure = _uniform_support_measure(variables, bounds)
    rng = random.Random(options.random_seed)
    sample_sum = 0.0
    sample_sum_squares = 0.0
    satisfied = 0

    for _ in range(options.num_samples):
        assignment = _uniform_sample_from_support(variables, bounds, rng)
        contribution = 0.0
        if assignment_satisfies(formula, assignment):
            contribution = float(density(assignment))
            satisfied += 1
        sample_sum += contribution
        sample_sum_squares += contribution * contribution

    estimate = measure * sample_sum / float(options.num_samples)
    error_bound = _running_error_bound(
        options.num_samples,
        sample_sum,
        sample_sum_squares,
        measure,
        _z_score(options.confidence_level),
    )
    return InferenceResult(
        value=estimate,
        exact=False,
        backend="wmi-bounded-support-monte-carlo",
        stats={
            "sample_count": options.num_samples,
            "satisfied_samples": satisfied,
            "support_measure": measure,
        },
        error_bound=error_bound,
    )


def _importance_sampling(
    formula: z3.ExprRef, density: Density, options: WMIOptions, variables: List[z3.ExprRef]
) -> InferenceResult:
    proposal = options.proposal or density
    rng = random.Random(options.random_seed)
    sample_sum = 0.0
    sample_sum_squares = 0.0
    satisfied = 0

    for _ in range(options.num_samples):
        assignment = proposal.sample_assignment(rng)
        missing = [str(var) for var in variables if str(var) not in assignment]
        if missing:
            raise ValueError(
                "Proposal density did not assign all formula variables: {}".format(
                    missing
                )
            )

        proposal_value = float(proposal(assignment))
        density_value = float(density(assignment))
        if proposal_value <= 0.0:
            if density_value > 0.0:
                raise ValueError(
                    "Proposal density assigned zero mass to a positive-density sample"
                )
            contribution = 0.0
        else:
            weight = density_value / proposal_value
            if assignment_satisfies(formula, assignment):
                contribution = weight
                satisfied += 1
            else:
                contribution = 0.0

        sample_sum += contribution
        sample_sum_squares += contribution * contribution

    estimate = sample_sum / float(options.num_samples)
    error_bound = _running_error_bound(
        options.num_samples,
        sample_sum,
        sample_sum_squares,
        1.0,
        _z_score(options.confidence_level),
    )
    return InferenceResult(
        value=estimate,
        exact=False,
        backend="wmi-importance-sampling",
        stats={
            "sample_count": options.num_samples,
            "satisfied_samples": satisfied,
            "proposal": proposal.__class__.__name__,
        },
        error_bound=error_bound,
    )


def _effective_method(density: Density, options: WMIOptions, variables: List[z3.ExprRef]) -> WMIMethod:
    method = _coerce_method(options.method)
    if method == WMIMethod.SAMPLING:
        method = WMIMethod.AUTO
    if method == WMIMethod.REGION:
        return WMIMethod.BOUNDED_SUPPORT_MONTE_CARLO
    if method != WMIMethod.AUTO:
        return method

    if isinstance(density, UniformDensity) and density.discrete:
        return WMIMethod.EXACT_DISCRETE

    bounds = density.support()
    if bounds is not None and finite_support(bounds):
        return WMIMethod.BOUNDED_SUPPORT_MONTE_CARLO
    return WMIMethod.IMPORTANCE_SAMPLING


def _validate_wmi_inputs(formula: z3.ExprRef, density: Density) -> List[z3.ExprRef]:
    if not z3.is_expr(formula):
        raise ValueError("Formula must be a Z3 expression")
    variables = _supported_formula_variables(formula)
    _validate_density(density)
    return variables


def wmi_integrate(
    formula: z3.ExprRef, density: Density, options: Optional[WMIOptions] = None
) -> InferenceResult:
    """
    Compute the probability mass of a formula under a normalized density.
    """

    opts = options or WMIOptions()
    variables = _validate_wmi_inputs(formula, density)
    method = _effective_method(density, opts, variables)

    if method == WMIMethod.EXACT_DISCRETE:
        if not isinstance(density, UniformDensity):
            raise ValueError("Exact discrete integration currently supports UniformDensity only")
        return _exact_discrete_mass(formula, density, variables)
    if method == WMIMethod.BOUNDED_SUPPORT_MONTE_CARLO:
        return _bounded_support_monte_carlo(formula, density, opts, variables)
    if method == WMIMethod.IMPORTANCE_SAMPLING:
        return _importance_sampling(formula, density, opts, variables)
    raise ValueError("Unsupported WMI method: {}".format(method))


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
    "Density",
    "UniformDensity",
    "GaussianDensity",
    "ExponentialDensity",
    "BetaDensity",
    "ProductDensity",
    "product_density",
    "WMIMethod",
    "WMIOptions",
    "wmi_integrate",
    "uniform_density",
    "gaussian_density",
    "exponential_density",
    "beta_density",
]
