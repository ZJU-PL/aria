"""
Moment queries for arithmetic probabilistic models.
"""

from __future__ import annotations

import math
import random
from typing import Dict, Optional

import z3

from aria.prob.core._helpers import assignment_satisfies, evaluate_term
from aria.prob.core.density import Density
from aria.prob.core.results import InferenceResult
from aria.utils.z3.expr import get_variables

from ._config import WMIMethod, WMIOptions
from ._exact_backend import _exact_discrete_expectation
from ._selection import _effective_method, _validate_wmi_inputs, _validate_wmi_options
from ._sampling_utils import _uniform_sample_from_support, _z_score


def _variables_for_terms(formula: z3.ExprRef, *terms: z3.ExprRef) -> list[z3.ExprRef]:
    variables = {str(var): var for var in get_variables(formula)}
    for term in terms:
        for var in get_variables(term):
            variables[str(var)] = var
    return [variables[name] for name in sorted(variables)]


def _estimate_error_bound(
    normalized_values_sum: float,
    normalized_values_sum_squares: float,
    sample_count: int,
    confidence_level: float,
) -> Optional[float]:
    if sample_count <= 1:
        return None
    mean = normalized_values_sum / float(sample_count)
    variance = max(
        normalized_values_sum_squares / float(sample_count) - mean * mean, 0.0
    )
    return _z_score(confidence_level) * math.sqrt(variance / float(sample_count))


def _effective_sample_size(weight_sum: float, weight_sum_squares: float) -> Optional[float]:
    if weight_sum <= 0.0 or weight_sum_squares <= 0.0:
        return None
    return (weight_sum * weight_sum) / weight_sum_squares


def _shared_sample_stats(
    estimates: Dict[str, float],
    per_draw_weighted_sums: Dict[str, float],
    per_draw_weighted_square_sums: Dict[str, float],
    sample_count: int,
    satisfied_samples: int,
    conditioning_weight_sum: float,
    conditioning_weight_sum_squares: float,
    conditioning_mass_scale: float,
    backend: str,
    confidence_level: float,
    proposal_name: Optional[str] = None,
) -> Dict[str, object]:
    error_bounds = {}
    for name in estimates:
        error_bounds[name] = _estimate_error_bound(
            per_draw_weighted_sums[name],
            per_draw_weighted_square_sums[name],
            sample_count,
            confidence_level,
        )
    conditioning_mass_estimate = (
        conditioning_mass_scale * conditioning_weight_sum / float(sample_count)
    )
    conditioning_mass_half_width = _estimate_error_bound(
        conditioning_weight_sum,
        conditioning_weight_sum_squares,
        sample_count,
        confidence_level,
    )
    if conditioning_mass_half_width is not None:
        conditioning_mass_half_width *= conditioning_mass_scale
    effective_sample_size = None
    if backend == "wmi-importance-sampling":
        effective_sample_size = _effective_sample_size(
            conditioning_weight_sum, conditioning_weight_sum_squares
        )
    return {
        "sample_count": sample_count,
        "satisfied_samples": satisfied_samples,
        "accepted_samples": satisfied_samples,
        "effective_conditioning_weight": conditioning_mass_estimate,
        "conditioning_mass_estimate": conditioning_mass_estimate,
        "conditioning_probability_estimate": conditioning_mass_estimate,
        "conditioning_mass_confidence_half_width": conditioning_mass_half_width,
        "raw_conditioning_weight_sum": conditioning_weight_sum,
        "raw_conditioning_weight_square_sum": conditioning_weight_sum_squares,
        "conditioning_mass_scale": conditioning_mass_scale,
        "estimates": dict(estimates),
        "per_draw_weighted_moment_sums": dict(per_draw_weighted_sums),
        "per_draw_weighted_moment_square_sums": dict(per_draw_weighted_square_sums),
        "moment_confidence_half_widths": error_bounds,
        "estimator_error_bounds": dict(error_bounds),
        "backend_family": backend,
        "proposal_name": proposal_name,
        "approx_effective_sample_size": effective_sample_size,
    }


def _sample_moment_estimates(
    terms: Dict[str, z3.ExprRef],
    formula: z3.ExprRef,
    density: Density,
    options: WMIOptions,
) -> Dict[str, object]:
    variables = _variables_for_terms(formula, *terms.values())
    _validate_wmi_inputs(formula, density)
    _validate_wmi_options(options, density, variables)
    method = _effective_method(density, options, variables)
    if method == WMIMethod.EXACT_DISCRETE:
        raise ValueError("Exact discrete single-pass estimation is handled separately")

    rng = random.Random(options.random_seed)
    weighted_sums = {name: 0.0 for name in terms}
    per_draw_weighted_sums = {name: 0.0 for name in terms}
    per_draw_weighted_square_sums = {name: 0.0 for name in terms}
    weight_sum = 0.0
    weight_sum_squares = 0.0
    satisfied_samples = 0
    conditioning_mass_scale = 1.0
    proposal_name = None

    if method == WMIMethod.BOUNDED_SUPPORT_MONTE_CARLO:
        bounds = density.support()
        if bounds is None:
            raise ValueError(
                "Bounded-support Monte Carlo expectation requires finite support"
            )
        backend = "wmi-bounded-support-monte-carlo"
        conditioning_mass_scale = 1.0
        for var in variables:
            min_val, max_val = bounds[str(var)]
            if var.sort() == z3.IntSort():
                conditioning_mass_scale *= int(max_val) - int(min_val) + 1
            else:
                conditioning_mass_scale *= float(max_val) - float(min_val)
        for _ in range(options.num_samples):
            assignment = _uniform_sample_from_support(variables, bounds, rng)
            sample_weight = 0.0
            values = {}
            if assignment_satisfies(formula, assignment):
                sample_weight = float(density(assignment))
                satisfied_samples += 1
                for name, term in terms.items():
                    term_value = evaluate_term(term, assignment)
                    if not isinstance(term_value, (int, float)):
                        raise ValueError(
                            "Expectation term must evaluate to a numeric value"
                        )
                    values[name] = float(term_value)
            weight_sum += sample_weight
            weight_sum_squares += sample_weight * sample_weight
            for name in terms:
                normalized = sample_weight * values.get(name, 0.0)
                weighted_sums[name] += normalized
                per_draw_weighted_sums[name] += normalized
                per_draw_weighted_square_sums[name] += normalized * normalized
    else:
        proposal = options.proposal or density
        backend = "wmi-importance-sampling"
        proposal_name = proposal.__class__.__name__
        for _ in range(options.num_samples):
            assignment = proposal.sample_assignment(rng)
            proposal_value = float(proposal(assignment))
            target_value = float(density(assignment))
            if proposal_value <= 0.0:
                if target_value > 0.0 and assignment_satisfies(formula, assignment):
                    raise ValueError(
                        "Proposal density assigned zero mass to a positive-density sample"
                    )
                continue
            if not assignment_satisfies(formula, assignment):
                continue
            sample_weight = target_value / proposal_value
            satisfied_samples += 1
            weight_sum += sample_weight
            weight_sum_squares += sample_weight * sample_weight
            for name, term in terms.items():
                term_value = evaluate_term(term, assignment)
                if not isinstance(term_value, (int, float)):
                    raise ValueError("Expectation term must evaluate to a numeric value")
                normalized = sample_weight * float(term_value)
                weighted_sums[name] += normalized
                per_draw_weighted_sums[name] += normalized
                per_draw_weighted_square_sums[name] += normalized * normalized

    if weight_sum == 0.0:
        raise ValueError("Expectation is undefined because the conditioning event is empty")

    estimates = {name: weighted_sums[name] / weight_sum for name in terms}
    return {
        "estimates": estimates,
        "stats": _shared_sample_stats(
            estimates,
            per_draw_weighted_sums,
            per_draw_weighted_square_sums,
            options.num_samples,
            satisfied_samples,
            weight_sum,
            weight_sum_squares,
            conditioning_mass_scale,
            backend,
            options.confidence_level,
            proposal_name=proposal_name,
        ),
        "backend": backend,
    }


def moment(
    term: z3.ExprRef,
    order: int,
    formula: z3.ExprRef,
    density: Density,
    options: Optional[WMIOptions] = None,
) -> InferenceResult:
    """
    Compute E[term^order | formula] under the given density.
    """

    if not isinstance(order, int) or isinstance(order, bool) or order < 1:
        raise ValueError("Moment order must be a positive integer")

    opts = options or WMIOptions()
    variables = _variables_for_terms(formula, term)
    _validate_wmi_options(opts, density, variables)
    method = _effective_method(density, opts, variables)
    if method == WMIMethod.EXACT_DISCRETE:
        powered_term = term if order == 1 else term ** order
        return _exact_discrete_expectation(powered_term, formula, density, variables)

    powered_term = term if order == 1 else term ** order
    sampled = _sample_moment_estimates({"moment": powered_term}, formula, density, opts)

    return InferenceResult(
        value=sampled["estimates"]["moment"],
        exact=False,
        backend=sampled["backend"],
        stats=dict(sampled["stats"], order=order),
        error_bound=sampled["stats"]["moment_confidence_half_widths"]["moment"],
    )


def expectation(
    term: z3.ExprRef,
    formula: z3.ExprRef,
    density: Density,
    options: Optional[WMIOptions] = None,
) -> InferenceResult:
    """
    Compute E[term | formula] under the given density.
    """

    return moment(term, 1, formula, density, options)


def covariance(
    term_x: z3.ExprRef,
    term_y: z3.ExprRef,
    formula: z3.ExprRef,
    density: Density,
    options: Optional[WMIOptions] = None,
) -> InferenceResult:
    """
    Compute Cov(term_x, term_y | formula) under the given density.
    """

    opts = options or WMIOptions()
    variables = _variables_for_terms(formula, term_x, term_y)
    _validate_wmi_options(opts, density, variables)
    method = _effective_method(density, opts, variables)
    if method == WMIMethod.EXACT_DISCRETE:
        first_x = moment(term_x, 1, formula, density, opts)
        first_y = moment(term_y, 1, formula, density, opts)
        mixed = moment(term_x * term_y, 1, formula, density, opts)
        covariance_value = float(mixed) - float(first_x) * float(first_y)
        return InferenceResult(
            value=covariance_value,
            exact=mixed.exact and first_x.exact and first_y.exact,
            backend=mixed.backend,
            stats=dict(
                mixed.stats,
                first_x=float(first_x),
                first_y=float(first_y),
                mixed_moment=float(mixed),
            ),
            error_bound=0.0,
        )

    sampled = _sample_moment_estimates(
        {"x": term_x, "y": term_y, "xy": term_x * term_y},
        formula,
        density,
        opts,
    )
    covariance_value = (
        sampled["estimates"]["xy"]
        - sampled["estimates"]["x"] * sampled["estimates"]["y"]
    )

    return InferenceResult(
        value=covariance_value,
        exact=False,
        backend=sampled["backend"],
        stats=dict(
            sampled["stats"],
            first_x=sampled["estimates"]["x"],
            first_y=sampled["estimates"]["y"],
            mixed_moment=sampled["estimates"]["xy"],
        ),
        error_bound=sampled["stats"]["moment_confidence_half_widths"]["xy"],
    )


def variance(
    term: z3.ExprRef,
    formula: z3.ExprRef,
    density: Density,
    options: Optional[WMIOptions] = None,
) -> InferenceResult:
    """
    Compute Var(term | formula) under the given density.
    """

    opts = options or WMIOptions()
    variables = _variables_for_terms(formula, term)
    _validate_wmi_options(opts, density, variables)
    method = _effective_method(density, opts, variables)
    if method == WMIMethod.EXACT_DISCRETE:
        result = covariance(term, term, formula, density, opts)
        return InferenceResult(
            value=float(result),
            exact=result.exact,
            backend=result.backend,
            stats=dict(
                result.stats,
                first_moment=result.stats["first_x"],
                second_moment=result.stats["mixed_moment"],
            ),
            error_bound=result.error_bound,
        )

    sampled = _sample_moment_estimates(
        {"x": term, "x2": term * term},
        formula,
        density,
        opts,
    )
    variance_value = sampled["estimates"]["x2"] - sampled["estimates"]["x"] ** 2
    return InferenceResult(
        value=variance_value,
        exact=False,
        backend=sampled["backend"],
        stats=dict(
            sampled["stats"],
            first_moment=sampled["estimates"]["x"],
            second_moment=sampled["estimates"]["x2"],
        ),
        error_bound=sampled["stats"]["moment_confidence_half_widths"]["x2"],
    )


__all__ = ["moment", "expectation", "covariance", "variance"]
