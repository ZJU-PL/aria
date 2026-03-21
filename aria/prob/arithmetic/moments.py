"""
Moment queries for arithmetic probabilistic models.
"""

from __future__ import annotations

import random
from typing import Optional

import z3

from aria.prob.core._helpers import assignment_satisfies, evaluate_term
from aria.prob.core.density import Density, UniformDensity
from aria.prob.core.results import InferenceResult
from aria.utils.z3_expr_utils import get_variables

from ._config import WMIMethod, WMIOptions
from ._exact_backend import _exact_discrete_expectation
from ._selection import _effective_method
from ._sampling_utils import _uniform_sample_from_support


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
    variables = sorted(get_variables(formula), key=str)

    method = _effective_method(density, opts, variables)
    if (
        method == WMIMethod.EXACT_DISCRETE
        and isinstance(density, UniformDensity)
        and density.discrete
    ):
        powered_term = term if order == 1 else term ** order
        return _exact_discrete_expectation(powered_term, formula, density, variables)

    powered_term = term if order == 1 else term ** order

    rng = random.Random(opts.random_seed)
    weighted_sum = 0.0
    weight_sum = 0.0
    satisfied_samples = 0

    if method == WMIMethod.BOUNDED_SUPPORT_MONTE_CARLO:
        bounds = density.support()
        if bounds is None:
            raise ValueError(
                "Bounded-support Monte Carlo expectation requires finite support"
            )

        for _ in range(opts.num_samples):
            assignment = _uniform_sample_from_support(variables, bounds, rng)
            if not assignment_satisfies(formula, assignment):
                continue
            term_value = evaluate_term(powered_term, assignment)
            if not isinstance(term_value, (int, float)):
                raise ValueError("Expectation term must evaluate to a numeric value")
            weight = float(density(assignment))
            weighted_sum += weight * float(term_value)
            weight_sum += weight
            satisfied_samples += 1
        backend = "wmi-bounded-support-monte-carlo"
    else:
        proposal = opts.proposal or density
        for _ in range(opts.num_samples):
            assignment = proposal.sample_assignment(rng)
            if not assignment_satisfies(formula, assignment):
                continue
            proposal_value = float(proposal(assignment))
            if proposal_value <= 0.0:
                continue
            term_value = evaluate_term(powered_term, assignment)
            if not isinstance(term_value, (int, float)):
                raise ValueError("Expectation term must evaluate to a numeric value")
            weight = float(density(assignment)) / proposal_value
            weighted_sum += weight * float(term_value)
            weight_sum += weight
            satisfied_samples += 1
        backend = "wmi-importance-sampling"

    if weight_sum == 0.0:
        raise ValueError("Expectation is undefined because the conditioning event is empty")

    return InferenceResult(
        value=weighted_sum / weight_sum,
        exact=False,
        backend=backend,
        stats={
            "order": order,
            "sample_count": opts.num_samples,
            "satisfied_samples": satisfied_samples,
        },
        error_bound=None,
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

    first_x = moment(term_x, 1, formula, density, options)
    first_y = moment(term_y, 1, formula, density, options)
    mixed = moment(term_x * term_y, 1, formula, density, options)
    covariance_value = float(mixed) - float(first_x) * float(first_y)

    return InferenceResult(
        value=covariance_value,
        exact=mixed.exact and first_x.exact and first_y.exact,
        backend=mixed.backend,
        stats={
            "first_x": float(first_x),
            "first_y": float(first_y),
            "mixed_moment": float(mixed),
        },
        error_bound=None,
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

    result = covariance(term, term, formula, density, options)
    return InferenceResult(
        value=float(result),
        exact=result.exact,
        backend=result.backend,
        stats={
            "first_moment": result.stats["first_x"],
            "second_moment": result.stats["mixed_moment"],
        },
        error_bound=None,
    )


__all__ = ["moment", "expectation", "covariance", "variance"]
