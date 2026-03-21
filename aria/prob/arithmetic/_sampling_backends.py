"""
Sampling-based backends for arithmetic probabilistic inference.
"""

from __future__ import annotations

import random
from typing import List

import z3

from aria.prob.core._helpers import assignment_satisfies, finite_support
from aria.prob.core.density import Density
from aria.prob.core.results import InferenceResult

from ._config import WMIOptions
from ._sampling_utils import (
    _running_error_bound,
    _uniform_sample_from_support,
    _uniform_support_measure,
    _z_score,
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


__all__ = ["_bounded_support_monte_carlo", "_importance_sampling"]
