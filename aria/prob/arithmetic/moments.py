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
from aria.utils.z3_expr_utils import get_variables, z3_value_to_python

from ._config import WMIMethod, WMIOptions
from ._sampling_utils import _uniform_sample_from_support
from ._selection import _effective_method


def _exact_discrete_expectation(
    term: z3.ExprRef,
    formula: z3.ExprRef,
    density: UniformDensity,
) -> InferenceResult:
    variables = sorted(get_variables(formula), key=str)
    if any(var.sort() != z3.IntSort() for var in variables):
        raise ValueError("Exact discrete expectation currently supports Int variables only")

    bounds = density.support()
    solver = z3.Solver()
    support_constraints = []
    for var in variables:
        min_val, max_val = bounds[str(var)]
        support_constraints.append(var >= int(min_val))
        support_constraints.append(var <= int(max_val))
    solver.add(z3.And(formula, *support_constraints))

    count = 0
    total = 0.0
    while solver.check() == z3.sat:
        model = solver.model()
        assignment = {}
        block = []
        for var in variables:
            value = model.eval(var, model_completion=True)
            python_value = z3_value_to_python(value)
            assignment[str(var)] = python_value
            block.append(var != value)

        term_value = evaluate_term(term, assignment)
        if not isinstance(term_value, (int, float)):
            raise ValueError("Expectation term must evaluate to a numeric value")
        total += float(term_value)
        count += 1
        solver.add(z3.Or(block))

    if count == 0:
        raise ValueError("Expectation is undefined because the conditioning event is empty")

    return InferenceResult(
        value=total / float(count),
        exact=True,
        backend="wmi-exact-discrete-uniform",
        stats={"model_count": count},
        error_bound=0.0,
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

    opts = options or WMIOptions()
    variables = sorted(get_variables(formula), key=str)

    method = _effective_method(density, opts, variables)
    if (
        method == WMIMethod.EXACT_DISCRETE
        and isinstance(density, UniformDensity)
        and density.discrete
    ):
        return _exact_discrete_expectation(term, formula, density)

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
            term_value = evaluate_term(term, assignment)
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
            term_value = evaluate_term(term, assignment)
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
            "sample_count": opts.num_samples,
            "satisfied_samples": satisfied_samples,
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

    first_moment = expectation(term, formula, density, options)
    second_moment = expectation(term * term, formula, density, options)
    variance_value = float(second_moment) - float(first_moment) * float(first_moment)

    return InferenceResult(
        value=variance_value,
        exact=first_moment.exact and second_moment.exact,
        backend=first_moment.backend,
        stats={
            "first_moment": float(first_moment),
            "second_moment": float(second_moment),
        },
        error_bound=None,
    )


__all__ = ["expectation", "variance"]
