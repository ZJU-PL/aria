"""
High-level probability and expectation queries.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Union

import z3
from pysat.formula import CNF

from aria.prob._helpers import (
    assignment_satisfies,
    evaluate_term,
    merge_cnfs,
)
from aria.prob.density import Density, UniformDensity
from aria.prob.results import InferenceResult
from aria.prob.wmc.base import LiteralWeights, WMCBackend, WMCOptions
from aria.prob.wmc.wmc import CompiledWMC, compile_wmc, wmc_count
from aria.prob.wmc.wmi import WMIMethod, WMIOptions, _effective_method, wmi_integrate
from aria.utils.z3_expr_utils import get_variables, z3_value_to_python


def _strict_wmc_options(options: Optional[WMCOptions]) -> WMCOptions:
    opts = options or WMCOptions()
    if opts.strict_complements:
        return opts
    return replace(opts, strict_complements=True)


def _literal_sequence(value: Optional[Union[int, Sequence[int]]]) -> List[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    return [int(lit) for lit in value]


def probability(
    formula: Union[CNF, z3.ExprRef, int, Sequence[int]],
    model: Union[CompiledWMC, LiteralWeights, Density],
    evidence: Optional[Union[CNF, z3.ExprRef, int, Sequence[int]]] = None,
    options: Optional[Union[WMCOptions, WMIOptions]] = None,
) -> InferenceResult:
    """
    Compute P(formula | evidence) under a Boolean weighted model or arithmetic density.
    """

    if isinstance(model, CompiledWMC):
        if isinstance(formula, CNF) or isinstance(evidence, CNF):
            raise ValueError(
                "CompiledWMC probability queries expect literals, not CNF formulas"
            )
        return model.probability(
            query=_literal_sequence(formula),
            evidence=_literal_sequence(evidence),
        )

    if isinstance(formula, CNF):
        if not isinstance(model, dict):
            raise ValueError("CNF probability queries require a literal weight map")
        if evidence is not None and not isinstance(evidence, CNF):
            raise ValueError("CNF evidence must also be a CNF formula")

        opts = _strict_wmc_options(options if isinstance(options, WMCOptions) else None)
        if opts.backend == WMCBackend.DNNF:
            numerator_compiled = compile_wmc(
                merge_cnfs(formula, evidence), model, opts
            )
            numerator = numerator_compiled.count()
            if evidence is None:
                denominator = 1.0
            else:
                denominator = compile_wmc(evidence, model, opts).count()
            if denominator == 0.0:
                raise ValueError("Evidence CNF has zero probability under the weights")
            return InferenceResult(
                value=numerator / denominator,
                exact=True,
                backend="wmc-dnnf",
                stats={"numerator": numerator, "denominator": denominator},
                error_bound=0.0,
            )

        numerator = wmc_count(merge_cnfs(formula, evidence), model, opts)
        if evidence is None:
            denominator = 1.0
        else:
            denominator = wmc_count(evidence, model, opts)
        if denominator == 0.0:
            raise ValueError("Evidence CNF has zero probability under the weights")
        exact = opts.model_limit is None
        return InferenceResult(
            value=numerator / denominator,
            exact=exact,
            backend="wmc-enumeration",
            stats={
                "numerator": numerator,
                "denominator": denominator,
                "model_limit": opts.model_limit,
            },
            error_bound=None,
        )

    if not isinstance(formula, z3.ExprRef):
        raise ValueError("Arithmetic probability queries require a Z3 formula")
    if not isinstance(model, Density):
        raise ValueError("Arithmetic probability queries require a density model")

    wmi_options = options if isinstance(options, WMIOptions) else None
    numerator_formula = z3.And(formula, evidence) if evidence is not None else formula
    numerator = wmi_integrate(numerator_formula, model, wmi_options)

    if evidence is None:
        if model.is_normalized():
            denominator_value = 1.0
            exact = numerator.exact
            error_bound = numerator.error_bound
        else:
            denominator = wmi_integrate(z3.BoolVal(True), model, wmi_options)
            denominator_value = float(denominator)
            exact = numerator.exact and denominator.exact
            error_bound = None
    else:
        denominator = wmi_integrate(evidence, model, wmi_options)
        denominator_value = float(denominator)
        exact = numerator.exact and denominator.exact
        error_bound = None

    if denominator_value == 0.0:
        raise ValueError("Evidence has zero probability under the density")

    return InferenceResult(
        value=float(numerator) / denominator_value,
        exact=exact,
        backend=numerator.backend,
        stats={
            "numerator": float(numerator),
            "denominator": denominator_value,
        },
        error_bound=error_bound if evidence is None else None,
    )


def conditional_probability(
    query: Union[CNF, z3.ExprRef, int, Sequence[int]],
    evidence: Union[CNF, z3.ExprRef, int, Sequence[int]],
    model: Union[CompiledWMC, LiteralWeights, Density],
    options: Optional[Union[WMCOptions, WMIOptions]] = None,
) -> InferenceResult:
    return probability(query, model, evidence=evidence, options=options)


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
        from aria.prob.wmc.wmi import _uniform_sample_from_support

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
