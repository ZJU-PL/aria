"""
Exact discrete helpers and backends for arithmetic probabilistic inference.
"""

from __future__ import annotations

from itertools import product
from typing import Any, Dict, Iterator, List, Optional, Tuple

import z3

from aria.counting.arith.arith_counting_latte import count_lia_models
from aria.prob.core._helpers import assignment_satisfies, evaluate_term
from aria.prob.core.density import DiscreteFactorizedDensity, UniformDensity
from aria.prob.core.results import InferenceResult


def _exact_conditioning_stats(
    conditioning_probability: float,
    satisfying_assignment_count: int,
    total_assignment_count: Optional[int] = None,
    numerator_count: Optional[int] = None,
    denominator_count: Optional[int] = None,
    numerator_weight: Optional[float] = None,
    denominator_weight: Optional[float] = None,
) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "conditioning_mass": conditioning_probability,
        "conditioning_probability": conditioning_probability,
        "effective_conditioning_weight": conditioning_probability,
        "satisfying_assignment_count": satisfying_assignment_count,
    }
    if total_assignment_count is not None:
        stats["support_assignment_count"] = total_assignment_count
    if numerator_count is not None:
        stats["numerator_count"] = numerator_count
    if denominator_count is not None:
        stats["denominator_count"] = denominator_count
    if numerator_weight is not None:
        stats["numerator_weight"] = numerator_weight
    if denominator_weight is not None:
        stats["denominator_weight"] = denominator_weight
    return stats


def _uniform_support_formula(
    variables: List[z3.ExprRef], bounds: Dict[str, Tuple[float, float]]
) -> z3.BoolRef:
    constraints = []
    for var in variables:
        var_name = str(var)
        if var_name not in bounds:
            raise ValueError(
                "Density support is missing bounds for variable '{}'".format(var_name)
            )
        min_val, max_val = bounds[var_name]
        if not float(min_val).is_integer() or not float(max_val).is_integer():
            raise ValueError(
                "Exact discrete integration requires integer bounds for '{}'".format(
                    var_name
                )
            )
        constraints.append(var >= int(min_val))
        constraints.append(var <= int(max_val))
    return z3.And(*constraints) if constraints else z3.BoolVal(True)


def _validate_exact_discrete_density(
    density: Any, variables: List[z3.ExprRef]
) -> Any:
    if any(var.sort() != z3.IntSort() for var in variables):
        raise ValueError("Exact discrete integration currently supports Int variables only")

    if isinstance(density, UniformDensity):
        if not density.discrete:
            raise ValueError(
                "Exact discrete integration requires UniformDensity(discrete=True)"
            )
        return density.support()

    if isinstance(density, DiscreteFactorizedDensity):
        support = density.discrete_support()
        for var in variables:
            if str(var) not in support:
                raise ValueError(
                    "Density support is missing values for variable '{}'".format(str(var))
                )
        return support

    raise ValueError("Exact discrete integration requires a supported discrete density")


def _iter_discrete_assignments(
    density: Any, variables: List[z3.ExprRef]
) -> Iterator[Dict[str, int]]:
    support = _validate_exact_discrete_density(density, variables)
    if isinstance(density, UniformDensity):
        value_lists = []
        for var in variables:
            min_val, max_val = support[str(var)]
            value_lists.append(list(range(int(min_val), int(max_val) + 1)))
    else:
        value_lists = [list(support[str(var)]) for var in variables]

    for values in product(*value_lists):
        yield {str(var): int(value) for var, value in zip(variables, values)}


def _exact_discrete_solver(
    formula: z3.ExprRef, density: UniformDensity, variables: List[z3.ExprRef]
) -> z3.Solver:
    bounds = _validate_exact_discrete_density(density, variables)
    support_formula = _uniform_support_formula(variables, bounds)
    solver = z3.Solver()
    solver.add(z3.And(formula, support_formula))
    return solver


def _exact_discrete_expectation(
    term: z3.ExprRef,
    formula: z3.ExprRef,
    density: Any,
    variables: List[z3.ExprRef],
) -> InferenceResult:
    total = 0.0
    total_weight = 0.0
    model_count = 0
    for assignment in _iter_discrete_assignments(density, variables):
        if not assignment_satisfies(formula, assignment):
            continue
        weight = float(density(assignment))
        if weight <= 0.0:
            continue
        term_value = evaluate_term(term, assignment)
        if not isinstance(term_value, (int, float)):
            raise ValueError("Expectation term must evaluate to a numeric value")
        total += weight * float(term_value)
        total_weight += weight
        model_count += 1

    if total_weight == 0.0:
        raise ValueError("Expectation is undefined because the conditioning event is empty")

    return InferenceResult(
        value=total / total_weight,
        exact=True,
        backend="wmi-exact-discrete"
        if isinstance(density, DiscreteFactorizedDensity)
        else "wmi-exact-discrete-uniform",
        stats=_exact_conditioning_stats(
            conditioning_probability=total_weight,
            satisfying_assignment_count=model_count,
        ),
        error_bound=0.0,
    )


def _exact_discrete_mass(
    formula: z3.ExprRef, density: Any, variables: List[z3.ExprRef]
) -> InferenceResult:
    if isinstance(density, UniformDensity):
        bounds = _validate_exact_discrete_density(density, variables)
        support_formula = _uniform_support_formula(variables, bounds)
        numerator = count_lia_models(z3.And(formula, support_formula))
        denominator = count_lia_models(support_formula)
        if denominator == 0:
            raise ValueError("Discrete uniform support is empty")

        return InferenceResult(
            value=float(numerator) / float(denominator),
            exact=True,
            backend="wmi-exact-discrete-uniform",
            stats=dict(
                _exact_conditioning_stats(
                    conditioning_probability=float(numerator) / float(denominator),
                    satisfying_assignment_count=numerator,
                    total_assignment_count=denominator,
                    numerator_count=numerator,
                    denominator_count=denominator,
                ),
                num_variables=len(variables),
            ),
            error_bound=0.0,
        )

    _validate_exact_discrete_density(density, variables)
    numerator_weight = 0.0
    denominator_weight = 0.0
    support_size = 0
    satisfying_support_size = 0
    for assignment in _iter_discrete_assignments(density, variables):
        weight = float(density(assignment))
        if weight <= 0.0:
            continue
        support_size += 1
        denominator_weight += weight
        if assignment_satisfies(formula, assignment):
            numerator_weight += weight
            satisfying_support_size += 1
    if denominator_weight == 0.0:
        raise ValueError("Discrete support is empty")

    return InferenceResult(
        value=numerator_weight / denominator_weight,
        exact=True,
        backend="wmi-exact-discrete",
        stats=dict(
            _exact_conditioning_stats(
                conditioning_probability=numerator_weight / denominator_weight,
                satisfying_assignment_count=satisfying_support_size,
                total_assignment_count=support_size,
                numerator_weight=numerator_weight,
                denominator_weight=denominator_weight,
            ),
            num_variables=len(variables),
        ),
        error_bound=0.0,
    )


__all__ = [
    "_exact_discrete_expectation",
    "_exact_discrete_mass",
    "_exact_discrete_solver",
    "_uniform_support_formula",
]
