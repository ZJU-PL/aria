"""
Exact discrete helpers and backends for arithmetic probabilistic inference.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import z3

from aria.counting.arith.arith_counting_latte import count_lia_models
from aria.prob.core._helpers import evaluate_term
from aria.prob.core.density import UniformDensity
from aria.prob.core.results import InferenceResult
from aria.utils.z3_expr_utils import z3_value_to_python


def _support_formula(
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


def _validate_exact_discrete_density(
    density: UniformDensity, variables: List[z3.ExprRef]
) -> Dict[str, Tuple[float, float]]:
    if not density.discrete:
        raise ValueError("Exact discrete integration requires UniformDensity(discrete=True)")

    if any(var.sort() != z3.IntSort() for var in variables):
        raise ValueError("Exact discrete integration currently supports Int variables only")

    return density.support()


def _exact_discrete_solver(
    formula: z3.ExprRef, density: UniformDensity, variables: List[z3.ExprRef]
) -> z3.Solver:
    bounds = _validate_exact_discrete_density(density, variables)
    support_formula = _support_formula(variables, bounds)
    solver = z3.Solver()
    solver.add(z3.And(formula, support_formula))
    return solver


def _exact_discrete_expectation(
    term: z3.ExprRef,
    formula: z3.ExprRef,
    density: UniformDensity,
    variables: List[z3.ExprRef],
) -> InferenceResult:
    solver = _exact_discrete_solver(formula, density, variables)

    count = 0
    total = 0.0
    while solver.check() == z3.sat:
        model = solver.model()
        assignment = {}
        block = []
        for var in variables:
            value = model.eval(var, model_completion=True)
            assignment[str(var)] = z3_value_to_python(value)
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


def _exact_discrete_mass(
    formula: z3.ExprRef, density: UniformDensity, variables: List[z3.ExprRef]
) -> InferenceResult:
    bounds = _validate_exact_discrete_density(density, variables)
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


__all__ = [
    "_exact_discrete_expectation",
    "_exact_discrete_mass",
    "_exact_discrete_solver",
    "_support_formula",
]
