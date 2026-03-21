"""
Exact discrete backends for arithmetic probabilistic inference.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import z3

from aria.counting.arith.arith_counting_latte import count_lia_models
from aria.prob.core.density import UniformDensity
from aria.prob.core.results import InferenceResult


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


__all__ = ["_exact_discrete_mass", "_support_formula"]
