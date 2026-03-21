"""
Backend selection and input validation for arithmetic inference.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import z3

from aria.prob.core._helpers import finite_support
from aria.prob.core.density import (
    Density,
    DiscreteFactorizedDensity,
    UniformDensity,
)
from aria.utils.z3_expr_utils import get_variables

from ._config import WMIMethod, WMIOptions


def _coerce_method(method: Any) -> WMIMethod:
    if isinstance(method, WMIMethod):
        return method
    legacy = {
        "sampling": WMIMethod.AUTO,
        "region": WMIMethod.BOUNDED_SUPPORT_MONTE_CARLO,
    }
    coerced = str(method)
    if coerced in legacy:
        return legacy[coerced]
    return WMIMethod(coerced)


def _validate_wmi_options(
    options: WMIOptions, density: Density, variables: List[z3.ExprRef]
) -> None:
    if options.num_samples <= 0:
        raise ValueError("WMI num_samples must be positive")
    if options.confidence_level <= 0.0 or options.confidence_level >= 1.0:
        raise ValueError("WMI confidence_level must lie strictly between 0 and 1")

    method = _coerce_method(options.method)
    if method != WMIMethod.EXACT_DISCRETE:
        return

    if isinstance(density, UniformDensity) and density.discrete:
        if any(var.sort() != z3.IntSort() for var in variables):
            raise ValueError(
                "Exact discrete integration currently supports Int variables only"
            )
        return
    if isinstance(density, DiscreteFactorizedDensity):
        if any(var.sort() != z3.IntSort() for var in variables):
            raise ValueError(
                "Exact discrete integration currently supports Int variables only"
            )
        return
    raise ValueError(
        "Exact discrete integration requires a discrete integer-valued density"
    )


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
        if not isinstance(min_val, (int, float)) or not isinstance(
            max_val, (int, float)
        ):
            raise ValueError(
                "Density support for '{}' must be numeric".format(var_name)
            )
        if math.isnan(min_val) or math.isnan(max_val):
            raise ValueError(
                "Density support for '{}' cannot contain NaN".format(var_name)
            )


def _effective_method(
    density: Density, options: WMIOptions, variables: List[z3.ExprRef]
) -> WMIMethod:
    method = _coerce_method(options.method)
    if method != WMIMethod.AUTO:
        return method

    if isinstance(density, UniformDensity) and density.discrete:
        return WMIMethod.EXACT_DISCRETE
    if isinstance(density, DiscreteFactorizedDensity):
        if any(var.sort() != z3.IntSort() for var in variables):
            return WMIMethod.IMPORTANCE_SAMPLING
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


__all__ = [
    "_coerce_method",
    "_effective_method",
    "_validate_wmi_inputs",
    "_validate_wmi_options",
]
