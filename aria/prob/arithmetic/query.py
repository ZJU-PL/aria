"""
Probability queries for arithmetic probabilistic models.
"""

from __future__ import annotations

from typing import Optional

import z3

from aria.prob.core.density import Density
from aria.prob.core.results import InferenceResult
from aria.prob.arithmetic.wmi import WMIOptions, wmi_integrate


def probability(
    formula: z3.ExprRef,
    density: Density,
    evidence: Optional[z3.ExprRef] = None,
    options: Optional[WMIOptions] = None,
) -> InferenceResult:
    """
    Compute P(formula | evidence) under an arithmetic density.
    """

    if not isinstance(formula, z3.ExprRef):
        raise ValueError("Arithmetic probability queries require a Z3 formula")
    if evidence is not None and not isinstance(evidence, z3.ExprRef):
        raise ValueError("Arithmetic evidence must be a Z3 formula")
    if not isinstance(density, Density):
        raise ValueError("Arithmetic probability queries require a density model")

    numerator_formula = z3.And(formula, evidence) if evidence is not None else formula
    numerator = wmi_integrate(numerator_formula, density, options)

    if evidence is None:
        if density.is_normalized():
            denominator_value = 1.0
            exact = numerator.exact
            error_bound = numerator.error_bound
        else:
            denominator = wmi_integrate(z3.BoolVal(True), density, options)
            denominator_value = float(denominator)
            exact = numerator.exact and denominator.exact
            error_bound = None
    else:
        denominator = wmi_integrate(evidence, density, options)
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
    query: z3.ExprRef,
    evidence: z3.ExprRef,
    density: Density,
    options: Optional[WMIOptions] = None,
) -> InferenceResult:
    return probability(query, density, evidence=evidence, options=options)


__all__ = ["probability", "conditional_probability"]
