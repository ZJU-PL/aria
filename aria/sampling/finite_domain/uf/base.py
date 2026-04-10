"""
Sampler for quantifier-free formulas with uninterpreted functions.

The sampler enumerates satisfying assignments over the ground uninterpreted
constants and function applications that occur in the formula.
"""

from typing import Any, List, Optional, Set

import z3

from aria.sampling.base import (
    Logic,
    Sampler,
    SamplingMethod,
    SamplingOptions,
    SamplingResult,
)
from aria.sampling.finite_domain.common import (
    collect_uf_observable_terms,
    enumerate_projected_models,
)
from aria.utils.z3.expr import get_variables


class UninterpretedFunctionSampler(Sampler):
    """Enumerate models for ground UF terms appearing in a formula."""

    def __init__(self, **_kwargs: Any) -> None:
        self.formula: Optional[z3.ExprRef] = None
        self.constants: List[z3.ExprRef] = []
        self.function_terms: List[z3.ExprRef] = []

    def supports_logic(self, logic: Logic) -> bool:
        return logic == Logic.QF_UF

    def init_from_formula(self, formula: z3.ExprRef) -> None:
        self.formula = formula
        self.constants = sorted(get_variables(formula), key=str)
        self.function_terms = collect_uf_observable_terms(formula)

    def sample(self, options: SamplingOptions) -> SamplingResult:
        if self.formula is None:
            raise ValueError("Sampler not initialized with a formula")

        tracked_terms = self.constants + self.function_terms
        return enumerate_projected_models(
            self.formula,
            options,
            tracked_terms,
        )

    def get_supported_methods(self) -> Set[SamplingMethod]:
        return {SamplingMethod.ENUMERATION}

    def get_supported_logics(self) -> Set[Logic]:
        return {Logic.QF_UF, Logic.QF_UFDT}
