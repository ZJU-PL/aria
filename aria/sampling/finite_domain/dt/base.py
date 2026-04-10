"""
Sampler for quantifier-free algebraic datatype formulas.
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
    collect_datatype_observable_terms,
    enumerate_projected_models,
)


class DatatypeSampler(Sampler):
    """Enumerate models over datatype-valued variables."""

    def __init__(self, **_kwargs: Any) -> None:
        self.formula: Optional[z3.ExprRef] = None
        self.default_terms: List[z3.ExprRef] = []

    def supports_logic(self, logic: Logic) -> bool:
        return logic == Logic.QF_DT

    def init_from_formula(self, formula: z3.ExprRef) -> None:
        self.formula = formula
        self.default_terms = collect_datatype_observable_terms(
            formula, include_selector_closure=False
        )

    def sample(self, options: SamplingOptions) -> SamplingResult:
        if self.formula is None:
            raise ValueError("Sampler not initialized with a formula")

        include_selector_closure = bool(
            options.additional_options.get("include_selector_closure", False)
        )
        observed_terms = collect_datatype_observable_terms(
            self.formula,
            include_selector_closure=include_selector_closure,
        )
        return enumerate_projected_models(
            self.formula,
            options,
            observed_terms,
            default_terms=self.default_terms,
            solver_factory=lambda: z3.SolverFor("QF_DT"),
        )

    def get_supported_methods(self) -> Set[SamplingMethod]:
        return {SamplingMethod.ENUMERATION}

    def get_supported_logics(self) -> Set[Logic]:
        return {Logic.QF_DT, Logic.QF_UFDT}
