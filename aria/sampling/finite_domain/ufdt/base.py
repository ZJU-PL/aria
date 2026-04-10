"""
Sampler for quantifier-free UF+datatype formulas.
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
    collect_ground_uf_terms,
    enumerate_projected_models,
)
from aria.utils.z3.expr import get_variables


def _dedupe_sorted_terms(terms: List[z3.ExprRef]) -> List[z3.ExprRef]:
    """Deduplicate tracked terms while preserving deterministic ordering."""
    return list({str(term): term for term in sorted(terms, key=str)}.values())


class MixedUFDatatypeSampler(Sampler):
    """Enumerate models for formulas mixing UF and algebraic datatypes."""

    def __init__(self, **_kwargs: Any) -> None:
        self.formula: Optional[z3.ExprRef] = None
        self.constants: List[z3.ExprRef] = []
        self.function_terms: List[z3.ExprRef] = []
        self.default_terms: List[z3.ExprRef] = []

    def supports_logic(self, logic: Logic) -> bool:
        return logic == Logic.QF_UFDT

    def init_from_formula(self, formula: z3.ExprRef) -> None:
        self.formula = formula
        self.constants = sorted(get_variables(formula), key=str)
        self.function_terms = collect_ground_uf_terms(formula)
        datatype_terms = collect_datatype_observable_terms(
            formula, include_selector_closure=False
        )
        self.default_terms = _dedupe_sorted_terms(
            self.constants + self.function_terms + datatype_terms
        )

    def sample(self, options: SamplingOptions) -> SamplingResult:
        if self.formula is None:
            raise ValueError("Sampler not initialized with a formula")

        include_selector_closure = bool(
            options.additional_options.get("include_selector_closure", False)
        )
        datatype_terms = collect_datatype_observable_terms(
            self.formula,
            include_selector_closure=include_selector_closure,
        )
        tracked_terms = _dedupe_sorted_terms(
            self.constants + self.function_terms + datatype_terms
        )
        return enumerate_projected_models(
            self.formula,
            options,
            tracked_terms,
            default_terms=self.default_terms,
        )

    def get_supported_methods(self) -> Set[SamplingMethod]:
        return {SamplingMethod.ENUMERATION}

    def get_supported_logics(self) -> Set[Logic]:
        return {Logic.QF_UFDT}
