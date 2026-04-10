"""
Sampler for quantifier-free formulas with uninterpreted functions.

The sampler enumerates satisfying assignments over the ground uninterpreted
constants and function applications that occur in the formula.
"""

from typing import Any, Dict, List, Optional, Set

import z3

from aria.sampling.base import (
    Logic,
    Sampler,
    SamplingMethod,
    SamplingOptions,
    SamplingResult,
)
from aria.sampling.finite_domain.common import (
    build_sample,
    block_model_on_terms,
    collect_ground_uf_terms,
    resolve_output_terms,
    resolve_projection_terms,
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
        self.function_terms = collect_ground_uf_terms(formula)

    def sample(self, options: SamplingOptions) -> SamplingResult:
        if self.formula is None:
            raise ValueError("Sampler not initialized with a formula")

        solver = z3.Solver()
        if options.random_seed is not None:
            solver.set("random_seed", options.random_seed)
            solver.set("seed", options.random_seed)
        solver.add(self.formula)

        tracked_terms = self.constants + self.function_terms
        projection_terms = resolve_projection_terms(
            tracked_terms, options.additional_options.get("projection_terms")
        )
        output_terms = resolve_output_terms(
            tracked_terms,
            options.additional_options.get("projection_terms"),
            options.additional_options.get("tracked_terms"),
            bool(options.additional_options.get("return_full_model", False)),
        )
        samples: List[Dict[str, Any]] = []
        stats: Dict[str, Any] = {
            "time_ms": 0,
            "iterations": 0,
            "projection_terms": [str(term) for term in projection_terms],
            "output_terms": [str(term) for term in output_terms],
        }

        for _ in range(options.num_samples):
            if solver.check() != z3.sat:
                break

            model = solver.model()
            sample = build_sample(model, output_terms)
            samples.append(sample)
            stats["iterations"] += 1

            if not block_model_on_terms(solver, model, projection_terms):
                break

        return SamplingResult(samples, stats)

    def get_supported_methods(self) -> Set[SamplingMethod]:
        return {SamplingMethod.ENUMERATION}

    def get_supported_logics(self) -> Set[Logic]:
        return {Logic.QF_UF, Logic.QF_UFDT}
