"""Floating-point sampler implementation."""

import random
from typing import Any, Dict, List, Optional, Set, cast

import z3

from aria.sampling.base import (
    Logic,
    Sampler,
    SamplingMethod,
    SamplingOptions,
    SamplingResult,
)
from .common import (
    get_fp_render_mode,
    get_fp_variables,
    fp_model_value,
    render_fp_value,
)


class FloatingPointSampler(Sampler):
    """Enumeration-based sampler for quantifier-free floating-point formulas."""

    def __init__(self, **_kwargs: Any) -> None:
        self.formula: Optional[z3.ExprRef] = None
        self.variables: List[z3.ExprRef] = []

    def supports_logic(self, logic: Logic) -> bool:
        return logic == Logic.QF_FP

    def init_from_formula(self, formula: z3.ExprRef) -> None:
        self.formula = formula
        self.variables = get_fp_variables(formula)

    def sample(self, options: SamplingOptions) -> SamplingResult:
        if self.formula is None:
            raise ValueError("Sampler not initialized with a formula")

        render_mode = get_fp_render_mode(options)

        if options.random_seed is not None:
            random.seed(options.random_seed)

        solver = z3.Solver()
        if options.random_seed is not None:
            solver.set("random_seed", options.random_seed)
            solver.set("seed", options.random_seed)
        solver.add(self.formula)

        samples: List[Dict[str, Any]] = []
        stats: Dict[str, Any] = {"time_ms": 0, "iterations": 0}

        for _ in range(options.num_samples):
            if solver.check() != z3.sat:
                break

            model = solver.model()
            sample: Dict[str, Any] = {}
            block: List[z3.ExprRef] = []

            for var in self.variables:
                value = fp_model_value(model, var)
                bits = model.evaluate(z3.fpToIEEEBV(var), model_completion=True)
                sample[str(var)] = render_fp_value(value, render_mode)
                block.append(cast(z3.ExprRef, z3.fpToIEEEBV(var) != bits))

            samples.append(sample)

            if block:
                solver.add(z3.Or(block))
            else:
                break
            stats["iterations"] += 1

        return SamplingResult(samples, stats)

    def get_supported_methods(self) -> Set[SamplingMethod]:
        return {SamplingMethod.ENUMERATION}

    def get_supported_logics(self) -> Set[Logic]:
        return {Logic.QF_FP}
