"""Hash-based sampling for quantifier-free floating-point formulas."""

import random
from typing import Any, Dict, List, Optional, Set

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
    get_uniform_samples_with_fp_xor,
    render_fp_value,
)


class HashBasedFPSampler(Sampler):
    """Approximate uniform sampler using XOR constraints over IEEE bits."""

    formula: Optional[z3.ExprRef]
    variables: List[z3.ExprRef]

    def __init__(self, **_kwargs: Any) -> None:
        self.formula = None
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

        if not self.variables:
            raise ValueError("No floating-point variables found in formula")

        if options.random_seed is not None:
            random.seed(options.random_seed)

        raw_samples = get_uniform_samples_with_fp_xor(
            self.variables,
            self.formula,
            options.num_samples,
        )

        samples: List[Dict[str, Any]] = []
        for raw_sample in raw_samples:
            sample: Dict[str, Any] = {}
            for var, value in zip(self.variables, raw_sample):
                sample[str(var)] = render_fp_value(value, render_mode)
            samples.append(sample)

        stats = {
            "time_ms": 0,
            "iterations": len(raw_samples),
            "attempted_samples": options.num_samples,
            "method": "hash_based_xor_ieee_bits",
        }
        return SamplingResult(samples, stats)

    def get_supported_methods(self) -> Set[SamplingMethod]:
        return {SamplingMethod.HASH_BASED}

    def get_supported_logics(self) -> Set[Logic]:
        return {Logic.QF_FP}
