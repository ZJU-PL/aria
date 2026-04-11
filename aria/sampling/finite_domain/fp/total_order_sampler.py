"""IEEE total-order-guided sampling for quantifier-free floating-point formulas."""

import random
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import z3

from aria.sampling.base import (
    Logic,
    Sampler,
    SamplingMethod,
    SamplingOptions,
    SamplingResult,
)

from .common import (
    enumerate_fp_assignments,
    fp_assignment_total_order_key,
    get_fp_render_mode,
    get_fp_variables,
    render_fp_assignment,
)


def _tuple_distance(left: Tuple[int, ...], right: Tuple[int, ...]) -> int:
    return sum(abs(lv - rv) for lv, rv in zip(left, right))


def _spread_assignments(assignments: Sequence[Sequence[z3.ExprRef]], count: int) -> List[int]:
    if count <= 0 or not assignments:
        return []

    keys = [fp_assignment_total_order_key(values) for values in assignments]
    if count >= len(keys):
        return list(range(len(keys)))

    selected = [0]
    if len(keys) > 1:
        farthest_from_first = max(
            range(1, len(keys)), key=lambda idx: _tuple_distance(keys[0], keys[idx])
        )
        if farthest_from_first not in selected:
            selected.append(farthest_from_first)

    while len(selected) < count:
        candidate = max(
            (idx for idx in range(len(keys)) if idx not in selected),
            key=lambda idx: min(_tuple_distance(keys[idx], keys[chosen]) for chosen in selected),
        )
        selected.append(candidate)

    return sorted(selected, key=lambda idx: keys[idx])


class TotalOrderFPSampler(Sampler):
    """Sampler that spreads samples using IEEE totalOrder over FP assignments."""

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
        if options.random_seed is not None:
            random.seed(options.random_seed)

        pool_factor = int(options.additional_options.get("candidate_pool_factor", 8))
        min_pool = int(options.additional_options.get("candidate_pool_min", 16))
        candidate_pool = int(
            options.additional_options.get(
                "candidate_pool_size",
                max(options.num_samples * pool_factor, min_pool),
            )
        )

        assignments = enumerate_fp_assignments(
            self.formula,
            self.variables,
            candidate_pool,
            random_seed=options.random_seed,
        )
        selected_indexes = _spread_assignments(assignments, options.num_samples)
        samples: List[Dict[str, Any]] = [
            render_fp_assignment(self.variables, assignments[idx], render_mode)
            for idx in selected_indexes
        ]

        stats = {
            "time_ms": 0,
            "iterations": len(assignments),
            "selected_samples": len(samples),
            "candidate_pool_size": candidate_pool,
            "method": "total_order_guided",
        }
        return SamplingResult(samples, stats)

    def get_supported_methods(self) -> Set[SamplingMethod]:
        return {SamplingMethod.TOTAL_ORDER}

    def get_supported_logics(self) -> Set[Logic]:
        return {Logic.QF_FP}
