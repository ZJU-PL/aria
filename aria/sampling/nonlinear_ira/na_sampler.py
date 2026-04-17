"""
Sampling for nonlinear arithmetic formulas.

This module provides a basic sampler for QF_NRA and QF_NIA formulas using
model enumeration and the existing search-tree sampler.
"""

from typing import Any, Dict, List, Optional, Set, cast

import z3

from aria.sampling.base import Logic, Sampler, SamplingMethod, SamplingOptions
from aria.sampling.base import SamplingResult
from aria.sampling.general_sampler.searchtree_sampler import search_tree_sample
from aria.utils.z3.expr import get_variables, is_int_sort, is_real_sort


def _z3_number_to_python(value: z3.ExprRef) -> Any:
    """Convert a Z3 numeric value into a Python value when practical."""
    if z3.is_int_value(value):
        return value.as_long()
    if z3.is_rational_value(value):
        return value.as_decimal(20)
    if isinstance(value, z3.AlgebraicNumRef):
        return value.approx(20).as_decimal(20)
    return str(value)


class NASampler(Sampler):
    """Sampler for nonlinear integer and real arithmetic formulas."""

    def __init__(self, **_kwargs):
        self.formula: Optional[z3.ExprRef] = None
        self.variables: List[z3.ExprRef] = []

    def supports_logic(self, logic: Logic) -> bool:
        return logic in {Logic.QF_NRA, Logic.QF_NIA}

    def init_from_formula(self, formula: z3.ExprRef) -> None:
        self.formula = formula
        self.variables = [
            var
            for var in get_variables(formula)
            if is_int_sort(var) or is_real_sort(var)
        ]
        self.variables.sort(key=str)

    def sample(self, options: SamplingOptions) -> SamplingResult:
        if self.formula is None:
            raise ValueError("Sampler not initialized with a formula")

        if options.method == SamplingMethod.SEARCH_TREE:
            return self._sample_via_search_tree(options)
        if options.method != SamplingMethod.ENUMERATION:
            raise ValueError(
                "NASampler only supports enumeration and search-tree sampling"
            )
        return self._sample_via_enumeration(options)

    def _sample_via_enumeration(self, options: SamplingOptions) -> SamplingResult:
        assert self.formula is not None
        solver = z3.Solver()
        if options.random_seed is not None:
            solver.set("random_seed", options.random_seed)
            solver.set("seed", options.random_seed)
        solver.add(self.formula)

        samples: List[Dict[str, Any]] = []
        stats: Dict[str, Any] = {
            "method": SamplingMethod.ENUMERATION.value,
            "iterations": 0,
        }

        for _ in range(options.num_samples):
            if solver.check() != z3.sat:
                break

            model = solver.model()
            samples.append(self._model_to_sample(model))
            blocking_literals = self._build_blocking_clause(model)
            if not blocking_literals:
                break
            solver.add(z3.Or(blocking_literals))
            stats["iterations"] += 1

        return SamplingResult(samples, stats)

    def _sample_via_search_tree(self, options: SamplingOptions) -> SamplingResult:
        assert self.formula is not None
        if not self.variables:
            return SamplingResult([], {"method": SamplingMethod.SEARCH_TREE.value})

        samples: List[Dict[str, Any]] = []
        stats: Dict[str, Any] = {
            "method": SamplingMethod.SEARCH_TREE.value,
            "iterations": 0,
        }

        for _ in range(options.num_samples):
            model = search_tree_sample(
                self.variables,
                cast(z3.BoolRef, self.formula),
                options.additional_options.get("search_tree_width", 2),
            )
            if model is None:
                break
            samples.append(self._model_to_sample(model))
            stats["iterations"] += 1

        return SamplingResult(samples, stats)

    def _model_to_sample(self, model: z3.ModelRef) -> Dict[str, Any]:
        sample: Dict[str, Any] = {}
        for var in self.variables:
            value = model.eval(var, model_completion=True)
            sample[str(var)] = _z3_number_to_python(value)
        return sample

    def _build_blocking_clause(self, model: z3.ModelRef) -> List[z3.BoolRef]:
        literals: List[z3.BoolRef] = []
        for var in self.variables:
            value = model.eval(var, model_completion=True)
            literals.append(var != value)
        return literals

    def get_supported_methods(self) -> Set[SamplingMethod]:
        return {SamplingMethod.ENUMERATION, SamplingMethod.SEARCH_TREE}

    def get_supported_logics(self) -> Set[Logic]:
        return {Logic.QF_NRA, Logic.QF_NIA}


if __name__ == "__main__":
    x, y = z3.Reals("x y")
    formula = z3.And(x * x + y * y < 1, x > 0, y > 0)
    sampler = NASampler()
    sampler.init_from_formula(formula)
    samples = sampler.sample(SamplingOptions(num_samples=5))
    for sample in samples:
        print(sample)
