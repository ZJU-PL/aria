"""
Sampler for formulas mixing algebraic datatypes and linear integer arithmetic.
"""

from typing import Any, Dict, List, Optional, Set, cast

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
from aria.utils.z3.expr import get_variables, is_int_sort

from .candidate_selector import (
    select_coverage_guided_subset,
    select_max_distance_subset,
)
from .shape_enumerator import (
    enumerate_datatype_shapes,
)


def _dedupe_sorted_terms(terms: List[z3.ExprRef]) -> List[z3.ExprRef]:
    """Deduplicate tracked terms while preserving deterministic ordering."""
    return list({str(term): term for term in sorted(terms, key=str)}.values())


class ADTLIASampler(Sampler):
    """Practical sampler for formulas combining ADTs with linear integers."""

    def __init__(self, **_kwargs: Any) -> None:
        self.formula: Optional[z3.ExprRef] = None
        self.int_variables: List[z3.ExprRef] = []
        self.datatype_roots: List[z3.ExprRef] = []
        self.default_terms: List[z3.ExprRef] = []

    def supports_logic(self, logic: Logic) -> bool:
        return logic == Logic.QF_DTLIA

    def init_from_formula(self, formula: z3.ExprRef) -> None:
        self.formula = formula
        self.int_variables = sorted(
            [var for var in get_variables(formula) if is_int_sort(var)],
            key=str,
        )
        self.datatype_roots = sorted(
            [
                var
                for var in get_variables(formula)
                if var.sort().kind() == z3.Z3_DATATYPE_SORT
            ],
            key=str,
        )
        datatype_terms = collect_datatype_observable_terms(
            formula, include_selector_closure=False
        )
        self.default_terms = _dedupe_sorted_terms(self.int_variables + datatype_terms)

    def sample(self, options: SamplingOptions) -> SamplingResult:
        if self.formula is None:
            raise ValueError("Sampler not initialized with a formula")
        formula = cast(z3.ExprRef, self.formula)

        if options.method == SamplingMethod.SEARCH_TREE:
            return self._sample_via_shapes(options)

        datatype_terms = collect_datatype_observable_terms(
            formula,
            include_selector_closure=bool(
                options.additional_options.get("include_selector_closure", False)
            ),
        )
        tracked_terms = _dedupe_sorted_terms(self.int_variables + datatype_terms)
        return enumerate_projected_models(
            formula,
            options,
            tracked_terms,
            default_terms=self.default_terms,
        )

    def get_supported_methods(self) -> Set[SamplingMethod]:
        return {SamplingMethod.ENUMERATION, SamplingMethod.SEARCH_TREE}

    def get_supported_logics(self) -> Set[Logic]:
        return {Logic.QF_DTLIA}

    def _sample_via_shapes(self, options: SamplingOptions) -> SamplingResult:
        if self.formula is None:
            raise ValueError("Sampler not initialized with a formula")
        formula = cast(z3.ExprRef, self.formula)

        include_selector_closure = bool(
            options.additional_options.get("include_selector_closure", False)
        )
        explicit_projection_terms = options.additional_options.get("projection_terms")
        explicit_tracked_terms = options.additional_options.get("tracked_terms")
        return_full_model = bool(
            options.additional_options.get("return_full_model", False)
        )
        tracked_terms = _dedupe_sorted_terms(
            self.int_variables
            + collect_datatype_observable_terms(
                formula,
                include_selector_closure=include_selector_closure,
            )
        )

        max_shapes = int(options.additional_options.get("max_shapes", options.num_samples))
        candidates_per_shape = int(
            options.additional_options.get("candidates_per_shape", options.num_samples)
        )
        diversity_mode = str(
            options.additional_options.get("diversity_mode", "enumeration")
        )

        shapes = enumerate_datatype_shapes(
            formula,
            self.datatype_roots,
            max_shapes=max_shapes,
            random_seed=options.random_seed,
            timeout=options.timeout,
        )
        if not shapes:
            return enumerate_projected_models(
                formula,
                SamplingOptions(
                    method=SamplingMethod.ENUMERATION,
                    num_samples=options.num_samples,
                    timeout=options.timeout,
                    random_seed=options.random_seed,
                    **options.additional_options,
                ),
                tracked_terms,
                default_terms=self.default_terms,
            )

        candidate_samples: List[Dict[str, Any]] = []
        candidate_samples_by_shape: List[List[Dict[str, Any]]] = []
        shape_sample_counts: List[int] = []
        shape_payload_terms: List[List[str]] = []
        candidate_shape_signatures: List[str] = []

        for shape in shapes:
            residual_formula = cast(
                z3.ExprRef,
                z3.And(formula, *shape.constraints) if shape.constraints else formula,
            )
            residual_tracked_terms = _dedupe_sorted_terms(
                self.int_variables
                + collect_datatype_observable_terms(
                    residual_formula,
                    include_selector_closure=include_selector_closure,
                )
            )
            residual_projection_terms = (
                explicit_projection_terms
                if explicit_projection_terms is not None
                else list(shape.payload_terms)
            )
            output_terms = (
                explicit_tracked_terms
                if explicit_tracked_terms is not None
                else residual_tracked_terms
            )
            residual_result = enumerate_projected_models(
                residual_formula,
                SamplingOptions(
                    method=SamplingMethod.ENUMERATION,
                    num_samples=candidates_per_shape,
                    timeout=options.timeout,
                    random_seed=options.random_seed,
                    projection_terms=residual_projection_terms,
                    tracked_terms=output_terms,
                    return_full_model=return_full_model,
                    include_selector_closure=include_selector_closure,
                ),
                residual_tracked_terms,
                default_terms=self.default_terms,
            )
            candidate_samples.extend(residual_result.samples)
            candidate_samples_by_shape.append(list(residual_result.samples))
            shape_sample_counts.append(len(residual_result.samples))
            shape_payload_terms.append([str(term) for term in shape.payload_terms])
            candidate_shape_signatures.extend(
                [str(shape.signature)] * len(residual_result.samples)
            )

        coverage_stats: Dict[str, Any] = {}
        if diversity_mode == "max_distance":
            selected_samples = select_max_distance_subset(
                candidate_samples, options.num_samples
            )
        elif diversity_mode == "coverage_guided":
            coverage_result = select_coverage_guided_subset(
                candidate_samples,
                options.num_samples,
                shape_signatures=candidate_shape_signatures,
            )
            selected_samples = coverage_result.samples
            coverage_stats = coverage_result.stats
        else:
            selected_samples = self._select_shape_first_round_robin(
                candidate_samples_by_shape,
                options.num_samples,
            )

        stats = {
            "time_ms": 0,
            "iterations": len(selected_samples),
            "method": options.method.value,
            "shape_count": len(shapes),
            "candidate_count": len(candidate_samples),
            "candidates_per_shape": candidates_per_shape,
            "diversity_mode": diversity_mode,
            "shape_signatures": [str(shape.signature) for shape in shapes],
            "shape_sample_counts": shape_sample_counts,
            "shape_payload_terms": shape_payload_terms,
            "residual_projection_mode": (
                "explicit"
                if explicit_projection_terms is not None
                else "payload_terms"
            ),
        }
        stats.update(coverage_stats)
        return SamplingResult(selected_samples, stats)

    @staticmethod
    def _select_shape_first_round_robin(
        candidate_samples_by_shape: List[List[Dict[str, Any]]],
        num_samples: int,
    ) -> List[Dict[str, Any]]:
        """Pick samples in round-robin order so shapes share the sample budget."""
        if num_samples <= 0:
            return []

        pending = [list(shape_samples) for shape_samples in candidate_samples_by_shape]
        selected: List[Dict[str, Any]] = []

        while len(selected) < num_samples:
            progressed = False
            for shape_samples in pending:
                if not shape_samples:
                    continue
                selected.append(shape_samples.pop(0))
                progressed = True
                if len(selected) >= num_samples:
                    break
            if not progressed:
                break

        return selected
