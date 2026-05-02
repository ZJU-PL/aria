"""Sampler for formulas mixing strings and linear integer arithmetic."""

from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, cast

import z3

from aria.sampling.base import (
    Logic,
    Sampler,
    SamplingMethod,
    SamplingOptions,
    SamplingResult,
)
from aria.sampling.finite_domain.common import (
    collect_string_observable_terms,
    enumerate_projected_models,
    resolve_output_terms,
    resolve_projection_terms,
)
from aria.utils.z3.expr import get_variables

from .config import SearchTreeConfig
from .features import (
    BoundaryProfile,
    SampleFeatureBundle,
    SampleFeatureExtractor,
    SearchCandidate,
    augment_boundary_features,
    select_candidates,
)
from .regex import RegexFeatureExtractor
from .shrink import SampleShrinker


class SLIASampler(Sampler):
    """Practical sampler for formulas combining strings and integers."""

    def __init__(self, **_kwargs: Any) -> None:
        self.formula: Optional[z3.ExprRef] = None
        self.string_variables: List[z3.SeqRef] = []
        self.int_variables: List[z3.ArithRef] = []
        self.length_terms: List[z3.ArithRef] = []
        self.default_terms: List[z3.ExprRef] = []
        self.observable_terms: List[z3.ExprRef] = []
        self._regex_feature_extractor = RegexFeatureExtractor()
        self._feature_extractor: Optional[SampleFeatureExtractor] = None
        self._sample_shrinker: Optional[SampleShrinker] = None

    def supports_logic(self, logic: Logic) -> bool:
        return logic == Logic.QF_SLIA

    def init_from_formula(self, formula: z3.ExprRef) -> None:
        self.formula = formula
        self.string_variables = sorted(
            [
                cast(z3.SeqRef, var)
                for var in get_variables(formula)
                if var.sort() == z3.StringSort()
            ],
            key=str,
        )
        self.int_variables = sorted(
            [
                cast(z3.ArithRef, var)
                for var in get_variables(formula)
                if var.sort() == z3.IntSort()
            ],
            key=str,
        )
        self.length_terms = [cast(z3.ArithRef, z3.Length(var)) for var in self.string_variables]
        self.observable_terms = collect_string_observable_terms(formula)
        self.default_terms = list(self.string_variables) + list(self.int_variables)
        if not self.default_terms:
            self.default_terms = list(self.observable_terms)

        self._regex_feature_extractor.initialize(formula)
        self._feature_extractor = SampleFeatureExtractor(
            self.string_variables,
            self.int_variables,
            self._regex_feature_extractor,
        )
        self._sample_shrinker = SampleShrinker(
            formula,
            self.string_variables,
            self.int_variables,
            self.observable_terms,
        )

    def sample(self, options: SamplingOptions) -> SamplingResult:
        if self.formula is None:
            raise ValueError("Sampler not initialized with a formula")

        if options.method == SamplingMethod.SEARCH_TREE:
            return self._sample_via_search_tree(options)

        return self._sample_via_enumeration(options)

    def get_supported_methods(self) -> Set[SamplingMethod]:
        return {SamplingMethod.ENUMERATION, SamplingMethod.SEARCH_TREE}

    def get_supported_logics(self) -> Set[Logic]:
        return {Logic.QF_SLIA}

    def _sample_via_enumeration(self, options: SamplingOptions) -> SamplingResult:
        assert self.formula is not None
        result = enumerate_projected_models(
            cast(z3.ExprRef, self.formula),
            options,
            self.observable_terms,
            default_terms=self.default_terms,
        )
        if not options.additional_options.get("shrink_samples", False):
            return result

        projection_terms = resolve_projection_terms(
            self.observable_terms,
            options.additional_options.get("projection_terms"),
            default_terms=self.default_terms,
        )
        shrunk_result = self._shrink_selected_samples(
            result.samples,
            (
                projection_terms
                if options.additional_options.get("shrink_preserve_projection", False)
                else []
            ),
            options,
        )
        merged_stats = dict(result.stats)
        merged_stats.update(shrunk_result[1])
        return SamplingResult(shrunk_result[0], merged_stats)

    def _sample_via_search_tree(self, options: SamplingOptions) -> SamplingResult:
        assert self.formula is not None
        formula = cast(z3.ExprRef, self.formula)
        started_at = perf_counter()
        cfg = SearchTreeConfig.from_options(options)

        final_projection_terms = resolve_projection_terms(
            self.observable_terms,
            cfg.explicit_projection_terms,
            default_terms=self.default_terms,
        )
        final_output_terms = resolve_output_terms(
            self.observable_terms,
            cfg.explicit_projection_terms,
            cfg.explicit_tracked_terms,
            cfg.return_full_model,
            default_terms=self.default_terms,
        )

        if not self.string_variables:
            fallback = self._sample_via_enumeration(
                SamplingOptions(
                    method=SamplingMethod.ENUMERATION,
                    num_samples=options.num_samples,
                    timeout=options.timeout,
                    random_seed=options.random_seed,
                    projection_terms=[str(term) for term in final_projection_terms],
                    tracked_terms=[str(term) for term in final_output_terms],
                    return_full_model=cfg.return_full_model,
                    shrink_samples=cfg.shrink_samples,
                    shrink_preserve_projection=cfg.shrink_preserve_projection,
                )
            )
            fallback.stats.setdefault("length_shape_count", 0)
            fallback.stats.setdefault("candidate_count", len(fallback.samples))
            fallback.stats.setdefault("shape_signature_count", 0)
            fallback.stats.setdefault("search_tree_fallback", True)
            return fallback

        length_shape_result = enumerate_projected_models(
            formula,
            SamplingOptions(
                method=SamplingMethod.ENUMERATION,
                num_samples=cfg.max_length_shapes,
                timeout=options.timeout,
                random_seed=options.random_seed,
                projection_terms=[str(term) for term in self.length_terms],
                tracked_terms=[str(term) for term in self.length_terms],
            ),
            self.observable_terms,
            default_terms=self.length_terms,
        )
        length_shape_samples = length_shape_result.samples
        if not length_shape_samples:
            stats = dict(length_shape_result.stats)
            stats.update(
                {
                    "length_shape_count": 0,
                    "candidate_count": 0,
                    "shape_signature_count": 0,
                    "coverage_ratio": 0.0,
                    "time_ms": int((perf_counter() - started_at) * 1000),
                }
            )
            return SamplingResult([], stats)

        candidates_by_shape: List[List[SearchCandidate]] = []
        candidate_count = 0
        residual_solver_checks = 0
        residual_termination_reasons: List[str] = []
        boundary_solver_checks = 0
        for length_shape_sample in length_shape_samples:
            residual_formula = self._length_shape_formula(formula, length_shape_sample)
            residual_result = enumerate_projected_models(
                residual_formula,
                SamplingOptions(
                    method=SamplingMethod.ENUMERATION,
                    num_samples=cfg.candidates_per_length_shape,
                    timeout=options.timeout,
                    random_seed=options.random_seed,
                    projection_terms=[str(term) for term in final_projection_terms],
                    return_full_model=True,
                ),
                self.observable_terms,
                default_terms=self.default_terms,
            )
            residual_solver_checks += int(residual_result.stats.get("solver_checks", 0))
            residual_termination_reasons.append(
                str(residual_result.stats.get("termination_reason", "unknown"))
            )
            if not residual_result.samples:
                continue

            length_shape_key = tuple(
                sorted((name, int(value)) for name, value in length_shape_sample.items())
            )
            length_shape_signature = ", ".join(
                f"{name}={value}" for name, value in length_shape_key
            )
            boundary_profile = self._compute_boundary_profile(
                length_shape_sample,
                radius=cfg.boundary_neighbor_radius,
                timeout=options.timeout,
                random_seed=options.random_seed,
            )
            boundary_solver_checks += boundary_profile.solver_checks

            group: List[SearchCandidate] = []
            for sample in residual_result.samples:
                feature_bundle = augment_boundary_features(
                    self._extract_features(sample),
                    boundary_profile,
                )
                group.append(
                    SearchCandidate(
                        sample=sample,
                        length_shape_key=length_shape_key,
                        length_shape_signature=length_shape_signature,
                        feature_bundle=feature_bundle,
                        boundary_score=boundary_profile.unsat_neighbors,
                        boundary_unsat_neighbors=boundary_profile.unsat_neighbors,
                        boundary_sat_neighbors=boundary_profile.sat_neighbors,
                    )
                )
            candidate_count += len(group)
            candidates_by_shape.append(group)

        selected_candidates = select_candidates(
            candidates_by_shape,
            options.num_samples,
            cfg.diversity_mode,
            boundary_focus=cfg.boundary_focus,
        )

        if cfg.shrink_samples:
            shrink_terms = (
                final_projection_terms
                if cfg.shrink_preserve_projection
                else list(self.length_terms)
            )
            selected_samples, shrink_stats = self._shrink_selected_samples(
                [candidate.sample for candidate in selected_candidates],
                shrink_terms,
                options,
            )
        else:
            selected_samples = [candidate.sample for candidate in selected_candidates]
            shrink_stats = {
                "shrink_passes": 0,
                "shrink_solver_checks": 0,
                "shrink_changed_samples": 0,
            }

        selected_samples = self._dedupe_samples(
            selected_samples,
            final_projection_terms,
            options.num_samples,
        )
        filtered_samples = [
            {str(term): sample[str(term)] for term in final_output_terms if str(term) in sample}
            for sample in selected_samples
        ]

        all_candidates = [candidate for group in candidates_by_shape for candidate in group]
        all_coverage = (
            set().union(
                *(candidate.feature_bundle.coverage_features for candidate in all_candidates)
            )
            if all_candidates
            else set()
        )
        selected_coverage = (
            set().union(
                *(self._extract_features(sample).coverage_features for sample in selected_samples)
            )
            if selected_samples
            else set()
        )

        stats: Dict[str, Any] = {
            "time_ms": int((perf_counter() - started_at) * 1000),
            "iterations": len(filtered_samples),
            "length_shape_count": len(candidates_by_shape),
            "candidate_count": candidate_count,
            "shape_signature_count": len(
                {candidate.feature_bundle.shape_signature for candidate in all_candidates}
            ),
            "coverage_ratio": (
                len(selected_coverage) / len(all_coverage) if all_coverage else 1.0
            ),
            "diversity_mode": cfg.diversity_mode.value,
            "projection_terms": [str(term) for term in final_projection_terms],
            "output_terms": [str(term) for term in final_output_terms],
            "length_shape_projection_terms": [str(term) for term in self.length_terms],
            "length_shape_signatures": [
                candidate.length_shape_signature for candidate in selected_candidates
            ],
            "length_shape_termination_reason": length_shape_result.stats.get(
                "termination_reason", "unknown"
            ),
            "length_shape_solver_checks": length_shape_result.stats.get("solver_checks", 0),
            "residual_solver_checks": residual_solver_checks,
            "residual_termination_reasons": residual_termination_reasons,
            "boundary_focus": cfg.boundary_focus,
            "boundary_neighbor_radius": cfg.boundary_neighbor_radius,
            "boundary_solver_checks": boundary_solver_checks,
            "boundary_candidate_count": sum(
                1 for candidate in all_candidates if candidate.boundary_unsat_neighbors > 0
            ),
            "selected_boundary_candidate_count": sum(
                1
                for candidate in selected_candidates
                if candidate.boundary_unsat_neighbors > 0
            ),
            "regex_atom_count": len(self._regex_feature_extractor.regex_atoms),
        }
        all_regex_features = {
            feature
            for candidate in all_candidates
            for feature in candidate.feature_bundle.coverage_features
            if feature.startswith("regex:")
        }
        selected_regex_features = {
            feature
            for sample in selected_samples
            for feature in self._extract_features(sample).coverage_features
            if feature.startswith("regex:")
        }
        stats["regex_branch_coverage_ratio"] = (
            len(selected_regex_features) / len(all_regex_features)
            if all_regex_features
            else 1.0
        )
        stats.update(shrink_stats)
        return SamplingResult(filtered_samples, stats)

    def _length_shape_formula(
        self,
        formula: z3.ExprRef,
        length_shape_sample: Dict[str, Any],
    ) -> z3.ExprRef:
        equalities: List[z3.BoolRef] = []
        for length_term in self.length_terms:
            name = str(length_term)
            if name not in length_shape_sample:
                continue
            equalities.append(length_term == z3.IntVal(int(length_shape_sample[name])))
        return cast(z3.ExprRef, z3.And(formula, *equalities) if equalities else formula)

    def _extract_features(self, sample: Dict[str, Any]) -> SampleFeatureBundle:
        assert self._feature_extractor is not None
        return self._feature_extractor.extract(sample)

    def _compute_boundary_profile(
        self,
        length_shape_sample: Dict[str, Any],
        radius: int,
        timeout: Optional[float],
        random_seed: Optional[int],
    ) -> BoundaryProfile:
        assert self.formula is not None
        unsat_neighbors = 0
        sat_neighbors = 0
        solver_checks = 0

        for length_term in self.length_terms:
            name = str(length_term)
            base_value = int(length_shape_sample[name])
            for delta in range(-radius, radius + 1):
                if delta == 0:
                    continue
                new_value = base_value + delta
                if new_value < 0:
                    continue
                neighbor_shape = dict(length_shape_sample)
                neighbor_shape[name] = new_value
                solver = z3.Solver()
                if timeout is not None:
                    solver.set("timeout", max(1, int(timeout * 1000)))
                if random_seed is not None:
                    solver.set("random_seed", random_seed)
                    solver.set("seed", random_seed)
                solver.add(self._length_shape_formula(cast(z3.ExprRef, self.formula), neighbor_shape))
                result = solver.check()
                solver_checks += 1
                if result == z3.sat:
                    sat_neighbors += 1
                elif result == z3.unsat:
                    unsat_neighbors += 1

        return BoundaryProfile(
            unsat_neighbors=unsat_neighbors,
            sat_neighbors=sat_neighbors,
            solver_checks=solver_checks,
        )

    def _shrink_selected_samples(
        self,
        samples: Sequence[Dict[str, Any]],
        fixed_terms: Sequence[z3.ExprRef],
        options: SamplingOptions,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        assert self._sample_shrinker is not None
        return self._sample_shrinker.shrink_selected_samples(samples, fixed_terms, options)

    def _dedupe_samples(
        self,
        samples: Sequence[Dict[str, Any]],
        projection_terms: Sequence[z3.ExprRef],
        sample_limit: int,
    ) -> List[Dict[str, Any]]:
        unique_samples: List[Dict[str, Any]] = []
        seen_keys: Set[Tuple[Any, ...]] = set()

        for sample in samples:
            key = tuple(sample.get(str(term)) for term in projection_terms)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique_samples.append(sample)
            if len(unique_samples) >= sample_limit:
                break

        return unique_samples
