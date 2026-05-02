"""Sampler for formulas mixing strings and linear integer arithmetic."""

from dataclasses import dataclass
from enum import Enum
from time import perf_counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, cast

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
    collect_string_observable_terms,
    enumerate_projected_models,
    resolve_output_terms,
    resolve_projection_terms,
)
from aria.utils.z3.expr import get_variables


class DiversityMode(str, Enum):
    """Supported candidate-selection modes for QF_SLIA search-tree sampling."""

    ENUMERATION = "enumeration"
    MAX_DISTANCE = "max_distance"
    COVERAGE_GUIDED = "coverage_guided"


@dataclass(frozen=True)
class SearchTreeConfig:
    """Knobs read from ``SamplingOptions.additional_options`` for search-tree sampling."""

    explicit_projection_terms: Optional[List[Any]]
    explicit_tracked_terms: Optional[List[Any]]
    return_full_model: bool
    max_length_shapes: int
    candidates_per_length_shape: int
    diversity_mode: DiversityMode
    shrink_samples: bool
    shrink_preserve_projection: bool
    boundary_focus: bool
    boundary_neighbor_radius: int

    @classmethod
    def from_options(cls, options: SamplingOptions) -> "SearchTreeConfig":
        opts = options.additional_options
        allowed_option_keys = {
            "projection_terms",
            "tracked_terms",
            "return_full_model",
            "max_length_shapes",
            "candidates_per_length_shape",
            "diversity_mode",
            "shrink_samples",
            "shrink_preserve_projection",
            "boundary_focus",
            "boundary_neighbor_radius",
        }
        unknown_keys = sorted(set(opts) - allowed_option_keys)
        if unknown_keys:
            raise ValueError(
                "Unknown SLIA search-tree options: " + ", ".join(unknown_keys)
            )

        max_length_shapes = int(opts.get("max_length_shapes", options.num_samples))
        candidates_per_length_shape = int(
            opts.get("candidates_per_length_shape", options.num_samples)
        )
        if max_length_shapes <= 0:
            raise ValueError("max_length_shapes must be positive")
        if candidates_per_length_shape <= 0:
            raise ValueError("candidates_per_length_shape must be positive")
        boundary_neighbor_radius = int(opts.get("boundary_neighbor_radius", 1))
        if boundary_neighbor_radius <= 0:
            raise ValueError("boundary_neighbor_radius must be positive")

        diversity_mode_value = str(
            opts.get("diversity_mode", DiversityMode.ENUMERATION.value)
        )
        try:
            diversity_mode = DiversityMode(diversity_mode_value)
        except ValueError as exc:
            allowed_modes = ", ".join(mode.value for mode in DiversityMode)
            raise ValueError(
                f"Unknown diversity_mode '{diversity_mode_value}'. "
                f"Expected one of: {allowed_modes}"
            ) from exc

        return cls(
            explicit_projection_terms=opts.get("projection_terms"),
            explicit_tracked_terms=opts.get("tracked_terms"),
            return_full_model=bool(opts.get("return_full_model", False)),
            max_length_shapes=max_length_shapes,
            candidates_per_length_shape=candidates_per_length_shape,
            diversity_mode=diversity_mode,
            shrink_samples=bool(opts.get("shrink_samples", False)),
            shrink_preserve_projection=bool(opts.get("shrink_preserve_projection", False)),
            boundary_focus=bool(opts.get("boundary_focus", False)),
            boundary_neighbor_radius=boundary_neighbor_radius,
        )


@dataclass
class SampleFeatureBundle:
    """Coverage and metric features extracted from a concrete sample."""

    coverage_features: Set[str]
    numeric_features: Tuple[int, ...]
    shape_signature: Tuple[Any, ...]
    complexity: int


@dataclass
class SearchCandidate:
    """A residual sample together with shape metadata and derived features."""

    sample: Dict[str, Any]
    length_shape_key: Tuple[Tuple[str, int], ...]
    length_shape_signature: str
    feature_bundle: SampleFeatureBundle
    boundary_score: int
    boundary_unsat_neighbors: int
    boundary_sat_neighbors: int


@dataclass(frozen=True)
class BoundaryProfile:
    """Neighborhood information for a length shape near the SAT/UNSAT frontier."""

    unsat_neighbors: int
    sat_neighbors: int
    solver_checks: int


def _char_class(char: str) -> str:
    if char.islower():
        return "lower"
    if char.isupper():
        return "upper"
    if char.isdigit():
        return "digit"
    if char.isspace():
        return "space"
    if char.isalpha():
        return "alpha_other"
    if char.isprintable():
        return "punct"
    return "other"


def _string_shape(value: str) -> Tuple[Any, ...]:
    classes = tuple(sorted({_char_class(char) for char in value}))
    if len(value) == 0:
        length_bucket = "empty"
    elif len(value) == 1:
        length_bucket = "singleton"
    elif len(value) <= 4:
        length_bucket = "short"
    elif len(value) <= 8:
        length_bucket = "medium"
    else:
        length_bucket = "long"

    if not value:
        repetition = "empty"
    elif len(set(value)) == len(value):
        repetition = "all_distinct"
    elif len(set(value)) == 1:
        repetition = "constant"
    else:
        repetition = "mixed_repeat"

    return (
        length_bucket,
        len(value),
        repetition,
        classes,
        value == value[::-1],
        _char_class(value[0]) if value else "none",
        _char_class(value[-1]) if value else "none",
    )


def _round_robin(groups: Sequence[Sequence[SearchCandidate]]) -> List[SearchCandidate]:
    ordered: List[SearchCandidate] = []
    max_group_size = max((len(group) for group in groups), default=0)
    for index in range(max_group_size):
        for group in groups:
            if index < len(group):
                ordered.append(group[index])
    return ordered


def _feature_distance(left: SampleFeatureBundle, right: SampleFeatureBundle) -> int:
    coverage_distance = len(left.coverage_features.symmetric_difference(right.coverage_features))
    numeric_distance = sum(
        abs(left_value - right_value)
        for left_value, right_value in zip(
            left.numeric_features, right.numeric_features, strict=False
        )
    )
    return coverage_distance + numeric_distance


def _preferred_char_order(sample_strings: Iterable[str]) -> List[str]:
    preferred = list("abcdefghijklmnopqrstuvwxyz")
    preferred.extend(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
    preferred.extend(list("0123456789"))
    preferred.extend([" ", "-", "_", ".", "/"])
    extras = sorted(
        {
            character
            for sample_string in sample_strings
            for character in sample_string
            if character not in preferred
        }
    )
    return preferred + extras


def _walk_exprs(formula: z3.ExprRef) -> List[z3.ExprRef]:
    """Return all reachable application nodes in deterministic DFS order."""
    stack = [formula]
    seen_ids: Set[int] = set()
    ordered: List[z3.ExprRef] = []

    while stack:
        expr = stack.pop()
        expr_id = z3.Z3_get_ast_id(expr.ctx.ref(), expr.as_ast())
        if expr_id in seen_ids:
            continue
        seen_ids.add(expr_id)
        ordered.append(expr)

        if z3.is_quantifier(expr):
            stack.append(cast(Any, expr).body())
            continue
        if not z3.is_app(expr):
            continue
        stack.extend(expr.children())

    return ordered


class SLIASampler(Sampler):
    """Practical sampler for formulas combining strings and integers."""

    def __init__(self, **_kwargs: Any) -> None:
        self.formula: Optional[z3.ExprRef] = None
        self.string_variables: List[z3.SeqRef] = []
        self.int_variables: List[z3.ArithRef] = []
        self.length_terms: List[z3.ArithRef] = []
        self.default_terms: List[z3.ExprRef] = []
        self.observable_terms: List[z3.ExprRef] = []
        self.regex_atoms: List[z3.BoolRef] = []
        self._regex_membership_cache: Dict[Tuple[str, str], bool] = {}

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
        self.regex_atoms = self._collect_regex_atoms(formula)
        self._regex_membership_cache = {}
        if not self.default_terms:
            self.default_terms = list(self.observable_terms)

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
            projection_terms if options.additional_options.get("shrink_preserve_projection", False) else [],
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
        boundary_profiles: Dict[Tuple[Tuple[str, int], ...], BoundaryProfile] = {}

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
            boundary_profiles[length_shape_key] = boundary_profile
            boundary_solver_checks += boundary_profile.solver_checks
            group: List[SearchCandidate] = []
            for sample in residual_result.samples:
                feature_bundle = self._extract_features(sample)
                feature_bundle = self._augment_boundary_features(
                    feature_bundle,
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

        selected_candidates = self._select_candidates(
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
        all_coverage = set().union(
            *(candidate.feature_bundle.coverage_features for candidate in all_candidates)
        ) if all_candidates else set()
        selected_coverage = set().union(
            *(
                self._extract_features(sample).coverage_features
                for sample in selected_samples
            )
        ) if selected_samples else set()

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
            "regex_atom_count": len(self.regex_atoms),
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
        coverage_features: Set[str] = set()
        numeric_features: List[int] = []
        shape_parts: List[Any] = []
        complexity = 0

        string_values: List[Tuple[str, str]] = []
        for variable in self.string_variables:
            name = str(variable)
            value = sample.get(name, "")
            if not isinstance(value, str):
                value = str(value)
            string_values.append((name, value))
            shape = _string_shape(value)
            shape_parts.append((name, shape))
            complexity += len(value)
            numeric_features.extend(
                [
                    len(value),
                    len(set(value)),
                    int(any(char.isdigit() for char in value)),
                    int(any(char.islower() for char in value)),
                    int(any(char.isupper() for char in value)),
                    int(any(not char.isalnum() and not char.isspace() for char in value)),
                    int(value == value[::-1]),
                    int(len(set(value)) < len(value) if value else 0),
                ]
            )
            coverage_features.add(f"{name}:len={len(value)}")
            coverage_features.add(f"{name}:bucket={shape[0]}")
            coverage_features.add(f"{name}:repeat={shape[2]}")
            for char_class in shape[3]:
                coverage_features.add(f"{name}:class={char_class}")
            if value:
                coverage_features.add(f"{name}:first={shape[5]}")
                coverage_features.add(f"{name}:last={shape[6]}")
            else:
                coverage_features.add(f"{name}:first=none")
                coverage_features.add(f"{name}:last=none")

        for left_index, (left_name, left_value) in enumerate(string_values):
            for right_name, right_value in string_values[left_index + 1 :]:
                coverage_features.add(f"{left_name}=={right_name}:{left_value == right_value}")
                coverage_features.add(
                    f"{left_name}prefixof{right_name}:{right_value.startswith(left_value)}"
                )
                coverage_features.add(
                    f"{left_name}suffixof{right_name}:{right_value.endswith(left_value)}"
                )
                coverage_features.add(
                    f"{left_name}contains{right_name}:{left_value in right_value}"
                )
                shape_parts.append(
                    (
                        left_name,
                        right_name,
                        left_value == right_value,
                        right_value.startswith(left_value),
                        right_value.endswith(left_value),
                    )
                )

        for variable in self.int_variables:
            name = str(variable)
            value = int(sample.get(name, 0))
            complexity += abs(value)
            numeric_features.append(value)
            if value == 0:
                sign = "zero"
            elif value < 0:
                sign = "neg"
            else:
                sign = "pos"
            coverage_features.add(f"{name}:sign={sign}")
            coverage_features.add(f"{name}:value={value}")
            shape_parts.append((name, value))

        regex_features = self._extract_regex_features(sample)
        coverage_features.update(regex_features)
        shape_parts.extend(sorted(regex_features))

        return SampleFeatureBundle(
            coverage_features=coverage_features,
            numeric_features=tuple(numeric_features),
            shape_signature=tuple(shape_parts),
            complexity=complexity,
        )

    def _augment_boundary_features(
        self,
        feature_bundle: SampleFeatureBundle,
        boundary_profile: BoundaryProfile,
    ) -> SampleFeatureBundle:
        coverage_features = set(feature_bundle.coverage_features)
        coverage_features.add(
            f"boundary:unsat_neighbors={boundary_profile.unsat_neighbors}"
        )
        coverage_features.add(
            f"boundary:sat_neighbors={boundary_profile.sat_neighbors}"
        )
        coverage_features.add(
            f"boundary:has_unsat_neighbor={boundary_profile.unsat_neighbors > 0}"
        )
        numeric_features = feature_bundle.numeric_features + (
            boundary_profile.unsat_neighbors,
            boundary_profile.sat_neighbors,
        )
        shape_signature = feature_bundle.shape_signature + (
            ("boundary", boundary_profile.unsat_neighbors, boundary_profile.sat_neighbors),
        )
        return SampleFeatureBundle(
            coverage_features=coverage_features,
            numeric_features=numeric_features,
            shape_signature=shape_signature,
            complexity=feature_bundle.complexity,
        )

    def _collect_regex_atoms(self, formula: z3.ExprRef) -> List[z3.BoolRef]:
        regex_atoms: List[z3.BoolRef] = []
        for expr in _walk_exprs(formula):
            if not z3.is_app(expr):
                continue
            if expr.decl().kind() == z3.Z3_OP_SEQ_IN_RE:
                regex_atoms.append(cast(z3.BoolRef, expr))
        return regex_atoms

    def _extract_regex_features(self, sample: Dict[str, Any]) -> Set[str]:
        features: Set[str] = set()
        for atom_index, atom in enumerate(self.regex_atoms):
            string_term = atom.arg(0)
            regex_term = atom.arg(1)
            string_name = str(string_term)
            if string_name not in sample:
                continue
            string_value = sample[string_name]
            if not isinstance(string_value, str):
                string_value = str(string_value)
            label = f"regex:{atom_index}"
            features.update(
                self._regex_branch_features(label, string_value, regex_term, "root")
            )
        return features

    def _regex_branch_features(
        self,
        label: str,
        string_value: str,
        regex_term: z3.ExprRef,
        path: str,
    ) -> Set[str]:
        if not self._concrete_in_regex(string_value, regex_term):
            return set()

        kind = regex_term.decl().kind()
        features = {f"{label}:{path}:accept", f"{label}:{path}:kind={kind}"}

        if kind == z3.Z3_OP_RE_UNION:
            matched_branches: List[int] = []
            for index, child in enumerate(regex_term.children()):
                if self._concrete_in_regex(string_value, child):
                    matched_branches.append(index)
                    features.add(f"{label}:{path}:union_branch={index}")
                    features.update(
                        self._regex_branch_features(
                            label,
                            string_value,
                            child,
                            f"{path}/union[{index}]",
                        )
                    )
            features.add(f"{label}:{path}:union_count={len(matched_branches)}")
            return features

        if kind == z3.Z3_OP_RE_OPTION:
            if string_value == "":
                features.add(f"{label}:{path}:option_empty")
            child = regex_term.arg(0)
            if self._concrete_in_regex(string_value, child):
                features.add(f"{label}:{path}:option_some")
                features.update(
                    self._regex_branch_features(
                        label,
                        string_value,
                        child,
                        f"{path}/option",
                    )
                )
            return features

        if kind == z3.Z3_OP_RE_STAR:
            features.add(
                f"{label}:{path}:star={'empty' if string_value == '' else 'nonempty'}"
            )
            child = regex_term.arg(0)
            if string_value and self._concrete_in_regex(
                string_value, z3.Loop(child, 2, max(2, len(string_value)))
            ):
                features.add(f"{label}:{path}:star_multi")
            elif string_value:
                features.add(f"{label}:{path}:star_single")
            return features

        if kind == z3.Z3_OP_RE_PLUS:
            child = regex_term.arg(0)
            if self._concrete_in_regex(
                string_value, z3.Loop(child, 2, max(2, len(string_value)))
            ):
                features.add(f"{label}:{path}:plus_multi")
            else:
                features.add(f"{label}:{path}:plus_single")
            return features

        if kind == z3.Z3_OP_RE_CONCAT and len(regex_term.children()) == 2:
            left, right = regex_term.children()
            matched_splits: List[int] = []
            for split_index in range(len(string_value) + 1):
                prefix = string_value[:split_index]
                suffix = string_value[split_index:]
                if self._concrete_in_regex(prefix, left) and self._concrete_in_regex(
                    suffix, right
                ):
                    matched_splits.append(split_index)
            if matched_splits:
                features.add(f"{label}:{path}:concat_split_count={len(matched_splits)}")
                features.add(f"{label}:{path}:concat_first_split={matched_splits[0]}")
            return features

        if kind == z3.Z3_OP_SEQ_TO_RE and regex_term.num_args() == 1:
            literal = regex_term.arg(0)
            if z3.is_string_value(literal):
                literal_value = cast(Any, literal).as_string()
                features.add(f"{label}:{path}:literal_len={len(literal_value)}")
            return features

        if kind == z3.Z3_OP_RE_RANGE and regex_term.num_args() == 2:
            lower = regex_term.arg(0)
            upper = regex_term.arg(1)
            if z3.is_string_value(lower) and z3.is_string_value(upper):
                features.add(
                    f"{label}:{path}:range={cast(Any, lower).as_string()}-{cast(Any, upper).as_string()}"
                )
            return features

        return features

    def _concrete_in_regex(self, string_value: str, regex_term: z3.ExprRef) -> bool:
        cache_key = (string_value, regex_term.sexpr())
        if cache_key in self._regex_membership_cache:
            return self._regex_membership_cache[cache_key]

        solver = z3.Solver()
        solver.add(z3.InRe(z3.StringVal(string_value), regex_term))
        is_member = solver.check() == z3.sat
        self._regex_membership_cache[cache_key] = is_member
        return is_member

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

    def _select_candidates(
        self,
        candidates_by_shape: Sequence[Sequence[SearchCandidate]],
        sample_limit: int,
        diversity_mode: DiversityMode,
        boundary_focus: bool = False,
    ) -> List[SearchCandidate]:
        if sample_limit <= 0:
            return []

        flattened = [candidate for group in candidates_by_shape for candidate in group]
        if not flattened:
            return []

        if diversity_mode == DiversityMode.ENUMERATION:
            groups = list(candidates_by_shape)
            if boundary_focus:
                groups = sorted(
                    groups,
                    key=lambda group: (
                        max(candidate.boundary_score for candidate in group),
                        -min(candidate.feature_bundle.complexity for candidate in group),
                    ),
                    reverse=True,
                )
            return _round_robin(groups)[:sample_limit]

        if diversity_mode == DiversityMode.MAX_DISTANCE:
            remaining = list(flattened)
            if boundary_focus:
                first = max(
                    remaining,
                    key=lambda candidate: (
                        candidate.boundary_score,
                        -candidate.feature_bundle.complexity,
                    ),
                )
            else:
                first = min(
                    remaining,
                    key=lambda candidate: candidate.feature_bundle.complexity,
                )
            selected = [first]
            remaining.remove(first)
            while remaining and len(selected) < sample_limit:
                next_candidate = max(
                    remaining,
                    key=lambda candidate: (
                        candidate.boundary_score if boundary_focus else 0,
                        min(
                            _feature_distance(
                                candidate.feature_bundle,
                                existing.feature_bundle,
                            )
                            for existing in selected
                        ),
                        -candidate.feature_bundle.complexity,
                    ),
                )
                selected.append(next_candidate)
                remaining.remove(next_candidate)
            return selected

        if diversity_mode == DiversityMode.COVERAGE_GUIDED:
            remaining = list(flattened)
            selected: List[SearchCandidate] = []
            seen_features: Set[str] = set()
            while remaining and len(selected) < sample_limit:
                next_candidate = max(
                    remaining,
                    key=lambda candidate: (
                        candidate.boundary_score if boundary_focus else 0,
                        len(candidate.feature_bundle.coverage_features - seen_features),
                        -candidate.feature_bundle.complexity,
                    ),
                )
                selected.append(next_candidate)
                seen_features.update(next_candidate.feature_bundle.coverage_features)
                remaining.remove(next_candidate)
            return selected

        return flattened[:sample_limit]

    def _shrink_selected_samples(
        self,
        samples: Sequence[Dict[str, Any]],
        fixed_terms: Sequence[z3.ExprRef],
        options: SamplingOptions,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        shrunk_samples: List[Dict[str, Any]] = []
        shrink_solver_checks = 0
        shrink_changed_samples = 0

        for sample in samples:
            shrunk_sample, solver_checks = self._shrink_sample(sample, fixed_terms, options)
            shrink_solver_checks += solver_checks
            if shrunk_sample != sample:
                shrink_changed_samples += 1
            shrunk_samples.append(shrunk_sample)

        return (
            shrunk_samples,
            {
                "shrink_passes": len(samples),
                "shrink_solver_checks": shrink_solver_checks,
                "shrink_changed_samples": shrink_changed_samples,
            },
        )

    def _shrink_sample(
        self,
        sample: Dict[str, Any],
        fixed_terms: Sequence[z3.ExprRef],
        options: SamplingOptions,
    ) -> Tuple[Dict[str, Any], int]:
        assert self.formula is not None
        optimizer = z3.Optimize()
        optimizer.set(priority="lex")
        if options.timeout is not None:
            optimizer.set(timeout=max(1, int(options.timeout * 1000)))
        optimizer.add(cast(z3.ExprRef, self.formula))

        for term in fixed_terms:
            name = str(term)
            if name not in sample:
                continue
            optimizer.add(term == self._python_value_to_z3(term.sort(), sample[name]))

        for string_var in self.string_variables:
            optimizer.minimize(z3.Length(string_var))
        for int_var in self.int_variables:
            optimizer.minimize(z3.Abs(int_var))

        distinct_chars = sorted(
            {
                character
                for string_var in self.string_variables
                for character in str(sample.get(str(string_var), ""))
            }
        )
        for character in distinct_chars:
            usage_terms = [
                z3.If(z3.Contains(string_var, z3.StringVal(character)), 1, 0)
                for string_var in self.string_variables
            ]
            if usage_terms:
                optimizer.minimize(z3.Sum(*usage_terms))

        preferred_chars = _preferred_char_order(
            str(sample.get(str(string_var), "")) for string_var in self.string_variables
        )
        for string_var in self.string_variables:
            string_name = str(string_var)
            current_value = str(sample.get(string_name, ""))
            for index in range(len(current_value)):
                char_expr = z3.SubString(string_var, index, 1)
                rank_expr: z3.ArithRef = z3.IntVal(len(preferred_chars) + 1)
                for rank, character in reversed(list(enumerate(preferred_chars))):
                    rank_expr = z3.If(
                        char_expr == z3.StringVal(character),
                        z3.IntVal(rank),
                        rank_expr,
                    )
                position_expr = z3.If(
                    z3.IntVal(index) < z3.Length(string_var),
                    rank_expr,
                    z3.IntVal(0),
                )
                optimizer.minimize(position_expr)

        check_result = optimizer.check()
        if check_result != z3.sat:
            return dict(sample), 1

        model = optimizer.model()
        return build_sample(model, self.observable_terms), 1

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

    def _python_value_to_z3(self, sort: z3.SortRef, value: Any) -> z3.ExprRef:
        if sort == z3.StringSort():
            return z3.StringVal(str(value))
        if sort == z3.IntSort():
            return z3.IntVal(int(value))
        raise ValueError(f"Unsupported fixed-term sort for shrinking: {sort}")
