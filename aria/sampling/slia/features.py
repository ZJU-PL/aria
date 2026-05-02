"""Feature extraction and candidate selection for SLIA sampling."""

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Set, Tuple

import z3

from .config import DiversityMode
from .regex import RegexFeatureExtractor


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


class SampleFeatureExtractor:
    """Build feature bundles for string/integer samples."""

    def __init__(
        self,
        string_variables: Sequence[z3.SeqRef],
        int_variables: Sequence[z3.ArithRef],
        regex_extractor: RegexFeatureExtractor,
    ) -> None:
        self.string_variables = list(string_variables)
        self.int_variables = list(int_variables)
        self.regex_extractor = regex_extractor

    def extract(self, sample: Dict[str, Any]) -> SampleFeatureBundle:
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

        regex_features = self.regex_extractor.extract_features(sample)
        coverage_features.update(regex_features)
        shape_parts.extend(sorted(regex_features))

        return SampleFeatureBundle(
            coverage_features=coverage_features,
            numeric_features=tuple(numeric_features),
            shape_signature=tuple(shape_parts),
            complexity=complexity,
        )


def augment_boundary_features(
    feature_bundle: SampleFeatureBundle,
    boundary_profile: BoundaryProfile,
) -> SampleFeatureBundle:
    coverage_features = set(feature_bundle.coverage_features)
    coverage_features.add(f"boundary:unsat_neighbors={boundary_profile.unsat_neighbors}")
    coverage_features.add(f"boundary:sat_neighbors={boundary_profile.sat_neighbors}")
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


def select_candidates(
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
