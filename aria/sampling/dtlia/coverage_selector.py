"""
Select a coverage-guided candidate subset for DTLIA sampling.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Set, Tuple

from .distance_selector import _build_distance_matrix, _next_max_distance_index


@dataclass(frozen=True)
class CoverageSelectionResult:
    """Result of coverage-guided sample selection."""

    samples: List[Dict[str, Any]]
    stats: Dict[str, Any]


def _collect_path_values(
    value: Any,
    path: str,
    int_values_by_path: DefaultDict[str, List[int]],
) -> None:
    """Collect integer values reachable from a sampled value."""
    if isinstance(value, dict):
        constructor = value.get("constructor")
        fields = value.get("fields", [])
        if constructor is not None:
            for index, field_value in enumerate(fields):
                _collect_path_values(field_value, f"{path}.{index}", int_values_by_path)
        return

    if isinstance(value, bool):
        return

    if isinstance(value, int):
        int_values_by_path[path].append(value)


def _collect_sample_features(
    value: Any,
    path: str,
    int_stats: Dict[str, Dict[str, Any]],
    features: Set[str],
) -> None:
    """Collect coverage features for a sampled value."""
    if isinstance(value, dict):
        constructor = value.get("constructor")
        fields = value.get("fields", [])
        if constructor is not None:
            features.add(f"ctor:{path}={constructor}")
            for index, field_value in enumerate(fields):
                _collect_sample_features(
                    field_value,
                    f"{path}.{index}",
                    int_stats,
                    features,
                )
        return

    if isinstance(value, bool):
        features.add(f"bool:{path}={value}")
        return

    if isinstance(value, int):
        stats = int_stats.get(path)
        features.add(f"int:any:{path}")
        if stats is None:
            return

        min_value = stats["min"]
        max_value = stats["max"]
        unique_values = stats["unique_values"]
        midpoint = stats["midpoint"]

        if value == min_value:
            features.add(f"int:min:{path}")
        if value == max_value:
            features.add(f"int:max:{path}")
        if min_value < value < max_value:
            features.add(f"int:interior:{path}")
        if value == 0:
            features.add(f"int:zero:{path}")
        if value < 0:
            features.add(f"int:negative:{path}")
        if value > 0:
            features.add(f"int:positive:{path}")
        if value % 2 == 0:
            features.add(f"int:even:{path}")
        else:
            features.add(f"int:odd:{path}")
        if value <= midpoint:
            features.add(f"int:lower_half:{path}")
        if value >= midpoint:
            features.add(f"int:upper_half:{path}")
        if len(unique_values) <= 4:
            features.add(f"int:value:{path}={value}")
        return

    features.add(f"atom:{path}={value}")


def _feature_priority(feature: str) -> int:
    """Return a coarse priority for coverage features."""
    if feature.startswith("shape:"):
        return 6
    if ":min:" in feature or ":max:" in feature:
        return 5
    if ":negative:" in feature or ":positive:" in feature or ":zero:" in feature:
        return 4
    if ":lower_half:" in feature or ":upper_half:" in feature:
        return 3
    if ":interior:" in feature or ":even:" in feature or ":odd:" in feature:
        return 2
    if ":value:" in feature:
        return 1
    return 0


def _priority_signature(features: Set[str]) -> Tuple[int, ...]:
    """Summarize feature coverage counts by descending priority."""
    counts = [0] * 7
    for feature in features:
        counts[_feature_priority(feature)] += 1
    return tuple(reversed(counts))


def _build_candidate_feature_sets(
    candidates: Sequence[Dict[str, Any]],
    shape_signatures: Optional[Sequence[str]] = None,
) -> List[Set[str]]:
    """Build coverage feature sets for candidate samples."""
    int_values_by_path: DefaultDict[str, List[int]] = defaultdict(list)
    for candidate in candidates:
        for key, value in candidate.items():
            _collect_path_values(value, key, int_values_by_path)

    int_stats: Dict[str, Dict[str, Any]] = {}
    for path, values in int_values_by_path.items():
        unique_values = sorted(set(values))
        min_value = min(unique_values)
        max_value = max(unique_values)
        int_stats[path] = {
            "min": min_value,
            "max": max_value,
            "midpoint": (min_value + max_value) / 2.0,
            "unique_values": unique_values,
        }

    candidate_feature_sets: List[Set[str]] = []
    for index, candidate in enumerate(candidates):
        features: Set[str] = set()
        if shape_signatures is not None and index < len(shape_signatures):
            features.add(f"shape:{shape_signatures[index]}")
        for key, value in candidate.items():
            _collect_sample_features(value, key, int_stats, features)
        candidate_feature_sets.append(features)

    return candidate_feature_sets


def _select_shape_seed_indices(
    candidate_feature_sets: Sequence[Set[str]],
    num_samples: int,
) -> List[int]:
    """Seed coverage selection with distinct constructor-shape features."""
    shape_to_index: Dict[str, int] = {}

    for index, feature_set in enumerate(candidate_feature_sets):
        shape_features = sorted(
            feature for feature in feature_set if feature.startswith("shape:")
        )
        if not shape_features:
            continue
        shape_feature = shape_features[0]
        previous_index = shape_to_index.get(shape_feature)
        if previous_index is None:
            shape_to_index[shape_feature] = index
            continue

        current_key = (
            _priority_signature(feature_set),
            len(feature_set),
            -index,
        )
        previous_key = (
            _priority_signature(candidate_feature_sets[previous_index]),
            len(candidate_feature_sets[previous_index]),
            -previous_index,
        )
        if current_key > previous_key:
            shape_to_index[shape_feature] = index

    seeded_indices = [
        index
        for _, index in sorted(
            shape_to_index.items(),
            key=lambda item: (
                _priority_signature(candidate_feature_sets[item[1]]),
                len(candidate_feature_sets[item[1]]),
                -item[1],
            ),
            reverse=True,
        )
    ]
    return seeded_indices[:num_samples]


def select_coverage_guided_subset(
    candidates: Sequence[Dict[str, Any]],
    num_samples: int,
    shape_signatures: Optional[Sequence[str]] = None,
) -> CoverageSelectionResult:
    """Greedily select a subset that maximizes feature coverage."""
    if num_samples <= 0 or not candidates:
        return CoverageSelectionResult(
            samples=[],
            stats={
                "coverage_feature_count": 0,
                "coverage_selected_feature_count": 0,
                "coverage_ratio": 0.0,
                "coverage_selection": "weighted_set_cover",
            },
        )

    if len(candidates) <= num_samples:
        candidate_feature_sets = _build_candidate_feature_sets(
            candidates, shape_signatures=shape_signatures
        )
        total_features = sorted(
            {feature for feature_set in candidate_feature_sets for feature in feature_set}
        )
        return CoverageSelectionResult(
            samples=list(candidates),
            stats={
                "coverage_feature_count": len(total_features),
                "coverage_selected_feature_count": len(total_features),
                "coverage_ratio": 1.0 if total_features else 0.0,
                "coverage_selection": "all_candidates",
            },
        )

    candidate_feature_sets = _build_candidate_feature_sets(
        candidates, shape_signatures=shape_signatures
    )
    distance_matrix = _build_distance_matrix(candidates)
    feature_counts = Counter(
        feature
        for feature_set in candidate_feature_sets
        for feature in feature_set
    )
    feature_weights = {
        feature: 1.0 / count for feature, count in feature_counts.items() if count > 0
    }

    uncovered_features = set(feature_weights)
    selected_indices = _select_shape_seed_indices(candidate_feature_sets, num_samples)
    remaining_indices = list(range(len(candidates)))
    for selected_index in selected_indices:
        if selected_index in remaining_indices:
            remaining_indices.remove(selected_index)
            uncovered_features -= candidate_feature_sets[selected_index]

    while remaining_indices and len(selected_indices) < num_samples:
        best_index: Optional[int] = None
        best_gain = -1.0
        best_priority_signature: Tuple[int, ...] = tuple([-1] * 7)
        best_new_feature_count = -1
        best_total_weight = -1.0

        for index in remaining_indices:
            feature_set = candidate_feature_sets[index]
            new_features = feature_set & uncovered_features
            gain = sum(feature_weights[feature] for feature in new_features)
            priority_signature = _priority_signature(new_features)
            total_weight = sum(feature_weights[feature] for feature in feature_set)
            candidate_key = (
                priority_signature,
                gain,
                len(new_features),
                total_weight,
                len(feature_set),
                -index,
            )
            best_key = (
                best_priority_signature,
                best_gain,
                best_new_feature_count,
                best_total_weight,
                -1,
                1,
            )
            if candidate_key > best_key:
                best_index = index
                best_gain = gain
                best_priority_signature = priority_signature
                best_new_feature_count = len(new_features)
                best_total_weight = total_weight

        if best_index is None or best_gain <= 0.0:
            break

        selected_indices.append(best_index)
        remaining_indices.remove(best_index)
        uncovered_features -= candidate_feature_sets[best_index]

    # Fall back to max-distance selection once coverage is saturated.
    while remaining_indices and len(selected_indices) < num_samples:
        next_index = _next_max_distance_index(
            candidates,
            selected_indices,
            remaining_indices,
            distance_matrix=distance_matrix,
        )
        if next_index is None:
            break
        selected_indices.append(next_index)
        remaining_indices.remove(next_index)

    selected_features = {
        feature
        for index in selected_indices
        for feature in candidate_feature_sets[index]
    }
    total_feature_count = len(feature_weights)
    coverage_ratio = (
        float(len(selected_features)) / float(total_feature_count)
        if total_feature_count
        else 0.0
    )

    return CoverageSelectionResult(
        samples=[candidates[index] for index in selected_indices],
        stats={
            "coverage_feature_count": total_feature_count,
            "coverage_selected_feature_count": len(selected_features),
            "coverage_ratio": coverage_ratio,
            "coverage_selection": "weighted_set_cover",
        },
    )
