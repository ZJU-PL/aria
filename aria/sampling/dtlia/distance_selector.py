"""Select a diverse candidate subset by maximizing minimum pairwise distance."""

from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple


def _collect_int_ranges(
    value: Any,
    path: str,
    int_values_by_path: DefaultDict[str, List[int]],
) -> None:
    """Collect integer values by structural path for distance normalization."""
    if isinstance(value, dict):
        for index, field_value in enumerate(value.get("fields", [])):
            _collect_int_ranges(field_value, f"{path}.{index}", int_values_by_path)
        return

    if isinstance(value, bool):
        return

    if isinstance(value, int):
        int_values_by_path[path].append(value)


def _build_int_ranges(candidates: Sequence[Dict[str, Any]]) -> Dict[str, Tuple[int, int]]:
    """Build min/max integer ranges per structural path."""
    int_values_by_path: DefaultDict[str, List[int]] = defaultdict(list)
    for candidate in candidates:
        for key, value in candidate.items():
            _collect_int_ranges(value, key, int_values_by_path)

    int_ranges: Dict[str, Tuple[int, int]] = {}
    for path, values in int_values_by_path.items():
        if values:
            int_ranges[path] = (min(values), max(values))
    return int_ranges


def sample_distance(
    left: Dict[str, Any],
    right: Dict[str, Any],
    int_ranges: Optional[Dict[str, Tuple[int, int]]] = None,
) -> float:
    """Compute a mixed-theory distance between two sampled models."""

    def value_distance(left_value: Any, right_value: Any, path: str) -> float:
        if isinstance(left_value, dict) and isinstance(right_value, dict):
            distance = 0.0
            if left_value.get("constructor") != right_value.get("constructor"):
                distance += 1.0

            left_fields = left_value.get("fields", [])
            right_fields = right_value.get("fields", [])
            for index, (left_field, right_field) in enumerate(
                zip(left_fields, right_fields)
            ):
                distance += value_distance(
                    left_field,
                    right_field,
                    f"{path}.{index}",
                )
            distance += abs(len(left_fields) - len(right_fields))
            return distance

        if isinstance(left_value, bool) and isinstance(right_value, bool):
            return 0.0 if left_value == right_value else 1.0

        if isinstance(left_value, int) and isinstance(right_value, int):
            if int_ranges is None:
                return float(abs(left_value - right_value))
            min_value, max_value = int_ranges.get(path, (left_value, right_value))
            span = max_value - min_value
            if span <= 0:
                return 0.0 if left_value == right_value else 1.0
            return float(abs(left_value - right_value)) / float(span)

        return 0.0 if left_value == right_value else 1.0

    keys = sorted(set(left) | set(right))
    return sum(value_distance(left.get(key), right.get(key), key) for key in keys)


def _build_distance_matrix(candidates: Sequence[Dict[str, Any]]) -> List[List[float]]:
    """Precompute pairwise distances for a candidate pool."""
    int_ranges = _build_int_ranges(candidates)
    matrix: List[List[float]] = [
        [0.0 for _ in range(len(candidates))] for _ in range(len(candidates))
    ]
    for left_index in range(len(candidates)):
        for right_index in range(left_index + 1, len(candidates)):
            distance = sample_distance(
                candidates[left_index],
                candidates[right_index],
                int_ranges=int_ranges,
            )
            matrix[left_index][right_index] = distance
            matrix[right_index][left_index] = distance
    return matrix


def _next_max_distance_index(
    candidates: Sequence[Dict[str, Any]],
    selected_indices: Sequence[int],
    remaining_indices: Sequence[int],
    distance_matrix: Optional[Sequence[Sequence[float]]] = None,
) -> Optional[int]:
    """Pick the next remaining candidate by max-min distance."""
    if not remaining_indices:
        return None

    def pair_distance(left_index: int, right_index: int) -> float:
        if distance_matrix is not None:
            return distance_matrix[left_index][right_index]
        return sample_distance(candidates[left_index], candidates[right_index])

    if not selected_indices:
        return max(
            remaining_indices,
            key=lambda index: (
                sum(
                    pair_distance(index, other_index)
                    for other_index in remaining_indices
                    if other_index != index
                ),
                -index,
            ),
        )

    return max(
        remaining_indices,
        key=lambda index: (
            min(
                pair_distance(index, selected_index)
                for selected_index in selected_indices
            ),
            -index,
        ),
    )


def select_max_distance_subset(
    candidates: Sequence[Dict[str, Any]], num_samples: int
) -> List[Dict[str, Any]]:
    """Greedily select a diverse subset by maximizing minimum distance."""
    if num_samples <= 0 or not candidates:
        return []
    if len(candidates) <= num_samples:
        return list(candidates)

    distance_matrix = _build_distance_matrix(candidates)
    remaining_indices = list(range(len(candidates)))
    selected_indices: List[int] = []

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

    return [candidates[index] for index in selected_indices]
