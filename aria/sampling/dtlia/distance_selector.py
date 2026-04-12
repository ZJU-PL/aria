"""
Select a diverse candidate subset by maximizing minimum pairwise distance.
"""

from typing import Any, Dict, List, Optional, Sequence


def sample_distance(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    """Compute a mixed-theory distance between two sampled models."""

    def value_distance(left_value: Any, right_value: Any) -> float:
        if isinstance(left_value, dict) and isinstance(right_value, dict):
            distance = 0.0
            if left_value.get("constructor") != right_value.get("constructor"):
                distance += 1.0

            left_fields = left_value.get("fields", [])
            right_fields = right_value.get("fields", [])
            for left_field, right_field in zip(left_fields, right_fields):
                distance += value_distance(left_field, right_field)
            distance += abs(len(left_fields) - len(right_fields))
            return distance

        if isinstance(left_value, bool) and isinstance(right_value, bool):
            return 0.0 if left_value == right_value else 1.0

        if isinstance(left_value, int) and isinstance(right_value, int):
            return float(abs(left_value - right_value))

        return 0.0 if left_value == right_value else 1.0

    keys = sorted(set(left) | set(right))
    return sum(value_distance(left.get(key), right.get(key)) for key in keys)


def _next_max_distance_index(
    candidates: Sequence[Dict[str, Any]],
    selected_indices: Sequence[int],
    remaining_indices: Sequence[int],
) -> Optional[int]:
    """Pick the next remaining candidate by max-min distance."""
    if not remaining_indices:
        return None

    if not selected_indices:
        return max(
            remaining_indices,
            key=lambda index: (
                sum(
                    sample_distance(candidates[index], candidates[other_index])
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
                sample_distance(candidates[index], candidates[selected_index])
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

    remaining = list(candidates)
    first_candidate = max(
        remaining,
        key=lambda candidate: sum(
            sample_distance(candidate, other)
            for other in remaining
            if other is not candidate
        ),
    )
    selected = [first_candidate]
    remaining.remove(first_candidate)

    while remaining and len(selected) < num_samples:
        next_candidate = max(
            remaining,
            key=lambda candidate: min(
                sample_distance(candidate, selected_candidate)
                for selected_candidate in selected
            ),
        )
        selected.append(next_candidate)
        remaining.remove(next_candidate)

    return selected
