"""Configuration for SLIA search-tree sampling."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional

from aria.sampling.base import SamplingOptions


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
