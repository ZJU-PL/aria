"""
Shared sampling and error-estimation helpers for arithmetic inference.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

import z3


def _z_score(confidence_level: float) -> float:
    if confidence_level >= 0.99:
        return 2.576
    if confidence_level >= 0.95:
        return 1.960
    if confidence_level >= 0.90:
        return 1.645
    return 1.0


def _uniform_support_measure(
    variables: List[z3.ExprRef], bounds: Dict[str, Tuple[float, float]]
) -> float:
    measure = 1.0
    for var in variables:
        min_val, max_val = bounds[str(var)]
        if var.sort() == z3.IntSort():
            measure *= int(max_val) - int(min_val) + 1
        else:
            measure *= float(max_val) - float(min_val)
    return measure


def _uniform_sample_from_support(
    variables: List[z3.ExprRef],
    bounds: Dict[str, Tuple[float, float]],
    rng: random.Random,
) -> Dict[str, Any]:
    assignment = {}
    for var in variables:
        min_val, max_val = bounds[str(var)]
        if var.sort() == z3.IntSort():
            assignment[str(var)] = rng.randint(int(min_val), int(max_val))
        else:
            assignment[str(var)] = rng.uniform(float(min_val), float(max_val))
    return assignment


def _running_error_bound(
    sample_count: int,
    sample_sum: float,
    sample_sum_squares: float,
    scale: float,
    z_score: float,
) -> Optional[float]:
    if sample_count <= 1:
        return None
    mean = sample_sum / float(sample_count)
    variance = max(sample_sum_squares / float(sample_count) - mean * mean, 0.0)
    std = math.sqrt(variance)
    return scale * z_score * std / math.sqrt(float(sample_count))


__all__ = [
    "_z_score",
    "_uniform_support_measure",
    "_uniform_sample_from_support",
    "_running_error_bound",
]
