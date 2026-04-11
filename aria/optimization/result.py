"""Shared result types for optimization solvers."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import z3


class OptimizationStatus(Enum):
    """Normalized optimization statuses."""

    OPTIMAL = "optimal"
    UNSAT = "unsat"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class OptimizationResult:
    """Normalized result object for optimization backends."""

    status: OptimizationStatus
    value: Optional[Any] = None
    model: Optional[z3.ModelRef] = None
    cost: Optional[float] = None
    engine: Optional[str] = None
    solver: Optional[str] = None
    detail: Optional[str] = None

    @property
    def is_sat(self) -> bool:
        """Return True when the backend found an optimal satisfying assignment."""
        return self.status == OptimizationStatus.OPTIMAL

    def require_value(self) -> Any:
        """Return the optimization value or raise if none is available."""
        if self.value is None:
            raise ValueError(f"Optimization result has no value: {self.status.value}")
        return self.value


def maxsmt_result_tuple(
    result: OptimizationResult,
) -> tuple[bool, Optional[z3.ModelRef], float]:
    """Convert a normalized result into the legacy MaxSMT tuple form."""
    if result.status == OptimizationStatus.OPTIMAL:
        return True, result.model, float(result.cost or 0.0)
    return False, None, float("inf")
