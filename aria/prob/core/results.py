"""
Result types for probabilistic inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class InferenceResult:
    """
    Numeric inference result with backend metadata.

    The object behaves like a float for common formatting and arithmetic so
    existing call sites can keep treating low-level inference results as
    numbers while richer callers can inspect the metadata.
    """

    value: float
    exact: bool
    backend: str
    stats: Dict[str, Any] = field(default_factory=dict)
    error_bound: Optional[float] = None

    def __float__(self) -> float:
        return float(self.value)

    def __format__(self, format_spec: str) -> str:
        return format(self.value, format_spec)

    def __repr__(self) -> str:
        return (
            "InferenceResult(value={!r}, exact={!r}, backend={!r}, "
            "error_bound={!r})"
        ).format(self.value, self.exact, self.backend, self.error_bound)

    def _coerce_other(self, other: Any) -> Any:
        if isinstance(other, InferenceResult):
            return other.value
        return other

    def __add__(self, other: Any) -> float:
        return self.value + self._coerce_other(other)

    def __radd__(self, other: Any) -> float:
        return self._coerce_other(other) + self.value

    def __sub__(self, other: Any) -> float:
        return self.value - self._coerce_other(other)

    def __rsub__(self, other: Any) -> float:
        return self._coerce_other(other) - self.value

    def __mul__(self, other: Any) -> float:
        return self.value * self._coerce_other(other)

    def __rmul__(self, other: Any) -> float:
        return self._coerce_other(other) * self.value

    def __truediv__(self, other: Any) -> float:
        return self.value / self._coerce_other(other)

    def __rtruediv__(self, other: Any) -> float:
        return self._coerce_other(other) / self.value

    def __neg__(self) -> float:
        return -self.value

    def __abs__(self) -> float:
        return abs(self.value)

    def __lt__(self, other: Any) -> bool:
        return self.value < self._coerce_other(other)

    def __le__(self, other: Any) -> bool:
        return self.value <= self._coerce_other(other)

    def __gt__(self, other: Any) -> bool:
        return self.value > self._coerce_other(other)

    def __ge__(self, other: Any) -> bool:
        return self.value >= self._coerce_other(other)

    def __eq__(self, other: Any) -> bool:
        return self.value == self._coerce_other(other)
