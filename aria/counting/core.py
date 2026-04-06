"""Shared result types for model counting backends."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence


@dataclass
class CountResult:
    """Structured result for model counting backends."""

    status: str
    count: Optional[float]
    backend: str
    exact: bool
    runtime_s: Optional[float] = None
    projection: Optional[Sequence[str]] = None
    error_bound: Optional[float] = None
    confidence: Optional[float] = None
    reason: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


def exact_count_result(
    count: float,
    backend: str,
    runtime_s: Optional[float] = None,
    projection: Optional[Sequence[str]] = None,
    reason: str = "",
    **metadata: Any,
) -> CountResult:
    """Create a successful exact counting result."""

    return CountResult(
        status="exact",
        count=count,
        backend=backend,
        exact=True,
        runtime_s=runtime_s,
        projection=projection,
        reason=reason,
        metadata=metadata,
    )


def unsupported_count_result(
    backend: str,
    reason: str,
    runtime_s: Optional[float] = None,
    projection: Optional[Sequence[str]] = None,
    **metadata: Any,
) -> CountResult:
    """Create an unsupported counting result."""

    return CountResult(
        status="unsupported",
        count=None,
        backend=backend,
        exact=False,
        runtime_s=runtime_s,
        projection=projection,
        reason=reason,
        metadata=metadata,
    )
