"""
Shared configuration types for arithmetic probabilistic inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from aria.prob.core.density import Density


class WMIMethod(str, Enum):
    """Available WMI backends."""

    AUTO = "auto"
    BOUNDED_SUPPORT_MONTE_CARLO = "bounded_support_monte_carlo"
    IMPORTANCE_SAMPLING = "importance_sampling"
    EXACT_DISCRETE = "exact_discrete"


@dataclass
class WMIOptions:
    """Options for WMI and probability queries over arithmetic formulas."""

    method: WMIMethod = WMIMethod.AUTO
    num_samples: int = 10000
    timeout: Optional[float] = None
    random_seed: Optional[int] = None
    confidence_level: float = 0.95
    proposal: Optional[Density] = None


__all__ = ["WMIMethod", "WMIOptions"]
