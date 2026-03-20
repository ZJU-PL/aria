"""
Base classes and types for Weighted Model Counting (WMC).

This module defines the core interfaces and options for WMC computation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class WMCBackend(str, Enum):
    """Backend methods for WMC computation."""

    DNNF = "dnnf"
    ENUMERATION = "enumeration"


@dataclass
class WMCOptions:
    """Options for WMC computation."""

    backend: WMCBackend = WMCBackend.DNNF
    # Limit for enumeration backend; None means enumerate all
    model_limit: int | None = None
    # Reject explicit complementary weights that do not sum to 1.0
    strict_complements: bool = False


# A literal weight map: maps int literal to probability weight in [0,1]
LiteralWeights = Dict[int, float]
