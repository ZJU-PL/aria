from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class WMCBackend(str, Enum):
    DNNF = "dnnf"
    ENUMERATION = "enumeration"


@dataclass
class WMCOptions:
    backend: WMCBackend = WMCBackend.DNNF
    # Limit for enumeration backend; None means enumerate all
    model_limit: int | None = None


# A literal weight map: maps int literal to probability weight in [0,1]
LiteralWeights = Dict[int, float]
