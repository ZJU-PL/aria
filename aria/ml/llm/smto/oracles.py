"""Oracle type definitions for PS_SMTO."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import z3


@dataclass
class OracleInfo:
    """Information about an oracle function."""

    name: str
    input_types: List[z3.SortRef]
    output_type: z3.SortRef
    description: str
    examples: List[Dict] = field(default_factory=list)


@dataclass
class WhiteboxOracleInfo(OracleInfo):
    """Oracle with source code available for specification synthesis."""

    source_code: Optional[str] = None
