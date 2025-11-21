"""
Oracle type definitions and data structures for SMTO.
"""

from typing import Dict, List, Optional, Callable, Any
import z3
from dataclasses import dataclass, field
from enum import Enum


class OracleType(Enum):
    """Types of oracles supported"""
    FUNCTION = "function"  # Callable pure function
    LLM = "llm"  # LLM-based oracle
    EXTERNAL = "external"  # External API or service
    WHITEBOX = "whitebox"  # Analyzed component


class OracleAnalysisMode(Enum):
    """Analysis modes for whitebox oracles"""
    BLACKBOX = "blackbox"  # Traditional input-output mapping
    DOCUMENTATION = "documentation"  # Analysis of component documentation
    SOURCE_CODE = "source_code"  # Analysis of component source code
    BINARY = "binary"  # Analysis of binary code
    MIXED = "mixed"  # Combined analysis approaches


@dataclass
class OracleInfo:
    """Information about an oracle function (blackbox case)"""
    name: str
    input_types: List[z3.SortRef]
    output_type: z3.SortRef
    description: str
    examples: List[Dict]  # List of input-output examples
    oracle_type: OracleType = OracleType.LLM
    function: Optional[Callable] = None  # Optional direct implementation
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WhiteboxOracleInfo(OracleInfo):
    """Information about a whitebox oracle with additional analysis capabilities"""
    analysis_mode: OracleAnalysisMode = OracleAnalysisMode.DOCUMENTATION
    documentation: Optional[str] = None  # Component documentation
    source_code: Optional[str] = None  # Component source code
    binary_code: Optional[bytes] = None  # Component binary code
    external_knowledge: Optional[List[str]] = None  # Additional information (forums, papers, etc.)
    symbolic_model: Optional[str] = None  # Symbolic model derived from analysis
    
    def __post_init__(self):
        # Set oracle type to whitebox
        self.oracle_type = OracleType.WHITEBOX
        
        # Initialize external knowledge if None
        if self.external_knowledge is None:
            self.external_knowledge = [] 