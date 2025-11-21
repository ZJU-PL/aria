"""
SMTO (Satisfiability Modulo Theories and Oracles) solver using LLM as oracle handler.

Supports two modes:
1. Blackbox mode: Traditional SMTO where we can only observe input-output behavior
2. Whitebox mode: Enhanced SMTO where we use LLM to analyze available component information

Related: "Satisfiability and Synthesis Modulo Oracles" [Polgreen/Reynolds/Seshia VMCAI 2022]
"""

# Import main solver
from arlib.llm.smto.smto import OraxSolver

# Import oracle definitions
from arlib.llm.smto.oracles import (
    OracleInfo,
    WhiteboxOracleInfo,
    OracleType,
    OracleAnalysisMode
)

# Import utility classes
from arlib.llm.smto.utils import OracleCache, ExplanationLogger

__all__ = [
    # Main solver
    "OraxSolver",
    
    # Oracle definitions
    "OracleInfo",
    "WhiteboxOracleInfo",
    "OracleType",
    "OracleAnalysisMode",
    
    # Utility classes
    "OracleCache",
    "ExplanationLogger"
] 