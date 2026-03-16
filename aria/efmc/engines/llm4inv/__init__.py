"""
LLM4Inv: LLM-guided Invariant Synthesis Engine

This module implements a simple "guess-and-check" approach:
- LLM proposes concrete, hole-free invariant candidates
- SMT solver verifies each candidate for validity
- Iterative refinement based on failed candidates
- No template completion or hole-filling required

The module focuses on core functionality with a streamlined
LLM-SMT hybrid approach for invariant generation.
"""

from .bv.llm4inv_prover import LLM4InvProver
from .bv.cegis_loop import LLMInvariantCEGIS
from .bv.llm_interface import LLMInterface

__all__ = ["LLM4InvProver", "LLMInvariantCEGIS", "LLMInterface"]
