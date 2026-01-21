"""Specification synthesis from oracle artifacts.

This module synthesizes SMT specifications from oracle information:
- Source code (optional)
- Documentation
- I/O examples

The LLM synthesizes a specification that captures the oracle behavior
holistically, without brittle path decomposition.
"""

from .synthesizer import SpecSynthesizer, SynthesizedSpec, spec_from_examples

__all__ = [
    "SpecSynthesizer",
    "SynthesizedSpec",
    "spec_from_examples",
]
