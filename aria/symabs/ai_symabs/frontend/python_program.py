"""High-level entry point for analyzing small Python programs.

Supported subset:
- integers only
- assignments/aug-assignments
- if/else, while, for over range
- arithmetic +, -, *, //, %, bitwise &, |, ^, <<, >>, and/or/not
Unsupported constructs (for now): functions, returns, breaks/continues, and
non-range for-loops.
"""

from __future__ import annotations

from ..domains.core import ConjunctiveDomain
from ..domains.core.abstract import AbstractState
from .cfg import CFG, analyze_cfg
from .python_cfg import build_python_cfg


class PythonProgram:
    """Represents a Python snippet lowered to a CFG for analysis."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.cfg, self.variables = build_python_cfg(source)

    def transform(
        self, domain: ConjunctiveDomain, input_state: AbstractState
    ) -> AbstractState:
        """Run abstract interpretation from the given input abstract state."""
        missing = set(self.variables) - set(domain.variables)  # type: ignore[attr-defined]
        if missing:
            raise ValueError(f"Domain is missing variables: {sorted(missing)}")
        return analyze_cfg(self.cfg, domain, input_state)


__all__ = ["PythonProgram", "build_python_cfg", "CFG"]
