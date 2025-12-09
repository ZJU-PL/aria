"""LIA model counting via LattE integration (stub).
https://www.math.ucdavis.edu/~latte/
https://github.com/latte-int/latte
"""

import shutil
import subprocess
from typing import Optional
import z3
from aria.utils.z3_expr_utils import get_variables


class LatteCounter:
    """Counter for linear integer arithmetic formulas using LattE (not implemented)."""

    def __init__(self, latte_path: Optional[str] = None):
        self.latte_path = latte_path or self._find_latte()

    def _find_latte(self) -> Optional[str]:
        """Find a LattE executable in PATH, if available."""
        for cmd in ("count", "latte-count", "latte-int"):
            path = shutil.which(cmd)
            if path:
                return path
        return None

    def _formula_to_polytope(self, formula: z3.ExprRef) -> str:
        """Convert Z3 formula to LattE polytope format (not implemented)."""
        raise NotImplementedError("Conversion to LattE polytope format is not implemented.")

    def count_models(self, formula: z3.ExprRef) -> int:
        """Count models of a linear integer arithmetic formula (not implemented)."""
        raise NotImplementedError("LattE-based LIA counting is not implemented.")

    def _is_lia_formula(self, formula: z3.ExprRef) -> bool:
        """Best-effort check for LIA fragment (placeholder)."""
        return True


def count_lia_models(formula: z3.ExprRef) -> int:
    """Convenience wrapper for LattE-based counting (not implemented)."""
    return LatteCounter().count_models(formula)
