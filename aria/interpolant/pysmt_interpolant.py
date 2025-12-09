"""PySMT-based interpolant synthesis module."""
from typing import Optional, List
import z3
from aria.utils.pysmt_solver import PySMTSolver


class PySMTInterpolantSynthesizer:
    """PySMT-based interpolant synthesizer using pysmt_solver.py."""
    SOLVER_NAME = "pysmt"

    def __init__(self, solver_name: str = "z3", logic=None) -> None:
        self.solver_name = solver_name
        self.logic = logic
        self._solver = PySMTSolver()

    def interpolate(self, A: z3.BoolRef, B: z3.BoolRef) -> Optional[z3.ExprRef]:
        """Generate a binary interpolant for formulas A and B."""
        if not A or not B:
            raise ValueError("Both formulas A and B must be provided")
        return self._solver.binary_interpolant(A, B, solver_name=self.solver_name, logic=self.logic)

    def sequence_interpolate(self, formulas: List[z3.ExprRef]) -> Optional[List[z3.ExprRef]]:
        """Generate a sequence interpolant for a list of formulas."""
        if not formulas:
            raise ValueError("At least one formula must be provided")
        return self._solver.sequence_interpolant(formulas)


# Backward compatibility alias
InterpolantSynthesizer = PySMTInterpolantSynthesizer
