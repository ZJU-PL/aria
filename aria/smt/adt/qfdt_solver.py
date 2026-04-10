"""
Quantifier-free algebraic datatype solver front-end.
"""

import logging
from typing import Optional

import z3

from aria.utils import SolverResult

logger = logging.getLogger(__name__)


class QFDTSolver:
    """
    Solver wrapper for formulas over algebraic datatypes.

    The default route uses Z3's `QF_DT` solver. When the requested logic is not
    recognized by Z3, the wrapper falls back to a generic solver so mixed ADT
    fragments such as UF+DT remain usable.
    """

    def __init__(self, logic: str = "QF_DT") -> None:
        self.logic = logic
        self.model: Optional[z3.ModelRef] = None

    def solve_smt_file(
        self, filepath: str, logic: Optional[str] = None
    ) -> SolverResult:
        """
        Solve an SMT-LIB problem from a file.
        """
        formulas = z3.parse_smt2_file(filepath)
        return self.check_sat(z3.And(formulas), logic=logic)

    def solve_smt_string(
        self, smt_str: str, logic: Optional[str] = None
    ) -> SolverResult:
        """
        Solve an SMT-LIB problem from a string.
        """
        formulas = z3.parse_smt2_string(smt_str)
        return self.check_sat(z3.And(formulas), logic=logic)

    def solve_smt_formula(
        self, fml: z3.ExprRef, logic: Optional[str] = None
    ) -> SolverResult:
        """
        Solve a Z3 formula directly.
        """
        return self.check_sat(fml, logic=logic)

    def check_sat(
        self, fml: z3.ExprRef, logic: Optional[str] = None
    ) -> SolverResult:
        """
        Check satisfiability of a datatype formula.
        """
        solver = self._make_solver(logic)
        solver.add(fml)
        result = solver.check()

        if result == z3.sat:
            self.model = solver.model()
            return SolverResult.SAT
        if result == z3.unsat:
            self.model = None
            return SolverResult.UNSAT

        self.model = None
        return SolverResult.UNKNOWN

    def _make_solver(self, logic: Optional[str]) -> z3.Solver:
        requested_logic = logic or self.logic
        if requested_logic in ("", "ALL"):
            return z3.Solver()

        try:
            return z3.SolverFor(requested_logic)
        except z3.Z3Exception:
            logger.warning(
                "Z3 does not recognize logic %s; falling back to the generic solver",
                requested_logic,
            )
            return z3.Solver()
