# coding: utf-8
"""
For testing the QF_FP solver
"""

from aria.tests import TestCase, main
from aria.smt.fp.qffp_solver import QFFPSolver
from aria.utils import SolverResult

import z3


def is_simple_formula(fml: z3.ExprRef):
    # for pruning sample formulas that can be solved by the pre-processing
    clauses = z3.Then("simplify", "propagate-values")(fml)
    after_simp = clauses.as_expr()
    if z3.is_false(after_simp) or z3.is_true(after_simp):
        return True
    return False


def solve_with_z3(smt2string: str):
    fml = z3.And(z3.parse_smt2_string(smt2string))
    sol = z3.Solver()
    sol.add(fml)
    res = sol.check()
    if res == z3.sat:
        return SolverResult.SAT
    elif res == z3.unsat:
        return SolverResult.UNSAT
    else:
        return SolverResult.UNKNOWN


# Deterministic test formulas for QF_FP
QF_FP_TEST_FORMULAS = [
    # Test with specific values (SAT)
    """
    (set-logic QF_FP)
    (declare-const x (_ FloatingPoint 8 24))
    (assert (fp.eq x ((_ to_fp 8 24) roundNearestTiesToEven 1.0)))
    (check-sat)
    """,
    # Test with subtraction (SAT)
    """
    (set-logic QF_FP)
    (declare-const x (_ FloatingPoint 8 24))
    (declare-const y (_ FloatingPoint 8 24))
    (declare-const z (_ FloatingPoint 8 24))
    (assert (fp.eq (fp.sub roundNearestTiesToEven x y) z))
    (check-sat)
    """,
    # Test with multiplication (SAT)
    """
    (set-logic QF_FP)
    (declare-const x (_ FloatingPoint 8 24))
    (declare-const y (_ FloatingPoint 8 24))
    (declare-const z (_ FloatingPoint 8 24))
    (assert (fp.eq (fp.mul roundNearestTiesToEven x y) z))
    (check-sat)
    """,
    # Test with multiple constraints (SAT)
    """
    (set-logic QF_FP)
    (declare-const x (_ FloatingPoint 8 24))
    (declare-const y (_ FloatingPoint 8 24))
    (declare-const z (_ FloatingPoint 8 24))
    (assert (fp.lt x y))
    (assert (fp.lt y z))
    (assert (fp.lt x z))
    (check-sat)
    """,
    # Test with contradictory constraints (UNSAT)
    """
    (set-logic QF_FP)
    (declare-const x (_ FloatingPoint 8 24))
    (declare-const y (_ FloatingPoint 8 24))
    (assert (fp.lt x y))
    (assert (fp.lt y x))
    (check-sat)
    """,
    # Test with special values (SAT)
    """
    (set-logic QF_FP)
    (declare-const x (_ FloatingPoint 8 24))
    (assert (fp.isNaN x))
    (check-sat)
    """,
    # Test with rounding modes (SAT)
    """
    (set-logic QF_FP)
    (declare-const x (_ FloatingPoint 8 24))
    (declare-const y (_ FloatingPoint 8 24))
    (declare-const z1 (_ FloatingPoint 8 24))
    (declare-const z2 (_ FloatingPoint 8 24))
    (assert (fp.eq (fp.add roundNearestTiesToEven x y) z1))
    (assert (fp.eq (fp.add roundTowardZero x y) z2))
    (assert (not (fp.eq z1 z2)))
    (check-sat)
    """,
]


class TestQFFP(TestCase):
    """
    Test the bit-blasting based solver
    """

    def test_qffp_solver(self):
        for i, smt2string in enumerate(QF_FP_TEST_FORMULAS):
            try:
                fml = z3.And(z3.parse_smt2_string(smt2string))
                if is_simple_formula(fml):
                    print(f"Formula {i + 1} is a simple formula, skipping...")
                    continue
            except Exception as ex:
                print(ex)
                print(smt2string)
                continue

            print(f"!!!Solving {i + 1}-th formula!!!")

            sol = QFFPSolver()
            res = sol.solve_smt_string(smt2string)
            res_z3 = solve_with_z3(smt2string)
            print(res, res_z3)
            if res != res_z3:
                print("inconsistent!!")


if __name__ == "__main__":
    main()
