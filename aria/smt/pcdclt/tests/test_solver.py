"""Tests for the simplified CDCL(T) SMT solver"""

from aria.tests import TestCase, main
from aria.smt.pcdclt import CDCLTSolver, solve
from aria.utils import SolverResult
from aria.global_params import SMT_SOLVERS_PATH


class TestCDCLTSolver(TestCase):
    """Test the CDCL(T) solver with various formulas"""

    def setUp(self):
        """Check if Z3 is available"""
        z3_config = SMT_SOLVERS_PATH.get('z3', {})
        if not z3_config.get('available', False):
            self.skipTest("Z3 not available")

    def test_simple_sat(self):
        """Test simple satisfiable formula"""
        formula = """
        (set-logic QF_LRA)
        (declare-fun x () Real)
        (declare-fun y () Real)
        (assert (> x 0))
        (assert (< y 10))
        (assert (>= (+ x y) 5))
        """
        result = solve(formula, logic="QF_LRA")
        self.assertEqual(result, SolverResult.SAT)

    def test_simple_unsat(self):
        """Test simple unsatisfiable formula"""
        formula = """
        (set-logic QF_LRA)
        (declare-fun x () Real)
        (assert (and (> x 0) (< x 0)))
        """
        result = solve(formula, logic="QF_LRA")
        self.assertEqual(result, SolverResult.UNSAT)

    def test_complex_sat(self):
        """Test complex satisfiable formula"""
        formula = """
        (set-logic QF_LRA)
        (declare-fun x () Real)
        (declare-fun y () Real)
        (declare-fun z () Real)
        (assert (>= x 0))
        (assert (<= y 10))
        (assert (>= z 2))
        (assert (<= z 8))
        (assert (or (>= (+ x y) 15) (and (>= x 5) (<= y 3))))
        """
        result = solve(formula, logic="QF_LRA")
        self.assertEqual(result, SolverResult.SAT)

    def test_complex_unsat(self):
        """Test complex unsatisfiable formula"""
        formula = """
        (set-logic QF_LRA)
        (declare-fun x () Real)
        (declare-fun y () Real)
        (declare-fun z () Real)
        (assert (and (>= x 0) (<= x 5)))
        (assert (and (>= y 0) (<= y 5)))
        (assert (and (>= z 0) (<= z 5)))
        (assert (>= (+ x y z) 20))
        """
        result = solve(formula, logic="QF_LRA")
        self.assertEqual(result, SolverResult.UNSAT)

    def test_system_of_equations_sat(self):
        """Test satisfiable system of equations"""
        formula = """
        (set-logic QF_LRA)
        (declare-fun x () Real)
        (declare-fun y () Real)
        (declare-fun z () Real)
        (assert (= (+ x y) 10))
        (assert (= (+ y z) 20))
        (assert (= (+ x z) 15))
        """
        result = solve(formula, logic="QF_LRA")
        self.assertEqual(result, SolverResult.SAT)

    def test_system_of_equations_unsat(self):
        """Test unsatisfiable system of equations"""
        formula = """
        (set-logic QF_LRA)
        (declare-fun x () Real)
        (declare-fun y () Real)
        (declare-fun z () Real)
        (assert (= x 5))
        (assert (= y 10))
        (assert (= z 20))
        (assert (= (+ x y z) 10))
        """
        result = solve(formula, logic="QF_LRA")
        self.assertEqual(result, SolverResult.UNSAT)

    def test_boolean_combinations(self):
        """Test formula with Boolean combinations"""
        formula = """
        (set-logic QF_LRA)
        (declare-fun x () Real)
        (declare-fun y () Real)
        (assert (or (and (> x 0) (< y 0))
                    (and (< x 0) (> y 0))))
        (assert (or (and (> x 1) (> y -1))
                    (and (< x -1) (< y -2))))
        """
        result = solve(formula, logic="QF_LRA")
        self.assertEqual(result, SolverResult.SAT)

    def test_many_variables(self):
        """Test with many variables"""
        formula = """
        (set-logic QF_LRA)
        (declare-fun a () Real)
        (declare-fun b () Real)
        (declare-fun c () Real)
        (declare-fun d () Real)
        (declare-fun e () Real)
        (assert (>= a 0))
        (assert (>= b 0))
        (assert (>= c 0))
        (assert (>= d 0))
        (assert (>= e 0))
        (assert (<= (+ a b c d e) 10))
        (assert (>= (+ a b) 3))
        (assert (>= (+ c d) 4))
        (assert (>= e 2))
        """
        result = solve(formula, logic="QF_LRA")
        self.assertEqual(result, SolverResult.SAT)

    def test_solver_class(self):
        """Test using the CDCLTSolver class"""
        solver = CDCLTSolver()

        formula = """
        (set-logic QF_LRA)
        (declare-fun x () Real)
        (assert (> x 5))
        (assert (< x 10))
        """

        result = solver.solve_smt2_string(formula, logic="QF_LRA")
        self.assertEqual(result, SolverResult.SAT)


if __name__ == '__main__':
    main()
