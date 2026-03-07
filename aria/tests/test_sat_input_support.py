from pysat.formula import CNF

from aria.bool.features.sat_instance import SATInstance
from aria.bool.sat.z3sat_solver import Z3MaxSATSolver
from aria.tests import TestCase, main


class TestSATInputSupport(TestCase):
    def assert_clause_set_equal(self, actual, expected):
        self.assertEqual(
            {tuple(sorted(clause)) for clause in actual},
            {tuple(sorted(clause)) for clause in expected},
        )

    def test_sat_instance_accepts_inline_cnf_string(self):
        cnf = "p cnf 2 2\n1 2 0\n-1 2 0\n"
        sat_inst = SATInstance(cnf)
        self.assertEqual(sat_inst.c, 2)
        self.assertEqual(sat_inst.v, 2)
        self.assertIsNone(sat_inst.path_to_cnf)
        self.assert_clause_set_equal(sat_inst.clauses, [[1, 2], [-1, 2]])

    def test_sat_instance_accepts_numeric_clauses(self):
        sat_inst = SATInstance([[1, 2, 0], [-1, 2, 0]])
        self.assertEqual(sat_inst.c, 2)
        self.assertEqual(sat_inst.v, 2)
        self.assertIsNone(sat_inst.path_to_cnf)
        self.assert_clause_set_equal(sat_inst.clauses, [[1, 2], [-1, 2]])

    def test_sat_instance_accepts_pysat_cnf(self):
        cnf = CNF(from_clauses=[[1, 2], [-1, 2]])
        sat_inst = SATInstance(cnf)
        self.assertEqual(sat_inst.c, 2)
        self.assertEqual(sat_inst.v, 2)
        self.assertIsNone(sat_inst.path_to_cnf)
        self.assert_clause_set_equal(sat_inst.clauses, [[1, 2], [-1, 2]])

    def test_z3_maxsat_accepts_dimacs_terminated_clauses(self):
        solver = Z3MaxSATSolver()
        solver.from_int_clauses(
            hard=[[1, 0], [-1, 2, 0]],
            soft=[[2, 0], [-2, 0]],
            weight=[1, 2],
        )
        self.assertEqual(solver.check(), 2)


if __name__ == "__main__":
    main()
