from sympy import symbols

from aria.tests import TestCase, main, skipIf

try:
    from pysat.solvers import Solver as _PySATSolver
except ImportError:
    _PySATSolver = None

from aria.smt.arith.cad import solve_bool_system_cdclt


@skipIf(_PySATSolver is None, "pysat is not available")
class TestArithmeticCDCLT(TestCase):
    def test_sat_cube(self):
        x = symbols("x", real=True)

        result = solve_bool_system_cdclt((x**2 - 2 < 0) & (x > 0), [x])

        self.assertEqual(len(result), 1)
        sample = result[0]
        self.assertTrue(sample[x] > 0)
        self.assertTrue(sample[x] ** 2 - 2 < 0)

    def test_unsat_conjunction(self):
        x = symbols("x", real=True)

        result = solve_bool_system_cdclt((x**2 + 1 < 0) & (x > 0), [x])

        self.assertEqual(result, [])

    def test_boolean_branching(self):
        x, y = symbols("x y", real=True)
        formula = ((x**2 - 2 < 0) & (x > 0)) | ((y**2 + 1 < 0) & (y > 0))

        result = solve_bool_system_cdclt(formula, [x, y])

        self.assertEqual(len(result), 1)
        sample = result[0]
        self.assertTrue(sample[x] > 0)
        self.assertTrue(sample[x] ** 2 - 2 < 0)

    def test_unsat_requires_learning(self):
        x = symbols("x", real=True)
        formula = (x > 0) & (x < 0)

        result = solve_bool_system_cdclt(formula, [x])

        self.assertEqual(result, [])


if __name__ == "__main__":
    main()
