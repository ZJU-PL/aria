# coding: utf-8
"""
For testing the QF_BV solver
"""

from aria.tests import TestCase, main
from aria.smt.bv import QFBVSolver
from aria.utils import SolverResult


class TestBVSat(TestCase):
    """
    Test the bit-blasting based solver
    """

    def test_bvsat(self):
        import z3
        x, y, z = z3.BitVecs("x y z", 5)
        # More complex constraints with three variables
        fml = z3.And(5 < x, x < y, y < z, z < 15, (x + y) % 7 == 3)
        sol = QFBVSolver()
        print(sol.solve_smt_formula(fml))
        # s = z3.Solver()
        # s.add(fml)
        # print(s.to_smt2())

    def test_bvsat_simple_constraints(self):
        """Test simple bit-vector constraints"""
        import z3
        x, y, z = z3.BitVecs("x y z", 8)
        # Added more constraints and a third variable
        fml = z3.And(
            x > 10,
            y > 20,
            z < 15,
            x + y + z == 50,
            x * 2 == y - 5
        )
        sol = QFBVSolver()
        result = sol.solve_smt_formula(fml)
        self.assertEqual(result, SolverResult.SAT)

    def test_bvsat_contradiction(self):
        """Test a direct contradiction"""
        import z3
        x, y = z3.BitVecs("x y", 8)
        # More complex contradiction with multiple variables
        fml = z3.And(
            x == y,
            x != y,
            x + y == 100,
            x * 2 == y * 2
        )
        sol = QFBVSolver()
        result = sol.solve_smt_formula(fml)
        self.assertEqual(result, SolverResult.UNSAT)

    def test_bvsat_bitwise_operations(self):
        """Test bit-vector bitwise operations"""
        import z3
        x, y = z3.BitVecs("x y", 8)
        # Simplified bitwise operations
        fml = z3.And(
            (x & 0x0F) == 0x05,  # Lower 4 bits of x are 0101
            (y | 0xF0) == 0xFF  # Upper 4 bits of y are set
        )
        sol = QFBVSolver()
        result = sol.solve_smt_formula(fml)
        self.assertEqual(result, SolverResult.SAT)

    def test_bvsat_arithmetic(self):
        """Test bit-vector arithmetic operations"""
        import z3
        x, y = z3.BitVecs("x y", 8)
        # Simplified arithmetic
        fml = z3.And(
            x + y == 100,
            x > 30,
            y > 30
        )
        sol = QFBVSolver()
        result = sol.solve_smt_formula(fml)
        self.assertEqual(result, SolverResult.SAT)

    def test_bvsat_shift_operations(self):
        """Test bit-vector shift operations"""
        import z3
        x, y = z3.BitVecs("x y", 8)
        fml = z3.And(
            (x << 2) == 20,  # Left shift
            (y >> 1) == x,  # Right shift
            x > 0,
            y > 0
        )
        sol = QFBVSolver()
        result = sol.solve_smt_formula(fml)
        self.assertEqual(result, SolverResult.SAT)

    def test_bvsat_right_shift(self):
        """Test bit-vector right shift operations"""
        import z3
        x, y = z3.BitVecs("x y", 8)
        fml = z3.And(
            (x >> 1) == 10,  # Right shift
            (y >> 2) == (x >> 1),  # Nested right shifts
            x > 0,
            y > 0
        )
        sol = QFBVSolver()
        result = sol.solve_smt_formula(fml)
        self.assertEqual(result, SolverResult.SAT)

    def test_bvsat_rotation(self):
        """Test bit-vector rotation operations"""
        import z3
        x, y = z3.BitVecs("x y", 8)
        fml = z3.And(
            z3.RotateLeft(x, 2) == 20,
            z3.RotateRight(y, 3) == x,
            x > 0,
            y > 0
        )
        sol = QFBVSolver()
        result = sol.solve_smt_formula(fml)
        self.assertEqual(result, SolverResult.SAT)

    def test_bvsat_extraction(self):
        """Test bit-vector extraction operations"""
        import z3
        x, y = z3.BitVecs("x y", 8)
        # More complex extraction operations
        fml = z3.And(
            z3.Extract(3, 1, x) == 3,  # Extract bits 3:1
            z3.Extract(7, 4, y) == z3.Extract(3, 0, x),  # Compare extractions
            x > 0,
            y > 0
        )
        sol = QFBVSolver()
        result = sol.solve_smt_formula(fml)
        self.assertEqual(result, SolverResult.SAT)

    def test_bvsat_multiple_variables(self):
        """Test with multiple bit-vector variables"""
        import z3
        x, y, z = z3.BitVecs("x y z", 8)
        # More variables and complex constraints
        fml = z3.And(
            x + y + z == 100,
            x > 10,
            y > 20,
            z > 30
        )
        sol = QFBVSolver()
        result = sol.solve_smt_formula(fml)
        self.assertEqual(result, SolverResult.SAT)

    def test_bvsat_complex_formula(self):
        """Test a more complex formula with multiple operations"""
        import z3
        x, y = z3.BitVecs("x y", 8)

        # Simplified complex formula
        fml = z3.And(
            # Arithmetic
            x + y == 100,

            # Bitwise operations
            (x & 0x0F) == 0x05,  # Lower 4 bits of x are 0101
            (y | 0xF0) == 0xFF,  # Upper 4 bits of y are set

            # Shifts and rotations
            (x << 1) > 50,

            # Extractions
            z3.Extract(7, 4, x) != 0,  # Upper 4 bits of x are not all zeros
            z3.Extract(3, 0, y) != 0  # Lower 4 bits of y are not all zeros
        )

        sol = QFBVSolver()
        result = sol.solve_smt_formula(fml)
        self.assertEqual(result, SolverResult.SAT)

    def test_bvsat_combined_operations(self):
        """Test a combination of different bit-vector operations"""
        import z3
        x, y = z3.BitVecs("x y", 8)

        # Formula combining different operations
        fml = z3.And(
            # Basic constraints
            x > 20,
            y < 100,

            # Arithmetic and bitwise combined
            (x + (y & 0x0F)) == 50,

            # Shift and arithmetic combined
            ((x << 1) - y) < 30,

            # Extraction and comparison
            z3.ULT(z3.Extract(3, 0, x), z3.Extract(3, 0, y)),

            # Rotation and bitwise
            (z3.RotateLeft(x, 2) & 0xF0) != 0
        )

        sol = QFBVSolver()
        result = sol.solve_smt_formula(fml)
        self.assertEqual(result, SolverResult.SAT)


if __name__ == '__main__':
    main()
