"""
Test cases for auxiliary invariant generation using EF and Houdini approaches.
"""

import z3

from aria.efmc.tests import TestCase, main
from aria.efmc.sts import TransitionSystem
from aria.efmc.engines.kinduction.aux_invariant_generator import InvariantGenerator
from aria.efmc.tests.simple_sts import get_int_sys1, get_int_sys2, get_int_sys3, get_int_sys4


class TestAuxInvariantGenerator(TestCase):
    """Test cases for auxiliary invariant generation."""

    def setUp(self):
        super().setUp()
        # Create a simple transition system for testing
        self.simple_sts = TransitionSystem()
        x, y, xp, yp = z3.Ints("x y x! y!")
        init = z3.And(x >= 0, y >= 0)
        trans = z3.And(xp == x + 1, yp == y + 2)
        post = z3.And(x >= 0, y >= 0)
        self.simple_sts.from_z3_cnts([[x, y, xp, yp], init, trans, post])

    def test_init(self):
        """Test initialization of InvariantGenerator."""
        generator = InvariantGenerator(self.simple_sts)
        self.assertEqual(generator.sts, self.simple_sts)

    def test_generate_via_ef(self):
        """Test EF-based auxiliary invariant generation."""
        generator = InvariantGenerator(self.simple_sts)
        result = generator.generate_via_ef()
        self.assertIsInstance(result, z3.ExprRef)
        print(f"EF generated invariant: {result}")

    def test_generate_via_houdini(self):
        """Test Houdini-based auxiliary invariant generation."""
        generator = InvariantGenerator(self.simple_sts)
        result = generator.generate_via_houdini(timeout=10)
        self.assertIsInstance(result, z3.ExprRef)
        print(f"Houdini generated invariant: {result}")

    def test_generate_methods(self):
        """Test generate method with different parameters."""
        generator = InvariantGenerator(self.simple_sts)

        # Test EF method
        result_ef = generator.generate(method="ef")
        self.assertIsInstance(result_ef, z3.ExprRef)
        print(f"EF method result: {result_ef}")

        # Test Houdini method
        result_houdini = generator.generate(method="houdini", timeout=10)
        self.assertIsInstance(result_houdini, z3.ExprRef)
        print(f"Houdini method result: {result_houdini}")

        # Test invalid method
        with self.assertRaises(ValueError) as context:
            generator.generate(method="invalid")
        self.assertIn("Unknown method", str(context.exception))

    def test_different_system_types(self):
        """Test generation with different types of transition systems."""
        systems = [
            # Integer system 1
            (get_int_sys1(), "int_sys1"),
            # Integer system 2
            (get_int_sys2(), "int_sys2"),
            # Integer system 3
            (get_int_sys3(), "int_sys3"),
            # Integer system 4
            (get_int_sys4(), "int_sys4"),
            # Bit-vector system
            (self._create_bv_system(), "bv_system"),
            # Boolean system
            (self._create_bool_system(), "bool_system"),
        ]

        for system_data, system_name in systems:
            print(f"\nTesting {system_name}:")
            sts = TransitionSystem()
            sts.from_z3_cnts(list(system_data))

            generator = InvariantGenerator(sts)

            # Test EF method
            print(f"  Testing EF for {system_name}...")
            result_ef = generator.generate_via_ef()
            self.assertIsInstance(result_ef, z3.ExprRef, f"EF failed for {system_name}")
            print(f"  EF result: {result_ef}")

            # Test Houdini method
            print(f"  Testing Houdini for {system_name}...")
            result_houdini = generator.generate_via_houdini(timeout=10)
            self.assertIsInstance(
                result_houdini, z3.ExprRef, f"Houdini failed for {system_name}"
            )
            print(f"  Houdini result: {result_houdini}")

    def _create_bv_system(self):
        """Create a simple bit-vector system."""
        x, y, xp, yp = z3.BitVecs("x y x! y!", 8)
        init = z3.And(x >= 0, y >= 0)
        trans = z3.And(xp == x + 1, yp == y + 2)
        post = z3.And(x >= 0, y >= 0)
        return [x, y, xp, yp], init, trans, post

    def _create_bool_system(self):
        """Create a simple boolean system."""
        x, y, xp, yp = z3.Bools("x y x! y!")
        init = z3.And(x, y)
        trans = z3.And(xp == z3.Not(x), yp == z3.Not(y))
        post = z3.Or(x, y)
        return [x, y, xp, yp], init, trans, post


if __name__ == "__main__":
    main()
