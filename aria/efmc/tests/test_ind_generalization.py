"""Unit tests for ind_generalization module."""

import z3
from aria.efmc.utils.ind_generalization import InductiveGeneralizer
from aria.efmc.sts import TransitionSystem
from aria.efmc.tests import TestCase, main


class TestInductiveGeneralizer(TestCase):
    """Test cases for InductiveGeneralizer functionality."""

    def setUp(self):
        """Set up test fixtures."""
        x = z3.Int("x")
        x_prime = z3.Int("x_prime")

        self.sts = TransitionSystem(
            variables=[x],
            prime_variables=[x_prime],
            init=x == 0,
            trans=x_prime == x + 1,
            post=x < 10,
        )
        self.generalizer = InductiveGeneralizer(self.sts)

    def test_init_and_timeout(self):
        """Test initialization and timeout setting."""
        self.assertEqual(self.generalizer.sts, self.sts)
        self.assertEqual(self.generalizer.timeout, 10000)

        self.generalizer.set_timeout(5000)
        self.assertEqual(self.generalizer.timeout, 5000)

    def test_check_invariant(self):
        """Test invariant checking."""
        x = z3.Int("x")

        # Test valid invariant
        inv = x < 10
        valid, condition, model = self.generalizer.check_invariant(inv)
        self.assertIsInstance(valid, bool)
        self.assertTrue(condition is None or isinstance(condition, str))
        self.assertTrue(model is None or hasattr(model, "eval"))

        # Test invalid invariant
        inv = x < 5
        valid, condition, model = self.generalizer.check_invariant(inv)
        self.assertFalse(valid)
        self.assertIn(condition, ["initiation", "inductiveness", "safety"])

    def test_cti_operations(self):
        """Test counterexample to induction operations."""
        x = z3.Int("x")
        inv = x < 5

        # Test getting CTI
        condition, model = self.generalizer.get_cti(inv)
        self.assertIsNotNone(condition)
        self.assertIsNotNone(model)

        # Test strengthening and weakening (simplified)
        inv = x < 10
        model = z3.Model()  # Simplified model
        self.assertIsNotNone(self.generalizer.strengthen_from_cti(inv, model))
        self.assertIsNotNone(self.generalizer.weaken_from_cti(inv, model))

    def test_generalization_methods(self):
        """Test generalization methods."""
        x = z3.Int("x")

        # Test dropping literals
        inv = z3.And(x >= 0, x < 10)
        generalized = self.generalizer.generalize_by_dropping_literals(inv)
        self.assertIsNotNone(generalized)
        self.assertTrue(z3.is_expr(generalized))

    def test_learn_invariant(self):
        """Test invariant learning."""
        inv = self.generalizer.learn_invariant(max_iterations=10)
        self.assertTrue(inv is None or z3.is_expr(inv))

    def test_unsafe_system(self):
        """Test with unsafe system."""
        x = z3.Int("x")
        x_prime = z3.Int("x_prime")

        unsafe_sts = TransitionSystem(
            variables=[x],
            prime_variables=[x_prime],
            init=x == 0,
            trans=x_prime == x + 1,
            post=x < 0,  # Impossible post condition
        )
        unsafe_generalizer = InductiveGeneralizer(unsafe_sts)

        inv = x < 10
        valid, condition, _ = unsafe_generalizer.check_invariant(inv)
        self.assertFalse(valid)
        self.assertIn(condition, ["safety", "inductiveness", "initiation"])


if __name__ == "__main__":
    main()
