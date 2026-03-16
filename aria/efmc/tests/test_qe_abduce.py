"""
Test cases for the qe_abduce function.

This module provides comprehensive test cases for the quantifier elimination-based
abduction implementation. Each test case validates that the abduction result satisfies
the two key properties of abduction:

1. Consistency: Γ ∧ ψ is satisfiable (the abduction is consistent with the precondition)
2. Sufficiency: Γ ∧ ψ |= φ (the abduction together with the precondition entails the postcondition)

Where:
- Γ (Gamma) is the precondition
- φ (Phi) is the postcondition
- ψ (Psi) is the abduced formula
"""

import z3

from aria.efmc.tests import TestCase, main
from aria.efmc.engines.abduction.abductor.qe_abduct import qe_abduce
from aria.efmc.utils import is_sat, is_entail


def validate_abduction(pre_cond, post_cond, abduction):
    """
    Validate that an abduction result satisfies the required properties.

    Args:
        pre_cond: Precondition Γ
        post_cond: Postcondition φ
        abduction: Abduction result ψ

    Returns:
        Tuple[bool, bool]: (consistency, sufficiency)
    """
    # Check consistency: Γ ∧ ψ is satisfiable
    consistency = is_sat(z3.And(pre_cond, abduction))

    # Check sufficiency: Γ ∧ ψ |= φ
    sufficiency = is_entail(z3.And(pre_cond, abduction), post_cond)

    return consistency, sufficiency


class TestQEAbduce(TestCase):

    def test_simple_linear_constraints(self):
        """Test abduction with simple linear constraints."""
        x, y, z = z3.Reals("x y z")

        pre_cond = z3.And(x <= 0, y > 1)
        post_cond = 2 * x - y + 3 * z <= 10

        result = qe_abduce(pre_cond, post_cond)
        self.assertIsNotNone(result, "Abduction should succeed")

        consistency, sufficiency = validate_abduction(pre_cond, post_cond, result)
        self.assertTrue(consistency, "Abduction should be consistent with precondition")
        self.assertTrue(
            sufficiency,
            "Abduction should be sufficient with precondition to entail postcondition",
        )

        # The result should be a formula over z only
        result_vars = set(str(v) for v in z3.z3util.get_vars(result))
        self.assertEqual(
            result_vars,
            {"z"},
            "Abduction should only contain variables from post_cond - pre_cond",
        )

        print(f"Abduction result: {result}")

    def test_integer_constraints(self):
        """Test abduction with integer constraints."""
        a, b, c = z3.Ints("a b c")

        pre_cond = z3.And(a >= 0, b >= 0)
        post_cond = a + b + c <= 10

        result = qe_abduce(pre_cond, post_cond)
        self.assertIsNotNone(result, "Abduction should succeed")

        consistency, sufficiency = validate_abduction(pre_cond, post_cond, result)
        self.assertTrue(consistency, "Abduction should be consistent with precondition")
        self.assertTrue(
            sufficiency,
            "Abduction should be sufficient with precondition to entail postcondition",
        )

        print(f"Abduction result: {result}")

    def test_boolean_constraints(self):
        """Test abduction with boolean constraints."""
        p, q, r = z3.Bools("p q r")

        pre_cond = z3.And(p, q)
        post_cond = z3.Or(p, r)

        result = qe_abduce(pre_cond, post_cond)
        self.assertIsNotNone(result, "Abduction should succeed")

        consistency, sufficiency = validate_abduction(pre_cond, post_cond, result)
        self.assertTrue(consistency, "Abduction should be consistent with precondition")
        self.assertTrue(
            sufficiency,
            "Abduction should be sufficient with precondition to entail postcondition",
        )

        print(f"Abduction result: {result}")

    def test_mixed_constraints(self):
        """Test abduction with mixed integer and boolean constraints."""
        x, y = z3.Ints("x y")
        p, q = z3.Bools("p q")

        pre_cond = z3.And(x > 0, p)
        post_cond = z3.Implies(p, z3.And(x + y > 0, q))

        result = qe_abduce(pre_cond, post_cond)
        self.assertIsNotNone(result, "Abduction should succeed")

        consistency, sufficiency = validate_abduction(pre_cond, post_cond, result)
        self.assertTrue(consistency, "Abduction should be consistent with precondition")
        self.assertTrue(
            sufficiency,
            "Abduction should be sufficient with precondition to entail postcondition",
        )

        print(f"Abduction result: {result}")

    def test_unsatisfiable_precondition(self):
        """Test abduction with unsatisfiable precondition."""
        x, y = z3.Ints("x y")

        pre_cond = z3.And(x > 0, x < 0)  # Unsatisfiable
        post_cond = y > 0

        result = qe_abduce(pre_cond, post_cond)
        self.assertIsNotNone(
            result, "Abduction should succeed even with unsatisfiable precondition"
        )

        # Since pre_cond is unsatisfiable, any formula is a valid abduction
        # We only check sufficiency here
        sufficiency = is_entail(z3.And(pre_cond, result), post_cond)
        self.assertTrue(
            sufficiency,
            "Abduction should be sufficient with precondition to entail postcondition",
        )

        print(f"Abduction result: {result}")

    def test_tautological_postcondition(self):
        """Test abduction with tautological postcondition."""
        x, y = z3.Ints("x y")

        pre_cond = x > 0
        post_cond = z3.Or(y > 0, y <= 0)  # Tautology

        result = qe_abduce(pre_cond, post_cond)
        self.assertIsNotNone(
            result, "Abduction should succeed with tautological postcondition"
        )

        consistency, sufficiency = validate_abduction(pre_cond, post_cond, result)
        self.assertTrue(consistency, "Abduction should be consistent with precondition")
        self.assertTrue(
            sufficiency,
            "Abduction should be sufficient with precondition to entail postcondition",
        )

        # The result should be a tautology
        self.assertTrue(
            is_entail(z3.BoolVal(True), result),
            "Abduction result should be a tautology",
        )

        print(f"Abduction result: {result}")

    def test_nonlinear_constraints(self):
        """Test abduction with nonlinear constraints."""
        x, y, z = z3.Reals("x y z")

        pre_cond = x * x + y * y <= 1  # Circle
        post_cond = z >= x * x  # z is at least x^2

        result = qe_abduce(pre_cond, post_cond)
        self.assertIsNotNone(
            result, "Abduction should succeed with nonlinear constraints"
        )

        consistency, sufficiency = validate_abduction(pre_cond, post_cond, result)
        self.assertTrue(consistency, "Abduction should be consistent with precondition")
        self.assertTrue(
            sufficiency,
            "Abduction should be sufficient with precondition to entail postcondition",
        )

        print(f"Abduction result: {result}")

    def test_array_constraints(self):
        """Test abduction with array constraints."""
        a = z3.Array("a", z3.IntSort(), z3.IntSort())
        i, v = z3.Ints("i v")

        pre_cond = z3.And(i >= 0, i < 10)
        post_cond = z3.Select(z3.Store(a, i, v), i) == v

        result = qe_abduce(pre_cond, post_cond)
        self.assertIsNotNone(result, "Abduction should succeed with array constraints")

        consistency, sufficiency = validate_abduction(pre_cond, post_cond, result)
        self.assertTrue(consistency, "Abduction should be consistent with precondition")
        self.assertTrue(
            sufficiency,
            "Abduction should be sufficient with precondition to entail postcondition",
        )

        print(f"Abduction result: {result}")

    def test_minimal_abduction(self):
        """Test that the abduction is minimal (not unnecessarily strong)."""
        x, y, z = z3.Reals("x y z")

        pre_cond = z3.And(x >= 0, y >= 0)
        post_cond = x + y + z <= 10

        result = qe_abduce(pre_cond, post_cond)
        self.assertIsNotNone(result, "Abduction should succeed")

        # A stronger but valid abduction would be z <= 0
        stronger = z <= 0

        # Check that our result is not unnecessarily strong
        self.assertFalse(
            is_entail(result, stronger), "Abduction should not be unnecessarily strong"
        )

        print(f"Abduction result: {result}")
        print(f"Stronger alternative: {stronger}")

    def test_multiple_solutions(self):
        """Test a case where multiple abductions are possible."""
        x, y, z1, z2 = z3.Reals("x y z1 z2")

        pre_cond = z3.And(x >= 0, y >= 0)
        post_cond = z3.Or(z1 <= 5, z2 <= 5)

        result = qe_abduce(pre_cond, post_cond)
        self.assertIsNotNone(result, "Abduction should succeed")

        consistency, sufficiency = validate_abduction(pre_cond, post_cond, result)
        self.assertTrue(consistency, "Abduction should be consistent with precondition")
        self.assertTrue(
            sufficiency,
            "Abduction should be sufficient with precondition to entail postcondition",
        )

        # Alternative valid abductions
        alt1 = z1 <= 5
        alt2 = z2 <= 5

        print(f"Abduction result: {result}")
        print(f"Alternative 1: {alt1}")
        print(f"Alternative 2: {alt2}")


if __name__ == "__main__":
    main()
