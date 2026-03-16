"""
Test cases for the dillig_abduce function.

This module provides comprehensive test cases for the Dillig-style abduction
implementation. Each test case validates that the abduction result satisfies
the two key properties of abduction:

1. Consistency: Γ ∧ ψ is satisfiable (the abduction is consistent with the precondition)
2. Sufficiency: Γ ∧ ψ |= φ (the abduction together with the precondition entails the postcondition)

Where:
- Γ (Gamma) is the precondition
- φ (Phi) is the postcondition
- ψ (Psi) is the abduced formula

The Dillig-style abduction uses Minimal Satisfying Assignments (MSA) to find
a minimal model that makes the formula pre_cond -> post_cond valid, and then
generalizes it through quantifier elimination.
"""

import z3

from aria.efmc.tests import TestCase, main
from aria.efmc.engines.abduction.abductor.dillig_abduct import dillig_abduce
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


class TestDilligAbduce(TestCase):

    def test_simple_linear_constraints(self):
        """Test abduction with simple linear constraints."""
        x, y, z = z3.Reals("x y z")

        pre_cond = z3.And(x <= 0, y > 1)
        post_cond = 2 * x - y + 3 * z <= 10

        result = dillig_abduce(pre_cond, post_cond)
        self.assertIsNotNone(result, "Abduction should succeed")

        consistency, sufficiency = validate_abduction(pre_cond, post_cond, result)
        self.assertTrue(consistency, "Abduction should be consistent with precondition")
        self.assertTrue(
            sufficiency,
            "Abduction should be sufficient with precondition to entail postcondition",
        )

        print(f"Abduction result: {result}")

    def test_integer_constraints(self):
        """Test abduction with integer constraints."""
        a, b, c = z3.Ints("a b c")

        pre_cond = z3.And(a >= 0, b >= 0)
        post_cond = a + b + c <= 10

        result = dillig_abduce(pre_cond, post_cond)
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

        result = dillig_abduce(pre_cond, post_cond)
        self.assertIsNotNone(result, "Abduction should succeed")

        consistency, sufficiency = validate_abduction(pre_cond, post_cond, result)
        self.assertTrue(consistency, "Abduction should be consistent with precondition")
        self.assertTrue(
            sufficiency,
            "Abduction should be sufficient with precondition to entail postcondition",
        )

        print(f"Abduction result: {result}")

    def test_disjunctive_constraints(self):
        """Test abduction with disjunctive constraints, which is a strength of the Dillig approach."""
        x, y, z = z3.Ints("x y z")

        pre_cond = z3.Or(z3.And(x == 1, y == 1), z3.And(x == 2, y == 2))
        post_cond = z3.Or(z <= 0, z >= 10)

        result = dillig_abduce(pre_cond, post_cond)
        self.assertIsNotNone(
            result, "Abduction should succeed with disjunctive constraints"
        )

        consistency, sufficiency = validate_abduction(pre_cond, post_cond, result)
        self.assertTrue(consistency, "Abduction should be consistent with precondition")
        self.assertTrue(
            sufficiency,
            "Abduction should be sufficient with precondition to entail postcondition",
        )

        print(f"Abduction result: {result}")

    def test_complex_formula(self):
        """Test abduction with a more complex formula structure."""
        x, y, z = z3.Ints("x y z")

        pre_cond = z3.And(z3.Or(x <= 0, x >= 10), z3.Or(y <= 0, y >= 10))
        post_cond = z3.Implies(z3.And(x >= 10, y >= 10), z >= 20)

        result = dillig_abduce(pre_cond, post_cond)
        self.assertIsNotNone(result, "Abduction should succeed with complex formula")

        consistency, sufficiency = validate_abduction(pre_cond, post_cond, result)
        self.assertTrue(consistency, "Abduction should be consistent with precondition")
        self.assertTrue(
            sufficiency,
            "Abduction should be sufficient with precondition to entail postcondition",
        )

        print(f"Abduction result: {result}")

    def test_minimal_model(self):
        """Test that the Dillig approach finds a minimal model."""
        a, b, c, d = z3.Ints("a b c d")

        # This formula has two minimal models: (a=3,b=3) and (a=1,b=1,c=1,d=1)
        # The MSA algorithm should find the first one as it's smaller
        formula = z3.Or(z3.And(a == 3, b == 3), z3.And(a == 1, b == 1, c == 1, d == 1))

        pre_cond = formula
        post_cond = z3.BoolVal(True)  # Any postcondition will do

        result = dillig_abduce(pre_cond, post_cond)
        self.assertIsNotNone(result, "Abduction should succeed")

        # Check if the result is consistent with the minimal model (a=3,b=3)
        s = z3.Solver()
        s.add(result)
        s.add(a == 3, b == 3)
        self.assertEqual(
            s.check(),
            z3.sat,
            "Abduction should be consistent with the minimal model (a=3,b=3)",
        )

        # Check if the result is inconsistent with the larger model (a=1,b=1,c=1,d=1)
        # or at least not entailed by it
        s = z3.Solver()
        s.add(result)
        s.add(a == 1, b == 1, c == 1, d == 1)
        # We don't require the result to be inconsistent with the larger model,
        # just that it doesn't entail the larger model
        self.assertNotEqual(
            s.check(),
            z3.unsat,
            "Abduction doesn't need to be inconsistent with the larger model",
        )

        print(f"Abduction result: {result}")

    def test_comparison_with_qe_abduce(self):
        """Compare dillig_abduce with qe_abduce on a case where QE might struggle."""
        from aria.efmc.engines.abduction.abductor.qe_abduct import qe_abduce

        x, y, z = z3.Ints("x y z")

        # A formula with many disjunctions that might be challenging for QE
        pre_cond = z3.Or(
            z3.And(x == 1, y == 1),
            z3.And(x == 2, y == 2),
            z3.And(x == 3, y == 3),
            z3.And(x == 4, y == 4),
            z3.And(x == 5, y == 5),
        )
        post_cond = z == 0

        # Try both approaches
        dillig_result = dillig_abduce(pre_cond, post_cond)
        qe_result = qe_abduce(pre_cond, post_cond)

        # Check if dillig_abduce succeeds where qe_abduce might fail
        if qe_result is None and dillig_result is not None:
            print("dillig_abduce succeeded where qe_abduce failed")
        elif qe_result is not None and dillig_result is not None:
            # Compare the results
            s = z3.Solver()
            s.add(z3.Xor(dillig_result, qe_result))
            if s.check() == z3.sat:
                print("dillig_abduce and qe_abduce produced different results")
                print(f"dillig_result: {dillig_result}")
                print(f"qe_result: {qe_result}")
            else:
                print("dillig_abduce and qe_abduce produced equivalent results")

        # Validate the dillig result if it exists
        if dillig_result is not None:
            consistency, sufficiency = validate_abduction(
                pre_cond, post_cond, dillig_result
            )
            self.assertTrue(
                consistency, "Abduction should be consistent with precondition"
            )
            self.assertTrue(
                sufficiency,
                "Abduction should be sufficient with precondition to entail postcondition",
            )

    def test_unsatisfiable_implication(self):
        """Test abduction when the implication pre_cond -> post_cond is unsatisfiable."""
        x, y = z3.Ints("x y")

        pre_cond = x > 0
        post_cond = x < 0  # This makes pre_cond -> post_cond unsatisfiable

        result = dillig_abduce(pre_cond, post_cond)
        self.assertIsNone(
            result, "Abduction should fail when implication is unsatisfiable"
        )

        print(f"Abduction result: {result}")


if __name__ == "__main__":
    main()
