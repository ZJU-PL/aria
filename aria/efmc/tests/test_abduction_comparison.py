"""
Comparison tests for different abduction approaches.

This module compares the QE-based abduction and Dillig-style abduction approaches
on the same set of test cases. It evaluates:
1. Success rate (whether abduction succeeds)
2. Correctness (consistency and sufficiency)
3. Minimality (whether the abduction is unnecessarily strong)
4. Performance (execution time)

The comparison helps understand the strengths and weaknesses of each approach
and when one might be preferred over the other.
"""

import time
import z3

from aria.efmc.tests import TestCase, main
from aria.efmc.engines.abduction.abductor.qe_abduct import qe_abduce
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
    if abduction is None:
        return False, False

    # Check consistency: Γ ∧ ψ is satisfiable
    consistency = is_sat(z3.And(pre_cond, abduction))

    # Check sufficiency: Γ ∧ ψ |= φ
    sufficiency = is_entail(z3.And(pre_cond, abduction), post_cond)

    return consistency, sufficiency


def compare_abductions(pre_cond, post_cond, qe_result, dillig_result):
    """
    Compare two abduction results.

    Args:
        pre_cond: Precondition Γ
        post_cond: Postcondition φ
        qe_result: Result from QE-based abduction
        dillig_result: Result from Dillig-style abduction

    Returns:
        dict: Comparison results
    """
    result = {
        "qe_success": qe_result is not None,
        "dillig_success": dillig_result is not None,
        "equivalent": False,
        "qe_stronger": False,
        "dillig_stronger": False,
        "qe_consistency": False,
        "qe_sufficiency": False,
        "dillig_consistency": False,
        "dillig_sufficiency": False,
    }

    # Check correctness
    if qe_result is not None:
        result["qe_consistency"], result["qe_sufficiency"] = validate_abduction(
            pre_cond, post_cond, qe_result
        )

    if dillig_result is not None:
        result["dillig_consistency"], result["dillig_sufficiency"] = validate_abduction(
            pre_cond, post_cond, dillig_result
        )

    # Compare strength if both succeeded
    if qe_result is not None and dillig_result is not None:
        # Check if they're equivalent
        s = z3.Solver()
        # Use z3.Xor instead of z3.Not(z3.Iff) - Xor is the negation of equivalence
        s.add(z3.Xor(qe_result, dillig_result))
        if s.check() == z3.unsat:
            result["equivalent"] = True
        else:
            # Check if one is stronger than the other
            if is_entail(qe_result, dillig_result) and not is_entail(
                dillig_result, qe_result
            ):
                result["qe_stronger"] = True
            elif is_entail(dillig_result, qe_result) and not is_entail(
                qe_result, dillig_result
            ):
                result["dillig_stronger"] = True

    return result


class TestAbductionComparison(TestCase):

    def run_comparison(self, name, pre_cond, post_cond):
        """Run a comparison test and print results."""
        print(f"\n=== Test case: {name} ===")
        print(f"Pre-condition: {pre_cond}")
        print(f"Post-condition: {post_cond}")

        # Run QE-based abduction
        qe_start = time.time()
        qe_result = qe_abduce(pre_cond, post_cond)
        qe_time = time.time() - qe_start

        # Run Dillig-style abduction
        dillig_start = time.time()
        dillig_result = dillig_abduce(pre_cond, post_cond)
        dillig_time = time.time() - dillig_start

        # Compare results
        comparison = compare_abductions(pre_cond, post_cond, qe_result, dillig_result)

        # Print results
        print("\nResults:")
        print(
            f"QE-based abduction: {'Success' if comparison['qe_success'] else 'Failed'} in {qe_time:.4f}s"
        )
        if comparison["qe_success"]:
            print(f"  Result: {qe_result}")
            print(f"  Consistency: {comparison['qe_consistency']}")
            print(f"  Sufficiency: {comparison['qe_sufficiency']}")

        print(
            f"Dillig-style abduction: {'Success' if comparison['dillig_success'] else 'Failed'} in {dillig_time:.4f}s"
        )
        if comparison["dillig_success"]:
            print(f"  Result: {dillig_result}")
            print(f"  Consistency: {comparison['dillig_consistency']}")
            print(f"  Sufficiency: {comparison['dillig_sufficiency']}")

        if comparison["qe_success"] and comparison["dillig_success"]:
            if comparison["equivalent"]:
                print("The results are equivalent")
            elif comparison["qe_stronger"]:
                print("QE-based result is stronger (more restrictive)")
            elif comparison["dillig_stronger"]:
                print("Dillig-style result is stronger (more restrictive)")
            else:
                print("The results are incomparable")

        return comparison

    def test_simple_linear(self):
        """Compare approaches on simple linear constraints."""
        x, y, z = z3.Reals("x y z")

        pre_cond = z3.And(x <= 0, y > 1)
        post_cond = 2 * x - y + 3 * z <= 10

        comparison = self.run_comparison(
            "Simple linear constraints", pre_cond, post_cond
        )

        # Both approaches should succeed
        self.assertTrue(comparison["qe_success"], "QE-based abduction should succeed")
        self.assertTrue(
            comparison["dillig_success"], "Dillig-style abduction should succeed"
        )

        # Both results should be correct
        self.assertTrue(
            comparison["qe_consistency"], "QE-based result should be consistent"
        )
        self.assertTrue(
            comparison["qe_sufficiency"], "QE-based result should be sufficient"
        )
        self.assertTrue(
            comparison["dillig_consistency"], "Dillig-style result should be consistent"
        )
        self.assertTrue(
            comparison["dillig_sufficiency"], "Dillig-style result should be sufficient"
        )

    def test_disjunctive(self):
        """Compare approaches on disjunctive constraints."""
        x, y, z = z3.Ints("x y z")

        pre_cond = z3.Or(z3.And(x == 1, y == 1), z3.And(x == 2, y == 2))
        post_cond = z3.Or(z <= 0, z >= 10)

        comparison = self.run_comparison("Disjunctive constraints", pre_cond, post_cond)

        # Both approaches should succeed, but Dillig might handle disjunctions better
        self.assertTrue(
            comparison["dillig_success"],
            "Dillig-style abduction should succeed with disjunctions",
        )

        # If both succeed, both results should be correct
        if comparison["qe_success"] and comparison["dillig_success"]:
            self.assertTrue(
                comparison["qe_consistency"], "QE-based result should be consistent"
            )
            self.assertTrue(
                comparison["qe_sufficiency"], "QE-based result should be sufficient"
            )
            self.assertTrue(
                comparison["dillig_consistency"],
                "Dillig-style result should be consistent",
            )
            self.assertTrue(
                comparison["dillig_sufficiency"],
                "Dillig-style result should be sufficient",
            )

    def test_complex_formula(self):
        """Compare approaches on a complex formula structure."""
        x, y, z = z3.Ints("x y z")

        pre_cond = z3.And(z3.Or(x <= 0, x >= 10), z3.Or(y <= 0, y >= 10))
        post_cond = z3.Implies(z3.And(x >= 10, y >= 10), z >= 20)

        comparison = self.run_comparison("Complex formula", pre_cond, post_cond)

        # At least one approach should succeed
        self.assertTrue(
            comparison["qe_success"] or comparison["dillig_success"],
            "At least one abduction approach should succeed",
        )

        # If both succeed, both results should be correct
        if comparison["qe_success"] and comparison["dillig_success"]:
            self.assertTrue(
                comparison["qe_consistency"], "QE-based result should be consistent"
            )
            self.assertTrue(
                comparison["qe_sufficiency"], "QE-based result should be sufficient"
            )
            self.assertTrue(
                comparison["dillig_consistency"],
                "Dillig-style result should be consistent",
            )
            self.assertTrue(
                comparison["dillig_sufficiency"],
                "Dillig-style result should be sufficient",
            )

    def test_nonlinear(self):
        """Compare approaches on nonlinear constraints."""
        x, y, z = z3.Reals("x y z")

        pre_cond = x * x + y * y <= 1  # Circle
        post_cond = z >= x * x  # z is at least x^2

        comparison = self.run_comparison("Nonlinear constraints", pre_cond, post_cond)

        # At least one approach should succeed
        self.assertTrue(
            comparison["qe_success"] or comparison["dillig_success"],
            "At least one abduction approach should succeed with nonlinear constraints",
        )

    def test_many_disjunctions(self):
        """Compare approaches on a formula with many disjunctions."""
        x, y, z = z3.Ints("x y z")

        pre_cond = z3.Or(
            z3.And(x == 1, y == 1),
            z3.And(x == 2, y == 2),
            z3.And(x == 3, y == 3),
            z3.And(x == 4, y == 4),
            z3.And(x == 5, y == 5),
            z3.And(x == 6, y == 6),
            z3.And(x == 7, y == 7),
            z3.And(x == 8, y == 8),
            z3.And(x == 9, y == 9),
            z3.And(x == 10, y == 10),
        )
        post_cond = z == 0

        comparison = self.run_comparison("Many disjunctions", pre_cond, post_cond)

        # Dillig-style abduction might handle this better
        if comparison["qe_success"] and comparison["dillig_success"]:
            self.assertTrue(
                comparison["qe_consistency"], "QE-based result should be consistent"
            )
            self.assertTrue(
                comparison["qe_sufficiency"], "QE-based result should be sufficient"
            )
            self.assertTrue(
                comparison["dillig_consistency"],
                "Dillig-style result should be consistent",
            )
            self.assertTrue(
                comparison["dillig_sufficiency"],
                "Dillig-style result should be sufficient",
            )

    def test_unsatisfiable(self):
        """Compare approaches on an unsatisfiable implication."""
        x, y = z3.Ints("x y")

        pre_cond = x > 0
        post_cond = x < 0  # This makes pre_cond -> post_cond unsatisfiable

        comparison = self.run_comparison(
            "Unsatisfiable implication", pre_cond, post_cond
        )

        # Both approaches should fail
        self.assertFalse(
            comparison["qe_success"],
            "QE-based abduction should fail with unsatisfiable implication",
        )
        self.assertFalse(
            comparison["dillig_success"],
            "Dillig-style abduction should fail with unsatisfiable implication",
        )

    def test_array_constraints(self):
        """Compare approaches on array constraints."""
        a = z3.Array("a", z3.IntSort(), z3.IntSort())
        i, v = z3.Ints("i v")

        pre_cond = z3.And(i >= 0, i < 10)
        post_cond = z3.Select(z3.Store(a, i, v), i) == v

        comparison = self.run_comparison("Array constraints", pre_cond, post_cond)

        # At least one approach should succeed
        self.assertTrue(
            comparison["qe_success"] or comparison["dillig_success"],
            "At least one abduction approach should succeed with array constraints",
        )


if __name__ == "__main__":
    main()
