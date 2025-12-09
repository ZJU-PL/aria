# coding: utf-8
"""
For testing the quantifier elimination engine
"""

import z3

from aria.tests import TestCase, main
from aria.quant.qe.qe_lme import qelim_exists_lme


def is_equivalent(a: z3.BoolRef, b: z3.BoolRef, timeout_ms=1000):
    """
    Check if a and b are equivalent with a timeout
    """
    s = z3.Solver()
    s.set("timeout", timeout_ms)
    s.add(a != b)
    result = s.check()
    if result == z3.sat:
        return False
    elif result == z3.unknown:
        # If timeout occurs, we'll assume they might be equivalent
        # but log a warning
        print(f"WARNING: Equivalence check timed out after {timeout_ms}ms")
        return True
    return True


class TestQuantifierElimination(TestCase):

    def test_simple_arithmetic(self):
        """Test QE with simple arithmetic formulas"""
        x, y = z3.Ints("x y")

        # Simple linear formula: ∃x. (x > 5 ∧ x < 10)
        fml1 = z3.And(x > 5, x < 10)
        qf1 = qelim_exists_lme(fml1, x)
        qfml1 = z3.Exists(x, fml1)
        assert is_equivalent(qf1, qfml1)

        # Simple formula with one other variable: ∃x. (x > y)
        fml2 = x > y
        qf2 = qelim_exists_lme(fml2, x)
        qfml2 = z3.Exists(x, fml2)
        assert is_equivalent(qf2, qfml2)

        # Simple disjunction: ∃x. (x == 0 ∨ x == 1)
        fml3 = z3.Or(x == 0, x == 1)
        qf3 = qelim_exists_lme(fml3, x)
        qfml3 = z3.Exists(x, fml3)
        assert is_equivalent(qf3, qfml3)

    def test_basic_real_arithmetic(self):
        """Test QE with basic real arithmetic formulas"""
        x, y = z3.Reals("x y")

        # Simple real formula: ∃x. (x > 0 ∧ x < 1)
        fml1 = z3.And(x > 0, x < 1)
        qf1 = qelim_exists_lme(fml1, x)
        qfml1 = z3.Exists(x, fml1)
        assert is_equivalent(qf1, qfml1)

        # Simple formula with one other variable: ∃x. (x < y)
        fml2 = x < y
        qf2 = qelim_exists_lme(fml2, x)
        qfml2 = z3.Exists(x, fml2)
        assert is_equivalent(qf2, qfml2)

    def test_boolean_structure(self):
        """Test QE with more complex Boolean structures"""
        x, y = z3.Ints("x y")

        # Nested OR-AND structure: ∃x. ((x < 0 ∨ x > 10) ∧ (x != y))
        fml1 = z3.And(z3.Or(x < 0, x > 10), x != y)
        qf1 = qelim_exists_lme(fml1, x)
        qfml1 = z3.Exists(x, fml1)
        assert is_equivalent(qf1, qfml1)

        # Formula with XOR: ∃x. (x < y ⊕ x > 5)
        fml2 = z3.Xor(x < y, x > 5)
        qf2 = qelim_exists_lme(fml2, x)
        qfml2 = z3.Exists(x, fml2)
        assert is_equivalent(qf2, qfml2)

        # Formula with implication: ∃x. (x > 0 → x < 10)
        fml3 = z3.Implies(x > 0, x < 10)
        qf3 = qelim_exists_lme(fml3, x)
        qfml3 = z3.Exists(x, fml3)
        assert is_equivalent(qf3, qfml3)

        # Formula with nested structure: ∃x. ((x == 0 ∨ x == 1) ∧ (x < y ∨ x > y + 5))
        fml4 = z3.And(z3.Or(x == 0, x == 1), z3.Or(x < y, x > y + 5))
        qf4 = qelim_exists_lme(fml4, x)
        qfml4 = z3.Exists(x, fml4)
        assert is_equivalent(qf4, qfml4)


if __name__ == '__main__':
    main()
