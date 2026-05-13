"""
Targeted tests for solvable polynomial closed forms.
"""

import unittest
from fractions import Fraction

from aria.utils.srk.algebra.polynomial import Polynomial as QQX
from aria.utils.srk.algebra.solvablePolynomial import (
    Block,
    closure_ocrs,
    closure_periodic_rational,
    standard_basis_prsd,
)


def var(dim: int, total_dim: int) -> QQX:
    return QQX.of_dim(dim, total_dim)


class TestSolvablePolynomialClosedForms(unittest.TestCase):
    def test_diagonal_homogeneous_closed_form(self):
        block = Block([[Fraction(2)]], [QQX.zero()])

        closed = closure_periodic_rational([block])

        self.assertEqual(closed[0].eval(0), var(0, 1))
        self.assertEqual(closed[0].eval(3), var(0, 1).scalar_mul(Fraction(8)))

    def test_nonunit_affine_constant_closed_form(self):
        block = Block([[Fraction(2)]], [QQX.scalar(Fraction(3))])

        closed = closure_periodic_rational([block])

        self.assertEqual(
            closed[0].eval(4).evaluate({0: Fraction(5)}),
            Fraction(2) ** 4 * Fraction(5) + Fraction(3) * (Fraction(2) ** 4 - 1),
        )

    def test_periodic_affine_constant_closed_form(self):
        block = Block([[Fraction(-1)]], [QQX.scalar(Fraction(6))])

        closed = closure_periodic_rational([block])

        self.assertEqual(closed[0].eval(2).evaluate({0: Fraction(7)}), Fraction(7))
        self.assertEqual(closed[0].eval(3).evaluate({0: Fraction(7)}), Fraction(-1))

    def test_unit_affine_constant_closed_form(self):
        block = Block([[Fraction(1)]], [QQX.scalar(Fraction(5))])

        closed = closure_periodic_rational([block])

        self.assertEqual(closed[0].eval(0), var(0, 1))
        self.assertEqual(closed[0].eval(3), var(0, 1) + Fraction(15))

    def test_later_block_can_sum_invariant_previous_block(self):
        x_block = Block([[Fraction(1)]], [QQX.zero()])
        y_block = Block([[Fraction(1)]], [var(0, 2)])

        closed = closure_periodic_rational([x_block, y_block])

        self.assertEqual(
            closed[1].eval(4).evaluate({0: Fraction(3), 1: Fraction(5)}),
            Fraction(17),
        )

    def test_closure_ocrs_uses_supported_closed_forms(self):
        block = Block([[Fraction(-1)]], [QQX.zero()])

        closed = closure_ocrs([block])

        self.assertEqual(closed[0].eval(2), var(0, 1))
        self.assertEqual(closed[0].eval(3), var(0, 1).scalar_mul(Fraction(-1)))

    def test_non_diagonal_block_is_explicitly_unsupported(self):
        block = Block(
            [[Fraction(1), Fraction(1)], [Fraction(0), Fraction(1)]],
            [QQX.zero(), QQX.zero()],
        )

        with self.assertRaises(NotImplementedError):
            closure_periodic_rational([block])

    def test_standard_basis_prsd_for_diagonal_matrix(self):
        decomposition = standard_basis_prsd(
            [[Fraction(1), Fraction(0)], [Fraction(0), Fraction(-1)]],
            2,
        )

        by_eigenvalue = {
            eigenvalue: (period, vectors)
            for period, eigenvalue, vectors in decomposition
        }
        self.assertEqual(
            by_eigenvalue[Fraction(1)], (1, [[Fraction(1), Fraction(0)]])
        )
        self.assertEqual(
            by_eigenvalue[Fraction(-1)], (2, [[Fraction(0), Fraction(1)]])
        )

    def test_standard_basis_prsd_rejects_non_diagonal_matrix(self):
        with self.assertRaises(NotImplementedError):
            standard_basis_prsd(
                [[Fraction(1), Fraction(1)], [Fraction(0), Fraction(1)]],
                2,
            )

    def test_standard_basis_prsd_for_signed_permutation(self):
        decomposition = standard_basis_prsd(
            [[Fraction(0), Fraction(1)], [Fraction(1), Fraction(0)]],
            2,
        )

        self.assertEqual(
            decomposition,
            [
                (
                    2,
                    Fraction(1),
                    [[Fraction(1), Fraction(0)], [Fraction(0), Fraction(1)]],
                )
            ],
        )

    def test_standard_basis_prsd_rejects_nonperiodic_monomial_matrix(self):
        with self.assertRaises(NotImplementedError):
            standard_basis_prsd(
                [[Fraction(0), Fraction(2)], [Fraction(1), Fraction(0)]],
                2,
            )

    # --- Non-diagonal (signed permutation) block tests ---

    def test_swap_matrix_homogeneous(self):
        """Swap matrix [[0,1],[1,0]] with zero additive terms."""
        block = Block(
            [[Fraction(0), Fraction(1)], [Fraction(1), Fraction(0)]],
            [QQX.zero(), QQX.zero()],
        )
        closed = closure_periodic_rational([block])
        vals = {0: Fraction(3), 1: Fraction(5)}
        # x_0(k) alternates: x_0(0)=3, x_0(1)=5, x_0(2)=3, ...
        self.assertEqual(closed[0].eval(0).evaluate(vals), Fraction(3))
        self.assertEqual(closed[0].eval(1).evaluate(vals), Fraction(5))
        self.assertEqual(closed[0].eval(2).evaluate(vals), Fraction(3))
        self.assertEqual(closed[0].eval(3).evaluate(vals), Fraction(5))
        # x_1(k) alternates: x_1(0)=5, x_1(1)=3, x_1(2)=5, ...
        self.assertEqual(closed[1].eval(0).evaluate(vals), Fraction(5))
        self.assertEqual(closed[1].eval(1).evaluate(vals), Fraction(3))
        self.assertEqual(closed[1].eval(2).evaluate(vals), Fraction(5))
        self.assertEqual(closed[1].eval(3).evaluate(vals), Fraction(3))

    def test_swap_matrix_with_additive_constants(self):
        """Swap matrix with constant additive terms."""
        block = Block(
            [[Fraction(0), Fraction(1)], [Fraction(1), Fraction(0)]],
            [QQX.scalar(Fraction(2)), QQX.scalar(Fraction(4))],
        )
        closed = closure_periodic_rational([block])
        vals = {0: Fraction(1), 1: Fraction(1)}
        # Manual trace:
        # x_0(0)=1, x_1(0)=1
        # x_0(1)=x_1(0)+2=3, x_1(1)=x_0(0)+4=5
        # x_0(2)=x_1(1)+2=7, x_1(2)=x_0(1)+4=7
        # x_0(3)=x_1(2)+2=9, x_1(3)=x_0(2)+4=11
        expected_x0 = [1, 3, 7, 9]
        expected_x1 = [1, 5, 7, 11]
        for k in range(4):
            self.assertEqual(
                closed[0].eval(k).evaluate(vals),
                Fraction(expected_x0[k]),
                f"x_0({k})",
            )
            self.assertEqual(
                closed[1].eval(k).evaluate(vals),
                Fraction(expected_x1[k]),
                f"x_1({k})",
            )

    def test_negative_swap_matrix(self):
        """Negative swap [[0,-1],[-1,0]]: x_0'=-x_1, x_1'=-x_0."""
        block = Block(
            [[Fraction(0), Fraction(-1)], [Fraction(-1), Fraction(0)]],
            [QQX.zero(), QQX.zero()],
        )
        closed = closure_periodic_rational([block])
        vals = {0: Fraction(6), 1: Fraction(2)}
        # Trace: x_0(0)=6, x_1(0)=2
        # x_0(1)=-x_1(0)=-2, x_1(1)=-x_0(0)=-6
        # x_0(2)=-x_1(1)=6, x_1(2)=-x_0(1)=2  (period 2 in values)
        expected_x0 = [6, -2, 6, -2]
        expected_x1 = [2, -6, 2, -6]
        for k in range(4):
            self.assertEqual(
                closed[0].eval(k).evaluate(vals),
                Fraction(expected_x0[k]),
                f"x_0({k})",
            )
            self.assertEqual(
                closed[1].eval(k).evaluate(vals),
                Fraction(expected_x1[k]),
                f"x_1({k})",
            )

    def test_swap_matrix_zero_additive(self):
        """Swap matrix with one zero and one nonzero additive term."""
        block = Block(
            [[Fraction(0), Fraction(1)], [Fraction(1), Fraction(0)]],
            [QQX.scalar(Fraction(10)), QQX.zero()],
        )
        closed = closure_periodic_rational([block])
        vals = {0: Fraction(0), 1: Fraction(0)}
        # Trace:
        # x_0(0)=0, x_1(0)=0
        # x_0(1)=x_1(0)+10=10, x_1(1)=x_0(0)+0=0
        # x_0(2)=x_1(1)+10=10, x_1(2)=x_0(1)+0=10
        # x_0(3)=x_1(2)+10=20, x_1(3)=x_0(2)+0=10
        expected_x0 = [0, 10, 10, 20]
        expected_x1 = [0, 0, 10, 10]
        for k in range(4):
            self.assertEqual(
                closed[0].eval(k).evaluate(vals),
                Fraction(expected_x0[k]),
                f"x_0({k})",
            )
            self.assertEqual(
                closed[1].eval(k).evaluate(vals),
                Fraction(expected_x1[k]),
                f"x_1({k})",
            )


if __name__ == "__main__":
    unittest.main()
