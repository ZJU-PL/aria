"""
Targeted tests for solvable polynomial closed forms.
"""

import unittest
from fractions import Fraction

from aria.srk.polynomial import Polynomial as QQX
from aria.srk.solvablePolynomial import (
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


if __name__ == "__main__":
    unittest.main()
