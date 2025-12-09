"""
Tests for scalar arithmetic operations.

This module tests scalar operations that are fundamental to SRK's arithmetic.
"""

import unittest
from fractions import Fraction
from aria.srk.qQ import QQ


class TestQQ(unittest.TestCase):
    """Test rational number operations."""

    def test_creation(self):
        """Test creating rational numbers."""
        q1 = QQ.of_int(5)           # Integer
        q2 = QQ.of_int(3)/QQ.of_int(4)        # Fraction
        q3 = QQ.of_int(2)/QQ.of_int(3)  # From Fraction

        self.assertEqual(q1, QQ.of_int(5))
        self.assertEqual(q2, QQ.of_int(3)/QQ.of_int(4))
        self.assertEqual(q3, QQ.of_int(2)/QQ.of_int(3))

    def test_arithmetic(self):
        """Test basic arithmetic operations."""
        q1 = QQ.of_int(1)/QQ.of_int(2)
        q2 = QQ.of_int(1)/QQ.of_int(3)

        # Addition
        self.assertEqual(q1 + q2, QQ.of_int(5)/QQ.of_int(6))

        # Subtraction
        self.assertEqual(q1 - q2, QQ.of_int(1)/QQ.of_int(6))

        # Multiplication
        self.assertEqual(q1 * q2, QQ.of_int(1)/QQ.of_int(6))

        # Division
        self.assertEqual(q1 / q2, QQ.of_int(3)/QQ.of_int(2))

    def test_comparison(self):
        """Test comparison operations."""
        q1 = QQ.of_int(1)/QQ.of_int(2)
        q2 = QQ.of_int(1)/QQ.of_int(3)
        q3 = QQ.of_int(1)/QQ.of_int(2)

        self.assertTrue(q1 > q2)
        self.assertTrue(q2 < q1)
        self.assertTrue(q1 == q3)
        self.assertTrue(q1 != q2)

    def test_zero_one(self):
        """Test zero and one constants."""
        zero = QQ.zero()
        one = QQ.one()

        self.assertEqual(zero, QQ.of_int(0))
        self.assertEqual(one, QQ.of_int(1))


if __name__ == '__main__':
    unittest.main()
