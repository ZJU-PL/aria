"""
Tests for the interval arithmetic module.
"""

import unittest
from fractions import Fraction
from aria.srk.interval import Interval


class TestInterval(unittest.TestCase):
    """Test interval operations."""

    def test_creation(self):
        """Test interval creation."""
        # Point interval
        i1 = Interval.const(Fraction(5, 2))
        self.assertEqual(i1.lower, Fraction(5, 2))
        self.assertEqual(i1.upper, Fraction(5, 2))

        # Bounded interval
        i2 = Interval.make_bounded(Fraction(1), Fraction(3))
        self.assertEqual(i2.lower, Fraction(1))
        self.assertEqual(i2.upper, Fraction(3))

        # Infinite bounds
        i3 = Interval.top()
        self.assertIsNone(i3.lower)
        self.assertIsNone(i3.upper)

    def test_bottom_interval(self):
        """Test empty interval detection."""
        bottom = Interval.bottom()
        self.assertTrue(bottom.is_bottom())

        # Invalid interval becomes bottom
        invalid = Interval(Fraction(3), Fraction(1))
        self.assertTrue(invalid.is_bottom())

    def test_top_interval(self):
        """Test universal interval detection."""
        top = Interval.top()
        self.assertTrue(top.is_top())

    def test_point_interval(self):
        """Test point interval detection."""
        point = Interval.const(Fraction(2))
        self.assertTrue(point.is_point())

        not_point = Interval(Fraction(1), Fraction(2))
        self.assertFalse(not_point.is_point())

    def test_contains(self):
        """Test interval membership."""
        i = Interval.make_bounded(Fraction(1), Fraction(3))

        self.assertTrue(i.contains(Fraction(2)))
        self.assertTrue(i.contains(Fraction(1)))
        self.assertTrue(i.contains(Fraction(3)))
        self.assertFalse(i.contains(Fraction(0)))
        self.assertFalse(i.contains(Fraction(4)))

    def test_addition(self):
        """Test interval addition."""
        i1 = Interval.make_bounded(Fraction(1), Fraction(2))
        i2 = Interval.make_bounded(Fraction(3), Fraction(4))

        result = i1 + i2
        expected = Interval.make_bounded(Fraction(4), Fraction(6))

        self.assertEqual(result.lower, expected.lower)
        self.assertEqual(result.upper, expected.upper)

    def test_negation(self):
        """Test interval negation."""
        i = Interval.make_bounded(Fraction(1), Fraction(3))
        neg_i = -i

        self.assertEqual(neg_i.lower, Fraction(-3))
        self.assertEqual(neg_i.upper, Fraction(-1))

    def test_multiplication(self):
        """Test interval multiplication."""
        i1 = Interval.make_bounded(Fraction(2), Fraction(3))
        i2 = Interval.make_bounded(Fraction(1), Fraction(2))

        result = i1 * i2
        # [2, 3] * [1, 2] = [2, 6]
        expected = Interval.make_bounded(Fraction(2), Fraction(6))

        self.assertEqual(result.lower, expected.lower)
        self.assertEqual(result.upper, expected.upper)

    def test_union(self):
        """Test interval union."""
        i1 = Interval.make_bounded(Fraction(1), Fraction(2))
        i2 = Interval.make_bounded(Fraction(3), Fraction(4))

        result = i1.union(i2)
        expected = Interval.make_bounded(Fraction(1), Fraction(4))

        self.assertEqual(result.lower, expected.lower)
        self.assertEqual(result.upper, expected.upper)

    def test_intersection(self):
        """Test interval intersection."""
        i1 = Interval.make_bounded(Fraction(1), Fraction(4))
        i2 = Interval.make_bounded(Fraction(2), Fraction(3))

        result = i1.intersection(i2)
        expected = Interval.make_bounded(Fraction(2), Fraction(3))

        self.assertEqual(result.lower, expected.lower)
        self.assertEqual(result.upper, expected.upper)

        # Test disjoint intervals
        i3 = Interval.make_bounded(Fraction(5), Fraction(6))
        result2 = i1.intersection(i3)
        self.assertTrue(result2.is_bottom())


if __name__ == '__main__':
    unittest.main()
