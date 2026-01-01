"""
Tests for ring theory operations.

This module tests ring operations and algebraic structures used in SRK.
"""

import unittest
from aria.srk.ring import (
    IntegerRing,
    RationalRing,
    RingMap,
    RingVector,
    RingMatrix,
    is_ring,
    is_commutative_semigroup,
    is_associative_semigroup,
)


class TestIntegerRing(unittest.TestCase):
    """Test integer ring operations."""

    def test_creation(self):
        """Test creating the integer ring."""
        ring = IntegerRing()
        self.assertIsNotNone(ring)

    def test_ring_operations(self):
        """Test basic ring operations."""
        ring = IntegerRing()

        # Test addition
        self.assertEqual(ring.add(2, 3), 5)
        self.assertEqual(ring.add(-1, 1), 0)

        # Test multiplication
        self.assertEqual(ring.mul(2, 3), 6)
        self.assertEqual(ring.mul(-2, 3), -6)

        # Test zero and one
        self.assertEqual(ring.zero, 0)
        self.assertEqual(ring.one, 1)


class TestRationalRing(unittest.TestCase):
    """Test rational ring operations."""

    def test_creation(self):
        """Test creating the rational ring."""
        ring = RationalRing()
        self.assertIsNotNone(ring)

    def test_ring_operations(self):
        """Test basic ring operations."""
        ring = RationalRing()

        # Test addition
        self.assertEqual(ring.add(1, 2), 3)  # 1/1 + 2/1 = 3/1

        # Test multiplication
        self.assertEqual(ring.mul(2, 3), 6)  # 2/1 * 3/1 = 6/1


class TestRingMap(unittest.TestCase):
    """Test ring homomorphism functionality."""

    def test_identity_map(self):
        """Test identity ring map."""
        ring = IntegerRing()
        identity = RingMap.identity(ring)

        self.assertEqual(identity.map(5), 5)
        self.assertEqual(identity.map(-3), -3)


class TestRingVector(unittest.TestCase):
    """Test ring vector operations."""

    def test_creation(self):
        """Test creating ring vectors."""
        ring = IntegerRing()
        vec = RingVector(ring, [1, 2, 3])

        self.assertEqual(vec.ring, ring)
        self.assertEqual(vec.elements, [1, 2, 3])


class TestRingMatrix(unittest.TestCase):
    """Test ring matrix operations."""

    def test_creation(self):
        """Test creating ring matrices."""
        ring = IntegerRing()
        matrix = RingMatrix(ring, [[1, 2], [3, 4]])

        self.assertEqual(matrix.ring, ring)
        self.assertEqual(matrix.elements, [[1, 2], [3, 4]])


class TestRingProperties(unittest.TestCase):
    """Test ring property checking."""

    def test_is_ring(self):
        """Test ring property checking."""
        int_ring = IntegerRing()
        rat_ring = RationalRing()

        self.assertTrue(is_ring(int_ring))
        self.assertTrue(is_ring(rat_ring))

    def test_semigroup_properties(self):
        """Test semigroup property checking."""
        int_ring = IntegerRing()

        # Addition should be a commutative semigroup
        self.assertTrue(is_commutative_semigroup(int_ring, int_ring.add))
        self.assertTrue(is_associative_semigroup(int_ring, int_ring.add))

        # Multiplication should be a semigroup but not necessarily commutative
        self.assertTrue(is_associative_semigroup(int_ring, int_ring.mul))


if __name__ == "__main__":
    unittest.main()
