"""
Tests for the algebraic structures module.
"""

import unittest
from aria.srk.algebra import (
    Semigroup,
    Ring,
    Semilattice,
    Lattice,
    IntegerRing,
    RationalRing,
    is_commutative_semigroup,
    is_associative_semigroup,
    is_ring,
)


class TestIntegerRing(unittest.TestCase):
    """Test IntegerRing implementation."""

    def test_ring_properties(self):
        """Test basic ring properties."""
        # Test additive identity
        self.assertEqual(IntegerRing.add(IntegerRing.zero, 5), 5)
        self.assertEqual(IntegerRing.add(5, IntegerRing.zero), 5)

        # Test multiplicative identity
        self.assertEqual(IntegerRing.mul(IntegerRing.one, 5), 5)
        self.assertEqual(IntegerRing.mul(5, IntegerRing.one), 5)

        # Test distributivity
        a, b, c = 2, 3, 4
        self.assertEqual(
            IntegerRing.mul(a, IntegerRing.add(b, c)),
            IntegerRing.add(IntegerRing.mul(a, b), IntegerRing.mul(a, c)),
        )

    def test_operations(self):
        """Test basic arithmetic operations."""
        self.assertEqual(IntegerRing.add(3, 4), 7)
        self.assertEqual(IntegerRing.mul(3, 4), 12)
        self.assertEqual(IntegerRing.negate(5), -5)
        self.assertTrue(IntegerRing.equal(5, 5))
        self.assertFalse(IntegerRing.equal(5, 6))


class TestRationalRing(unittest.TestCase):
    """Test RationalRing implementation."""

    def test_ring_properties(self):
        """Test basic ring properties with floating point tolerance."""
        # Test additive identity
        self.assertAlmostEqual(RationalRing.add(RationalRing.zero, 3.5), 3.5)
        self.assertAlmostEqual(RationalRing.add(3.5, RationalRing.zero), 3.5)

        # Test multiplicative identity
        self.assertAlmostEqual(RationalRing.mul(RationalRing.one, 3.5), 3.5)
        self.assertAlmostEqual(RationalRing.mul(3.5, RationalRing.one), 3.5)

    def test_operations(self):
        """Test basic arithmetic operations."""
        self.assertAlmostEqual(RationalRing.add(2.5, 3.7), 6.2, places=10)
        self.assertAlmostEqual(RationalRing.mul(2.5, 3.0), 7.5, places=10)
        self.assertAlmostEqual(RationalRing.negate(4.5), -4.5, places=10)
        self.assertTrue(RationalRing.equal(3.5, 3.5))
        self.assertFalse(RationalRing.equal(3.5, 3.6))


class TestAlgebraicUtilities(unittest.TestCase):
    """Test utility functions for algebraic structures."""

    def test_is_commutative_semigroup(self):
        """Test commutativity checking."""
        # Test with IntegerRing multiplication
        self.assertTrue(is_commutative_semigroup(IntegerRing, IntegerRing.mul))

        # Test with addition (should also be commutative)
        self.assertTrue(is_commutative_semigroup(IntegerRing, IntegerRing.add))

    def test_is_associative_semigroup(self):
        """Test associativity checking."""
        # Test with IntegerRing multiplication
        self.assertTrue(is_associative_semigroup(IntegerRing, IntegerRing.mul))

        # Test with addition (should also be associative)
        self.assertTrue(is_associative_semigroup(IntegerRing, IntegerRing.add))

    def test_is_ring(self):
        """Test ring property checking."""
        self.assertTrue(is_ring(IntegerRing, IntegerRing.zero, IntegerRing.one))
        self.assertTrue(is_ring(RationalRing, RationalRing.zero, RationalRing.one))

    def test_invalid_cases(self):
        """Test that invalid cases return False."""

        # Test with a class that doesn't have mul method
        class BadRing:
            pass

        self.assertFalse(is_commutative_semigroup(BadRing, None))

        # Test with function that raises exception
        def bad_mul(a, b):
            raise RuntimeError("Bad multiplication")

        self.assertFalse(is_commutative_semigroup(IntegerRing, bad_mul))


class TestProtocols(unittest.TestCase):
    """Test protocol definitions (basic smoke tests)."""

    def test_semigroup_protocol(self):
        """Test that Semigroup protocol is properly defined."""

        # This is mainly a smoke test since protocols are structural
        class TestSemigroup:
            def mul(self, other):
                return other

        # Should not raise TypeError when used as Semigroup
        sg = TestSemigroup()
        self.assertEqual(sg.mul(5), 5)

    def test_ring_protocol(self):
        """Test that Ring protocol is properly defined."""

        class TestRing:
            def equal(self, other):
                return self.value == other

            def add(self, other):
                return self.value + other

            def negate(self):
                return -self.value

            def zero(self):
                return 0

            def mul(self, other):
                return self.value * other

            def one(self):
                return 1

        ring = TestRing()
        ring.value = 3
        self.assertEqual(ring.add(2), 5)
        self.assertEqual(ring.mul(2), 6)

    def test_semilattice_protocol(self):
        """Test that Semilattice protocol is properly defined."""

        class TestSemilattice:
            def join(self, other):
                return max(self.value, other)

            def equal(self, other):
                return self.value == other

        sl = TestSemilattice()
        sl.value = 5
        self.assertEqual(sl.join(3), 5)
        self.assertEqual(sl.join(7), 7)

    def test_lattice_protocol(self):
        """Test that Lattice protocol is properly defined."""

        class TestLattice:
            def join(self, other):
                return max(self.value, other)

            def meet(self, other):
                return min(self.value, other)

            def equal(self, other):
                return self.value == other

        lat = TestLattice()
        lat.value = 5
        self.assertEqual(lat.join(3), 5)
        self.assertEqual(lat.meet(3), 3)


if __name__ == "__main__":
    unittest.main()
