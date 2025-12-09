"""
Tests for the coordinate system module.
"""

import unittest
from aria.srk.coordinateSystem import (
    CoordinateSystem, CSTerm, CSTermType, TermType, mk_empty, get_context
)
from aria.srk.syntax import Context
from aria.srk.linear import QQVector
from fractions import Fraction


class TestCoordinateSystem(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        self.context = Context()
        self.cs = mk_empty(self.context)

    def test_empty_coordinate_system(self):
        """Test creating an empty coordinate system."""
        self.assertEqual(self.cs.dim, 0)
        self.assertEqual(self.cs.int_dim, 0)
        self.assertEqual(self.cs.real_dim, 0)

    def test_coordinate_system_copy(self):
        """Test copying a coordinate system."""
        cs_copy = self.cs.copy()
        self.assertEqual(cs_copy.dim, 0)
        self.assertEqual(cs_copy.context, self.cs.context)

    def test_cs_term_creation(self):
        """Test creating coordinate system terms."""
        # Create some vectors for testing
        vec1 = QQVector().add_term(Fraction(1), 0)
        vec2 = QQVector().add_term(Fraction(2), 1)

        # Test basic term creation
        mul_term = CSTerm.mul(vec1, vec2)
        self.assertEqual(mul_term.term_type, CSTermType.MUL)
        self.assertEqual(len(mul_term.vectors), 2)

        inv_term = CSTerm.inv(vec1)
        self.assertEqual(inv_term.term_type, CSTermType.INV)
        self.assertEqual(len(inv_term.vectors), 1)

        one_term = CSTerm.one()
        self.assertEqual(one_term.term_type, CSTermType.ONE)
        self.assertEqual(len(one_term.vectors), 0)

        zero_term = CSTerm.zero()
        self.assertEqual(zero_term.term_type, CSTermType.ZERO)
        self.assertEqual(len(zero_term.vectors), 0)

    def test_coordinate_admission(self):
        """Test admitting coordinate system terms."""
        # First admit some basic terms to create coordinates 0 and 1
        one_term = CSTerm.one()
        coord_id_0 = self.cs.admit_cs_term(one_term)

        vec1 = QQVector().add_term(Fraction(1), coord_id_0)
        vec2 = QQVector().add_term(Fraction(2), coord_id_0)

        # Admit a multiplication term
        mul_term = CSTerm.mul(vec1, vec2)
        coord_id = self.cs.admit_cs_term(mul_term)

        self.assertEqual(coord_id, 1)  # Should be the second coordinate
        self.assertEqual(self.cs.dim, 2)

        # Try to admit the same term again (should return same ID)
        coord_id2 = self.cs.admit_cs_term(mul_term)
        self.assertEqual(coord_id, coord_id2)

    def test_coordinate_destruction(self):
        """Test getting coordinate system terms back."""
        # First admit some basic terms to create coordinates 0 and 1
        one_term = CSTerm.one()
        coord_id_0 = self.cs.admit_cs_term(one_term)

        vec1 = QQVector().add_term(Fraction(1), coord_id_0)
        vec2 = QQVector().add_term(Fraction(2), coord_id_0)

        mul_term = CSTerm.mul(vec1, vec2)
        coord_id = self.cs.admit_cs_term(mul_term)

        # Get the term back
        retrieved_term = self.cs.destruct_coordinate(coord_id)
        self.assertEqual(retrieved_term, mul_term)

    def test_coordinate_types(self):
        """Test coordinate type analysis."""
        # First admit some basic terms to create coordinates 0 and 1
        one_term = CSTerm.one()
        coord_id_0 = self.cs.admit_cs_term(one_term)

        vec1 = QQVector().add_term(Fraction(1), coord_id_0)
        vec2 = QQVector().add_term(Fraction(2), coord_id_0)

        # Admit a multiplication term
        mul_term = CSTerm.mul(vec1, vec2)
        coord_id = self.cs.admit_cs_term(mul_term)

        # Check type (should be real since we don't have integer context)
        typ = self.cs.type_of_id(coord_id)
        self.assertEqual(typ, TermType.TY_REAL)

    def test_subcoordinates(self):
        """Test finding subcoordinates."""
        # Create a simple coordinate system with a multiplication
        # First admit some basic terms
        one_term = CSTerm.one()
        coord_id_0 = self.cs.admit_cs_term(one_term)

        vec1 = QQVector().add_term(Fraction(1), coord_id_0)
        vec2 = QQVector().add_term(Fraction(2), coord_id_0)

        mul_term = CSTerm.mul(vec1, vec2)
        coord_id = self.cs.admit_cs_term(mul_term)

        # Get subcoordinates (should include coord_id_0)
        subcoords = self.cs.subcoordinates(coord_id)
        self.assertIn(coord_id_0, subcoords)

        # Get direct subcoordinates (should also include coord_id_0)
        direct_subcoords = self.cs.direct_subcoordinates(coord_id)
        self.assertIn(coord_id_0, direct_subcoords)

    def test_term_conversion(self):
        """Test converting coordinates to terms."""
        # First admit a basic term
        one_term = CSTerm.one()
        coord_id_0 = self.cs.admit_cs_term(one_term)

        # Convert back to term (should be a constant)
        term = self.cs.term_of_coordinate(coord_id_0)
        self.assertIsNotNone(term)


if __name__ == '__main__':
    unittest.main()
