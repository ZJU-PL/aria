"""
Tests for the path expression module.
"""

import unittest
from aria.srk.pathexpr import (
    PathExprContext,
    mk_context,
    mk_table,
    mk_one,
    mk_zero,
    mk_edge,
    mk_mul,
    mk_add,
    mk_star,
    mk_omega,
    mk_segment,
    accept_epsilon,
    first,
    derivative,
    show,
    EdgeAlg,
    OneAlg,
    ZeroAlg,
)


class TestPathExpressions(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        self.context = mk_context()

    def test_basic_constructors(self):
        """Test basic path expression constructors."""
        # Test basic constructors
        one = mk_one(self.context)
        zero = mk_zero(self.context)
        edge = mk_edge(self.context, 0, 1)

        self.assertIsNotNone(one)
        self.assertIsNotNone(zero)
        self.assertIsNotNone(edge)

    def test_simplification_rules(self):
        """Test path expression simplification rules."""
        zero = mk_zero(self.context)
        one = mk_one(self.context)
        edge = mk_edge(self.context, 0, 1)

        # Test multiplication simplifications
        mul_zero_left = mk_mul(self.context, zero, edge)
        mul_zero_right = mk_mul(self.context, edge, zero)
        mul_one_left = mk_mul(self.context, one, edge)
        mul_one_right = mk_mul(self.context, edge, one)

        # Should simplify to zero
        self.assertEqual(mul_zero_left, zero)
        self.assertEqual(mul_zero_right, zero)

        # Should simplify to edge
        self.assertEqual(mul_one_left, edge)
        self.assertEqual(mul_one_right, edge)

        # Test addition simplifications
        add_zero_left = mk_add(self.context, zero, edge)
        add_zero_right = mk_add(self.context, edge, zero)

        # Should simplify to edge
        self.assertEqual(add_zero_left, edge)
        self.assertEqual(add_zero_right, edge)

        # Test star simplifications
        star_zero = mk_star(self.context, zero)
        star_one = mk_star(self.context, one)

        # Should simplify to one
        self.assertEqual(star_zero, one)
        self.assertEqual(star_one, one)

    def test_accept_epsilon(self):
        """Test epsilon acceptance."""
        zero = mk_zero(self.context)
        one = mk_one(self.context)
        edge = mk_edge(self.context, 0, 1)
        star_edge = mk_star(self.context, edge)

        self.assertFalse(accept_epsilon(zero))
        self.assertTrue(accept_epsilon(one))
        self.assertFalse(accept_epsilon(edge))
        self.assertTrue(accept_epsilon(star_edge))

    def test_first_function(self):
        """Test the first function."""
        edge = mk_edge(self.context, 0, 1)
        zero = mk_zero(self.context)
        one = mk_one(self.context)

        # Test edge first set
        first_edge = first(edge)
        self.assertEqual(first_edge, {(0, 1)})

        # Test zero and one first sets
        self.assertEqual(first(zero), set())
        self.assertEqual(first(one), set())

    def test_derivative(self):
        """Test path expression derivatives."""
        edge = mk_edge(self.context, 0, 1)
        one = mk_one(self.context)
        zero = mk_zero(self.context)

        # Derivative of edge with itself should be one
        deriv_same = derivative(self.context, (0, 1), edge)
        self.assertEqual(deriv_same, one)

        # Derivative of edge with different edge should be zero
        deriv_different = derivative(self.context, (0, 2), edge)
        self.assertEqual(deriv_different, zero)

    def test_show_function(self):
        """Test string representation."""
        edge = mk_edge(self.context, 0, 1)
        one = mk_one(self.context)

        edge_str = show(edge)
        one_str = show(one)

        self.assertIsInstance(edge_str, str)
        self.assertIsInstance(one_str, str)
        self.assertTrue(len(edge_str) > 0)
        self.assertTrue(len(one_str) > 0)

    def test_context_hashconsing(self):
        """Test that hash-consing works correctly."""
        # Create the same edge twice
        edge1 = mk_edge(self.context, 0, 1)
        edge2 = mk_edge(self.context, 0, 1)

        # Should be the same object due to hash-consing
        self.assertEqual(edge1, edge2)
        self.assertIs(edge1, edge2)

    def test_complex_expressions(self):
        """Test more complex path expressions."""
        edge1 = mk_edge(self.context, 0, 1)
        edge2 = mk_edge(self.context, 1, 2)

        # Create a complex expression: (0->1) + (1->2)*
        expr = mk_add(self.context, edge1, mk_star(self.context, edge2))

        self.assertIsNotNone(expr)

        # Test that it's not simplified away
        self.assertNotEqual(expr, mk_one(self.context))
        self.assertNotEqual(expr, mk_zero(self.context))


if __name__ == "__main__":
    unittest.main()
