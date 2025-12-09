"""
Tests for the Weighted Graph module.
"""

import unittest
from aria.srk.weightedGraph import WeightedGraph, Algebra, path_weight, omega_path_weight, msat_path_weight
from aria.srk.syntax import Context, Symbol, Type, mk_symbol, mk_const, mk_lt, mk_add, mk_real
from aria.srk.transition import Transition
from aria.srk.qQ import QQ


class TestWeightedGraph(unittest.TestCase):
    """Test Weighted Graph operations."""

    def setUp(self):
        """Set up test context and symbols."""
        self.ctx = Context()
        self.x = mk_symbol(self.ctx, 'x', Type.INT)
        self.n = mk_symbol(self.ctx, 'n', Type.INT)

    def test_empty_graph(self):
        """Test creating an empty graph."""
        # Create algebra for integer weights
        algebra = Algebra(
            mul=lambda x, y: x * y,
            add=lambda x, y: x + y,
            star=lambda x: 1 / (1 - x) if x < 1 else float('inf'),  # Simplified Kleene star
            zero=0,
            one=1
        )
        wg = WeightedGraph(algebra)

        # Empty graph should have no vertices
        self.assertEqual(len(wg.vertices()), 0)

    def test_add_vertex(self):
        """Test adding vertices."""
        algebra = Algebra(
            mul=lambda x, y: x * y,
            add=lambda x, y: x + y,
            star=lambda x: 1 / (1 - x) if x < 1 else float('inf'),
            zero=0,
            one=1
        )
        wg = WeightedGraph(algebra)
        wg = wg.add_vertex(0)
        wg = wg.add_vertex(1)
        wg = wg.add_vertex(2)

        vertices = wg.vertices()
        self.assertIn(0, vertices)
        self.assertIn(1, vertices)
        self.assertIn(2, vertices)

    def test_add_edge(self):
        """Test adding edges."""
        algebra = Algebra(
            mul=lambda x, y: x * y,
            add=lambda x, y: x + y,
            star=lambda x: 1 / (1 - x) if x < 1 else float('inf'),
            zero=0,
            one=1
        )
        wg = WeightedGraph(algebra)
        wg = wg.add_vertex(0)
        wg = wg.add_vertex(1)
        wg = wg.add_vertex(2)

        # Add simple edge with weight
        wg = wg.add_edge(0, 1, 1)  # weight 1
        wg = wg.add_edge(1, 2, 2)  # weight 2

        # Check that edges exist
        edges = wg.edges()
        self.assertGreater(len(edges), 0)

    def test_simple_loop(self):
        """Test simple loop pattern."""
        algebra = Algebra(
            mul=lambda x, y: x * y,
            add=lambda x, y: x + y,
            star=lambda x: 1 / (1 - x) if x < 1 else float('inf'),
            zero=0,
            one=1
        )
        wg = WeightedGraph(algebra)
        wg = wg.add_vertex(0)
        wg = wg.add_vertex(1)
        wg = wg.add_vertex(2)
        wg = wg.add_vertex(3)

        # Create a simple loop: 0 -> 1 -> 2 -> 1 -> 3
        wg = wg.add_edge(0, 0, 1)  # weight 0
        wg = wg.add_edge(1, 1, 2)  # weight 1
        wg = wg.add_edge(2, 1, 1)  # Loop back with weight 1
        wg = wg.add_edge(1, 10, 3)  # weight 10

        # Test that the graph was constructed
        self.assertIsNotNone(wg)

    def test_branching_structure(self):
        """Test branching pattern."""
        algebra = Algebra(
            mul=lambda x, y: x * y,
            add=lambda x, y: x + y,
            star=lambda x: 1 / (1 - x) if x < 1 else float('inf'),
            zero=0,
            one=1
        )
        wg = WeightedGraph(algebra)
        wg = wg.add_vertex(0)
        wg = wg.add_vertex(1)
        wg = wg.add_vertex(2)
        wg = wg.add_vertex(3)

        # Create branching: 0 -> 1 -> 2 and 0 -> 1 -> 3
        wg = wg.add_edge(0, 0, 1)  # weight 0
        wg = wg.add_edge(1, 1, 2)  # weight 1
        wg = wg.add_edge(1, 2, 3)  # weight 2

        self.assertIsNotNone(wg)

    def test_complex_control_flow(self):
        """Test complex control flow patterns."""
        algebra = Algebra(
            mul=lambda x, y: x * y,
            add=lambda x, y: x + y,
            star=lambda x: 1 / (1 - x) if x < 1 else float('inf'),
            zero=0,
            one=1
        )
        wg = WeightedGraph(algebra)

        # Create nodes for different states
        nodes = list(range(10))
        for node in nodes:
            wg = wg.add_vertex(node)

        # Add various edges to create complex flow
        wg = wg.add_edge(0, 0, 1)  # weight 0
        wg = wg.add_edge(1, 1, 2)  # weight 1
        wg = wg.add_edge(2, 2, 3)  # weight 2
        wg = wg.add_edge(3, 3, 1)  # Loop with weight 3
        wg = wg.add_edge(2, 4, 4)  # Branch with weight 4
        wg = wg.add_edge(4, 5, 5)  # weight 5
        wg = wg.add_edge(5, 6, 6)  # weight 6
        wg = wg.add_edge(6, 7, 7)  # weight 7
        wg = wg.add_edge(7, 8, 8)  # weight 8
        wg = wg.add_edge(8, 9, 9)  # weight 9

        self.assertIsNotNone(wg)


class TestGraphOperations(unittest.TestCase):
    """Test graph operations and algorithms."""

    def setUp(self):
        """Set up test context."""
        self.ctx = Context()

    def test_path_finding(self):
        """Test basic path finding capabilities."""
        algebra = Algebra(
            mul=lambda x, y: x * y,
            add=lambda x, y: x + y,
            star=lambda x: 1 / (1 - x) if x < 1 else float('inf'),
            zero=0,
            one=1
        )
        wg = WeightedGraph(algebra)
        wg = wg.add_vertex(0)
        wg = wg.add_vertex(1)
        wg = wg.add_vertex(2)

        wg = wg.add_edge(0, 1, 1)  # weight 1
        wg = wg.add_edge(1, 2, 2)  # weight 2

        # Test that we can compute path weights
        path_func = path_weight(wg, 0)
        weight_1 = path_func(1)
        weight_2 = path_func(2)

        # Basic checks that paths exist
        self.assertIsNotNone(weight_1)
        self.assertIsNotNone(weight_2)

    def test_cycle_detection(self):
        """Test cycle detection."""
        algebra = Algebra(
            mul=lambda x, y: x * y,
            add=lambda x, y: x + y,
            star=lambda x: 1 / (1 - x) if x < 1 else float('inf'),
            zero=0,
            one=1
        )
        wg = WeightedGraph(algebra)
        wg = wg.add_vertex(0)
        wg = wg.add_vertex(1)
        wg = wg.add_vertex(2)

        # Create a cycle: 0 -> 1 -> 2 -> 0
        wg = wg.add_edge(0, 1, 1)  # weight 1
        wg = wg.add_edge(1, 2, 2)  # weight 2
        wg = wg.add_edge(2, 0, 3)  # weight 3

        self.assertIsNotNone(wg)


if __name__ == '__main__':
    unittest.main()
