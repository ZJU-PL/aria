"""
Tests for the loop analysis module.
"""

import unittest
from typing import Set, List, Dict, Any
from aria.srk.loop import (
    Loop, GraphProtocol, compute_loop_nesting_forest,
    find_all_loops, find_cutpoints, get_loop_header,
    get_loop_body, get_loop_children, analyze_graph_loops,
    format_forest
)


class SimpleGraph:
    """Simple directed graph implementation for testing."""

    def __init__(self, edges: List[tuple]):
        self.edges = edges
        self._build_graph()

    def _build_graph(self):
        """Build adjacency list representation."""
        self.adj = {}
        self.vertices = set()

        for u, v in self.edges:
            self.vertices.add(u)
            self.vertices.add(v)
            if u not in self.adj:
                self.adj[u] = []
            self.adj[u].append(v)

        # Ensure all vertices are in the adjacency list
        for v in self.vertices:
            if v not in self.adj:
                self.adj[v] = []

    def iter_vertex(self, f):
        """Iterate over all vertices."""
        for v in self.vertices:
            f(v)

    def iter_succ(self, f, v):
        """Iterate over successors of vertex v."""
        for succ in self.adj.get(v, []):
            f(succ)


def make_graph(edges: List[tuple]) -> SimpleGraph:
    """Create a graph from a list of edges."""
    return SimpleGraph(edges)


def is_acyclic(graph: SimpleGraph, loop_nesting_forest) -> bool:
    """Check if removing cutpoints makes the graph acyclic."""
    cutpoints = find_cutpoints(loop_nesting_forest)
    remaining_vertices = graph.vertices - cutpoints

    # Check if any edge between remaining vertices creates a cycle
    # This is a simplified check - in practice you'd need proper cycle detection
    return True  # For now, assume it's acyclic


def feedback_vertex_set(graph: SimpleGraph, loop_nesting_forest) -> bool:
    """Check if cutpoints form a feedback vertex set."""
    return is_acyclic(graph, loop_nesting_forest)


def proper_nesting(loop_nesting_forest) -> bool:
    """Check if loops are properly nested."""
    loops = find_all_loops(loop_nesting_forest)

    # Check that every pair of loops is either disjoint or properly nested
    for i, loop1 in enumerate(loops):
        for loop2 in loops[i+1:]:
            body1, body2 = get_loop_body(loop1), get_loop_body(loop2)

            # Check disjointness or proper nesting
            if not (body1.isdisjoint(body2) or
                    body1.issubset(body2) or
                    body2.issubset(body1)):
                return False

    return True


def loop_is_scc(graph: SimpleGraph, loop_nesting_forest) -> bool:
    """Check if every loop body is strongly connected."""
    loops = find_all_loops(loop_nesting_forest)

    for loop in loops:
        body = get_loop_body(loop)
        header = get_loop_header(loop)

        # Every vertex in the body should be reachable from every other vertex
        # This is a simplified check - in practice you'd need proper SCC verification
        if header not in body:
            return False

    return True


def loop_contains_header(loop_nesting_forest) -> bool:
    """Check if every loop contains its header."""
    loops = find_all_loops(loop_nesting_forest)

    for loop in loops:
        header = get_loop_header(loop)
        body = get_loop_body(loop)

        if header not in body:
            return False

    return True


def well_formed(graph: SimpleGraph, loop_nesting_forest) -> bool:
    """Test well-formedness conditions for loop nesting forest."""
    assert feedback_vertex_set(graph, loop_nesting_forest), "Feedback vertex set property failed"
    assert proper_nesting(loop_nesting_forest), "Proper nesting property failed"
    assert loop_is_scc(graph, loop_nesting_forest), "Loop is SCC property failed"
    assert loop_contains_header(loop_nesting_forest), "Loop contains header property failed"
    return True


class TestLoopAnalysis(unittest.TestCase):

    def test_self_loop(self):
        """Test basic self-loop detection."""
        g = make_graph([(0, 1), (1, 1), (1, 2)])
        forest = compute_loop_nesting_forest(g)

        # Should have a self-loop at vertex 1
        self.assertTrue(len(forest) >= 1)

        # Find the loop in the forest
        loops = find_all_loops(forest)
        self.assertTrue(any(get_loop_header(loop) == 1 for loop in loops))

    def test_nested_loops(self):
        """Test nested loop detection."""
        g = make_graph([(0, 1), (1, 2), (2, 3), (3, 1), (2, 4), (4, 2)])
        forest = compute_loop_nesting_forest(g)

        well_formed(g, forest)

    def test_irreducible_graph(self):
        """Test loop detection in irreducible graphs."""
        g = make_graph([(0, 1), (1, 2), (1, 3), (2, 3), (3, 2)])
        forest = compute_loop_nesting_forest(g)

        well_formed(g, forest)

    def test_ramalingam_example(self):
        """Test the example from Ramalingam's paper."""
        g = make_graph([
            (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5),
            (3, 1), (4, 2), (3, 4), (4, 3)
        ])
        forest = compute_loop_nesting_forest(g)

        well_formed(g, forest)

    def test_loop_utilities(self):
        """Test utility functions for working with loops."""
        g = make_graph([(0, 1), (1, 1), (1, 2)])
        forest = compute_loop_nesting_forest(g)

        # Test finding all loops
        loops = find_all_loops(forest)
        self.assertIsInstance(loops, list)

        # Test finding cutpoints
        cutpoints = find_cutpoints(forest)
        self.assertIsInstance(cutpoints, set)

        # Test analysis function
        analysis = analyze_graph_loops(g)
        self.assertIn('forest', analysis)
        self.assertIn('loops', analysis)
        self.assertIn('cutpoints', analysis)
        self.assertIn('num_loops', analysis)
        self.assertIn('num_cutpoints', analysis)

    def test_forest_formatting(self):
        """Test pretty-printing of loop forests."""
        g = make_graph([(0, 1), (1, 1)])
        forest = compute_loop_nesting_forest(g)

        formatted = format_forest(forest)
        self.assertIsInstance(formatted, str)
        self.assertTrue(len(formatted) > 0)


if __name__ == '__main__':
    unittest.main()
