"""
Loop analysis for graphs.

This module provides functionality for computing loop nesting forests in directed graphs,
which is useful for program analysis, compiler optimizations, and static analysis.

The implementation follows the algorithm described in:
- "On Loops, Dominators, and Dominance Frontiers" by G. Ramalingam
- "Efficient Chaotic Iteration Strategies with Widenings" by F. Bourdoncle
"""

from __future__ import annotations
from typing import Protocol, TypeVar, Generic, List, Set, Dict, Optional, Union, Callable, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import copy
from enum import Enum

# Type variables for generic graph interface
V = TypeVar('V')
G = TypeVar('G')


class GraphProtocol(Protocol[V]):
    """Protocol defining the interface for graphs used in loop analysis."""

    def iter_vertex(self, f: Callable[[V], None]) -> None:
        """Iterate over all vertices in the graph."""
        ...

    def iter_succ(self, f: Callable[[V], None], v: V) -> None:
        """Iterate over successors of vertex v."""
        ...


@dataclass(frozen=True)
class Loop:
    """Represents a loop in a directed graph.

    A loop consists of a header vertex and a body of vertices that form
    a strongly connected component.
    """
    header: V
    children: List[Union[V, 'Loop']]  # Forest of nested loops/vertices
    body: Set[V]  # Set of vertices in the loop body


def find_strongly_connected_components(graph: GraphProtocol[V]) -> List[Set[V]]:
    """Find strongly connected components using Kosaraju's algorithm.

    This implementation uses two DFS passes: first to get finishing times,
    then on the transpose graph to find SCCs.
    """
    # Get all vertices
    vertices = set()
    graph.iter_vertex(lambda v: vertices.add(v))

    if not vertices:
        return []

    # Build adjacency list and transpose
    adj_list = {}
    transpose_adj = {}

    def build_graphs(v):
        if v not in adj_list:
            adj_list[v] = []
            transpose_adj[v] = []
        successors = []
        graph.iter_succ(lambda s: successors.append(s), v)
        for s in successors:
            if s not in adj_list[v]:
                adj_list[v].append(s)

        # Build transpose
        for succ in adj_list[v]:
            if succ not in transpose_adj:
                transpose_adj[succ] = []
            if v not in transpose_adj[succ]:
                transpose_adj[succ].append(v)

    for v in vertices:
        build_graphs(v)

    # First DFS pass to get finishing times
    visited = set()
    finishing_order = []

    def dfs1(v):
        if v in visited:
            return
        visited.add(v)

        for succ in adj_list.get(v, []):
            dfs1(succ)

        finishing_order.append(v)

    for v in vertices:
        dfs1(v)

    # Second DFS pass on transpose graph
    visited = set()
    components = []

    def dfs2(v, component):
        if v in visited:
            return
        visited.add(v)
        component.add(v)

        for pred in transpose_adj.get(v, []):
            dfs2(pred, component)

    # Process vertices in reverse finishing order
    for v in reversed(finishing_order):
        if v not in visited:
            component = set()
            dfs2(v, component)
            if component:
                components.append(component)

    return components


def compute_loop_nesting_forest(graph: GraphProtocol[V]) -> List[Union[V, Loop[V]]]:
    """Compute the loop nesting forest for a directed graph.

    Returns a forest where each element is either a vertex or a loop,
    representing the hierarchical structure of loops in the graph.
    """
    # Get all vertices
    vertices = set()
    graph.iter_vertex(lambda v: vertices.add(v))

    # Find SCCs (simplified for now)
    sccs = find_strongly_connected_components(graph)

    # Process each SCC
    forest = []

    for scc in sccs:
        if len(scc) == 1:
            v = next(iter(scc))

            # Collect successors to check for self-loop
            successors = []
            graph.iter_succ(lambda s: successors.append(s), v)

            if v in successors:
                # Self-loop detected
                loop = Loop(
                    header=v,
                    children=[],
                    body=scc
                )
                forest.append(loop)
            else:
                forest.append(v)
        else:
            # Multi-vertex SCC - need to find header and nested structure
            # For simplicity, pick the first vertex as header
            header = next(iter(scc))
            body = scc.copy()

            # Find nested loops in the remaining vertices
            subgraph_vertices = body - {header}
            if subgraph_vertices:
                # Create a subgraph for nested analysis
                # This is a simplified approach - a full implementation would
                # create a proper subgraph and recurse
                nested_forest = list(subgraph_vertices)  # Treat as individual vertices for now
            else:
                nested_forest = []

            loop = Loop(
                header=header,
                children=nested_forest,
                body=body
            )
            forest.append(loop)

    return forest


def get_loop_header(loop: Loop[V]) -> V:
    """Get the header vertex of a loop."""
    return loop.header


def get_loop_body(loop: Loop[V]) -> Set[V]:
    """Get the set of vertices in a loop body."""
    return loop.body.copy()


def get_loop_children(loop: Loop[V]) -> List[Union[V, Loop[V]]]:
    """Get the children (nested loops/vertices) of a loop."""
    return loop.children.copy()


def find_all_loops(forest: List[Union[V, Loop[V]]]) -> List[Loop[V]]:
    """Extract all loops from a loop nesting forest."""
    loops = []

    def extract_loops(item: Union[V, Loop[V]]) -> None:
        if isinstance(item, Loop):
            loops.append(item)
            for child in item.children:
                extract_loops(child)

    for item in forest:
        extract_loops(item)

    return loops


def find_cutpoints(forest: List[Union[V, Loop[V]]]) -> Set[V]:
    """Find all loop header vertices (cutpoints) in the forest."""
    cutpoints = set()

    def find_headers(item: Union[V, Loop[V]]) -> None:
        if isinstance(item, Loop):
            cutpoints.add(item.header)
            for child in item.children:
                find_headers(child)

    for item in forest:
        find_headers(item)

    return cutpoints


def format_forest(forest: List[Union[V, Loop[V]]],
                  vertex_formatter: Callable[[V], str] = str) -> str:
    """Pretty-print a loop nesting forest."""
    lines = []

    def format_item(item: Union[V, Loop[V]], indent: int = 0) -> None:
        spaces = "  " * indent
        if isinstance(item, Loop):
            lines.append(f"{spaces}Loop(header={vertex_formatter(item.header)}):")
            lines.append(f"{spaces}  Body: {{{', '.join(vertex_formatter(v) for v in sorted(item.body, key=str))}}}")
            lines.append(f"{spaces}  Children:")
            for child in item.children:
                format_item(child, indent + 2)
        else:
            lines.append(f"{spaces}Vertex({vertex_formatter(item)})")

    for item in forest:
        format_item(item)

    return "\n".join(lines)


# Convenience functions for working with specific graph types
def analyze_graph_loops(graph: GraphProtocol[V]) -> Dict[str, Any]:
    """Analyze loops in a graph and return comprehensive results."""
    forest = compute_loop_nesting_forest(graph)
    loops = find_all_loops(forest)
    cutpoints = find_cutpoints(forest)

    return {
        'forest': forest,
        'loops': loops,
        'cutpoints': cutpoints,
        'num_loops': len(loops),
        'num_cutpoints': len(cutpoints)
    }
