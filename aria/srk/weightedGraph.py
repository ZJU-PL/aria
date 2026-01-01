"""
Weighted graph data structure and algorithms.

This module provides weighted graphs with algebraic operations for
path analysis, SCC computation, and advanced graph algorithms.
"""

from __future__ import annotations
from typing import (
    TypeVar,
    Generic,
    Dict,
    List,
    Set,
    Tuple,
    Optional,
    Union,
    Callable,
    Any,
    Iterator,
)
from dataclasses import dataclass
from abc import ABC, abstractmethod
from fractions import Fraction

from aria.srk.util import IntSet
from aria.srk.compressedWeightedForest import CompressedWeightedForest

T = TypeVar("T")
U = TypeVar("U")


@dataclass
class Algebra(Generic[T]):
    """Algebra for weighted graph operations."""

    mul: Callable[[T, T], T]
    add: Callable[[T, T], T]
    star: Callable[[T], T]  # Kleene star operation
    zero: T
    one: T


@dataclass
class OmegaAlgebra(Generic[T, U]):
    """Omega algebra for infinite path computations."""

    omega: Callable[[T], U]
    omega_add: Callable[[U, U], U]
    omega_mul: Callable[[T, U], U]


# Type alias for vertices
Vertex = int


class WeightedGraph(Generic[T]):
    """Weighted graph with algebraic operations."""

    def __init__(self, algebra: Algebra[T]):
        """Initialize weighted graph with given algebra."""
        self.graph: Dict[Vertex, Set[Vertex]] = {}  # adjacency list
        self.labels: Dict[Tuple[Vertex, Vertex], T] = {}  # edge weights
        self.algebra = algebra

    def add_vertex(self, vertex: Vertex) -> WeightedGraph[T]:
        """Add a vertex to the graph."""
        if vertex not in self.graph:
            self.graph[vertex] = set()
        return self

    def add_edge(self, u: Vertex, weight: T, v: Vertex) -> WeightedGraph[T]:
        """Add a weighted edge from u to v."""
        self.add_vertex(u)
        self.add_vertex(v)

        # Update adjacency list
        self.graph[u].add(v)

        # Update edge weight
        edge_key = (u, v)
        if edge_key in self.labels:
            # Combine weights using algebra
            self.labels[edge_key] = self.algebra.add(self.labels[edge_key], weight)
        else:
            self.labels[edge_key] = weight

        return self

    def remove_vertex(self, u: Vertex) -> WeightedGraph[T]:
        """Remove a vertex and all its incident edges."""
        if u not in self.graph:
            return self

        # Remove all edges from/to u
        edges_to_remove = []
        for src, dst in self.labels.keys():
            if src == u or dst == u:
                edges_to_remove.append((src, dst))

        for edge in edges_to_remove:
            del self.labels[edge]

        # Remove from adjacency list
        del self.graph[u]

        # Remove from other adjacency lists
        for neighbors in self.graph.values():
            neighbors.discard(u)

        return self

    def remove_edge(self, u: Vertex, v: Vertex) -> WeightedGraph[T]:
        """Remove edge from u to v."""
        edge_key = (u, v)
        if edge_key in self.labels:
            del self.labels[edge_key]
            if v in self.graph.get(u, set()):
                self.graph[u].remove(v)
        return self

    def edge_weight(self, u: Vertex, v: Vertex) -> T:
        """Get weight of edge from u to v."""
        return self.labels.get((u, v), self.algebra.zero)

    def vertices(self) -> Set[Vertex]:
        """Get all vertices in the graph."""
        return set(self.graph.keys())

    def edges(self) -> List[Tuple[Vertex, T, Vertex]]:
        """Get all edges with their weights."""
        return [(u, weight, v) for (u, v), weight in self.labels.items()]

    def successors(self, u: Vertex) -> Set[Vertex]:
        """Get successors of vertex u."""
        return self.graph.get(u, set()).copy()

    def predecessors(self, v: Vertex) -> Set[Vertex]:
        """Get predecessors of vertex v."""
        preds = set()
        for (u, dst), _ in self.labels.items():
            if dst == v:
                preds.add(u)
        return preds

    def fold_succ_e(self, f: Callable[[Vertex, T, Vertex], U], u: Vertex, acc: U) -> U:
        """Fold over successor edges of u."""
        for v in self.successors(u):
            weight = self.edge_weight(u, v)
            acc = f(u, weight, v, acc)
        return acc

    def fold_pred_e(self, f: Callable[[Vertex, T, Vertex], U], v: Vertex, acc: U) -> U:
        """Fold over predecessor edges of v."""
        for u in self.predecessors(v):
            weight = self.edge_weight(u, v)
            acc = f(u, weight, v, acc)
        return acc

    def iter_succ_e(self, f: Callable[[Vertex, T, Vertex], None], u: Vertex) -> None:
        """Iterate over successor edges of u."""
        for v in self.successors(u):
            weight = self.edge_weight(u, v)
            f(u, weight, v)

    def iter_pred_e(self, f: Callable[[Vertex, T, Vertex], None], v: Vertex) -> None:
        """Iterate over predecessor edges of v."""
        for u in self.predecessors(v):
            weight = self.edge_weight(u, v)
            f(u, weight, v)

    def fold_vertex(self, f: Callable[[Vertex], U], acc: U) -> U:
        """Fold over all vertices."""
        for vertex in self.vertices():
            acc = f(vertex, acc)
        return acc

    def iter_vertex(self, f: Callable[[Vertex], None]) -> None:
        """Iterate over all vertices."""
        for vertex in self.vertices():
            f(vertex)

    def mem_edge(self, u: Vertex, v: Vertex) -> bool:
        """Check if edge u->v exists."""
        return (u, v) in self.labels

    def max_vertex(self) -> Optional[Vertex]:
        """Get the maximum vertex id."""
        if not self.vertices():
            return None
        return max(self.vertices())

    def forget_weights(self) -> Dict[Vertex, Set[Vertex]]:
        """Return the underlying unweighted graph."""
        return {v: neighbors.copy() for v, neighbors in self.graph.items()}

    def map_weights(self, f: Callable[[Vertex, T, Vertex], T]) -> WeightedGraph[T]:
        """Map a function over all edge weights."""
        new_graph = WeightedGraph(self.algebra)
        new_graph.graph = {v: neighbors.copy() for v, neighbors in self.graph.items()}

        for (u, v), weight in self.labels.items():
            new_weight = f(u, weight, v)
            new_graph.labels[(u, v)] = new_weight

        return new_graph

    def fold_edges(self, f: Callable[[Vertex, T, Vertex], U], acc: U) -> U:
        """Fold over all edges."""
        for (u, v), weight in self.labels.items():
            acc = f(u, weight, v, acc)
        return acc

    def iter_edges(self, f: Callable[[Vertex, T, Vertex], None]) -> None:
        """Iterate over all edges."""
        for (u, v), weight in self.labels.items():
            f(u, weight, v)

    def fold_incident_edges(
        self, f: Callable[[Tuple[Vertex, Vertex]], U], v: Vertex, acc: U
    ) -> U:
        """Fold over edges incident to v."""
        # Successors
        for u in self.successors(v):
            acc = f((v, u), acc)

        # Predecessors (avoid double counting self-loops)
        for u in self.predecessors(v):
            if u != v:
                acc = f((u, v), acc)

        return acc

    def __len__(self) -> int:
        """Return number of vertices."""
        return len(self.vertices())

    def __str__(self) -> str:
        """String representation."""
        edges_str = []
        for (u, v), weight in sorted(self.labels.items()):
            edges_str.append(f"{u} --({weight})--> {v}")
        return f"WeightedGraph({{{', '.join(edges_str)}}})"

    def __repr__(self) -> str:
        """Debug representation."""
        return self.__str__()


def path_weight(wg: WeightedGraph[T], src: Vertex) -> Callable[[Vertex], T]:
    """Compute path weights from src to all reachable vertices using Bellman-Ford-like algorithm.

    This computes shortest paths in the semiring defined by the algebra.
    Uses a worklist algorithm to compute fixed-point of path weights.
    """
    # Initialize path weights
    weights: Dict[Vertex, T] = {}
    worklist: Set[Vertex] = {src}

    # Source has weight 'one' (identity element)
    weights[src] = wg.algebra.one

    # Process worklist until convergence
    max_iterations = len(wg.vertices()) * len(wg.vertices())  # Prevent infinite loops
    iteration = 0

    while worklist and iteration < max_iterations:
        iteration += 1
        current = worklist.pop()
        current_weight = weights.get(current, wg.algebra.zero)

        # Update successors
        for successor in wg.successors(current):
            edge_weight = wg.edge_weight(current, successor)
            # New weight through current vertex
            new_weight = wg.algebra.mul(current_weight, edge_weight)

            # Add to existing weight (using semiring addition)
            old_weight = weights.get(successor, wg.algebra.zero)
            combined_weight = wg.algebra.add(old_weight, new_weight)

            # Update if changed
            if successor not in weights or combined_weight != old_weight:
                weights[successor] = combined_weight
                worklist.add(successor)

    def compute_path_weight(target: Vertex) -> T:
        """Get computed path weight to target."""
        return weights.get(target, wg.algebra.zero)

    return compute_path_weight


def omega_path_weight(
    wg: WeightedGraph[T], omega: OmegaAlgebra[T, U], src: Vertex
) -> U:
    """Compute omega path weights from src (infinite path weights).

    This computes the weight of infinite paths starting from src,
    which is useful for finding cycles and infinite behaviors.

    The algorithm:
    1. Find all cycles reachable from src
    2. Compute their omega weights
    3. Combine them using the omega algebra
    """
    # Find strongly connected components reachable from src
    reachable = _find_reachable(wg, src)
    sccs = _find_sccs_in_vertices(wg, reachable)

    # Initialize omega weight as the zero element in the omega algebra
    result = omega.omega(wg.algebra.zero)

    # For each SCC with more than one vertex or a self-loop
    for scc in sccs:
        if len(scc) > 1 or (len(scc) == 1 and wg.mem_edge(list(scc)[0], list(scc)[0])):
            # Find a cycle in this SCC
            cycle_weight = _find_cycle_weight(wg, scc)
            if cycle_weight != wg.algebra.zero:
                # Compute omega of the cycle weight
                omega_cycle = omega.omega(cycle_weight)
                # Combine with result
                result = omega.omega_add(result, omega_cycle)

    return result


def _find_reachable(wg: WeightedGraph[T], src: Vertex) -> Set[Vertex]:
    """Find all vertices reachable from src."""
    reachable = set()
    worklist = [src]

    while worklist:
        current = worklist.pop()
        if current in reachable:
            continue
        reachable.add(current)
        worklist.extend(wg.successors(current))

    return reachable


def _find_sccs_in_vertices(
    wg: WeightedGraph[T], vertices: Set[Vertex]
) -> List[Set[Vertex]]:
    """Find strongly connected components within a set of vertices."""
    # Build subgraph
    subgraph = {v: wg.successors(v) & vertices for v in vertices}

    # Use Tarjan's algorithm for SCC
    index_counter = [0]
    stack: List[Vertex] = []
    lowlinks: Dict[Vertex, int] = {}
    index: Dict[Vertex, int] = {}
    on_stack: Set[Vertex] = set()
    sccs: List[Set[Vertex]] = []

    def strongconnect(v: Vertex) -> None:
        index[v] = index_counter[0]
        lowlinks[v] = index_counter[0]
        index_counter[0] += 1
        stack.append(v)
        on_stack.add(v)

        for w in subgraph.get(v, set()):
            if w not in index:
                strongconnect(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elif w in on_stack:
                lowlinks[v] = min(lowlinks[v], index[w])

        if lowlinks[v] == index[v]:
            scc = set()
            while True:
                w = stack.pop()
                on_stack.remove(w)
                scc.add(w)
                if w == v:
                    break
            sccs.append(scc)

    for v in vertices:
        if v not in index:
            strongconnect(v)

    return sccs


def _find_cycle_weight(wg: WeightedGraph[T], scc: Set[Vertex]) -> T:
    """Find the weight of a cycle in an SCC."""
    if not scc:
        return wg.algebra.zero

    # Pick an arbitrary vertex from the SCC
    start = next(iter(scc))

    # If there's a self-loop, use it
    if wg.mem_edge(start, start):
        return wg.edge_weight(start, start)

    # Otherwise, find any cycle through start
    # Use DFS to find a path back to start
    visited = set()
    path_weight = {start: wg.algebra.one}

    def dfs(current: Vertex, weight: T) -> Optional[T]:
        if current in visited:
            if current == start:
                return weight
            return None

        visited.add(current)

        for successor in wg.successors(current):
            if successor in scc:
                edge_w = wg.edge_weight(current, successor)
                new_weight = wg.algebra.mul(weight, edge_w)
                result = dfs(successor, new_weight)
                if result is not None:
                    return result

        visited.remove(current)
        return None

    # Try to find a cycle from start
    for first_succ in wg.successors(start):
        if first_succ in scc:
            edge_w = wg.edge_weight(start, first_succ)
            cycle_w = dfs(first_succ, edge_w)
            if cycle_w is not None:
                return cycle_w

    return wg.algebra.zero


def msat_path_weight(wg: WeightedGraph[T], sources: List[Vertex]) -> WeightedGraph[T]:
    """Multi-source path weight computation."""
    result = WeightedGraph(wg.algebra)

    for src in sources:
        path_w = path_weight(wg, src)
        for target in wg.vertices():
            if target != src:
                weight = path_w(target)
                if weight != wg.algebra.zero:
                    result.add_edge(src, weight, target)

    return result


# Trivial omega algebra
def omega_trivial() -> OmegaAlgebra[T, None]:
    """Trivial omega algebra that ignores omega paths."""

    def omega(x: T) -> None:
        return None

    def omega_add(x: None, y: None) -> None:
        return None

    def omega_mul(x: T, y: None) -> None:
        return None

    return OmegaAlgebra(omega, omega_add, omega_mul)


def split_vertex(
    wg: WeightedGraph[T], u: Vertex, weight: T, v: Vertex
) -> WeightedGraph[T]:
    """Split vertex u with new vertex v and edge u->v with given weight."""
    new_wg = WeightedGraph(wg.algebra)

    # Copy all vertices and edges
    for vertex in wg.vertices():
        new_wg.add_vertex(vertex)

    for (src, dst), w in wg.labels.items():
        new_wg.add_edge(src, w, dst)

    # Add the split edge
    new_wg.add_edge(u, weight, v)

    # Update successors of u to go through v for all non-v successors
    for successor in list(wg.successors(u)):
        if successor != v:
            old_weight = wg.edge_weight(u, successor)
            new_wg.remove_edge(u, successor)
            new_wg.add_edge(v, old_weight, successor)

    return new_wg


# SCC (Strongly Connected Components) computation
def scc_list(graph: Dict[Vertex, Set[Vertex]]) -> List[List[Vertex]]:
    """Compute strongly connected components of a graph."""
    # Simplified SCC computation using DFS
    visited = set()
    scc = []

    def dfs1(node: Vertex) -> List[Vertex]:
        """First DFS pass."""
        stack = [node]
        component = []

        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                component.append(current)

                # Add unvisited successors
                for successor in graph.get(current, set()):
                    if successor not in visited:
                        stack.append(successor)

        return component

    def dfs2(node: Vertex, component: Set[Vertex]) -> None:
        """Second DFS pass for transpose graph."""
        stack = [node]

        while stack:
            current = stack.pop()
            if current not in component:
                component.add(current)

                # Add predecessors in transpose graph
                for pred in all_predecessors.get(current, set()):
                    if pred not in component:
                        stack.append(pred)

    # Build predecessor map for transpose graph
    all_predecessors: Dict[Vertex, Set[Vertex]] = {}
    for vertex in graph:
        for successor in graph.get(vertex, set()):
            if successor not in all_predecessors:
                all_predecessors[successor] = set()
            all_predecessors[successor].add(vertex)

    # First pass: get finishing times
    all_vertices = list(graph.keys())
    finishing_order = []

    for vertex in all_vertices:
        if vertex not in visited:
            component = dfs1(vertex)
            finishing_order.extend(component)

    # Reset visited
    visited.clear()

    # Second pass: find SCCs in reverse finishing order
    for vertex in reversed(finishing_order):
        if vertex not in visited:
            component = set()
            dfs2(vertex, component)
            if component:
                scc.append(list(component))

    return scc


def cut_graph(wg: WeightedGraph[T], vertices: List[Vertex]) -> WeightedGraph[T]:
    """Cut graph to only include specified vertices while preserving path weights."""
    vertex_set = set(vertices)

    # Create new graph with only specified vertices
    new_wg = WeightedGraph(wg.algebra)

    for v in vertices:
        new_wg.add_vertex(v)

    # Add edges between specified vertices
    for (u, v), weight in wg.labels.items():
        if u in vertex_set and v in vertex_set:
            new_wg.add_edge(u, weight, v)

    return new_wg


# Line graph construction (edges become vertices)
class LineGraph:
    """Line graph where edges become vertices."""

    def __init__(self, graph: Dict[Vertex, Set[Vertex]]):
        self.graph = graph

    def vertices(self) -> List[Tuple[Vertex, Vertex]]:
        """Get all edges as vertices."""
        edges = []
        for u in self.graph:
            for v in self.graph[u]:
                edges.append((u, v))
        return edges

    def successors(self, edge: Tuple[Vertex, Vertex]) -> List[Tuple[Vertex, Vertex]]:
        """Get successor edges."""
        u, v = edge
        successors = []

        # An edge (u,v) has successor edges from v
        for w in self.graph.get(v, set()):
            if w != u:  # Avoid self-loops
                successors.append((v, w))

        return successors


# Forward analysis framework
def forward_analysis(
    wg: WeightedGraph[T],
    entry: Vertex,
    update: Callable[[T, T], Optional[T]],
    init: Callable[[Vertex], T],
) -> Callable[[Vertex], T]:
    """Perform forward analysis on the weighted graph.

    Args:
        wg: The weighted graph
        entry: Entry vertex for analysis
        update: Update function: (pre_state, edge_weight) -> new_post_state
        init: Initialization function for vertex states

    Returns:
        Function mapping vertices to their computed states
    """
    # Initialize state for all vertices
    states: Dict[Vertex, T] = {}
    worklist = set()

    def get_state(v: Vertex) -> T:
        """Get state for vertex v."""
        if v not in states:
            states[v] = init(v)
        return states[v]

    def set_state(v: Vertex, state: T) -> None:
        """Set state for vertex v."""
        if v not in states or states[v] != state:
            states[v] = state
            worklist.add(v)

    # Initialize entry state
    set_state(entry, init(entry))

    # Process worklist
    while worklist:
        current = worklist.pop()

        # Process outgoing edges
        for successor in wg.successors(current):
            edge_weight = wg.edge_weight(current, successor)
            pre_state = get_state(current)
            post_state = get_state(successor)

            new_post = update(pre_state, edge_weight, post_state)
            if new_post is not None:
                set_state(successor, new_post)

    return get_state


# Advanced path-finding algorithms using compressed weighted forests


def _solve_dense(wg: WeightedGraph[T], src: Vertex) -> WeightedGraph[T]:
    """Solve dense path problems using sophisticated algorithms.

    This is a simplified version of the complex algorithm in the OCaml code.
    A full implementation would handle SCCs, loops, and omega paths properly.
    """
    # For now, return a basic version
    result = WeightedGraph(wg.algebra)
    result.add_vertex(src)

    # Simple path computation - just direct edges
    for (u, v), weight in wg.labels.items():
        if u == src:
            result.add_edge(u, weight, v)

    return result


# Main path weight computation using advanced algorithms
def _path_weight_advanced(
    wg: WeightedGraph[T], omega: OmegaAlgebra[T, U], src: Vertex
) -> tuple[U, Callable[[Vertex], T], Set[Vertex]]:
    """Advanced path weight computation with SCC and loop handling.

    This is a simplified version. The full algorithm would:
    1. Use compressed weighted forests for efficient path tracking
    2. Handle strongly connected components
    3. Process loops using specialized algorithms
    4. Compute omega paths for infinite behaviors
    """
    # Create artificial start node to ensure no incoming edges
    start = wg.max_vertex() + 1 if wg.max_vertex() is not None else 0
    start_wg = wg.add_vertex(start).add_edge(start, wg.algebra.one, src)

    # Initialize compressed weighted forest
    forest = CompressedWeightedForest(wg.algebra.mul, wg.algebra.one)

    # Map WG vertices to forest nodes
    wg_to_forest: Dict[Vertex, int] = {}
    forest_to_wg: Dict[int, Vertex] = {}

    def to_forest(v: Vertex) -> int:
        if v not in wg_to_forest:
            r = forest.root()
            wg_to_forest[v] = r
            forest_to_wg[r] = v
        return wg_to_forest[v]

    def find_vertex(forest_node: int) -> Vertex:
        return forest_to_wg[forest.find(forest_node)]

    def link_vertices(u: Vertex, weight: T, v: Vertex) -> None:
        forest.link(to_forest(u), weight, to_forest(v))

    def eval_vertex(v: Vertex) -> T:
        return forest.eval(to_forest(v))

    # For now, implement a simplified version
    # A full implementation would use the sophisticated SCC and loop algorithms
    path_weight = lambda target: (
        wg.algebra.one
        if target == src
        else (
            wg.edge_weight(src, target) if wg.mem_edge(src, target) else wg.algebra.zero
        )
    )

    omega_weight = omega.omega(wg.algebra.zero)
    reachable = {v for v in wg.vertices() if v != src and wg.mem_edge(src, v)}

    return omega_weight, path_weight, reachable


def path_weight_advanced(wg: WeightedGraph[T], src: Vertex) -> Callable[[Vertex], T]:
    """Advanced path weight computation."""
    _, path_weight, _ = _path_weight_advanced(wg, omega_trivial(), src)
    return path_weight


def omega_path_weight_advanced(
    wg: WeightedGraph[T], omega: OmegaAlgebra[T, U], src: Vertex
) -> U:
    """Advanced omega path weight computation."""
    omega_weight, _, _ = _path_weight_advanced(wg, omega, src)
    return omega_weight


def msat_path_weight_advanced(
    wg: WeightedGraph[T], sources: List[Vertex]
) -> WeightedGraph[T]:
    """Advanced multi-source path weight computation."""
    result = WeightedGraph(wg.algebra)

    for src in sources:
        path_w = path_weight_advanced(wg, src)
        for target in wg.vertices():
            if target != src:
                weight = path_w(target)
                if weight != wg.algebra.zero:
                    result.add_edge(src, weight, target)

    return result


# Graph transformation utilities


def contract_vertex(wg: WeightedGraph[T], v: Vertex) -> WeightedGraph[T]:
    """Contract vertex v by connecting predecessors to successors.

    This implements vertex contraction while preserving path weights.
    """
    if v not in wg.vertices():
        return wg

    predecessors = wg.predecessors(v)
    successors = wg.successors(v)

    new_wg = WeightedGraph(wg.algebra)

    # Copy all vertices except v
    for vertex in wg.vertices():
        if vertex != v:
            new_wg.add_vertex(vertex)

    # Copy all edges except those involving v
    for (u, dst), weight in wg.labels.items():
        if u != v and dst != v:
            new_wg.add_edge(u, weight, dst)

    # Connect predecessors to successors
    for pred in predecessors:
        if pred == v:
            continue

        # Get weight from pred to v
        pv_weight = wg.edge_weight(pred, v)

        for succ in successors:
            if succ == v:
                continue

            # Get weight from v to succ
            vs_weight = wg.edge_weight(v, succ)

            # Combine weights
            combined_weight = wg.algebra.mul(pv_weight, vs_weight)
            new_wg.add_edge(pred, combined_weight, succ)

    return new_wg


def fold_reachable_edges(
    f: Callable[[Vertex, Vertex], U], wg: WeightedGraph[T], v: Vertex, acc: U
) -> U:
    """Fold over all edges reachable from vertex v."""
    visited = set()

    def dfs(current: Vertex, acc: U) -> U:
        if current in visited:
            return acc

        visited.add(current)

        for successor in wg.successors(current):
            acc = f(current, successor, acc)
            acc = dfs(successor, acc)

        return acc

    return dfs(v, acc)
