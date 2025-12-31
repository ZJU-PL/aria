"""Various simple functions for graph input.

Each function's input graph G should be represented in such a way that
"for v in G" loops through the vertices, and "G[v]" produces a list of the
neighbors of v; for instance, G may be a dictionary mapping each vertex
to its neighbor set.

D. Eppstein, April 2004.
"""

# pylint: disable=invalid-name
# Module name kept as "Graphs" for backward compatibility

# sets.Set is deprecated in Python 3, use built-in set instead

def is_undirected(graph):
    """Check that graph represents a simple undirected graph."""
    for v in graph:
        if v in graph[v]:
            return False
        for w in graph[v]:
            if v not in graph[w]:
                return False
    return True

def max_degree(graph):
    """Return the maximum vertex (out)degree of graph G."""
    return max(len(graph[v]) for v in graph)

def min_degree(graph):
    """Return the minimum vertex (out)degree of graph G."""
    return min(len(graph[v]) for v in graph)

def copy_graph(graph, adjacency_list_type):
    """
    Make a copy of a graph G and return the copy.
    Any information stored in edges graph[v][w] is discarded.
    The second argument should be a callable that turns a sequence
    of neighbors into an appropriate representation of the adjacency list.
    Note that, while Set, list, and tuple are appropriate values for
    adjacency_list_type, dict is not -- use Util.map_to_constant instead.
    """
    return {v: adjacency_list_type(iter(graph[v])) for v in graph}

def induced_subgraph(vertices, graph, adjacency_list_type):
    """
    The subgraph consisting of all edges between pairs of vertices in vertices.
    """
    def neighbors(x):
        for y in graph[x]:
            if y in vertices:
                yield y
    return {x: adjacency_list_type(neighbors(x)) for x in graph if x in vertices}

def union(*graphs):
    """Return a graph having all edges from the argument graphs."""
    out = {}
    for graph in graphs:
        for v in graph:
            out.setdefault(v, set()).update(list(graph[v]))
    return out

def is_independent_set(vertices, graph):
    """
    True if vertices is an independent set of vertices in graph, False otherwise.
    """
    class NonIndependent(Exception):
        pass

    def test_independent(seq):
        for x in seq:
            raise NonIndependent

    try:
        induced_subgraph(vertices, graph, test_independent)
        return True
    except NonIndependent:
        return False
