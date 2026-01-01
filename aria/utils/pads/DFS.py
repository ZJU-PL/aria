"""Algorithms for depth first search in Python.
We need to search iteratively rather than recursively in
order to avoid Python's low recursion limit.

D. Eppstein, July 2004.
"""

# pylint: disable=invalid-name

# Types of edges in DFS traversal.
# The numerical values are used in DepthFirstSearcher, change with care.
FORWARD = 1  # traversing edge (v,w) from v to w
REVERSE = -1  # returning backwards on (v,w) from w to v
NONTREE = 0  # edge (v,w) is not part of the DFS tree

whole_graph = object()  # special flag object, do not use as a graph vertex


def search(graph, initial_vertex=whole_graph):
    """
    Generate sequence of triples (v,w,edgetype) for DFS of graph G.
    The subsequence for each root of each tree in the DFS forest starts
    with (root,root,forward) and ends with (root,root,reverse).
    If the initial vertex is given, it is used as the root and vertices
    not reachable from it are not searched.
    """
    visited = set()
    if initial_vertex == whole_graph:
        initials = graph
    else:
        initials = [initial_vertex]
    for v in initials:
        if v not in visited:
            yield v, v, FORWARD
            visited.add(v)
            stack = [(v, iter(graph[v]))]
            while stack:
                parent, children = stack[-1]
                try:
                    child = next(children)
                    if child in visited:
                        yield parent, child, NONTREE
                    else:
                        yield parent, child, FORWARD
                        visited.add(child)
                        stack.append((child, iter(graph[child])))
                except StopIteration:
                    stack.pop()
                    if stack:
                        yield stack[-1][0], parent, REVERSE
            yield v, v, REVERSE


def preorder(graph, initial_vertex=whole_graph):
    """Generate all vertices of graph G in depth-first preorder."""
    for v, w, edgetype in search(graph, initial_vertex):
        if edgetype is FORWARD:
            yield w


def postorder(graph, initial_vertex=whole_graph):
    """Generate all vertices of graph G in depth-first postorder."""
    for v, w, edgetype in search(graph, initial_vertex):
        if edgetype is REVERSE:
            yield w


def reachable(graph, v, w):
    """Can we get from v to w in graph G?"""
    return w in preorder(graph, v)


class Searcher:
    """
    Handler for performing general depth first searches of graphs.
    Some or all of the routines preorder, postorder, and backedge
    should be shadowed in order to make the search do something useful.
    """

    def preorder(self, parent, child):
        """
        Called when DFS visits child, before visiting all grandchildren.
        Parent==child when child is the root of each DFS tree.
        """

    def postorder(self, parent, child):
        """
        Called when DFS visits child, after visiting all grandchildren.
        Parent==child when child is the root of each DFS tree.
        """

    def backedge(self, source, destination):
        """Called when DFS discovers an edge to a non-child."""

    def __init__(self, graph):
        """Perform a depth first search of graph G."""
        dispatch = [self.backedge, self.preorder, self.postorder]
        for v, w, edgetype in search(graph):
            dispatch[edgetype](v, w)
