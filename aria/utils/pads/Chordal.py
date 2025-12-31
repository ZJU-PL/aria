"""Recognize and compute elimination ordering of chordal graphs, using
an algorithm from Habib, McConnell, Paul, and Viennot, "Lex-BFS and
Partition Refinement, with Applications to Transitive Orientation,
Interval Graph Recognition, and Consecutive Ones Testing", Theor.
Comput. Sci. 234:59-84 (2000), http://www.cs.colostate.edu/~rmm/lexbfs.ps

D. Eppstein, November 2003.
"""
# pylint: disable=invalid-name

from aria.utils.pads.LexBFS import lex_bfs


def perfect_elimination_ordering(graph):
    """Return a perfect elimination ordering, or None if graph is not chordal.
    graph should be represented in such a way that "for v in graph" loops through
    the vertices, and "graph[v]" produces a list of the neighbors of v; for
    instance, graph may be a dictionary mapping each vertex to its neighbor set.
    Running time is O(n+m) and additional space usage over graph is O(n+m).
    """
    already_processed = set()
    bfs_order = list(lex_bfs(graph))
    position = {bfs_order[i]: i for i in range(len(bfs_order))}
    left_neighbors = {}
    parent = {}
    for v in bfs_order:
        left_neighbors[v] = set(graph[v]) & already_processed
        already_processed.add(v)
        if left_neighbors[v]:
            parent[v] = bfs_order[max(position[w] for w in left_neighbors[v])]
            if not left_neighbors[v] - set([parent[v]]) <= left_neighbors[parent[v]]:
                return None
    bfs_order.reverse()
    return bfs_order
