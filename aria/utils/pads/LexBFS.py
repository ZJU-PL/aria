"""Lexicographic breadth-first-search traversal of a graph, as described
in Habib, McConnell, Paul, and Viennot, "Lex-BFS and Partition Refinement,
with Applications to Transitive Orientation, Interval Graph Recognition,
and Consecutive Ones Testing", Theor. Comput. Sci. 234:59-84 (2000),
http://www.cs.colostate.edu/~rmm/lexbfs.ps

D. Eppstein, November 2003.
"""
# pylint: disable=invalid-name

from aria.utils.pads.PartitionRefinement import PartitionRefinement
from aria.utils.pads.Sequence import Sequence
from aria.utils.pads.Util import arbitrary_item


def lex_bfs(graph):
    """Find lexicographic breadth-first-search traversal order of a graph.
    graph should be represented in such a way that "for v in graph" loops through
    the vertices, and "graph[v]" produces a sequence of the neighbors of v; for
    instance, graph may be a dictionary mapping each vertex to its neighbor set.
    Running time is O(n+m) and additional space usage over graph is O(n).
    """
    partition = PartitionRefinement(graph)
    sequence = Sequence(partition, key=id)
    while sequence:
        current_set = sequence[0]
        v = arbitrary_item(current_set)
        yield v
        partition.remove(v)
        if not current_set:
            sequence.remove(current_set)
        for new, old in partition.refine(graph[v]):
            sequence.insertBefore(old, new)
