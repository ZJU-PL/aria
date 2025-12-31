"""Two-color graphs and find related structures.
D. Eppstein, May 2004.
"""

import unittest

from aria.utils.pads import DFS, Graphs
from aria.utils.pads.Biconnectivity import BiconnectedComponents

class NonBipartite(Exception):
    pass

def two_color(graph):
    """
    Find a bipartition of graph, if one exists.
    Raises NonBipartite or returns dict mapping vertices
    to two colors (True and False).
    """
    color = {}
    for v, w, edgetype in DFS.search(graph):
        if edgetype is DFS.forward:
            color[w] = not color.get(v, False)
        elif edgetype is DFS.nontree and color[v] == color[w]:
            raise NonBipartite
    return color

def bipartition(graph):
    """
    Find a bipartition of graph, if one exists.
    Raises NonBipartite or returns sequence of vertices
    on one side of the bipartition.
    """
    color = two_color(graph)
    for v, color_val in color.items():
        if color_val:
            yield v

def is_bipartite(graph):
    """
    Return True if graph is bipartite, False otherwise.
    """
    try:
        two_color(graph)
        return True
    except NonBipartite:
        return False

def bipartite_orientation(graph, adjacency_list_type=set):
    """
    Given an undirected bipartite graph, return a directed graph in which
    the edges are oriented from one side of the bipartition to the other.
    The second argument has the same meaning as in Graphs.copy_graph.
    """
    bipartition_vertices = bipartition(graph)
    return {v: adjacency_list_type(iter(graph[v])) for v in bipartition_vertices}

def odd_core(graph):
    """
    Subgraph of vertices and edges that participate in odd cycles.
    Aka, the union of nonbipartite biconnected components.
    """
    return Graphs.union(*[C for C in BiconnectedComponents(graph)
                          if not is_bipartite(C)])

# If run as "python Bipartite.py", run tests on various small graphs
# and check that the correct results are obtained.

class BipartitenessTest(unittest.TestCase):
    def cycle(self, n):
        return {i: [(i-1) % n, (i+1) % n] for i in range(n)}

    def test_even_cycles(self):
        for i in range(4, 12, 2):
            self.assertEqual(is_bipartite(self.cycle(i)), True)

    def test_odd_cycles(self):
        for i in range(3, 12, 2):
            self.assertEqual(is_bipartite(self.cycle(i)), False)

if __name__ == "__main__":
    unittest.main()
