"""Various operations on partial orders and directed acyclic graphs.

D. Eppstein, July 2006.
"""


import unittest

from aria.utils.pads import BipartiteMatching
from aria.utils.pads.DFS import postorder, preorder


def is_topological_order(graph, ordering):
    """Check that ordering is a topological ordering of directed graph graph."""
    vnum = {}
    for i, vertex in enumerate(ordering):
        if vertex not in graph:
            return False
        vnum[vertex] = i
    for v in graph:
        if v not in vnum:
            return False
        for w in graph[v]:
            if w not in vnum or vnum[w] <= vnum[v]:
                return False
    return True

def topological_order(graph):
    """Find a topological ordering of directed graph graph."""
    ordering = list(postorder(graph))
    ordering.reverse()
    if not is_topological_order(graph, ordering):
        raise ValueError("topological_order: graph is not acyclic.")
    return ordering

def is_acyclic(graph):
    """Return True if graph is a directed acyclic graph, False otherwise."""
    ordering = list(postorder(graph))
    ordering.reverse()
    return is_topological_order(graph, ordering)

def transitive_closure(graph):
    """
    The transitive closure of graph graph.
    This is a graph on the same vertex set containing an edge (v,w)
    whenever v != w and there is a directed path from v to w in graph.
    """
    tc = {v: set(preorder(graph, v)) for v in graph}
    for v in graph:
        tc[v].remove(v)
    return tc

def trace_paths(graph):
    """
    Turn a DAG with indegree and outdegree <= 1 into a sequence of lists.
    """
    path = []
    for v in topological_order(graph):
        if path and v not in graph[path[-1]]:
            yield path
            path = []
        path.append(v)
    if path:
        yield path

def minimum_path_decomposition(graph):
    """
    Cover a directed acyclic graph with a minimum number of paths.
    """
    matching, set_a, set_b = BipartiteMatching.matching(graph)
    dag = {v: [] for v in graph}
    for v in graph:
        if v in matching:
            dag[matching[v]] = (v,)
    return trace_paths(dag)

def minimum_chain_decomposition(graph):
    """
    Cover a partial order with a minimum number of chains.
    By Dilworth's theorem the number of chains equals the size
    of the largest antichain of the order. The input should be
    a directed acyclic graph, not necessarily transitively closed.
    """
    return minimum_path_decomposition(transitive_closure(graph))

def maximum_antichain(graph):
    """
    Find a maximum antichain in the given directed acyclic graph.
    """
    if not is_acyclic(graph):
        raise ValueError("maximum_antichain: input is not acyclic.")
    tc = transitive_closure(graph)
    matching, set_a, set_b = BipartiteMatching.matching(transitive_closure(graph))
    return set(set_a).intersection(set_b)

class PartialOrderTest(unittest.TestCase):
    cube = {i: [] for i in range(16)}
    for i in range(16):
        for b in (1,2,4,8):
            cube[min(i,i^b)].append(max(i,i^b))

    def test_hypercube_acyclic(self):
        self.assertTrue(is_acyclic(self.cube))

    def test_hypercube_closure(self):
        tc = transitive_closure(self.cube)
        for i in range(16):
            self.assertEqual(tc[i],
                {j for j in range(16) if i & j == i and i != j})

    def test_hypercube_antichain(self):
        antichain = maximum_antichain(self.cube)
        self.assertEqual(antichain,set((3,5,6,9,10,12)))

    def test_hypercube_dilworth(self):
        chain_decomp = list(minimum_chain_decomposition(self.cube))
        print(chain_decomp)
        self.assertEqual(len(chain_decomp),6)

if __name__ == "__main__":
    unittest.main()
