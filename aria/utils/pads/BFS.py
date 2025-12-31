"""Breadth First Search. See also LexBFS.py.

D. Eppstein, May 2007.
"""

def breadth_first_levels(graph, root):
    """
    Generate a sequence of bipartite directed graphs, each consisting
    of the edges from level i to level i+1 of graph. Edges that connect
    vertices within the same level are not included in the output.
    The vertices in each level can be listed by iterating over each
    output graph.
    """
    visited = set()
    current_level = [root]
    while current_level:
        for v in current_level:
            visited.add(v)
        next_level = set()
        level_graph = {v: set() for v in current_level}
        for v in current_level:
            for w in graph[v]:
                if w not in visited:
                    level_graph[v].add(w)
                    next_level.add(w)
        yield level_graph
        current_level = next_level
