"""Find maximum cardinality matchings in general undirected graphs.

D. Eppstein, UC Irvine, September 6, 2003.
"""

# import sys
# pylint: disable=invalid-name
from aria.utils.pads.UnionFind import UnionFind
from aria.utils.pads.Util import arbitrary_item


def matching(graph, initial_matching=None):
    """Find a maximum cardinality matching in a graph.

    graph is represented in modified GvR form: iter(graph) lists its vertices;
    iter(graph[v]) lists the neighbors of v; w in graph[v] tests adjacency.
    For maximal efficiency, graph and graph[v] should be dictionaries, so
    that adjacency tests take constant time each.
    The output is a dictionary mapping vertices to their matches;
    unmatched vertices are omitted from the dictionary.

    We use Edmonds' blossom-contraction algorithm, as described e.g.
    in Galil's 1986 Computing Surveys paper.
    """

    # Copy initial matching so we can use it nondestructively
    # and augment it greedily to reduce main loop iterations
    result_matching = greedy_matching(graph, initial_matching)

    def augment():
        """Search for a single augmenting path.
        Returns true if the matching size was increased, false otherwise.
        """

        # Data structures for augmenting path search:
        #
        # leader: union-find structure; the leader of a blossom is one
        # of its vertices (not necessarily topmost), and leader[v] always
        # points to the leader of the largest blossom containing v
        #
        # s_blossoms: dictionary of blossoms at even levels of the structure tree.
        # Dictionary keys are names of blossoms (as returned by the union-find
        # data structure) and values are the structure tree parent of the blossom
        # (a t_vertex, or the top vertex if the blossom is a root of a structure tree).
        #
        # t_vertices: dictionary of vertices at odd levels of the structure tree.
        # Dictionary keys are the vertices; t_vertices[x] is a vertex with an unmatched
        # edge to x.  To find the parent in the structure tree, use
        # leader[t_vertices[x]].
        #
        # unexplored: collection of unexplored vertices within blossoms of s_blossoms
        #
        # base: if x was originally a t_vertex, but becomes part of a blossom,
        # base[t] will be the pair (v,w) at the base of the blossom, where v and t
        # are on the same side of the blossom and w is on the other side.

        leader = UnionFind()
        s_blossoms = {}
        t_vertices = {}
        unexplored = []
        base = {}

        # Subroutines for augmenting path search.
        # Many of these are called only from one place, but are split out
        # as subroutines to improve modularization and readability.

        def blossom(v, w, a):
            """Create a new blossom from edge v-w with common ancestor a."""

            def find_side(v_vertex, w_vertex):
                path = [leader[v_vertex]]
                b = (v_vertex, w_vertex)  # new base for all T nodes found on the path
                while path[-1] != a:
                    tnode = s_blossoms[path[-1]]
                    path.append(tnode)
                    base[tnode] = b
                    unexplored.append(tnode)
                    path.append(leader[t_vertices[tnode]])
                return path

            a = leader[a]  # sanity check
            path1, path2 = find_side(v, w), find_side(w, v)
            leader.union(*path1)
            leader.union(*path2)
            s_blossoms[leader[a]] = s_blossoms[a]  # update structure tree

        topless = object()  # should be unequal to any graph vertex

        def alternating_path(start, goal=topless):
            """Return sequence of vertices on alternating path from start to goal.
            The goal must be a T node along the path from the start to
            the root of the structure tree. If goal is omitted, we find
            an alternating path to the structure tree root.
            """
            path = []
            while 1:
                while start in t_vertices:
                    v, w = base[start]
                    vs = alternating_path(v, start)
                    vs.reverse()
                    path += vs
                    start = w
                path.append(start)
                if start not in result_matching:
                    return path  # reached top of structure tree, done!
                tnode = result_matching[start]
                path.append(tnode)
                if tnode == goal:
                    return path  # finished recursive subpath
                start = t_vertices[tnode]

        def alternate(v):
            """Make v unmatched by alternating the path to the root of its structure
            tree."""
            path = alternating_path(v)
            path.reverse()
            for i in range(0, len(path) - 1, 2):
                result_matching[path[i]] = path[i + 1]
                result_matching[path[i + 1]] = path[i]

        def add_match(v, w):
            """Here with an S-S edge vw connecting vertices in different structure
            trees. Find the corresponding augmenting path and use it to augment the
            matching.
            """
            alternate(v)
            alternate(w)
            result_matching[v] = w
            result_matching[w] = v

        def ss(v, w):
            """Handle detection of an S-S edge in augmenting path search.
            Like augment(), returns true iff the matching size was increased.
            """

            if leader[v] == leader[w]:
                return False  # self-loop within blossom, ignore

            # parallel search up two branches of structure tree
            # until we find a common ancestor of v and w
            path1, head1 = {}, v
            path2, head2 = {}, w

            def step(path, head):
                head = leader[head]
                parent = leader[s_blossoms[head]]
                if parent == head:
                    return head  # found root of structure tree
                path[head] = parent
                path[parent] = leader[t_vertices[parent]]
                return path[parent]

            while 1:
                head1 = step(path1, head1)
                head2 = step(path2, head2)

                if head1 == head2:
                    blossom(v, w, head1)
                    return False

                if (
                    leader[s_blossoms[head1]] == head1
                    and leader[s_blossoms[head2]] == head2
                ):
                    add_match(v, w)
                    return True

                if head1 in path2:
                    blossom(v, w, head1)
                    return False

                if head2 in path1:
                    blossom(v, w, head2)
                    return False

        # Start of main augmenting path search code.

        for v in graph:
            if v not in result_matching:
                s_blossoms[v] = v
                unexplored.append(v)

        current = 0  # index into unexplored, in FIFO order so we get short paths
        while current < len(unexplored):
            v = unexplored[current]
            current += 1

            for w in graph[v]:
                if leader[w] in s_blossoms:  # S-S edge: blossom or augmenting path
                    if ss(v, w):
                        return True

                elif w not in t_vertices:  # previously unexplored node, add as T-node
                    t_vertices[w] = v
                    u = result_matching[w]
                    if leader[u] not in s_blossoms:
                        s_blossoms[u] = w  # and add its match as an S-node
                        unexplored.append(u)

        return False  # ran out of graph without finding an augmenting path

    # augment the matching until it is maximum
    while augment():
        pass

    return result_matching


def greedy_matching(graph, initial_matching=None):
    """Near-linear-time greedy heuristic for creating high-cardinality matching.
    If there is any vertex with one unmatched neighbor, we match it.
    Otherwise, if there is a vertex with two unmatched neighbors, we contract
    it away and store the contraction on a stack for later matching.
    If neither of these two cases applies, we match an arbitrary edge.
    """

    # Copy initial matching so we can use it nondestructively
    result_matching = {}
    if initial_matching:
        for x in initial_matching:
            result_matching[x] = initial_matching[x]

    # Copy graph to new subgraph of available edges
    # Representation: nested dictionary rep->rep->pair
    # where the reps are representative vertices for merged clusters
    # and the pair is an unmatched original pair of vertices
    avail = {}
    has_edge = False
    for v in graph:
        if v not in result_matching:
            avail[v] = {}
            for w in graph[v]:
                if w not in result_matching:
                    avail[v][w] = (v, w)
                    has_edge = True
            if not avail[v]:
                del avail[v]
    if not has_edge:
        return result_matching

    # make sets of degree one and degree two vertices
    deg1 = {
        v for v, neighbors in avail.items() if len(neighbors) == 1
    }  # pylint: disable=consider-using-dict-items
    deg2 = {
        v for v, neighbors in avail.items() if len(neighbors) == 2
    }  # pylint: disable=consider-using-dict-items
    d2edges = []

    def update_degree(v):
        """Cluster degree changed, update sets."""
        if v in deg1:
            deg1.remove(v)
        elif v in deg2:
            deg2.remove(v)
        if len(avail[v]) == 0:
            del avail[v]
        elif len(avail[v]) == 1:
            deg1.add(v)
        elif len(avail[v]) == 2:
            deg2.add(v)

    def add_match(v, w):
        """Add edge connecting two given cluster reps, update avail."""
        p, q = avail[v][w]
        result_matching[p] = q
        result_matching[q] = p
        for x in avail[v]:
            if x != w:
                del avail[x][v]
                update_degree(x)
        for x in avail[w]:
            if x != v:
                del avail[x][w]
                update_degree(x)
        avail[v] = avail[w] = {}
        update_degree(v)
        update_degree(w)

    def contract(v):
        """Handle degree two vertex."""
        u, w = avail[v]  # find reps for two neighbors
        d2edges.extend([avail[v][u], avail[v][w]])
        del avail[u][v]
        del avail[w][v]
        if len(avail[u]) > len(avail[w]):
            u, w = w, u  # swap to preserve near-linear time bound
        for x in avail[u]:
            del avail[x][u]
            if x in avail[w]:
                update_degree(x)
            elif x != w:
                avail[x][w] = avail[w][x] = avail[u][x]
        avail[u] = avail[v] = {}
        update_degree(u)
        update_degree(v)
        update_degree(w)

    # loop adding edges or contracting deg2 clusters
    while avail:
        if deg1:
            v = arbitrary_item(deg1)
            w = arbitrary_item(avail[v])
            add_match(v, w)
        elif deg2:
            v = arbitrary_item(deg2)
            contract(v)
        else:
            v = arbitrary_item(avail)
            w = arbitrary_item(avail[v])
            add_match(v, w)

    # at this point the edges listed in d2edges form a matchable tree
    # repeat the degree one part of the algorithm only on those edges
    avail = {}
    d2edges = [
        (u, v)
        for u, v in d2edges
        if u not in result_matching and v not in result_matching
    ]
    for u, v in d2edges:
        avail[u] = {}
        avail[v] = {}
    for u, v in d2edges:
        avail[u][v] = avail[v][u] = (u, v)
    deg1 = {
        v for v, neighbors in avail.items() if len(neighbors) == 1
    }  # pylint: disable=consider-using-dict-items  # pylint: disable=consider-using-dict-items
    while deg1:
        v = arbitrary_item(deg1)
        w = arbitrary_item(avail[v])
        add_match(v, w)

    return result_matching
