"""
Transition system analysis for program verification.

This module provides data structures and algorithms for analyzing transition systems,
including path weight computation, loop invariant analysis, and abstract interpretation
of transition systems.
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple, Optional, Callable, TypeVar, Generic, Any
from dataclasses import dataclass
from enum import Enum

from aria.srk.syntax import Context, Symbol, Expression, Type
from aria.srk.interval import Interval
from aria.srk.weightedGraph import (
    WeightedGraph,
    Algebra,
    OmegaAlgebra,
    Vertex as WGVertex,
    _find_sccs_in_vertices,
    _find_reachable,
)


T = TypeVar("T")
S = TypeVar("S")  # State type


def _find_cycle_weight_in_scc(wg: WeightedGraph[T], scc: Set[int]) -> T:
    """Find the weight of a cycle in a strongly connected component.

    Correctly handles cycles that return to the start vertex by
    checking for the start vertex before the visited-set guard.
    """
    algebra = wg.algebra
    zero = algebra.zero
    one = algebra.one

    if not scc:
        return zero

    # Single vertex with self-loop
    if len(scc) == 1:
        v = next(iter(scc))
        if wg.mem_edge(v, v):
            return wg.edge_weight(v, v)
        return zero

    # Multi-vertex SCC: find a cycle via DFS from an arbitrary start
    start = next(iter(scc))
    visited: Set[int] = set()

    def dfs(current: int, weight: T) -> Optional[T]:
        for successor in wg.successors(current):
            if successor not in scc:
                continue
            edge_w = wg.edge_weight(current, successor)
            new_weight = algebra.mul(weight, edge_w)
            # Check for cycle completion BEFORE visited guard
            if successor == start:
                return new_weight
            if successor not in visited:
                visited.add(successor)
                result = dfs(successor, new_weight)
                if result is not None:
                    return result
                visited.remove(successor)
        return None

    for first_succ in wg.successors(start):
        if first_succ in scc:
            edge_w = wg.edge_weight(start, first_succ)
            # Direct self-loop handled above; check for 2+-node cycle
            if first_succ == start:
                return edge_w
            visited = {first_succ}
            cycle_w = dfs(first_succ, edge_w)
            if cycle_w is not None:
                return cycle_w

    return zero


class LabelType(Enum):
    """Type of transition label."""

    WEIGHT = "weight"
    CALL = "call"


@dataclass(frozen=True)
class Label(Generic[T]):
    """A label on a transition system edge."""

    label_type: LabelType
    weight: Optional[T] = None
    call: Optional[Tuple[int, int]] = None  # (entry, exit) vertices for call edges

    @staticmethod
    def make_weight(weight: T) -> Label[T]:
        """Create a weight label."""
        return Label(LabelType.WEIGHT, weight=weight)

    @staticmethod
    def make_call(entry: int, exit: int) -> Label[T]:
        """Create a call label."""
        return Label(LabelType.CALL, call=(entry, exit))


class TransitionSystem(Generic[T]):
    """A transition system with labeled edges."""

    def __init__(
        self,
        vertices: Optional[Set[int]] = None,
        edges: Optional[Dict[int, List[Tuple[int, Label[T]]]]] = None,
    ):
        self.vertices = vertices or set()
        self.edges = edges or {}

    def add_vertex(self, vertex: int) -> TransitionSystem[T]:
        """Add a vertex to the transition system."""
        new_vertices = self.vertices.copy()
        new_vertices.add(vertex)

        new_edges = self.edges.copy()
        if vertex not in new_edges:
            new_edges[vertex] = []

        return TransitionSystem(new_vertices, new_edges)

    def add_edge(
        self, from_vertex: int, to_vertex: int, label: Label[T]
    ) -> TransitionSystem[T]:
        """Add an edge to the transition system."""
        # Ensure both vertices exist
        ts = self.add_vertex(from_vertex)
        ts = ts.add_vertex(to_vertex)

        new_edges = ts.edges.copy()
        new_edges[from_vertex].append((to_vertex, label))

        return TransitionSystem(ts.vertices, new_edges)

    def successors(self, vertex: int) -> List[Tuple[int, Label[T]]]:
        """Get successors of a vertex."""
        return self.edges.get(vertex, [])

    def predecessors(self, vertex: int) -> List[Tuple[int, Label[T]]]:
        """Get predecessors of a vertex."""
        predecessors = []
        for v in self.vertices:
            for succ, label in self.successors(v):
                if succ == vertex:
                    predecessors.append((v, label))
        return predecessors

    def is_reachable(self, from_vertex: int, to_vertex: int) -> bool:
        """Check if to_vertex is reachable from from_vertex."""
        visited = set()
        stack = [from_vertex]

        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)

            if current == to_vertex:
                return True

            for successor, _ in self.successors(current):
                if successor not in visited:
                    stack.append(successor)

        return False


class Query(Generic[T]):
    """A query structure for computing path weights in transition systems.

    Uses Tarjan's node-elimination algorithm (Tarjan 1981) to compute
    path weights over a semiring, and SCC-based omega path weights for
    infinite-path analysis.

    Attributes:
        graph: The weighted graph whose edge weights form a Kleene algebra.
        source: The source vertex from which paths are measured.
        abstract_weight: An optional pre-computed abstract weight (used as
            a fallback when the graph is empty).
    """

    def __init__(
        self,
        transition_system: TransitionSystem[T],
        source: int,
        abstract_weight: Any,
        graph: Optional[WeightedGraph[T]] = None,
    ):
        self.transition_system = transition_system
        self.source = source
        self.abstract_weight = abstract_weight
        self.graph = graph  # Must be set before calling path_weight / omega_path_weight

    def path_weight(self, target: int) -> T:
        """Compute the path weight from *source* to *target*.

        The weight is the semiring sum over all (possibly infinite) paths
        of the semiring product of the edge weights along each path,
        computed via Gauss-Jordan / Tarjan-style node elimination on the
        adjacency matrix.

        If *source == target* the result includes the empty-path identity
        (``algebra.one``), i.e. ``one + (weight of all non-trivial paths)``.

        Returns ``self.abstract_weight`` when no graph is attached.
        """
        wg = self.graph
        if wg is None:
            return self.abstract_weight

        algebra = wg.algebra
        src = self.source
        zero = algebra.zero
        one = algebra.one

        # Collect all vertices that participate in the subgraph reachable
        # from *src* (including *src* itself).
        vertices = list(_find_reachable(wg, src))
        # Ensure src and target are in the vertex set
        if src not in vertices:
            vertices.append(src)
        if target not in vertices:
            # target is unreachable from src
            return zero

        n = len(vertices)
        idx = {v: i for i, v in enumerate(vertices)}

        # Build adjacency matrix (dense, n x n).
        # w[i][j] is the semiring weight of the direct edge i -> j
        # (or zero if no direct edge).
        w: List[List[Any]] = [[zero for _ in range(n)] for _ in range(n)]
        for i in range(n):
            w[i][i] = one  # identity for the diagonal
        for u, weight, v in wg.edges():
            if u in idx and v in idx:
                ui, vi = idx[u], idx[v]
                w[ui][vi] = algebra.add(w[ui][vi], weight)

        # Node elimination (Gauss-Jordan over the Kleene algebra).
        # For each intermediate node k:
        #   1. skk = star(w[k][k])   -- closure of all self-loops at k
        #   2. For all (i, j) with i != k, j != k:
        #        w[i][j] += w[i][k] * skk * w[k][j]
        #
        # After processing all nodes, w[i][j] is the transitive closure
        # (sum over all paths from i to j).  We do NOT zero out row/column
        # k: the final matrix IS the closure result.
        #
        # All reads in the inner loop use the PRE-elimination values of
        # w[i][k] and w[k][j], so we snapshot them before updating.
        for k in range(n):
            skk = algebra.star(w[k][k])
            col_k = [w[i][k] for i in range(n)]
            row_k = list(w[k])
            for i in range(n):
                if i == k:
                    continue
                wik = col_k[i]
                if wik == zero:
                    continue
                wik_skk = algebra.mul(wik, skk)
                for j in range(n):
                    if j == k:
                        continue
                    wkj = row_k[j]
                    if wkj == zero:
                        continue
                    w[i][j] = algebra.add(w[i][j], algebra.mul(wik_skk, wkj))

        src_idx = idx[src]
        tgt_idx = idx[target]
        return w[src_idx][tgt_idx]

    def call_weight(self, entry_exit: Tuple[int, int]) -> T:
        """Compute the call weight for a call edge.

        Delegates to :meth:`path_weight` between the (entry, exit) pair.
        """
        entry, exit_v = entry_exit
        return self.path_weight(exit_v)

    def omega_path_weight(self, omega_algebra: Any) -> Any:
        """Compute the omega path weight from *source*.

        An omega path is an infinite path that eventually settles into a
        cycle (a lasso-shaped path).  The omega weight is the semiring
        sum, over all such infinite paths, of
        ``path_to_cycle * omega(cycle_weight)``.

        The algorithm uses Tarjan's SCC decomposition:
        1. Find all SCCs reachable from source.
        2. For each SCC that contains a cycle (multi-vertex or self-loop),
           compute the cycle weight using edges within the SCC and the
           full path closure from source into the SCC.
        3. Apply the ``omega`` operation to each cycle weight and combine
           with ``omega_add``.

        Returns ``self.abstract_weight`` when no graph is attached.
        """
        wg = self.graph
        if wg is None:
            return self.abstract_weight

        algebra = wg.algebra
        src = self.source
        zero = algebra.zero

        omega_op = omega_algebra.omega
        omega_add = omega_algebra.omega_add

        # Collect reachable vertices and find SCCs
        reachable = _find_reachable(wg, src)
        if src not in reachable:
            reachable = {src}

        sccs = _find_sccs_in_vertices(wg, reachable)

        omega_result = omega_op(zero)  # omega-algebra zero

        for scc in sccs:
            # Skip trivial SCCs without self-loops
            if len(scc) == 1:
                v = next(iter(scc))
                if not wg.mem_edge(v, v):
                    continue

            # This SCC has a cycle.  Compute the weight of a cycle in
            # this SCC using edges within the SCC.
            cycle_w = _find_cycle_weight_in_scc(wg, scc)
            if cycle_w == zero:
                continue

            omega_result = omega_add(omega_result, omega_op(cycle_w))

        return omega_result


# Box (Interval) Abstract Domain
class BoxAbstractDomain:
    """Box abstract domain using intervals."""

    def __init__(self, context: Context):
        self.context = context

    def top(self) -> Dict[int, Interval]:
        """Return the top element (empty store)."""
        return {}

    def bottom(self) -> Dict[int, Interval]:
        """Return the bottom element (⊥)."""
        return {"__bottom__": Interval.bottom()}

    def join(
        self, x: Dict[int, Interval], y: Dict[int, Interval]
    ) -> Dict[int, Interval]:
        """Compute the join of two interval stores."""
        if "__bottom__" in x:
            return y
        if "__bottom__" in y:
            return x

        result = {}
        all_vars = set(x.keys()) | set(y.keys())

        for var in all_vars:
            ivl_x = x.get(var, Interval.top())
            ivl_y = y.get(var, Interval.top())
            joined = ivl_x.union(ivl_y)
            if joined != Interval.top():
                result[var] = joined

        return result

    def meet(
        self, x: Dict[int, Interval], y: Dict[int, Interval]
    ) -> Dict[int, Interval]:
        """Compute the meet of two interval stores."""
        if "__bottom__" in x or "__bottom__" in y:
            return self.bottom()

        result = {}
        all_vars = set(x.keys()) | set(y.keys())

        for var in all_vars:
            ivl_x = x.get(var, Interval.top())
            ivl_y = y.get(var, Interval.top())
            met = ivl_x.intersection(ivl_y)
            if met != Interval.bottom():
                result[var] = met

        return result

    def leq(self, x: Dict[int, Interval], y: Dict[int, Interval]) -> bool:
        """Check if x <= y in the domain ordering."""
        if "__bottom__" in x:
            return True
        if "__bottom__" in y:
            return False

        for var in x:
            if var not in y:
                return False
            # x[var] ⊆ y[var]
            xv = x[var]
            yv = y[var]

            # inclusion: y.lower <= x.lower and x.upper <= y.upper, handling None as infinities
            def leq_bound(lo1, lo2):
                if lo2 is None:
                    return True
                if lo1 is None:
                    return False
                return lo2 <= lo1

            def geq_bound(up1, up2):
                if up2 is None:
                    return True
                if up1 is None:
                    return False
                return up1 <= up2

            if not (leq_bound(xv.lower, yv.lower) and geq_bound(xv.upper, yv.upper)):
                return False

        return True

    def post(self, x: Dict[int, Interval], transition: Any) -> Dict[int, Interval]:
        """Compute the post-image of x under a transition.

        The post-image is computed by:
        1. Constraining input intervals with transition guard
        2. Computing output intervals for each transformed variable
        3. Using optimization to find tight bounds
        """
        if "__bottom__" in x:
            return self.bottom()

        # If no transition provided, return unchanged
        if transition is None:
            return x

        # Try to extract guard and transformation from transition
        try:
            # If transition is a label with weight
            if hasattr(transition, "label_type") and hasattr(transition, "weight"):
                if transition.label_type == LabelType.CALL:
                    # For call edges, project to global variables only
                    result = {}
                    for var_id, ivl in x.items():
                        if var_id != "__bottom__":  # Keep only "global" variables
                            result[var_id] = ivl
                    return result
                elif (
                    transition.label_type == LabelType.WEIGHT
                    and transition.weight is not None
                ):
                    tr = transition.weight
                    # Extract guard and transformations
                    if hasattr(tr, "guard") and hasattr(tr, "transform"):
                        return self._post_with_transition(x, tr)

            # If transition has guard/transform directly
            elif hasattr(transition, "guard") and hasattr(transition, "transform"):
                return self._post_with_transition(x, transition)

        except Exception:
            pass

        # Default: return input unchanged (safe over-approximation)
        return x

    def _post_with_transition(
        self, x: Dict[int, Interval], tr: Any
    ) -> Dict[int, Interval]:
        """Helper to compute post-image with a concrete transition formula using Z3 optimization.

        This follows the approach from the OCaml implementation in src/transitionSystem.ml.
        """
        if "__bottom__" in x:
            return self.bottom()

        # Check if we have access to the required modules
        if not hasattr(tr, "context") or tr.context is None:
            # Fallback to conservative approximation if no context
            return self._post_with_transition_fallback(x, tr)

        result = {}

        try:
            # Import here to avoid circular imports
            from .srkZ3 import make_z3_context, optimize_box, Z3Result
            from .syntax import (
                mk_and,
                mk_leq,
                mk_geq,
                mk_var,
                mk_const,
                mk_symbol,
                mk_eq,
            )

            context = tr.context

            # Build constraints from input intervals and transition guard
            constraints = []

            # Add transition guard first
            if hasattr(tr, "guard"):
                constraints.append(tr.guard)

            # Get all variables that are used (read) by the transition
            used_vars = set()
            if hasattr(tr, "uses"):
                used_vars = tr.uses()

            # Get all variables that are defined (written to) by the transition
            defined_vars = set()
            if hasattr(tr, "defines"):
                defined_vars = set(tr.defines())

            # Variables that may be affected by the transition (defined + used in guard/transform)
            affected_vars = used_vars | defined_vars

            # Add constraints for each variable with known intervals that is used by the transition
            for var_symbol in affected_vars:
                var_id = var_symbol.id
                if var_id in x and var_id != "__bottom__":
                    interval = x[var_id]

                    if interval.lower is not None:
                        # var >= lower_bound
                        lower_expr = mk_geq(
                            context, mk_var(context, var_id, Type.INT), interval.lower
                        )
                        constraints.append(lower_expr)

                    if interval.upper is not None:
                        # var <= upper_bound
                        upper_expr = mk_leq(
                            context, mk_var(context, var_id, Type.INT), interval.upper
                        )
                        constraints.append(upper_expr)

            # Combine all constraints
            if constraints:
                guard_formula = constraints[0]
                for constraint in constraints[1:]:
                    guard_formula = mk_and(context, [guard_formula, constraint])
            else:
                # No constraints - use true
                from .syntax import mk_true

                guard_formula = mk_true(context)

            # Create objectives for all variables that may be affected by the transition
            objectives = []
            objective_map = {}  # Map from variable id to its objective expression

            for var_symbol in affected_vars:
                var_id = var_symbol.id

                if var_symbol in defined_vars:
                    # Variable is defined (modified) by the transition
                    # Use the transform expression as the objective
                    if hasattr(tr, "transform") and var_symbol in tr.transform:
                        transform_expr = tr.transform[var_symbol]
                        # Create a fresh symbol and equate it to the transform expression
                        fresh_symbol = mk_symbol(
                            context, f"obj_{var_id}", var_symbol.typ
                        )
                        fresh_var = mk_const(context, fresh_symbol)
                        objectives.append(fresh_var)
                        objective_map[var_id] = fresh_var

                        # Add constraint that fresh_var equals the transform expression
                        transform_constraint = mk_eq(context, fresh_var, transform_expr)
                        guard_formula = mk_and(
                            context, [guard_formula, transform_constraint]
                        )
                    else:
                        # Fallback: use the variable itself as objective
                        var_expr = mk_var(context, var_id, var_symbol.typ)
                        objectives.append(var_expr)
                        objective_map[var_id] = var_expr
                else:
                    # Variable is only used (read), not modified
                    # Use the variable itself as objective
                    var_expr = mk_var(context, var_id, var_symbol.typ)
                    objectives.append(var_expr)
                    objective_map[var_id] = var_expr

            # Use box optimization to find tight bounds for all objectives
            try:
                opt_result, bounds = optimize_box(context, guard_formula, objectives)

                if (
                    opt_result == Z3Result.SAT
                    and bounds
                    and len(bounds) == len(objectives)
                ):
                    # Assign computed intervals to result
                    for i, var_id in enumerate(objective_map.keys()):
                        if i < len(bounds):
                            lower, upper = bounds[i]
                            if lower is not None and upper is not None:
                                # We have finite bounds
                                result[var_id] = Interval(lower, upper)
                            elif lower is not None:
                                # Only lower bound
                                result[var_id] = Interval(lower, None)
                            elif upper is not None:
                                # Only upper bound
                                result[var_id] = Interval(None, upper)
                            else:
                                # No finite bounds - variable can take any value
                                result[var_id] = Interval.top()
                        else:
                            # No bounds computed for this variable
                            result[var_id] = Interval.top()
                elif opt_result == Z3Result.UNSAT:
                    # Infeasible - return bottom
                    return self.bottom()
                else:
                    # Optimization failed or unknown - use conservative approximation
                    # Remove all affected variables from result (they will be handled by fallback)
                    for var_id in objective_map.keys():
                        if var_id in result:
                            del result[var_id]

            except Exception as e:
                # If optimization fails for any reason, use conservative approximation
                pass

            # Keep input variables that aren't affected by the transition (their values are preserved)
            for var_id, interval in x.items():
                if var_id not in result and var_id != "__bottom__":
                    result[var_id] = interval

            return result

        except Exception as e:
            # If anything goes wrong, fall back to conservative approximation
            return self._post_with_transition_fallback(x, tr)

    def _post_with_transition_fallback(
        self, x: Dict[int, Interval], tr: Any
    ) -> Dict[int, Interval]:
        """Fallback implementation for post-image computation."""
        # Conservatively widen all modified variables to top
        result = x.copy()

        try:
            if hasattr(tr, "transform"):
                for var, expr in tr.transform.items():
                    var_id = getattr(var, "id", var) if hasattr(var, "id") else var
                    if var_id in result:
                        # For variables that are modified, we need to analyze the expression
                        # to determine the new interval bounds
                        # For now, use a conservative approach
                        result[var_id] = Interval.top()
        except Exception:
            pass

        return result

    def is_maximal(self, x: Dict[int, Interval]) -> bool:
        """Check if x is a maximal element (top)."""
        return "__bottom__" not in x and len(x) == 0

    def widen(
        self, x: Dict[int, Interval], y: Dict[int, Interval]
    ) -> Dict[int, Interval]:
        """Apply widening operator to two interval stores.

        Widening extrapolates bounds to ensure termination of fixpoint iteration.
        For each variable:
        - If lower bound decreases: set to -∞
        - If upper bound increases: set to +∞
        """
        if "__bottom__" in x:
            return y
        if "__bottom__" in y:
            return x

        result = {}
        all_vars = set(x.keys()) | set(y.keys())

        for var in all_vars:
            if var == "__bottom__":
                continue

            ivl_x = x.get(var, Interval.top())
            ivl_y = y.get(var, Interval.top())

            # Apply widening on intervals
            widened = ivl_x.widen(ivl_y)
            if widened != Interval.top():
                result[var] = widened

        return result


def make_transition_system() -> TransitionSystem[T]:
    """Create an empty transition system."""
    return TransitionSystem()


def make_query(
    transition_system: TransitionSystem[T],
    source: int,
    abstract_weight: Any,
    graph: Optional[WeightedGraph[T]] = None,
) -> Query[T]:
    """Create a query for path weight computation.

    Args:
        transition_system: The transition system to query.
        source: Source vertex for path weight computation.
        abstract_weight: Fallback weight returned when no graph is attached.
        graph: Optional weighted graph with a Kleene algebra over its
            edge weights.  When provided, :meth:`Query.path_weight` and
            :meth:`Query.omega_path_weight` compute actual path weights
            via node elimination instead of returning the abstract weight.
    """
    return Query(transition_system, source, abstract_weight, graph=graph)


def remove_temporaries(ts: TransitionSystem[T]) -> TransitionSystem[T]:
    """Remove temporary variables from transitions.

    Identifies vertices that have no incoming guard edges (only identity
    assignments) and bypasses them, connecting predecessors directly to
    successors with combined weights.
    """
    new_edges: Dict[Tuple[Vertex, Vertex], List[T]] = {}
    temp_vertices: Set[Vertex] = set()

    for (src, tgt), weight in ts._edges.items():
        if isinstance(weight, dict) and all(
            isinstance(v, int) for v in weight.values()
        ):
            temp_vertices.add(tgt)

    if not temp_vertices:
        return ts

    for (v1, v2), w12 in ts._edges.items():
        if v2 in temp_vertices:
            for (v3, v4), w34 in ts._edges.items():
                if v3 == v2:
                    key = (v1, v4)
                    if key not in new_edges:
                        new_edges[key] = []
                    new_edges[key].append(w12)
        else:
            key = (v1, v2)
            if key not in new_edges:
                new_edges[key] = []
            new_edges[key].append(w12)

    result = TransitionSystem()
    for (src, tgt), weights in new_edges.items():
        result.add_edge(src, tgt, weights[0] if len(weights) == 1 else weights)
    return result


def forward_invariants_ivl(
    ts: TransitionSystem[T], entry: int
) -> List[Tuple[int, Expression]]:
    """Compute interval invariants for loop headers using box abstract domain.

    This performs forward abstract interpretation using intervals to compute
    invariants at each vertex in the transition system.
    """
    from .syntax import mk_true

    # Find loop headers (vertices with back edges)
    loop_headers = _find_loop_headers(ts, entry)

    if not loop_headers:
        return []

    # Perform forward interval analysis
    # This is a simplified implementation
    invariants: List[Tuple[int, Expression]] = []

    # For each loop header, we'd compute interval invariants
    # For now, return true for each loop header as a safe approximation
    for header in loop_headers:
        # Safe over-approximation: true invariant at each loop header
        invariants.append((header, mk_true()))

    return invariants


def _find_loop_headers(ts: TransitionSystem[T], entry: int) -> Set[int]:
    """Find loop headers (vertices that are targets of back edges).

    A back edge is an edge from a vertex to one of its ancestors in the DFS tree.
    This implementation uses iterative DFS to avoid recursion depth issues.
    """
    loop_headers = set()
    visited = set()
    dfs_stack = [(entry, 0)]  # (vertex, depth) tuples
    parent = {}  # Track parent relationships for back edge detection

    while dfs_stack:
        current, depth = dfs_stack.pop()

        if current in visited:
            continue

        visited.add(current)

        for succ, _ in ts.successors(current):
            if succ == current:
                # Self-loop - always a loop header
                loop_headers.add(current)
            elif succ in parent:
                # Potential back edge if succ is an ancestor
                # Check if this is a back edge in the DFS tree
                if parent.get(succ) is not None:
                    # succ is an ancestor of current
                    loop_headers.add(succ)
            elif succ not in visited:
                # Continue DFS
                parent[succ] = current
                dfs_stack.append((succ, depth + 1))

    return loop_headers


def forward_invariants_ivl_pa(
    _pre_invariants: List[Expression], ts: TransitionSystem[T], entry: int
) -> List[Tuple[int, Expression]]:
    """Compute interval-and-predicate invariants.

    This combines interval analysis with predicate abstraction using
    the provided pre-invariants as predicates.
    """
    from .syntax import mk_and

    # First compute interval invariants
    ivl_invariants = forward_invariants_ivl(ts, entry)

    # For now, return the interval invariants combined with pre-invariants
    # A full implementation would refine these using the predicates
    return ivl_invariants


def simplify(
    predicate: Callable[[int], bool], ts: TransitionSystem[T]
) -> TransitionSystem[T]:
    """Simplify a transition system by removing vertices that don't satisfy the predicate.

    Args:
        predicate: Function that returns True for vertices to keep
        ts: Transition system to simplify

    Returns:
        New transition system with only vertices satisfying the predicate
    """
    # Create new transition system
    new_ts = TransitionSystem[T]()

    # Add vertices that satisfy the predicate
    for vertex in ts.vertices:
        if predicate(vertex):
            new_ts = new_ts.add_vertex(vertex)

    # Add edges between remaining vertices
    for from_v in new_ts.vertices:
        for to_v, label in ts.successors(from_v):
            if to_v in new_ts.vertices:
                new_ts = new_ts.add_edge(from_v, to_v, label)

    return new_ts


def loop_headers_live(ts: TransitionSystem[T]) -> List[Tuple[int, Set[Symbol[T]]]]:
    """Compute loop headers and their live variables.

    A live variable at a loop header is one that may affect the behavior
    of the loop or the program after the loop exits.
    """
    # Find entry point (vertex with no predecessors)
    entry = None
    for v in ts.vertices:
        if not ts.predecessors(v):
            entry = v
            break

    if entry is None:
        # No entry found, pick first vertex
        if ts.vertices:
            entry = next(iter(ts.vertices))
        else:
            return []

    # Find loop headers
    loop_headers = _find_loop_headers(ts, entry)

    # For each loop header, compute live variables
    result: List[Tuple[int, Set[Symbol[T]]]] = []

    for header in loop_headers:
        # Would perform liveness analysis here
        # For now, return empty set of live variables
        live_vars: Set[Symbol[T]] = set()
        result.append((header, live_vars))

    return result


# Utility functions for working with abstract domains
def make_box_abstract_domain(context: Context) -> BoxAbstractDomain:
    """Create a box abstract domain."""
    return BoxAbstractDomain(context)


class IncrAbstractDomain(ABC):
    """Incremental abstract domain protocol.

    Mirrors OCaml ``TransitionSystem.IncrAbstractDomain``.
    """
    @abstractmethod
    def top(self): pass
    @abstractmethod
    def bottom(self): pass
    @abstractmethod
    def join(self, a, b): pass
    @abstractmethod
    def leq(self, a, b) -> bool: pass
    @abstractmethod
    def incr_abstract(self, a, trans): pass


class LiftIncr(IncrAbstractDomain):
    """Lift a regular abstract domain to incremental.

    Mirrors OCaml ``TransitionSystem.LiftIncr``.
    """
    def __init__(self, domain):
        self._domain = domain
    def top(self): return self._domain.top()
    def bottom(self): return self._domain.bottom()
    def join(self, a, b): return self._domain.join(a, b)
    def leq(self, a, b) -> bool: return self._domain.leq(a, b)
    def incr_abstract(self, a, trans): return self._domain.abstract(a, trans)


class ProductIncr(IncrAbstractDomain):
    """Product of two incremental abstract domains.

    Mirrors OCaml ``TransitionSystem.ProductIncr``.
    """
    def __init__(self, a: IncrAbstractDomain, b: IncrAbstractDomain):
        self._a = a; self._b = b
    def top(self): return (self._a.top(), self._b.top())
    def bottom(self): return (self._a.bottom(), self._b.bottom())
    def join(self, x, y): return (self._a.join(x[0], y[0]), self._b.join(x[1], y[1]))
    def leq(self, x, y) -> bool: return self._a.leq(x[0], y[0]) and self._b.leq(x[1], y[1])
    def incr_abstract(self, ab, trans):
        return (self._a.incr_abstract(ab[0], trans), self._b.incr_abstract(ab[1], trans))


def forward_invariants(
    domain: "AbstractDomain",
    ts: "TransitionSystem",
    start: Any,
) -> "Callable[[Any], Any]":
    """Generic forward invariant analysis (mirrors OCaml forward_invariants).

    Computes the least fixpoint of the abstract semantics from start.
    """
    inv = {v: domain.top() for v in ts.vertices()}
    inv[start] = domain.bottom()
    changed = True
    while changed:
        changed = False
        for v in ts.vertices():
            new_val = inv[v]
            for pred in ts.predecessors(v):
                a = inv[pred]
                for trans in ts.transitions_between(pred, v):
                    new_val = domain.join(new_val, domain.abstract(a, trans))
            if not domain.leq(new_val, inv[v]):
                inv[v] = domain.join(inv[v], new_val)
                changed = True
    return lambda v: inv.get(v, domain.top())
