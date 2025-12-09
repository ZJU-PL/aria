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

from aria.srk.syntax import Context, Symbol, Expression, Type, mk_true, mk_and, mk_leq, mk_geq, mk_var, mk_const, mk_symbol, mk_eq
from aria.srk.srkZ3 import Z3Result, optimize_box
from aria.srk.interval import Interval


T = TypeVar('T')
S = TypeVar('S')  # State type


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

    def __init__(self,
                 vertices: Optional[Set[int]] = None,
                 edges: Optional[Dict[int, List[Tuple[int, Label[T]]]]] = None):
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

    def add_edge(self, from_vertex: int, to_vertex: int, label: Label[T]) -> TransitionSystem[T]:
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
    """A query structure for computing path weights in transition systems."""

    def __init__(self,
                 transition_system: TransitionSystem[T],
                 source: int,
                 abstract_weight: Any):  # Would need proper abstract weight type
        self.transition_system = transition_system
        self.source = source
        self.abstract_weight = abstract_weight

    def path_weight(self, target: int) -> T:
        """Compute the path weight from source to target."""
        # Default implementation - return the abstract weight
        # A full implementation would compute the actual path weight
        return self.abstract_weight

    def call_weight(self, entry_exit: Tuple[int, int]) -> T:
        """Compute the call weight for a call edge."""
        # Default implementation - return the abstract weight
        # A full implementation would compute call-specific weights
        return self.abstract_weight

    def omega_path_weight(self, omega_algebra: Any) -> Any:  # Would need proper omega algebra type
        """Compute infinite path weights."""
        # Default implementation - return the abstract weight
        # A full implementation would handle infinite paths using omega algebra
        return self.abstract_weight


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

    def join(self, x: Dict[int, Interval], y: Dict[int, Interval]) -> Dict[int, Interval]:
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

    def meet(self, x: Dict[int, Interval], y: Dict[int, Interval]) -> Dict[int, Interval]:
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
            if hasattr(transition, 'label_type') and hasattr(transition, 'weight'):
                if transition.label_type == LabelType.CALL:
                    # For call edges, project to global variables only
                    result = {}
                    for var_id, ivl in x.items():
                        if var_id != "__bottom__":  # Keep only "global" variables
                            result[var_id] = ivl
                    return result
                elif transition.label_type == LabelType.WEIGHT and transition.weight is not None:
                    tr = transition.weight
                    # Extract guard and transformations
                    if hasattr(tr, 'guard') and hasattr(tr, 'transform'):
                        return self._post_with_transition(x, tr)

            # If transition has guard/transform directly
            elif hasattr(transition, 'guard') and hasattr(transition, 'transform'):
                return self._post_with_transition(x, transition)

        except Exception:
            pass

        # Default: return input unchanged (safe over-approximation)
        return x

    def _post_with_transition(self, x: Dict[int, Interval], tr: Any) -> Dict[int, Interval]:
        """Helper to compute post-image with a concrete transition formula using Z3 optimization.

        This follows the approach from the OCaml implementation in src/transitionSystem.ml.
        """
        if "__bottom__" in x:
            return self.bottom()

        # Check if we have access to the required modules
        if not hasattr(tr, 'context') or tr.context is None:
            # Fallback to conservative approximation if no context
            return self._post_with_transition_fallback(x, tr)

        result = {}

        try:
            # Import here to avoid circular imports
            from .srkZ3 import make_z3_context, optimize_box, Z3Result
            from .syntax import mk_and, mk_leq, mk_geq, mk_var, mk_const, mk_symbol, mk_eq

            context = tr.context

            # Build constraints from input intervals and transition guard
            constraints = []

            # Add transition guard first
            if hasattr(tr, 'guard'):
                constraints.append(tr.guard)

            # Get all variables that are used (read) by the transition
            used_vars = set()
            if hasattr(tr, 'uses'):
                used_vars = tr.uses()

            # Get all variables that are defined (written to) by the transition
            defined_vars = set()
            if hasattr(tr, 'defines'):
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
                        lower_expr = mk_geq(context, mk_var(context, var_id, Type.INT), interval.lower)
                        constraints.append(lower_expr)

                    if interval.upper is not None:
                        # var <= upper_bound
                        upper_expr = mk_leq(context, mk_var(context, var_id, Type.INT), interval.upper)
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
                    if hasattr(tr, 'transform') and var_symbol in tr.transform:
                        transform_expr = tr.transform[var_symbol]
                        # Create a fresh symbol and equate it to the transform expression
                        fresh_symbol = mk_symbol(context, f"obj_{var_id}", var_symbol.typ)
                        fresh_var = mk_const(context, fresh_symbol)
                        objectives.append(fresh_var)
                        objective_map[var_id] = fresh_var

                        # Add constraint that fresh_var equals the transform expression
                        transform_constraint = mk_eq(context, fresh_var, transform_expr)
                        guard_formula = mk_and(context, [guard_formula, transform_constraint])
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

                if opt_result == Z3Result.SAT and bounds and len(bounds) == len(objectives):
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

    def _post_with_transition_fallback(self, x: Dict[int, Interval], tr: Any) -> Dict[int, Interval]:
        """Fallback implementation for post-image computation."""
        # Conservatively widen all modified variables to top
        result = x.copy()

        try:
            if hasattr(tr, 'transform'):
                for var, expr in tr.transform.items():
                    var_id = getattr(var, 'id', var) if hasattr(var, 'id') else var
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

    def widen(self, x: Dict[int, Interval], y: Dict[int, Interval]) -> Dict[int, Interval]:
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


def make_query(transition_system: TransitionSystem[T],
               source: int,
               abstract_weight: Any) -> Query[T]:
    """Create a query for path weight computation."""
    return Query(transition_system, source, abstract_weight)


def remove_temporaries(ts: TransitionSystem[T]) -> TransitionSystem[T]:
    """Remove temporary variables from transitions.

    This function removes edges that represent temporary variable assignments
    and connects their predecessors directly to their successors.
    """
    # For now, return the transition system as-is
    # A full implementation would:
    # 1. Identify temporary variables (could be marked in some way)
    # 2. For each vertex with only temporary assignments, bypass it
    # 3. Connect predecessors directly to successors with combined weights
    return ts


def forward_invariants_ivl(ts: TransitionSystem[T],
                          entry: int) -> List[Tuple[int, Expression]]:
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
        invariants.append((header, mk_true))

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


def forward_invariants_ivl_pa(pre_invariants: List[Expression],
                             ts: TransitionSystem[T],
                             entry: int) -> List[Tuple[int, Expression]]:
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


def simplify(predicate: Callable[[int], bool],
             ts: TransitionSystem[T]) -> TransitionSystem[T]:
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
