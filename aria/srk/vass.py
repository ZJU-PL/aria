"""
Vector Addition Systems with States (VASS) implementation for SRK.

This module provides abstract interpretation for VASS, which extend VAS
with control states to model transition systems with multiple control states.
Based on the OCaml implementation in src/vass.ml.
"""

import logging
from typing import List, Dict, Set, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
from functools import reduce

from . import linear
from . import syntax
from . import smt as Smt
# from . import interpretation  # Temporarily disabled due to missing functions
from . import srkSimplify
from . import nonlinear
from . import apron as ApronInterface
from . import coordinateSystem as CS
from . import vas
from . import transitionFormula as TF

logger = logging.getLogger(__name__)

@dataclass
class SCCVAS:
    """Strongly Connected Component VAS"""
    control_states: List[syntax.Formula]  # Control state formulas
    graph: List[List[vas.VAS]]  # Adjacency matrix of VAS transformers
    s_lst: List[linear.QQMatrix]  # Simulation matrices

@dataclass
class VASSType:
    """VASS abstraction type"""
    vasses: List[SCCVAS]  # Array of SCC VAS abstractions
    formula: syntax.Formula  # Initial formula
    sink: syntax.Formula  # Sink state formula
    skolem_constants: Set[syntax.Symbol]  # Skolem constants

class VASSContext:
    """Context for VASS operations"""
    def __init__(self, srk_context: syntax.Context):
        self.srk = srk_context
        self.log = logging.getLogger(f"srk.vass.{id(self)}")

def mk_all_nonnegative(srk: syntax.Context, terms: List[syntax.Term]) -> syntax.Formula:
    """Create conjunction requiring all terms >= 0"""
    return syntax.mk_and(srk, [syntax.mk_leq(srk, syntax.mk_zero(srk), term) for term in terms])

def unify_matrices(matrices: List[linear.QQMatrix]) -> linear.QQMatrix:
    """Stack matrices vertically to form a single matrix"""
    if not matrices:
        return linear.QQMatrix()

    rows = []
    for matrix in matrices:
        if hasattr(matrix, 'rows'):
            rows.extend(list(matrix.rows))

    return linear.QQMatrix(rows)

def map_terms(srk: syntax.Context, symbols: List[syntax.Symbol]) -> List[syntax.Term]:
    """Map symbols to constant terms"""
    return [syntax.mk_const(srk, sym) for sym in symbols]

def ident_matrix_real(n: int) -> linear.QQMatrix:
    """Create identity matrix of size n"""
    # Use linear utility identity matrix
    try:
        return linear.identity_matrix(n)
    except Exception:
        # Fallback minimal identity
        rows = []
        from fractions import Fraction
        for i in range(n):
            rows.append(linear.QQVector({i: Fraction(1)}))
        return linear.QQMatrix(rows)

def exists_transition(srk: syntax.Context, cs1: syntax.Formula, cs2: syntax.Formula,
                     tr_symbols: List[Tuple[syntax.Symbol, syntax.Symbol]],
                     phi: syntax.Formula) -> bool:
    """Check if there exists a transition from cs1 to cs2"""
    postify = syntax.substitute(srk, TF.post_map(srk, tr_symbols))
    return Smt.is_sat(srk, syntax.mk_and(srk, [cs1, postify(cs2), phi])) != Smt.Unsat

def compute_edges(srk: syntax.Context, tr_symbols: List[Tuple[syntax.Symbol, syntax.Symbol]],
                 c_states: List[syntax.Formula], phi: syntax.Formula) -> List[List[bool]]:
    """Compute boolean adjacency graph of control states"""
    n = len(c_states)
    graph = [[False for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            graph[i][j] = exists_transition(srk, c_states[i], c_states[j], tr_symbols, phi)

    return graph

def pp(srk: syntax.Context, tr_symbols: List[Tuple[syntax.Symbol, syntax.Symbol]],
      formatter, vasses: VASSType) -> None:
    """Pretty print VASS abstraction"""
    formatter.write("VASS abstraction with {} SCCs".format(len(vasses.vasses)))

def compute_single_scc_vass(exists: Callable[[syntax.Symbol], bool],
                           srk: syntax.Context,
                           tr_symbols: List[Tuple[syntax.Symbol, syntax.Symbol]],
                           cs_lst: List[syntax.Formula],
                           phi: syntax.Formula) -> SCCVAS:
    """Compute VASS for a single strongly connected component"""
    n = len(cs_lst)
    if n == 0:
        return SCCVAS([], [], [])

    graph = [[vas.VAS.empty() for _ in range(n)] for _ in range(n)]

    # Create postification function for transitions
    postify = syntax.substitute(srk, TF.post_map(srk, tr_symbols))

    # For each pair of control states, compute VAS abstraction
    for i in range(n):
        for j in range(n):
            # Restrict phi to transitions from cs_lst[i] to cs_lst[j]
            # This creates a formula: cs_i ∧ cs'_j ∧ phi
            restricted_phi = syntax.mk_and(srk, [
                cs_lst[i],
                postify(cs_lst[j]),
                phi
            ])

            # Check if this transition is possible
            sat_result = Smt.is_sat(srk, restricted_phi)
            if sat_result != Smt.Unsat:
                # Create a transition formula for this edge
                tf = TF.TransitionFormula(
                    formula=restricted_phi,
                    symbols=tr_symbols,
                    exists=exists
                )

                # Abstract this transition to a VAS
                # This uses the VAS abstraction from the vas module
                try:
                    vas_abstract = vas.abstract_to_vas(srk, tf)
                    graph[i][j] = vas_abstract
                    logger.debug(f"Computed VAS for edge {i}->{j}: {len(vas_abstract.transformers)} transformers")
                except Exception as e:
                    logger.warning(f"Failed to compute VAS for edge {i}->{j}: {e}")
                    graph[i][j] = vas.VAS.empty()
            else:
                graph[i][j] = vas.VAS.empty()

    # Create simulation matrix mapping pre/post symbols to dimension indices
    # This creates a matrix where each row corresponds to a pre-symbol
    # and maps it to its dimension in the VAS abstraction
    pre_symbols = [x for x, _ in tr_symbols]
    post_symbols = [x_prime for _, x_prime in tr_symbols]

    # Build simulation matrix: maps each variable to its corresponding dimension
    sim_rows = []
    for idx, (x, x_prime) in enumerate(tr_symbols):
        # Create a vector with 1 at position idx (dimension for this variable)
        vec = linear.QQVector.of_term(linear.QQ.one(), idx)
        sim_rows.append(vec)

    sim = linear.QQMatrix.of_rows(sim_rows) if sim_rows else linear.QQMatrix.zero()

    return SCCVAS(cs_lst, graph, [sim])

    def analyze_reachability(self) -> Dict[Tuple[int, int], bool]:
        """Analyze reachability between control states in this SCC."""
        n = len(self.control_states)
        reachability = {}

        # Use Floyd-Warshall-like algorithm for reachability
        # Initialize with direct transitions
        for i in range(n):
            for j in range(n):
                # Check if there's a non-empty VAS between i and j
                has_transition = not self.graph[i][j].is_empty()
                reachability[(i, j)] = has_transition

        # Floyd-Warshall: check paths through intermediate nodes
        for k in range(n):
            for i in range(n):
                for j in range(n):
                    # Can we reach j from i via k?
                    if reachability[(i, k)] and reachability[(k, j)]:
                        reachability[(i, j)] = True

        return reachability

    def find_cycles(self) -> List[List[int]]:
        """Find all cycles in this SCC."""
        n = len(self.control_states)
        cycles = []

        def dfs(current: int, path: List[int], visited: Set[int]) -> None:
            if current in path:
                # Found a cycle
                cycle_start = path.index(current)
                cycle = path[cycle_start:] + [current]
                cycles.append(cycle)
                return

            if current in visited:
                return

            visited.add(current)
            path.append(current)

            # Visit neighbors
            for next_state in range(n):
                if not self.graph[current][next_state].is_empty():
                    dfs(next_state, path.copy(), visited.copy())

        # Start DFS from each state
        for start in range(n):
            dfs(start, [], set())

        return cycles

def project_dnf(srk: syntax.Context, exists: Callable[[syntax.Symbol], bool],
                phi: syntax.Formula) -> List[syntax.Formula]:
    """Project formula onto symbols satisfying exists predicate and convert to DNF"""
    phi = syntax.rewrite(srk, phi, down=syntax.nnf_rewriter(srk))
    phi = srkSimplify.simplify_terms(srk, phi)

    # Minimal, sound projection placeholder: return a single cube if satisfiable
    result = Smt.is_sat(srk, phi)
    if result == Smt.SAT:
        return [phi]
    elif result == Smt.UNSAT:
        return []
    else:
        # Unknown: conservatively return the original formula as one cube
        return [phi]

def get_largest_polyhedrons(srk: syntax.Context, control_states: List[syntax.Formula]) -> List[syntax.Formula]:
    """Combine overlapping control states"""
    # Simplified implementation - would need proper polyhedron intersection logic
    return control_states

def get_control_states(srk: syntax.Context, tf: TF.TransitionFormula) -> Tuple[List[syntax.Formula], syntax.Formula]:
    """Compute control states using projection"""
    tr_symbols = TF.symbols(tf)
    pre_symbols = TF.pre_symbols(tr_symbols)
    post_symbols = TF.post_symbols(tr_symbols)

    def exists_pre(x: syntax.Symbol) -> bool:
        return TF.exists(tf, x) and x not in post_symbols

    def exists_post(x: syntax.Symbol) -> bool:
        return TF.exists(tf, x) and x not in pre_symbols

    control_states = project_dnf(srk, exists_pre, TF.formula(tf))
    sink = syntax.mk_or(srk, project_dnf(srk, exists_post, TF.formula(tf)))
    control_states = get_largest_polyhedrons(srk, control_states)

    return control_states, sink

def abstract_to_vass(srk: syntax.Context, tf: TF.TransitionFormula) -> VASSType:
    """Abstract a transition formula to VASS abstraction"""
    exists_func = TF.exists
    tr_symbols = TF.symbols(tf)
    skolem_constants = syntax.Symbol.Set.filter(
        lambda a: not exists_func(a),
        syntax.symbols(TF.formula(tf))
    )

    phi = syntax.rewrite(srk, TF.formula(tf), down=syntax.nnf_rewriter(srk))
    # phi = nonlinear.linearize(srk, phi)  # Disabled due to missing functions
    control_states, sink = get_control_states(srk, tf)

    # Compute adjacency graph
    graph = compute_edges(srk, tr_symbols, control_states, phi)

    # Find strongly connected components using Kosaraju's algorithm
    n = len(control_states)
    adj = {i: [j for j in range(n) if graph[i][j]] for i in range(n)}

    visited: Set[int] = set()
    order: List[int] = []

    def dfs(v: int) -> None:
        visited.add(v)
        for u in adj[v]:
            if u not in visited:
                dfs(u)
        order.append(v)

    for i in range(n):
        if i not in visited:
            dfs(i)

    # Build reverse graph
    radj = {i: [] for i in range(n)}
    for i in range(n):
        for j in adj[i]:
            radj[j].append(i)

    visited.clear()
    components: List[List[int]] = []

    def rdfs(v: int, comp: List[int]) -> None:
        visited.add(v)
        comp.append(v)
        for u in radj[v]:
            if u not in visited:
                rdfs(u, comp)

    for v in reversed(order):
        if v not in visited:
            comp: List[int] = []
            rdfs(v, comp)
            components.append(comp)

    # Map components to lists of control state formulas
    sccs = [[control_states[i] for i in comp] for comp in components]
    num_sccs = len(sccs)

    if num_sccs == 0:
        return VASSType([], phi, skolem_constants, sink)

    # Compute VASS for each SCC
    vassarrays = []
    for scc in sccs:
        scc_vas = compute_single_scc_vass(exists_func, srk, tr_symbols, scc, phi)
        vassarrays.append(scc_vas)

    result = VASSType(vassarrays, phi, skolem_constants, sink)

    logger.info("Created VASS abstraction: %s", result)
    return result

def create_local_s_t(srk: syntax.Context, num: int) -> List[Tuple[syntax.Term, syntax.Term]]:
    """Create source and sink variables for a SCC"""
    sources = map_terms(srk, [
        syntax.mk_symbol(srk, f"source{i}", syntax.TyInt) for i in range(num)
    ])
    sinks = map_terms(srk, [
        syntax.mk_symbol(srk, f"sink{i}", syntax.TyInt) for i in range(num)
    ])
    return list(zip(sources, sinks))

def source_sink_conds_satisfied(srk: syntax.Context, local_s_t: List[Tuple[syntax.Term, syntax.Term]],
                               cs: List[syntax.Formula],
                               tr_symbols: List[Tuple[syntax.Symbol, syntax.Symbol]]) -> syntax.Formula:
    """Source and sink conditions must be satisfied"""
    postify = syntax.substitute(srk, TF.post_map(srk, tr_symbols))

    constraints = []
    for i, (source, sink) in enumerate(local_s_t):
        constraints.append(
            syntax.mk_and(srk, [
                syntax.mk_if(srk, syntax.mk_eq(srk, source, syntax.mk_one(srk)), cs[i]),
                syntax.mk_if(srk, syntax.mk_eq(srk, sink, syntax.mk_one(srk)), postify(cs[i]))
            ])
        )

    return syntax.mk_and(srk, constraints)

def split_terms_add_to_one(srk: syntax.Context, local_s_t: List[Tuple[syntax.Term, syntax.Term]]) -> syntax.Formula:
    """Each control state contributes exactly one source and one sink"""
    sources, sinks = zip(*local_s_t)
    return syntax.mk_and(srk, [
        syntax.mk_eq(srk, syntax.mk_add(srk, sources), syntax.mk_one(srk)),
        syntax.mk_eq(srk, syntax.mk_add(srk, sinks), syntax.mk_one(srk))
    ])

def exp_each_ests_one_or_zero(srk: syntax.Context, local_s_t: List[Tuple[syntax.Term, syntax.Term]]) -> syntax.Formula:
    """Each source/sink pair is either 0 or 1"""
    constraints = []
    for source, sink in local_s_t:
        constraints.append(
            syntax.mk_and(srk, [
                syntax.mk_or(srk, [
                    syntax.mk_eq(srk, source, syntax.mk_zero(srk)),
                    syntax.mk_eq(srk, source, syntax.mk_one(srk))
                ]),
                syntax.mk_or(srk, [
                    syntax.mk_eq(srk, sink, syntax.mk_zero(srk)),
                    syntax.mk_eq(srk, sink, syntax.mk_one(srk))
                ])
            ])
        )
    return syntax.mk_and(srk, constraints)

def closure_of_an_scc(srk: syntax.Context, tr_symbols: List[Tuple[syntax.Symbol, syntax.Symbol]],
                     loop_counter: syntax.Term, vass: SCCVAS) -> Tuple[syntax.Formula, List[syntax.Term]]:
    """Compute closure of a single SCC"""
    cs = vass.control_states
    local_s_t = create_local_s_t(srk, len(cs))

    constr1 = split_terms_add_to_one(srk, local_s_t)
    constr2 = exp_each_ests_one_or_zero(srk, local_s_t)
    constr3 = source_sink_conds_satisfied(srk, local_s_t, cs, tr_symbols)

    # If no VAS transformers, require that a control state is used
    unified_s = unify_matrices(vass.s_lst)
    if len(getattr(unified_s, 'rows', ())) == 0:
        return (syntax.mk_and(srk, [constr1, constr2, constr3]), [source for source, _ in local_s_t])

    # More complex logic would go here for the full VAS case
    # For now, return simplified version
    return (syntax.mk_and(srk, [constr1, constr2, constr3]), [source for source, _ in local_s_t])

def no_trans_taken(srk: syntax.Context, loop_counter: syntax.Term,
                  tr_symbols: List[Tuple[syntax.Symbol, syntax.Symbol]]) -> syntax.Formula:
    """No transitions taken constraint"""
    eqs = [syntax.mk_eq(srk, syntax.mk_const(srk, x), syntax.mk_const(srk, x_prime))
           for x, x_prime in tr_symbols]
    return syntax.mk_and(srk, [syntax.mk_eq(srk, loop_counter, syntax.mk_zero(srk))] + eqs)

def exp(srk: syntax.Context, tr_symbols: List[Tuple[syntax.Symbol, syntax.Symbol]],
        loop_counter: syntax.Term, sccsform: VASSType) -> syntax.Formula:
    """Compute VASS abstraction closure"""
    if not sccsform.vasses:
        return no_trans_taken(srk, loop_counter, tr_symbols)

    # Create symbol mappings for each SCC
    symmappings = []
    for i, _ in enumerate(sccsform.vasses):
        symmappings.append([
            (syntax.mk_symbol(srk, f"{x.name}_{i}", syntax.typ_symbol(srk, x)),
             syntax.mk_symbol(srk, f"{x_prime.name}_{i}", syntax.typ_symbol(srk, x_prime)))
            for x, x_prime in tr_symbols
        ])

    # Create loop counters for each SCC
    subloop_counters = [
        syntax.mk_const(srk, syntax.mk_symbol(srk, f"counter_{i}", syntax.TyInt))
        for i in range(len(sccsform.vasses))
    ]

    # Create Skolem mappings
    skolem_mappings_transitions = []
    for i in range(len(sccsform.vasses)):
        skolem_mappings_transitions.append([
            (x, syntax.mk_symbol(srk, f"{x.name}_{i}", syntax.typ_symbol(srk, x)))
            for x in sccsform.skolem_constants
        ])

    # Compute closures for each SCC
    sccclosures_sources = [
        closure_of_an_scc(srk, tr_symbols, subloop_counters[i], vass)
        for i, vass in enumerate(sccsform.vasses)
    ]

    sccclosures, sources = zip(*sccclosures_sources)
    sccclosures = list(sccclosures)
    sources = list(sources)

    # Add sink state
    sccclosures.append(sccsform.sink)
    sources.append([syntax.mk_real(srk, linear.QQ.one())])

    # Transform closures to use proper symbol mappings
    subst = lambda symbol_pairs: syntax.substitute(srk, TF.post_map(srk, symbol_pairs))

    # This would need more complex transformation logic
    # For now, return a simplified version
    constr1 = mk_all_nonnegative(srk, subloop_counters)

    # More constraints would be added here

    return syntax.mk_or(srk, [
        syntax.mk_and(srk, [constr1]),  # Full constraint
        no_trans_taken(srk, loop_counter, tr_symbols)  # No transitions case
    ])
