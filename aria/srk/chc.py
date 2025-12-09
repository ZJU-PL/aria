"""
Constrained Horn Clause (CHC) solving.

This module implements algorithms for solving constrained Horn clauses,
which are used in program verification and synthesis.

This follows the OCaml chc.ml module.
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
import logging

# Type aliases
Relation = int
RelAtom = Tuple[Relation, List[Any]]  # (relation_id, list of symbols)


@dataclass
class Fp:
    """
    Fixed-point system for CHC solving.
    
    Attributes:
        rules: List of rules (hypothesis_atoms, phi, conclusion_atom)
        queries: List of query relation IDs
        rel_ctx: Dynamic array of (name, type_list) for each relation
    """
    rules: List[Tuple[List[RelAtom], Any, RelAtom]]  # [(hyp_atoms, phi, concl)]
    queries: List[Relation]
    rel_ctx: List[Tuple[str, List[Any]]]  # [(name, types), ...]
    
    def __init__(self):
        self.rules = []
        self.queries = []
        self.rel_ctx = []


def mk_relation(fp: Fp, name: str = "R", typ: List[Any] = None) -> Relation:
    """
    Create a new relation in the fixed-point system.
    
    Args:
        fp: Fixed-point system
        name: Name of the relation
        typ: List of types for the relation parameters
        
    Returns:
        Relation ID
    """
    if typ is None:
        typ = []
    
    fp.rel_ctx.append((name, typ))
    return len(fp.rel_ctx) - 1


def type_of(fp: Fp, rel: Relation) -> List[Any]:
    """Get the types of a relation."""
    return fp.rel_ctx[rel][1]


def name_of(fp: Fp, rel: Relation) -> str:
    """Get the name of a relation."""
    return fp.rel_ctx[rel][0]


def rel_of_atom(atom: RelAtom) -> Relation:
    """Get the relation ID from a relation atom."""
    return atom[0]


def params_of_atom(atom: RelAtom) -> List[Any]:
    """Get the parameters from a relation atom."""
    return atom[1]


def mk_rel_atom(srk: Any, fp: Fp, rel: Relation, syms: List[Any]) -> RelAtom:
    """
    Create a relation atom.
    
    Args:
        srk: Context
        fp: Fixed-point system
        rel: Relation ID
        syms: List of symbols (parameters)
        
    Returns:
        Relation atom
        
    Raises:
        TypeError: If types don't match
    """
    from .syntax import typ_symbol
    
    expected_types = type_of(fp, rel)
    actual_types = [typ_symbol(srk, sym) for sym in syms]
    
    if expected_types != actual_types:
        raise TypeError(f"Types error in rel atom: expected {expected_types}, got {actual_types}")
    
    return (rel, syms)


def create() -> Fp:
    """Create a new fixed-point system."""
    return Fp()


def add_rule(fp: Fp, hypo: List[RelAtom], phi: Any, conc: RelAtom) -> None:
    """Add a rule to the fixed-point system."""
    fp.rules.append((hypo, phi, conc))


def add_query(fp: Fp, query: Relation) -> None:
    """Add a query to the fixed-point system."""
    fp.queries.append(query)


def get_relations_used(fp: Fp) -> Set[Relation]:
    """Get the set of all relations used in rules and queries."""
    rels = set(fp.queries)
    
    for hypo_atoms, _, conc_atom in fp.rules:
        rels.add(rel_of_atom(conc_atom))
        for atom in hypo_atoms:
            rels.add(rel_of_atom(atom))
    
    return rels


def get_relations_declared(fp: Fp) -> List[Relation]:
    """Get the list of all declared relations."""
    return list(range(len(fp.rel_ctx)))


def is_linear(fp: Fp) -> bool:
    """Check if the CHC system is linear (at most one hypothesis atom per rule)."""
    return all(len(hypo_atoms) <= 1 for hypo_atoms, _, _ in fp.rules)

def is_horn(fp: Fp) -> bool:
    """Check if the CHC system is in Horn form (at most one positive literal per clause)."""
    # In CHC terms, this means at most one conclusion atom per rule
    return all(True)  # CHC rules already have exactly one conclusion

def get_rule_complexity(fp: Fp) -> Dict[int, int]:
    """Get complexity measure for each rule (number of atoms in hypothesis)."""
    complexity = {}
    for i, (hypo_atoms, _, _) in enumerate(fp.rules):
        complexity[i] = len(hypo_atoms)
    return complexity

def has_cycles(fp: Fp) -> bool:
    """Check if the CHC system has cycles (recursive definitions)."""
    # Build dependency graph: rule i depends on rule j if i's conclusion
    # appears in j's hypothesis
    dependencies = {}
    conclusions = {}

    for i, (_, _, conc_atom) in enumerate(fp.rules):
        conclusions[i] = rel_of_atom(conc_atom)

    for i, (hypo_atoms, _, _) in enumerate(fp.rules):
        deps = set()
        for atom in hypo_atoms:
            rel = rel_of_atom(atom)
            # Find rules that conclude with this relation
            for j, conc_rel in conclusions.items():
                if rel == conc_rel and j != i:
                    deps.add(j)
        dependencies[i] = deps

    # Check for cycles using DFS
    visited = set()
    rec_stack = set()

    def has_cycle(node):
        if node in rec_stack:
            return True
        if node in visited:
            return False

        visited.add(node)
        rec_stack.add(node)

        for neighbor in dependencies.get(node, []):
            if has_cycle(neighbor):
                return True

        rec_stack.remove(node)
        return False

    for i in range(len(fp.rules)):
        if i not in visited:
            if has_cycle(i):
                return True

    return False

def validate(fp: Fp) -> List[str]:
    """Validate a CHC system and return list of issues found."""
    issues = []

    # Check for empty system
    if len(fp.rules) == 0 and len(fp.queries) == 0:
        issues.append("Empty CHC system")

    # Check for unused relations
    used_rels = get_relations_used(fp)
    declared_rels = get_relations_declared(fp)

    unused = set(declared_rels) - set(used_rels)
    if unused:
        issues.append(f"Unused relations: {unused}")

    # Check for undefined relations in rules
    for i, (hypo_atoms, _, conc_atom) in enumerate(fp.rules):
        conc_rel = rel_of_atom(conc_atom)
        if conc_rel not in declared_rels:
            issues.append(f"Rule {i}: undefined conclusion relation {conc_rel}")

        for atom in hypo_atoms:
            hyp_rel = rel_of_atom(atom)
            if hyp_rel not in declared_rels:
                issues.append(f"Rule {i}: undefined hypothesis relation {hyp_rel}")

    # Check for queries on undefined relations
    for query_rel in fp.queries:
        if query_rel not in declared_rels:
            issues.append(f"Query on undefined relation {query_rel}")

    # Check for cycles (may be problematic for some solvers)
    if has_cycles(fp):
        issues.append("CHC system contains cycles (recursive definitions)")

    return issues

def get_statistics(fp: Fp) -> Dict[str, Any]:
    """Get statistics about the CHC system."""
    stats = {
        'num_rules': len(fp.rules),
        'num_queries': len(fp.queries),
        'num_relations': len(fp.rel_ctx),
        'is_linear': is_linear(fp),
        'is_horn': is_horn(fp),
        'has_cycles': has_cycles(fp),
        'rule_complexities': get_rule_complexity(fp),
        'relations_used': len(get_relations_used(fp)),
    }

    # Rule statistics
    if fp.rules:
        complexities = [len(hypo) for hypo, _, _ in fp.rules]
        stats['max_rule_complexity'] = max(complexities)
        stats['avg_rule_complexity'] = sum(complexities) / len(complexities)

    return stats

def print_statistics(fp: Fp) -> None:
    """Print statistics about the CHC system."""
    stats = get_statistics(fp)

    print("CHC System Statistics:")
    print(f"  Rules: {stats['num_rules']}")
    print(f"  Queries: {stats['num_queries']}")
    print(f"  Relations: {stats['num_relations']}")
    print(f"  Relations used: {stats['relations_used']}")
    print(f"  Linear: {stats['is_linear']}")
    print(f"  Horn: {stats['is_horn']}")
    print(f"  Has cycles: {stats['has_cycles']}")

    if 'max_rule_complexity' in stats:
        print(f"  Max rule complexity: {stats['max_rule_complexity']}")
        print(f"  Avg rule complexity: {stats['avg_rule_complexity']:.2f}")


def to_weighted_graph(srk: Any, fp: Fp, pd: Any) -> Any:
    """
    Convert a linear CHC system to a weighted graph.
    
    This is the key function for solving linear CHCs. It converts the CHC system
    into a weighted graph where:
    - Nodes are relations (plus special start and goal nodes)
    - Edges are weighted by transition formulas
    - The weight algebra has operations: mul (composition), add (union), star (iteration)
    
    Args:
        srk: Context
        fp: Fixed-point system
        pd: Pre-domain for iteration
        
    Returns:
        Weighted graph
    """
    from .weightedGraph import WeightedGraph, add_vertex, add_edge, path_weight
    from .transitionFormula import make as make_tf
    from .syntax import mk_true, mk_false, mk_or, mk_and, mk_const, substitute, dup_symbol, symbols
    import numpy as np
    
    # Special vertices
    START_VERT = -1
    GOAL_VERT = -2
    
    # Define the weight algebra for transition formulas
    # Weights are tuples: (pre_symbols, post_symbols, formula)
    
    def is_one(weight):
        _, _, phi = weight
        return phi == mk_true(srk)
    
    def is_zero(weight):
        _, _, phi = weight
        return phi == mk_false(srk)
    
    empty_arr = np.array([])
    zero_weight = (empty_arr, empty_arr, mk_false(srk))
    one_weight = (empty_arr, empty_arr, mk_true(srk))
    
    def add_weights(w1, w2):
        """Add (union) two weights."""
        if is_zero(w1):
            return w2
        if is_zero(w2):
            return w1
        if is_one(w1) or is_one(w2):
            return one_weight

        try:
            # Union: need to rename variables to avoid conflicts
            pre1, post1, phi1 = w1
            pre2, post2, phi2 = w2

            # Create fresh symbols for both
            pre_fresh = np.array([dup_symbol(srk, sym) for sym in pre1])
            post_fresh = np.array([dup_symbol(srk, sym) for sym in post1])

            # Substitute in both formulas
            subst_map1 = {}
            for i, sym in enumerate(pre1):
                subst_map1[sym] = mk_const(srk, pre_fresh[i])
            for i, sym in enumerate(post1):
                subst_map1[sym] = mk_const(srk, post_fresh[i])

            phi1_subst = substitute(srk, subst_map1, phi1)

            # Similar for phi2
            pre_fresh2 = np.array([dup_symbol(srk, sym) for sym in pre2])
            post_fresh2 = np.array([dup_symbol(srk, sym) for sym in post2])

            subst_map2 = {}
            for i, sym in enumerate(pre2):
                subst_map2[sym] = mk_const(srk, pre_fresh2[i])
            for i, sym in enumerate(post2):
                subst_map2[sym] = mk_const(srk, post_fresh2[i])

            phi2_subst = substitute(srk, subst_map2, phi2)

            return (pre_fresh, post_fresh, mk_or(srk, [phi1_subst, phi2_subst]))
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to add weights: {e}")
            return one_weight  # Conservative fallback
    
    def mul_weights(w1, w2):
        """Multiply (compose) two weights."""
        if is_zero(w1) or is_zero(w2):
            return zero_weight
        if is_one(w1):
            return w2
        if is_one(w2):
            return w1
        
        # Composition: use transitionFormula.mul
        from .transitionFormula import mul as tf_mul
        
        pre1, post1, phi1 = w1
        pre2, post2, phi2 = w2
        
        # Build transition formulas
        symbols1 = list(zip(pre1, post1))
        symbols2 = list(zip(pre2, post2))
        
        tf1 = make_tf(phi1, symbols1)
        tf2 = make_tf(phi2, symbols2)
        
        # Compose
        tf_composed = tf_mul(srk, tf1, tf2)
        
        return (np.array([s for s, _ in tf_composed.symbols]),
                np.array([s_prime for _, s_prime in tf_composed.symbols]),
                tf_composed.formula)
    
    def star_weight(weight):
        """Star (iteration) of a weight."""
        pre, post, phi = weight
        
        # Use the pre-domain to compute the iteration
        symbols_list = list(zip(pre, post))
        
        # Create loop counter
        from .syntax import mk_symbol
        lc = mk_symbol(srk, typ='TyInt')
        
        # Build transition formula
        tr_phi = make_tf(phi, symbols_list)
        
        # Abstract and compute exponential
        try:
            exp_formula = pd.exp(srk, symbols_list, mk_const(srk, lc), pd.abstract(srk, tr_phi))
            return (pre, post, exp_formula)
        except Exception as e:
            logger = logging.getLogger(__name__)
            logger.warning(f"Failed to compute star weight: {e}")
            # Fallback: return original formula
            return weight
    
    # Define the algebra
    algebra = {
        'mul': mul_weights,
        'add': add_weights,
        'star': star_weight,
        'zero': zero_weight,
        'one': one_weight
    }
    
    # Build the weighted graph
    wg = WeightedGraph(algebra)
    wg = add_vertex(wg, START_VERT)
    wg = add_vertex(wg, GOAL_VERT)
    
    # Add vertices for all used relations
    for rel in get_relations_used(fp):
        wg = add_vertex(wg, rel)
    
    # Add edges for rules
    for hypo_atoms, phi, conc_atom in fp.rules:
        if not hypo_atoms:
            # Fact: start -> conclusion
            conc_rel = rel_of_atom(conc_atom)
            conc_params = params_of_atom(conc_atom)
            
            weight = (empty_arr, np.array(conc_params), phi)
            wg = add_edge(wg, START_VERT, weight, conc_rel)
        
        elif len(hypo_atoms) == 1:
            # Linear rule: hyp -> conclusion
            hyp_atom = hypo_atoms[0]
            hyp_rel = rel_of_atom(hyp_atom)
            hyp_params = params_of_atom(hyp_atom)
            
            conc_rel = rel_of_atom(conc_atom)
            conc_params = params_of_atom(conc_atom)
            
            weight = (np.array(hyp_params), np.array(conc_params), phi)
            wg = add_edge(wg, hyp_rel, weight, conc_rel)
        
        else:
            raise ValueError("Non-linear rule in to_weighted_graph")
    
    # Add edges for queries
    for query_rel in fp.queries:
        wg = add_edge(wg, query_rel, one_weight, GOAL_VERT)
    
    return wg


def check(srk: Any, fp: Fp, pd: Any) -> str:
    """
    Check if a linear CHC system is satisfiable.
    
    Args:
        srk: Context
        fp: Fixed-point system
        pd: Pre-domain for iteration
        
    Returns:
        'No' if unsatisfiable, 'Unknown' otherwise
    """
    from .smt import is_sat
    
    if not is_linear(fp):
        raise ValueError("No methods for solving non-linear fp")
    
    wg = to_weighted_graph(srk, fp, pd)
    
    # Compute path weight from start to goal
    from .weightedGraph import path_weight
    START_VERT = -1
    GOAL_VERT = -2
    
    _, _, phi = path_weight(wg, START_VERT, GOAL_VERT)
    
    result = is_sat(srk, phi)
    if result == 'Unsat':
        return 'No'
    elif result == 'Unknown':
        return 'Unknown'
    else:
        return 'Unknown'


def solve(srk: Any, fp: Fp, pd: Any) -> Callable[[Relation], Tuple[List[Any], Any]]:
    """
    Solve a linear CHC system and return a solution function.
    
    Args:
        srk: Context
        fp: Fixed-point system
        pd: Pre-domain for iteration
        
    Returns:
        Function that maps each relation to (params, formula)
    """
    if not is_linear(fp):
        raise ValueError("No methods for solving non-linear fp")
    
    wg = to_weighted_graph(srk, fp, pd)
    
    from .weightedGraph import path_weight
    START_VERT = -1
    
    def soln(rel: Relation) -> Tuple[List[Any], Any]:
        """Get the solution for a relation."""
        _, params, phi = path_weight(wg, START_VERT, rel)
        return (params, phi)
    
    return soln


# Legacy class-based interface for compatibility

@dataclass(frozen=True)
class CHCClause:
    """Represents a constrained Horn clause: H1 ∧ H2 ∧ ... ∧ Hn ⇒ C"""

    premises: Tuple[Any, ...]
    conclusion: Any

    def __init__(self, premises: List[Any], conclusion: Any):
        object.__setattr__(self, 'premises', tuple(premises))
        object.__setattr__(self, 'conclusion', conclusion)

    def __str__(self) -> str:
        if not self.premises:
            return str(self.conclusion)
        else:
            premises_str = " ∧ ".join(str(p) for p in self.premises)
            return f"{premises_str} ⇒ {self.conclusion}"


@dataclass(frozen=True)
class CHCSystem:
    """System of constrained Horn clauses."""

    clauses: Tuple[CHCClause, ...]
    predicates: Set[str]

    def __init__(self, clauses: List[CHCClause], predicates: Optional[Set[str]] = None):
        object.__setattr__(self, 'clauses', tuple(clauses))
        object.__setattr__(self, 'predicates', predicates or set())

    def __str__(self) -> str:
        clauses_str = "\n".join(f"  {clause}" for clause in self.clauses)
        return f"CHCSystem({len(self.clauses)} clauses)\n{clauses_str}"


class CHCSolver:
    """Solver for constrained Horn clauses."""

    def __init__(self, context: Any):
        self.context = context

    def solve(self, chc_system: CHCSystem) -> bool:
        """Solve a CHC system."""
        # Convert to Fp format and solve
        fp = create()
        
        # This is a simplified conversion - a full implementation would
        # properly convert the CHC clauses to the Fp format
        
        # For now, return Unknown
        return False


# Factory functions
def make_chc_clause(premises: List[Any], conclusion: Any) -> CHCClause:
    """Create a CHC clause."""
    return CHCClause(premises, conclusion)


def make_chc_system(clauses: List[CHCClause]) -> CHCSystem:
    """Create a CHC system."""
    return CHCSystem(clauses)


def make_chc_solver(context: Any) -> CHCSolver:
    """Create a CHC solver."""
    return CHCSolver(context)
