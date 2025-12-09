"""
SRK (Symbolic Reasoning Kit) - Python Edition

A comprehensive library for symbolic reasoning, program verification,
and automated theorem proving.

This package provides:
- Symbolic expression manipulation
- SMT solver integration
- Polynomial and linear algebra operations
- Abstract interpretation
- Termination analysis
- Quantifier reasoning

Main modules:
- syntax: Core expression and formula types
- polynomial: Polynomial operations and Groebner bases
- smt: SMT solver interfaces
- abstract: Abstract domains for program analysis
- quantifier: Quantifier elimination and strategy synthesis
"""

__version__ = "0.1.0"
__author__ = "SRK Python Migration"

# Core exports
from .syntax import *
from .polynomial import *
from .smt import *
from .abstract import *
from .linear import *
from .interval import *
from .polyhedron import *
from .srkSimplify import (
    Simplifier, NNFConverter, CNFConverter, ExpressionSimplifier,
    make_simplifier, make_nnf_converter, make_cnf_converter, make_expression_simplifier,
    simplify_expression, to_negation_normal_form, to_conjunctive_normal_form, eliminate_ite_expressions
)
from .sequence import *
from .vas import *
from .iteration import *
from .quantifier import *
from .interpretation import *
from .transition import *
from .transitionFormula import *
from .transitionSystem import *
from .wedge import *
from .termination import *
from .fixpoint import *
from .cache import *
from .disjointSet import *
from .log import *
from .util import *
from .srkUtil import *
from .sparseMap import *
from .memo import *
from .expPolynomial import *
from .chc import *
from .sequence import *
from .lts import *
from .polyhedron import *
from .zZ import *
from .qQ import *
from .bigO import *
from .compressedWeightedForest import *
from .weightedGraph import *
from .loop import *
from .pathexpr import *
from .srkZ3 import *
from .coordinateSystem import *
from .nonlinear import *
from .featureTree import *
from .srkParse import *
from .randomFormula import *  # Re-enabled after fixing imports

# CLI module is not exported in __all__ since it's mainly for command-line usage

__all__ = [
    # Core syntax exports will be added as we implement
    'Context',
    'Symbol',
    'Expression',
    'FormulaExpression',
    'TermExpression',
    'Type',
    'mk_symbol',
    'mk_const',
    'mk_add',
    'mk_eq',
    # Polynomial exports
    'Polynomial',
    'Monomial',
    # SMT exports
    'SMTResult',
    'SMTModel',
    'SMTSolver',
    'Z3Solver',
    'SMTInterface',
    # Abstract domain exports
    'AbstractDomain',
    'SignDomain',
    'AffineDomain',
    'ProductDomain',
    'PredicateAbstraction',
    'IntervalDomain',
    'Interval',
    'AffineRelation',
    'AbstractValue',
    # Abstract domain convenience functions
    'sign_domain',
    'affine_domain',
    'interval_domain',
    'predicate_abstraction',
    'product_domain',
    'top_sign',
    'bottom_sign',
    'top_interval',
    'bottom_interval',
    # Linear algebra exports
    'QQVector',
    'QQMatrix',
    'QQVectorSpace',
    # Interval arithmetic exports
    # Polyhedron exports
    'Polyhedron',
    'Constraint',
    # Sequence analysis exports
    'UltimatelyPeriodicSequence',
    'SequenceAnalyzer',
    # VAS and Petri net exports
    'Transformer',
    'VectorAdditionSystem',
    'PetriNet',
    'Place',
    'Transition',
    'ReachabilityResult',
    # Iteration exports
    'IterationEngine',
    'WedgeGuard',
    'PolyhedronGuard',
    # Quantifier exports
    'QuantifierEngine',
    'StrategyImprovementSolver',
    # Interpretation exports
    'Interpretation',
    'EvaluationContext',
    # Transition exports
    'Transition',
    'TransitionSystem',
    'TransitionFormula',
    'TransitionResult',
    # Transition creation functions
    'make_assume',
    'make_assign',
    'make_parallel_assign',
    'make_havoc',
    'make_zero',
    'make_one',
    # Transition formula exports
    'make_transition_formula',
    'identity_transition_formula',
    'zero_transition_formula',
    'compose_transition_formulas',
    'union_transition_formulas',
    'linearize_transition_formula',
    # Advanced transition system exports
    'Query',
    'Label',
    'make_transition_system',
    'make_query',
    'remove_temporaries',
    'forward_invariants_ivl',
    'forward_invariants_ivl_pa',
    'simplify',
    'loop_headers_live',
    'AbstractDomain',
    'IncrAbstractDomain',
    'forward_invariants',
    # Wedge exports
    'WedgeDomain',
    'WedgeElement',
    # Termination exports
    'TerminationAnalyzer',
    'RankingFunction',
    # Fixpoint exports
    'Lattice',
    'FixpointComputer',
    'SimpleLattice',
    'PowerSetLattice',
    'IntervalLattice',
    # Cache exports
    'LRUCache',
    'WeakKeyCache',
    'Memoize',
    'FunctionCache',
    'ExpressionCache',
    # Disjoint set exports
    'DisjointSet',
    # SRK utility exports
    'binary_search',
    'merge_arrays',
    'exp',
    'format_to_string',
    'print_enum',
    'print_list',
    'cartesian_product',
    'tuples',
    'adjacent_pairs',
    'distinct_pairs',
    'IntSet',
    'IntMap',
    'make_int_set',
    'make_int_map',
    'default_separator',
    'enum_to_list',
    'list_to_enum',
    # Sparse map exports
    'SparseMap',
    'ZeroValue',
    'IntSparseMap',
    'FloatSparseMap',
    'make_sparse_map',
    'make_sparse_map_from_dict',
    'make_sparse_map_from_enum',
    'sparse_map_add',
    'sparse_map_multiply',
    'sparse_map_scale',
    # Logging exports
    'SRKLogger',
    'LogContext',
    'PerformanceMonitor',
    # Utility exports
    'IntSet',
    'Counter',
    'Stack',
    'Queue',
    'PriorityQueue',
    'Graph',
    # Memoization exports
    'ExpressionMemoizer',
    'FunctionMemoizer',
    'MemoizationMonitor',
    # Exponential polynomial exports
    'ExpPolynomial',
    'ExpPolynomialVector',
    'ExpPolynomialMatrix',
    # CHC exports
    'CHCClause',
    'CHCSystem',
    'CHCSolver',
    # Sequence exports
    # LTS exports
    'LinearTransitionSystem',
    'LTSAnalysis',
    'PartialLinearMap',
    'DeterministicLTS',
    # Polyhedron exports
    'Polyhedron',
    'Constraint',
    'PolyhedronOperations',
    # VAS exports
    'Vector',
    'VAS',
    'VASS',
    'PetriNet',
    'VASAnalysis',
    # Sparse map exports
    'SparseMap',
    # AST exports
    # Apron exports
    # Simplification exports
    # Algebra exports
    # Ring theory exports
    # Big O complexity exports
]
