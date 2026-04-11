"""Package initialization for aria.cflobdd."""

from aria.cflobdd.fixpoint import (
    FixpointResult,
    least_fixed_point,
    reflexive_closure,
    reflexive_transitive_closure,
    transitive_closure,
)
from aria.cflobdd.grammar import (
    GrammarProduction,
    GrammarReachability,
    GrammarSolution,
)
from aria.cflobdd.pushdown import balanced_reachability
from aria.cflobdd.relation import Relation
from aria.cflobdd.witness import build_witness_tree, extract_edge_path, format_witness

__all__ = [
    "FixpointResult",
    "GrammarProduction",
    "GrammarReachability",
    "GrammarSolution",
    "Relation",
    "balanced_reachability",
    "build_witness_tree",
    "extract_edge_path",
    "format_witness",
    "least_fixed_point",
    "reflexive_closure",
    "reflexive_transitive_closure",
    "transitive_closure",
]
