"""Package initialization for aria.cflobdd."""

from aria.automata.cflobdd.cflobvdd import CFLOBVDD
from aria.automata.cflobdd.fixpoint import (
    FixpointResult,
    least_fixed_point,
    reflexive_closure,
    reflexive_transitive_closure,
    transitive_closure,
)
from aria.automata.cflobdd.grammar import (
    GrammarProduction,
    GrammarReachability,
    GrammarSolution,
)
from aria.automata.cflobdd.pushdown import balanced_reachability
from aria.automata.cflobdd.relation import Relation
from aria.automata.cflobdd.witness import build_witness_tree, extract_edge_path, format_witness

__all__ = [
    "CFLOBVDD",
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
