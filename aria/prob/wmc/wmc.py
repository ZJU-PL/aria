"""
Weighted Model Counting (WMC) for propositional CNF formulas.

This module provides WMC computation using different backends:
- DNNF compilation for exact counting
- Enumeration for approximate counting with model limits
"""
from __future__ import annotations

from typing import Iterable, List, Dict

from pysat.formula import CNF
from pysat.solvers import Solver

from aria.bool.knowledge_compiler.dtree import Dtree_Compiler
from aria.bool.knowledge_compiler.dnnf import DNNF_Compiler, DNF_Node

from .base import WMCBackend, WMCOptions, LiteralWeights


def _validate_weights(weights: LiteralWeights) -> None:
    for lit, w in weights.items():
        if w < 0.0 or w > 1.0:
            raise ValueError(f"Weight of literal {lit} must be in [0,1], got {w}")


def _ensure_complement_weights(weights: LiteralWeights, variables: Iterable[int]) -> LiteralWeights:
    completed: Dict[int, float] = dict(weights)
    for v in variables:
        pos = completed.get(v)
        neg = completed.get(-v)
        if pos is None and neg is None:
            # default to 0.5/0.5 if unspecified
            completed[v] = 0.5
            completed[-v] = 0.5
        elif pos is None:
            completed[v] = 1.0 - float(neg)
        elif neg is None:
            completed[-v] = 1.0 - float(pos)
        # if both given, keep as is (no constraint enforced)
    return completed


def _variables_of_cnf(cnf: CNF) -> List[int]:
    if cnf.nv:
        return list(range(1, cnf.nv + 1))
    vars_set = set()
    for cl in cnf.clauses:
        for lit in cl:
            vars_set.add(abs(lit))
    return sorted(vars_set)


def _wmc_on_dnnf(root: DNF_Node, weights: LiteralWeights) -> float:
    memo: Dict[int, float] = {}

    def eval_node(n: DNF_Node) -> float:
        if n.explore_id is not None and n.explore_id in memo:
            return memo[n.explore_id]

        if n.type == 'L':
            if isinstance(n.literal, bool):
                val = 1.0 if n.literal else 0.0
            else:
                val = float(weights[int(n.literal)])
        elif n.type == 'A':
            left = eval_node(n.left_child)
            right = eval_node(n.right_child)
            val = left * right
        elif n.type == 'O':
            left = eval_node(n.left_child)
            right = eval_node(n.right_child)
            val = left + right
        else:
            raise RuntimeError("Unknown DNNF node type")

        if n.explore_id is not None:
            memo[n.explore_id] = val
        return val

    # assign explore ids for memoization
    root.reset()
    root.count_node(0)
    return eval_node(root)


def _wmc_by_enumeration(cnf: CNF, weights: LiteralWeights, model_limit: int | None) -> float:
    total = 0.0
    with Solver(bootstrap_with=cnf) as s:
        count = 0
        while s.solve():
            model = s.get_model()
            prob = 1.0
            for lit in model:
                if abs(lit) > cnf.nv:
                    continue
                prob *= float(weights.get(lit, 0.5))
            total += prob
            count += 1
            if model_limit is not None and count >= model_limit:
                break
            s.add_clause([-l for l in model if abs(l) <= cnf.nv])
    return total


def wmc_count(cnf: CNF, weights: LiteralWeights, options: WMCOptions | None = None) -> float:
    """
    Compute weighted model count of a propositional CNF.

    Args:
        cnf: PySAT CNF formula
        weights: literal -> probability weight. If only one polarity is given,
                 the other is assumed to be 1-w.
        options: WMCOptions

    Returns:
        Weighted model count (float)
    """
    opts = options or WMCOptions()
    _validate_weights(weights)
    variables = _variables_of_cnf(cnf)
    w = _ensure_complement_weights(weights, variables)

    if opts.backend == WMCBackend.DNNF:
        # compile to DNNF using existing compiler
        clauses = cnf.clauses
        ordering = variables
        dt = Dtree_Compiler(clauses).el2dt(ordering)
        compiler = DNNF_Compiler(dt)
        root = compiler.compile()
        if root is None:
            return 0.0
        return _wmc_on_dnnf(root, w)
    if opts.backend == WMCBackend.ENUMERATION:
        return _wmc_by_enumeration(cnf, w, opts.model_limit)
    raise ValueError(f"Unsupported WMC backend: {opts.backend}")
