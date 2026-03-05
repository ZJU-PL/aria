#!/usr/bin/env python3
"""IR metadata extraction for finite-field formulas.

The translation backend uses this module to build cheap structural facts
about each expression node, primarily for modulo-reduction scheduling:

- ``depth``: expression depth in the AST/DAG
- ``fanout``: structural reuse count based on a deterministic key
- ``nonlinear``: whether the node (or descendants) introduces nonlinearity
- ``est_bits``: coarse upper bound on intermediate bit growth
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .ff_ast import (
    BoolAnd,
    BoolConst,
    BoolIte,
    BoolImplies,
    BoolNot,
    BoolOr,
    BoolVar,
    BoolXor,
    FieldAdd,
    FieldConst,
    FieldDiv,
    FieldEq,
    FieldExpr,
    FieldMul,
    FieldNeg,
    FieldPow,
    FieldSub,
    FieldVar,
)


@dataclass
class FFNodeStats:
    """Per-expression structural statistics for scheduling.

    Attributes:
        op: Node operator/class name.
        depth: Maximum distance to a leaf (>= 1).
        fanout: Number of times the structural key appears in the formula.
        nonlinear: Whether this subterm contains nonlinear arithmetic.
        est_bits: Conservative estimate of intermediate bit-width.
    """

    op: str
    depth: int
    fanout: int
    nonlinear: bool
    est_bits: int


@dataclass
class FFIRMetadata:
    """Whole-formula metadata keyed by structural node keys."""

    stats_by_key: Dict[Tuple[object, ...], FFNodeStats]


def expr_key(expr: FieldExpr) -> Tuple[object, ...]:
    """Return a deterministic structural key for a field expression."""
    if isinstance(expr, FieldVar):
        return ("FieldVar", expr.name)
    if isinstance(expr, FieldConst):
        return ("FieldConst", expr.value, expr.modulus)
    if isinstance(expr, BoolVar):
        return ("BoolVar", expr.name)
    if isinstance(expr, BoolConst):
        return ("BoolConst", expr.value)

    if isinstance(expr, FieldNeg):
        return ("FieldNeg", expr_key(expr.arg))
    if isinstance(expr, BoolNot):
        return ("BoolNot", expr_key(expr.arg))

    if isinstance(expr, FieldPow):
        return ("FieldPow", expr_key(expr.base), expr.exponent)

    if isinstance(expr, FieldEq):
        return ("FieldEq", expr_key(expr.left), expr_key(expr.right))

    if isinstance(expr, BoolImplies):
        return (
            "BoolImplies",
            expr_key(expr.antecedent),
            expr_key(expr.consequent),
        )

    if isinstance(expr, BoolIte):
        return (
            "BoolIte",
            expr_key(expr.cond),
            expr_key(expr.then_expr),
            expr_key(expr.else_expr),
        )

    if isinstance(expr, (FieldAdd, FieldMul, FieldSub, BoolAnd, BoolOr, BoolXor)):
        return (type(expr).__name__, tuple(expr_key(arg) for arg in expr.args))

    return (type(expr).__name__, id(expr))


def build_ir_metadata(assertions: Sequence[FieldExpr]) -> FFIRMetadata:
    """Compute metadata for a collection of assertions.

    The pass is intentionally lightweight and deterministic. It is designed
    to be cheap relative to SMT solving and robust across benchmark families.
    """
    fanout_by_key: Dict[Tuple[object, ...], int] = {}
    for assertion in assertions:
        _collect_fanout(assertion, fanout_by_key)

    cache: Dict[Tuple[object, ...], FFNodeStats] = {}
    for assertion in assertions:
        _build_stats(assertion, fanout_by_key, cache)

    return FFIRMetadata(stats_by_key=cache)


def _collect_fanout(expr: FieldExpr, fanout_by_key: Dict[Tuple[object, ...], int]) -> None:
    key = expr_key(expr)
    fanout_by_key[key] = fanout_by_key.get(key, 0) + 1
    for child in _children(expr):
        _collect_fanout(child, fanout_by_key)


def _build_stats(
    expr: FieldExpr,
    fanout_by_key: Dict[Tuple[object, ...], int],
    cache: Dict[Tuple[object, ...], FFNodeStats],
) -> FFNodeStats:
    key = expr_key(expr)
    if key in cache:
        return cache[key]

    child_stats: List[FFNodeStats] = []
    for child in _children(expr):
        child_stats.append(_build_stats(child, fanout_by_key, cache))

    depth = 1 + max((stat.depth for stat in child_stats), default=0)
    nonlinear = _is_nonlinear(expr) or any(stat.nonlinear for stat in child_stats)
    est_bits = _estimate_bits(expr, child_stats)
    stat = FFNodeStats(
        op=type(expr).__name__,
        depth=depth,
        fanout=fanout_by_key.get(key, 1),
        nonlinear=nonlinear,
        est_bits=est_bits,
    )
    cache[key] = stat
    return stat


def _children(expr: FieldExpr) -> Sequence[FieldExpr]:
    if isinstance(expr, (FieldAdd, FieldMul, FieldSub, BoolAnd, BoolOr, BoolXor)):
        return expr.args
    if isinstance(expr, FieldNeg):
        return [expr.arg]
    if isinstance(expr, FieldPow):
        return [expr.base]
    if isinstance(expr, FieldEq):
        return [expr.left, expr.right]
    if isinstance(expr, BoolNot):
        return [expr.arg]
    if isinstance(expr, BoolImplies):
        return [expr.antecedent, expr.consequent]
    if isinstance(expr, BoolIte):
        return [expr.cond, expr.then_expr, expr.else_expr]
    if isinstance(expr, FieldDiv):
        return [expr.num, expr.denom]
    return []


def _is_nonlinear(expr: FieldExpr) -> bool:
    if isinstance(expr, FieldMul):
        # Multiplication involving two non-constant factors is nonlinear.
        non_const = [arg for arg in expr.args if not isinstance(arg, FieldConst)]
        return len(non_const) >= 2
    if isinstance(expr, FieldPow):
        return expr.exponent >= 2
    return False


def _estimate_bits(expr: FieldExpr, child_stats: Sequence[FFNodeStats]) -> int:
    """Estimate a conservative bit-size bound for arithmetic intermediates."""
    if isinstance(expr, FieldConst):
        val = abs(expr.value)
        return max(1, val.bit_length())
    if isinstance(expr, FieldVar):
        return 64
    if isinstance(expr, FieldAdd):
        if not child_stats:
            return 1
        return max(stat.est_bits for stat in child_stats) + _ceil_log2(len(child_stats) + 1)
    if isinstance(expr, FieldSub):
        if not child_stats:
            return 1
        return max(stat.est_bits for stat in child_stats) + 1
    if isinstance(expr, FieldMul):
        if not child_stats:
            return 1
        return sum(stat.est_bits for stat in child_stats)
    if isinstance(expr, FieldNeg):
        return child_stats[0].est_bits + 1 if child_stats else 2
    if isinstance(expr, FieldPow):
        if not child_stats:
            return 1
        return max(1, child_stats[0].est_bits * max(1, expr.exponent))
    if isinstance(expr, FieldEq):
        return max((stat.est_bits for stat in child_stats), default=1)
    if isinstance(expr, BoolIte):
        # branches drive the arithmetic size in mixed-sort ITE terms
        if len(child_stats) >= 3:
            return max(child_stats[1].est_bits, child_stats[2].est_bits)
        return max((stat.est_bits for stat in child_stats), default=1)
    return max((stat.est_bits for stat in child_stats), default=1)


def _ceil_log2(value: int) -> int:
    if value <= 1:
        return 0
    return int(math.ceil(math.log2(float(value))))
