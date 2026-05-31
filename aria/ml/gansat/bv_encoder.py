"""
QF_BV Formula Encoder — bit-vector theory support for GANSAT.

QF_BV variables are fixed-width integers (e.g., bv8, bv32, bv64).
Assignment space per variable: [0, 2^width - 1]

Encoding layout:
  Variable block   : MAX_VARS × VAR_FEAT   = 64 × 4   = 256 dims
    [width_norm, lb_norm, ub_norm, is_signed]

  Constraint block : MAX_CONSTRAINTS × (MAX_VARS + OP_DIM + 2)
                   = 128 × (64 + 16 + 2)   = 128 × 82 = 10496 dims

  Total feature dim: 256 + 10496 = 10752

BV operations encoded (one-hot, 16 classes):
  0  bvult   (unsigned <)
  1  bvule   (unsigned <=)
  2  bvugt   (unsigned >)
  3  bvuge   (unsigned >=)
  4  bvslt   (signed <)
  5  bvsle   (signed <=)
  6  bvsgt   (signed >)
  7  bvsge   (signed >=)
  8  =       (equality)
  9  distinct (inequality)
  10 bvand   (bitwise AND result == 0)
  11 bvor    (bitwise OR)
  12 bvxor   (bitwise XOR)
  13 bvadd   (arithmetic)
  14 bvsub
  15 bvmul
"""

import numpy as np
import z3
from dataclasses import dataclass, field
from typing import Optional
from .parser import ParsedFormula

MAX_VARS        = 64
MAX_CONSTRAINTS = 128
VAR_FEAT        = 4    # per-variable feature size
OP_DIM          = 16   # one-hot operation encoding


def bv_feature_dim() -> int:
    return MAX_VARS * VAR_FEAT + MAX_CONSTRAINTS * (MAX_VARS + OP_DIM + 2)


# ─── BV operation type mapping ────────────────────────────────────────────────
_BV_OP_INDEX = {
    z3.Z3_OP_ULEQ:  0,   # unsigned <=
    z3.Z3_OP_UGEQ:  2,   # unsigned >=  (no ULT/UGT in Z3 directly, use ULEQ/UGEQ)
    z3.Z3_OP_SLEQ:  5,   # signed <=
    z3.Z3_OP_SGEQ:  7,   # signed >=
    z3.Z3_OP_EQ:    8,
    z3.Z3_OP_DISTINCT: 9,
    z3.Z3_OP_BAND:  10,
    z3.Z3_OP_BOR:   11,
    z3.Z3_OP_BXOR:  12,
    z3.Z3_OP_BADD:  13,
    z3.Z3_OP_BSUB:  14,
    z3.Z3_OP_BMUL:  15,
}


def bv_encode(formula: ParsedFormula) -> np.ndarray:
    vec = np.zeros(bv_feature_dim(), dtype=np.float32)
    var_index = {name: i for i, name in enumerate(formula.var_names[:MAX_VARS])}

    # Variable block
    for name, var in list(formula.variables.items())[:MAX_VARS]:
        idx = var_index[name]
        sort = var.sort()
        if z3.is_bv_sort(sort):
            width = sort.size()
            max_val = float((1 << width) - 1)
            base = idx * VAR_FEAT
            vec[base + 0] = width / 64.0          # normalized width
            vec[base + 1] = 0.0                   # lb always 0 for BV (unsigned)
            vec[base + 2] = 1.0                   # ub normalized to 1.0
            vec[base + 3] = 0.0                   # unsigned by default

    # Constraint block
    base_offset = MAX_VARS * VAR_FEAT
    row_size = MAX_VARS + OP_DIM + 2
    constraints = _extract_bv_constraints(formula, var_index)

    for ci, (var_mask, op_onehot, lhs_val, rhs_val) in enumerate(constraints[:MAX_CONSTRAINTS]):
        offset = base_offset + ci * row_size
        vec[offset : offset + MAX_VARS]             = var_mask
        vec[offset + MAX_VARS : offset + MAX_VARS + OP_DIM] = op_onehot
        vec[offset + MAX_VARS + OP_DIM]     = lhs_val
        vec[offset + MAX_VARS + OP_DIM + 1] = rhs_val

    return vec


def bv_decode_assignment(vec: np.ndarray, formula: ParsedFormula) -> dict:
    """Convert generator output [-1,1]^MAX_VARS → {var: int_value} for BV vars."""
    assignment = {}
    for i, name in enumerate(formula.var_names[:MAX_VARS]):
        var = formula.variables.get(name)
        if var is None:
            continue
        sort = var.sort()
        if z3.is_bv_sort(sort):
            width = sort.size()
            max_val = (1 << width) - 1
            raw = float(vec[i])
            value = int(round((raw + 1.0) / 2.0 * max_val))
            value = max(0, min(max_val, value))
            assignment[name] = value
    return assignment


def _extract_bv_constraints(formula: ParsedFormula, var_index: dict) -> list:
    constraints = []
    for assertion in formula.assertions:
        _parse_bv_constraint(assertion, var_index, formula, constraints)
    return constraints


def _parse_bv_constraint(expr: z3.ExprRef, var_index: dict, formula: ParsedFormula, out: list):
    if not z3.is_bool(expr):
        return
    decl = expr.decl().kind()

    if decl == z3.Z3_OP_AND:
        for child in expr.children():
            _parse_bv_constraint(child, var_index, formula, out)
        return

    op_idx = _BV_OP_INDEX.get(decl)
    if op_idx is None:
        return

    op_onehot = np.zeros(OP_DIM, dtype=np.float32)
    op_onehot[op_idx] = 1.0

    children = expr.children()
    if len(children) < 2:
        return

    lhs, rhs = children[0], children[1]

    # Variable mask: which variables appear in this constraint
    var_mask = np.zeros(MAX_VARS, dtype=np.float32)
    _collect_bv_vars(lhs, var_index, var_mask)
    _collect_bv_vars(rhs, var_index, var_mask)

    # Extract literal values if present
    lhs_val = _bv_literal_norm(lhs)
    rhs_val = _bv_literal_norm(rhs)

    out.append((var_mask, op_onehot, lhs_val, rhs_val))


def _collect_bv_vars(expr: z3.ExprRef, var_index: dict, mask: np.ndarray):
    if z3.is_const(expr) and expr.decl().kind() == z3.Z3_OP_UNINTERPRETED:
        idx = var_index.get(str(expr))
        if idx is not None and idx < MAX_VARS:
            mask[idx] = 1.0
    for child in expr.children():
        _collect_bv_vars(child, var_index, mask)


def _bv_literal_norm(expr: z3.ExprRef) -> float:
    if z3.is_bv_value(expr):
        width = expr.sort().size()
        max_val = float((1 << width) - 1)
        return float(expr.as_long()) / max_val if max_val > 0 else 0.0
    return 0.0
