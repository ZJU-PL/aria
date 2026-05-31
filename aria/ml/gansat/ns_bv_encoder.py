"""
QF_BV formula encoder using NeuroSym's own AST (no Z3).
Produces identical-shaped feature vectors as the original bv_encoder.py.

Layout (matches bv_encoder.py exactly):
  Variable block   : MAX_VARS × VAR_FEAT   = 256 dims
  Constraint block : MAX_CONSTRAINTS × (MAX_VARS + OP_DIM + 2) = 10496 dims
  Total            : 10752
"""

import numpy as np
from .ns_ast import (
    Term, BoolLit, IntLit, BVLit, Var, App,
    NsFormula, BVSort,
)

MAX_VARS        = 64
MAX_CONSTRAINTS = 128
VAR_FEAT        = 4
OP_DIM          = 16

# Operation → one-hot index (matches bv_encoder.py)
_BV_OP_INDEX = {
    'bvule': 0,   # unsigned <=
    'bvult': 1,   # unsigned <   (mapped from uleq in original; index 1 unused there)
    'bvuge': 2,   # unsigned >=
    'bvugt': 3,   # unsigned >   (index 3 unused in original)
    'bvsle': 5,   # signed <=
    'bvslt': 4,   # signed <
    'bvsge': 7,   # signed >=
    'bvsgt': 6,   # signed >
    '=':     8,
    'distinct': 9,
    'bvand': 10,
    'bvor':  11,
    'bvxor': 12,
    'bvadd': 13,
    'bvsub': 14,
    'bvmul': 15,
}


def bv_feature_dim() -> int:
    return MAX_VARS * VAR_FEAT + MAX_CONSTRAINTS * (MAX_VARS + OP_DIM + 2)


# ── Encoding entry point ───────────────────────────────────────────────────────

def bv_encode(formula: NsFormula) -> np.ndarray:
    vec       = np.zeros(bv_feature_dim(), dtype=np.float32)
    var_index = {name: i for i, name in enumerate(formula.var_names[:MAX_VARS])}

    # Variable block
    for name in formula.var_names[:MAX_VARS]:
        var = formula.variables.get(name)
        if var is None: continue
        sort = var.sort
        if isinstance(sort, BVSort):
            idx     = var_index[name]
            width   = sort.width
            base    = idx * VAR_FEAT
            vec[base + 0] = width / 64.0   # normalised width
            vec[base + 1] = 0.0            # unsigned lb = 0
            vec[base + 2] = 1.0            # ub normalised to 1.0
            vec[base + 3] = 0.0            # unsigned by default

    # Constraint block
    base_offset  = MAX_VARS * VAR_FEAT
    row_size     = MAX_VARS + OP_DIM + 2
    constraints  = _extract_bv_constraints(formula, var_index)
    for ci, (var_mask, op_onehot, lhs_val, rhs_val) in enumerate(constraints[:MAX_CONSTRAINTS]):
        offset = base_offset + ci * row_size
        vec[offset : offset + MAX_VARS]                    = var_mask
        vec[offset + MAX_VARS : offset + MAX_VARS + OP_DIM] = op_onehot
        vec[offset + MAX_VARS + OP_DIM]                    = lhs_val
        vec[offset + MAX_VARS + OP_DIM + 1]                = rhs_val

    return vec


def bv_decode_assignment(vec: np.ndarray, formula: NsFormula) -> dict:
    """Convert GAN output [-1,1]^MAX_VARS → {var: int_value} for BV vars."""
    assignment = {}
    for i, name in enumerate(formula.var_names[:MAX_VARS]):
        var = formula.variables.get(name)
        if var is None: continue
        sort = var.sort
        if isinstance(sort, BVSort):
            width   = sort.width
            max_val = (1 << width) - 1
            raw     = float(vec[i])
            value   = int(round((raw + 1.0) / 2.0 * max_val))
            value   = max(0, min(max_val, value))
            assignment[name] = value
    return assignment


# ── Constraint extraction ──────────────────────────────────────────────────────

def _extract_bv_constraints(formula: NsFormula, var_index: dict) -> list:
    constraints = []
    for assertion in formula.assertions:
        _parse_bv_constraint(assertion, var_index, formula, constraints)
    return constraints


def _parse_bv_constraint(expr: Term, var_index: dict, formula: NsFormula, out: list):
    if not isinstance(expr, App):
        return
    op = expr.op

    if op == 'and':
        for child in expr.args:
            _parse_bv_constraint(child, var_index, formula, out)
        return

    op_idx = _BV_OP_INDEX.get(op)
    if op_idx is None:
        return

    if len(expr.args) < 2:
        return

    op_onehot       = np.zeros(OP_DIM, dtype=np.float32)
    op_onehot[op_idx] = 1.0

    lhs, rhs = expr.args[0], expr.args[1]

    var_mask = np.zeros(MAX_VARS, dtype=np.float32)
    _collect_bv_vars(lhs, var_index, var_mask)
    _collect_bv_vars(rhs, var_index, var_mask)

    lhs_val = _bv_literal_norm(lhs)
    rhs_val = _bv_literal_norm(rhs)

    out.append((var_mask, op_onehot, lhs_val, rhs_val))


def _collect_bv_vars(expr: Term, var_index: dict, mask: np.ndarray):
    if isinstance(expr, Var):
        idx = var_index.get(expr.name)
        if idx is not None and idx < MAX_VARS:
            mask[idx] = 1.0
    elif isinstance(expr, App):
        for child in expr.args:
            _collect_bv_vars(child, var_index, mask)


def _bv_literal_norm(expr: Term) -> float:
    if isinstance(expr, BVLit):
        width   = expr.width
        max_val = float((1 << width) - 1)
        return float(expr.value) / max_val if max_val > 0 else 0.0
    return 0.0
