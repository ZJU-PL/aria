"""
QF_LIA formula encoder using NeuroSym's own AST (no Z3).
Produces identical-shaped feature vectors as the original encoder.py.

Layout (matches encoder.py exactly):
  Variable bounds block  : MAX_VARS * 2
  Constraint block       : MAX_CONSTRAINTS * (MAX_VARS + 2)
  Total                  : 8576  (default)
"""

import numpy as np
from .ns_ast import (
    Term, BoolLit, IntLit, BVLit, Var, App,
    NsFormula, BoolSort, IntSort, BVSort,
    BOOL, INT,
)

MAX_VARS        = 64
MAX_CONSTRAINTS = 128
BOUND_CLIP      = 1e4
COEFF_CLIP      = 1e4


def feature_dim() -> int:
    return MAX_VARS * 2 + MAX_CONSTRAINTS * (MAX_VARS + 2)


# ── Encoding entry point ───────────────────────────────────────────────────────

def encode(formula: NsFormula) -> np.ndarray:
    vec       = np.zeros(feature_dim(), dtype=np.float32)
    var_index = {name: i for i, name in enumerate(formula.var_names[:MAX_VARS])}

    # Variable bounds block
    bounds = _extract_bounds(formula)
    for name, (lb, ub) in bounds.items():
        idx = var_index.get(name)
        if idx is None:
            continue
        vec[idx * 2]     = np.clip(lb, -BOUND_CLIP, BOUND_CLIP) / BOUND_CLIP
        vec[idx * 2 + 1] = np.clip(ub, -BOUND_CLIP, BOUND_CLIP) / BOUND_CLIP

    # Constraint block
    base     = MAX_VARS * 2
    row_size = MAX_VARS + 2
    constraints = _extract_linear_constraints(formula, var_index)
    for ci, (coeffs, rhs, ctype) in enumerate(constraints[:MAX_CONSTRAINTS]):
        offset = base + ci * row_size
        for var_idx, coeff in coeffs.items():
            if var_idx < MAX_VARS:
                vec[offset + var_idx] = np.clip(coeff, -COEFF_CLIP, COEFF_CLIP) / COEFF_CLIP
        vec[offset + MAX_VARS]     = np.clip(rhs, -BOUND_CLIP, BOUND_CLIP) / BOUND_CLIP
        vec[offset + MAX_VARS + 1] = ctype   # 0=leq, 1=eq, 2=geq, 3=neq

    return vec


def decode_assignment(vec: np.ndarray, formula: NsFormula) -> dict:
    """Convert GAN output vector → variable assignment dict."""
    assignment = {}
    bounds     = _extract_bounds(formula)
    for i, name in enumerate(formula.var_names[:MAX_VARS]):
        raw      = float(vec[i])
        lb, ub   = bounds.get(name, (-BOUND_CLIP, BOUND_CLIP))
        value    = lb + (raw + 1.0) / 2.0 * (ub - lb)
        assignment[name] = int(round(value))
    return assignment


# ── Bounds extraction ──────────────────────────────────────────────────────────

def _extract_bounds(formula: NsFormula) -> dict:
    bounds = {name: (-BOUND_CLIP, BOUND_CLIP) for name in formula.var_names}
    for assertion in formula.assertions:
        _parse_bound(assertion, bounds)
    return bounds


def _parse_bound(expr: Term, bounds: dict):
    if not isinstance(expr, App):
        return

    op = expr.op

    if op == 'and':
        for child in expr.args:
            _parse_bound(child, bounds)
        return

    _LE_OPS = {'<=', '<'}
    _GE_OPS = {'>=', '>'}

    if op in _LE_OPS or op in _GE_OPS:
        if len(expr.args) != 2:
            return
        lhs, rhs = expr.args
        if isinstance(lhs, Var) and isinstance(rhs, IntLit):
            name, val = lhs.name, rhs.value
            if op in _LE_OPS:
                lb, ub = bounds.get(name, (-BOUND_CLIP, BOUND_CLIP))
                bounds[name] = (lb, min(ub, val))
            else:
                lb, ub = bounds.get(name, (-BOUND_CLIP, BOUND_CLIP))
                bounds[name] = (max(lb, val), ub)
        elif isinstance(rhs, Var) and isinstance(lhs, IntLit):
            name, val = rhs.name, lhs.value
            if op in _GE_OPS:
                lb, ub = bounds.get(name, (-BOUND_CLIP, BOUND_CLIP))
                bounds[name] = (lb, min(ub, val))
            else:
                lb, ub = bounds.get(name, (-BOUND_CLIP, BOUND_CLIP))
                bounds[name] = (max(lb, val), ub)


# ── Linear constraint extraction ───────────────────────────────────────────────

def _extract_linear_constraints(formula: NsFormula, var_index: dict) -> list:
    constraints = []
    for assertion in formula.assertions:
        _parse_linear(assertion, var_index, constraints)
    return constraints


def _parse_linear(expr: Term, var_index: dict, out: list):
    if not isinstance(expr, App):
        return
    op = expr.op

    if op == 'and':
        for child in expr.args:
            _parse_linear(child, var_index, out)
        return

    _TYPE_MAP = {'<=': 0, '=': 1, '>=': 2, 'distinct': 3}
    if op not in _TYPE_MAP:
        return

    ctype = _TYPE_MAP[op]
    if len(expr.args) < 2:
        return
    lhs, rhs = expr.args[0], expr.args[1]

    # Compute lhs - rhs coefficient vector
    coeffs  = {}
    constant = 0

    def _walk(e: Term, sign: int):
        nonlocal constant
        if isinstance(e, IntLit):
            constant += sign * e.value
        elif isinstance(e, Var):
            idx = var_index.get(e.name)
            if idx is not None:
                coeffs[idx] = coeffs.get(idx, 0) + sign
        elif isinstance(e, App):
            if e.op == '+':
                for c in e.args: _walk(c, sign)
            elif e.op == '-':
                if len(e.args) == 1:
                    _walk(e.args[0], -sign)
                else:
                    _walk(e.args[0], sign)
                    for c in e.args[1:]: _walk(c, -sign)
            elif e.op == '*':
                if len(e.args) == 2:
                    if isinstance(e.args[0], IntLit):
                        _walk(e.args[1], sign * e.args[0].value)
                    elif isinstance(e.args[1], IntLit):
                        _walk(e.args[0], sign * e.args[1].value)
            # ignore non-linear terms

    _walk(lhs,  1)
    _walk(rhs, -1)

    if coeffs:
        out.append((coeffs, -constant, ctype))
