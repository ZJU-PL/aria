"""
Formula encoder: converts a ParsedFormula into a fixed-size float tensor.

Encoding layout (QF_LIA focused):
  - Variable bounds block  : MAX_VARS * 2  (lb, ub per variable)
  - Constraint block       : MAX_CONSTRAINTS * (MAX_VARS + 2)
                              (coefficients + rhs + type per constraint)

Total feature dim = MAX_VARS*2 + MAX_CONSTRAINTS*(MAX_VARS+2)
                  = 64*2 + 128*(64+2) = 8576  (default config)
"""

import numpy as np
import z3
from .parser import ParsedFormula

MAX_VARS = 64
MAX_CONSTRAINTS = 128
BOUND_CLIP = 1e4
COEFF_CLIP = 1e4


def feature_dim() -> int:
    return MAX_VARS * 2 + MAX_CONSTRAINTS * (MAX_VARS + 2)


def encode(formula: ParsedFormula) -> np.ndarray:
    vec = np.zeros(feature_dim(), dtype=np.float32)
    var_index = {name: i for i, name in enumerate(formula.var_names[:MAX_VARS])}

    # Variable bounds block
    bounds = _extract_bounds(formula)
    for name, (lb, ub) in bounds.items():
        idx = var_index.get(name)
        if idx is None:
            continue
        vec[idx * 2]     = np.clip(lb, -BOUND_CLIP, BOUND_CLIP) / BOUND_CLIP
        vec[idx * 2 + 1] = np.clip(ub, -BOUND_CLIP, BOUND_CLIP) / BOUND_CLIP

    # Constraint block (starts after bounds block)
    base = MAX_VARS * 2
    row_size = MAX_VARS + 2
    constraints = _extract_linear_constraints(formula, var_index)
    for ci, (coeffs, rhs, ctype) in enumerate(constraints[:MAX_CONSTRAINTS]):
        offset = base + ci * row_size
        for var_idx, coeff in coeffs.items():
            if var_idx < MAX_VARS:
                vec[offset + var_idx] = np.clip(coeff, -COEFF_CLIP, COEFF_CLIP) / COEFF_CLIP
        vec[offset + MAX_VARS]     = np.clip(rhs, -BOUND_CLIP, BOUND_CLIP) / BOUND_CLIP
        vec[offset + MAX_VARS + 1] = ctype  # 0=leq, 1=eq, 2=geq, 3=neq

    return vec


def decode_assignment(vec: np.ndarray, formula: ParsedFormula) -> dict:
    """Convert generator output vector back to variable assignment dict."""
    assignment = {}
    bounds = _extract_bounds(formula)
    for i, name in enumerate(formula.var_names[:MAX_VARS]):
        raw = float(vec[i])
        lb, ub = bounds.get(name, (-BOUND_CLIP, BOUND_CLIP))
        value = lb + (raw + 1.0) / 2.0 * (ub - lb)
        assignment[name] = int(round(value))
    return assignment


def _extract_bounds(formula: ParsedFormula) -> dict:
    bounds = {name: (-BOUND_CLIP, BOUND_CLIP) for name in formula.var_names}
    for assertion in formula.assertions:
        _parse_bound(assertion, bounds)
    return bounds


def _parse_bound(expr: z3.ExprRef, bounds: dict):
    if not z3.is_bool(expr):
        return
    decl = expr.decl().kind()

    if decl == z3.Z3_OP_AND:
        for child in expr.children():
            _parse_bound(child, bounds)
        return

    if decl in (z3.Z3_OP_LE, z3.Z3_OP_GE, z3.Z3_OP_LT, z3.Z3_OP_GT):
        lhs, rhs = expr.children()
        if z3.is_const(lhs) and z3.is_int_value(rhs):
            name = str(lhs)
            val = rhs.as_long()
            if decl in (z3.Z3_OP_LE, z3.Z3_OP_LT):
                lb, ub = bounds.get(name, (-BOUND_CLIP, BOUND_CLIP))
                bounds[name] = (lb, min(ub, val))
            else:
                lb, ub = bounds.get(name, (-BOUND_CLIP, BOUND_CLIP))
                bounds[name] = (max(lb, val), ub)
        elif z3.is_const(rhs) and z3.is_int_value(lhs):
            name = str(rhs)
            val = lhs.as_long()
            if decl in (z3.Z3_OP_GE, z3.Z3_OP_GT):
                lb, ub = bounds.get(name, (-BOUND_CLIP, BOUND_CLIP))
                bounds[name] = (lb, min(ub, val))
            else:
                lb, ub = bounds.get(name, (-BOUND_CLIP, BOUND_CLIP))
                bounds[name] = (max(lb, val), ub)


def _extract_linear_constraints(formula: ParsedFormula, var_index: dict) -> list:
    constraints = []
    for assertion in formula.assertions:
        _parse_linear(assertion, var_index, constraints)
    return constraints


def _parse_linear(expr: z3.ExprRef, var_index: dict, out: list):
    if not z3.is_bool(expr):
        return
    decl = expr.decl().kind()

    if decl == z3.Z3_OP_AND:
        for child in expr.children():
            _parse_linear(child, var_index, out)
        return

    type_map = {
        z3.Z3_OP_LE: 0, z3.Z3_OP_EQ: 1,
        z3.Z3_OP_GE: 2, z3.Z3_OP_DISTINCT: 3,
    }
    if decl not in type_map:
        return

    ctype = type_map[decl]
    lhs, rhs = expr.children()
    diff = z3.simplify(lhs - rhs)
    coeffs, constant = _extract_coefficients(diff, var_index)
    if coeffs:
        out.append((coeffs, -constant, ctype))


def _extract_coefficients(expr: z3.ExprRef, var_index: dict) -> tuple:
    coeffs = {}
    constant = 0

    def walk(e, sign=1):
        nonlocal constant
        if z3.is_int_value(e):
            constant += sign * e.as_long()
        elif z3.is_const(e) and e.decl().kind() == z3.Z3_OP_UNINTERPRETED:
            idx = var_index.get(str(e))
            if idx is not None:
                coeffs[idx] = coeffs.get(idx, 0) + sign
        elif e.decl().kind() == z3.Z3_OP_ADD:
            for c in e.children():
                walk(c, sign)
        elif e.decl().kind() == z3.Z3_OP_SUB:
            children = list(e.children())
            walk(children[0], sign)
            for c in children[1:]:
                walk(c, -sign)
        elif e.decl().kind() == z3.Z3_OP_UMINUS:
            walk(e.children()[0], -sign)
        elif e.decl().kind() == z3.Z3_OP_MUL:
            children = list(e.children())
            if len(children) == 2 and z3.is_int_value(children[0]):
                walk(children[1], sign * children[0].as_long())
            elif len(children) == 2 and z3.is_int_value(children[1]):
                walk(children[0], sign * children[1].as_long())

    walk(expr)
    return coeffs, constant
