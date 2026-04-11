"""Shared helpers for QE projection variable handling."""

from collections.abc import Iterable
from typing import Any, List, Optional, cast

import z3

from aria.utils.z3.expr import get_expr_vars


def normalize_vars(vars_or_var: Optional[Any]) -> List[z3.ExprRef]:
    """Normalize a Z3 variable or iterable of variables to a list."""
    if vars_or_var is None:
        return []
    if z3.is_expr(vars_or_var):
        return [cast(z3.ExprRef, vars_or_var)]
    return [cast(z3.ExprRef, var) for var in cast(Iterable[Any], vars_or_var)]


def get_projection_vars(
    phi: Any,
    qvars: List[z3.ExprRef],
    keep_vars: Optional[List[z3.ExprRef]],
) -> List[z3.ExprRef]:
    """Return quantified and projected-away variables for one QE step."""
    if keep_vars is None:
        return qvars

    qvar_ids = {var.get_id() for var in qvars}
    keep_var_ids = {var.get_id() for var in keep_vars}
    if qvar_ids & keep_var_ids:
        raise ValueError("qvars and keep_vars must be disjoint")

    projection_vars = list(qvars)
    for var in get_expr_vars(phi):
        if var.get_id() in qvar_ids or var.get_id() in keep_var_ids:
            continue
        projection_vars.append(var)
    return projection_vars
