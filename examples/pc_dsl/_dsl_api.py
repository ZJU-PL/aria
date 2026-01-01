from __future__ import annotations

import sys
from typing import Any, Iterable

import z3

from ._dsl_nodes import Namespace, Value, Z3Wrapper  # type: ignore
from ._dsl_sorts import resolve_sort
from ._dsl_traverse import traverse

__all__ = [
    "get_constraints",
    "get_free_vars",
    "rename_free_vars",
    "to_z3",
    "from_z3",
]


def get_constraints(solver_cls: type) -> tuple[Any, ...]:
    return getattr(solver_cls, "_SolverMeta__constraints", ())


def get_free_vars(expr: z3.AstRef) -> list[z3.ExprRef]:
    seen: set[int] = set()
    out: list[z3.ExprRef] = []
    stack = [expr]
    while stack:
        e = stack.pop()
        try:
            key = z3.Z3_get_ast_id(e.ctx_ref(), e.as_ast())
        except Exception:
            key = id(e)
        if key in seen:
            continue
        seen.add(key)
        try:
            if (
                z3.is_const(e)
                and e.num_args() == 0
                and e.decl().kind() == z3.Z3_OP_UNINTERPRETED
            ):
                out.append(e)
        except Exception:
            pass
        try:
            stack.extend(list(e.children()))
        except Exception:
            pass
    return out


def rename_free_vars(expr: z3.AstRef, prefix: str = "", postfix: str = "") -> z3.AstRef:
    consts = get_free_vars(expr)
    subs = []
    for c in consts:
        try:
            name = c.decl().name()
        except Exception:
            name = str(c)
        new = z3.Const(f"{prefix}{name}{postfix}", c.sort())
        subs.append((c, new))
    if not subs:
        return expr
    return z3.substitute(expr, *subs)


def from_z3(expr: Any) -> Value:
    if isinstance(expr, Value):
        return expr
    ns = Namespace()
    return Z3Wrapper(expr, ns=lambda: ns)  # type: ignore


def to_z3(value: Any, env: dict[str, Any] | None = None) -> Any:
    if isinstance(value, z3.AstRef) or isinstance(value, z3.SortRef):
        return value
    if not isinstance(value, Value):
        return value
    if isinstance(value, Z3Wrapper):
        return value.expr
    # If env is not provided, try to infer from any Variable occurrences
    if env is None:
        env = {}
    vars: dict[str, Any] = {}
    for k, v in env.items():
        if isinstance(v, z3.AstRef):
            vars[k] = v
        else:
            sort = resolve_sort(v)
            vars[k] = z3.Const(k, sort)
    return traverse(value, vars)
