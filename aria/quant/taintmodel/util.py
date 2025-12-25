# sic_smt/util.py -------------------------------------------------------------
from __future__ import annotations
import itertools
from typing import Dict, Iterable, Iterator, Sequence, Tuple
from z3 import *  # type: ignore


# --------------------------------------------------------------- fresh const
_counter = itertools.count()


def fresh_const(sort, prefix: str = "sk") -> ExprRef:
    return Const(f"{prefix}_{next(_counter)}", sort)


# ------------------------------------------------------------- quant blocks
def iter_qblocks(f: ExprRef) -> Iterator[QuantifierRef]:
    """
    Depth-first iterator over **outermost** quantifier blocks appearing in *f*.
    """
    stack = [f]
    while stack:
        n = stack.pop()
        if is_quantifier(n):
            yield n
        else:
            stack.extend(reversed(n.children()))


# ---------------------------------------------------------------- skolemise
def skolemize_block(q: QuantifierRef) -> Tuple[ExprRef, Dict[str, ExprRef]]:
    """
    Skolemise ONE block; return (body_with_consts, mapping).

    Works for ∀ and ∃ (pure 1st-order, no dependencies on previous vars
    because we process outside-in).
    """
    names = [q.var_name(i) or f"V{i}" for i in range(q.num_vars())]
    sorts = [q.var_sort(i) for i in range(q.num_vars())]
    fresh = {n: fresh_const(s, "sk") for n, s in zip(names, sorts)}

    # substitute_vars takes them in *reverse* order
    body = substitute_vars(q.body(), *reversed(list(fresh.values())))
    return body, fresh


# -------------------------------------------------------------- projection
def project_model(model: ModelRef, keep: Sequence[ExprRef]) -> ModelRef:
    """
    Return a *shallow* copy of model restricted to declarations in *keep*.
    """
    m = Model()
    for d in model.decls():
        c = d()
        if c in keep:
            m[d] = model[d]
    return m
