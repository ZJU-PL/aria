# sic_smt/util.py -------------------------------------------------------------
from __future__ import annotations
import itertools
from collections import defaultdict
from typing import Dict, Iterator, Sequence, Tuple
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


def iter_prefix_blocks(f: ExprRef) -> Iterator[QuantifierRef]:
    """
    Iterate over quantifier blocks in a prenex-style prefix.

    Stops once a non-quantifier body is reached.
    """
    cur = f
    while is_quantifier(cur):
        yield cur
        cur = cur.body()


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


# -------------------------------------------------------------- let sharing
def let_sharing(expr: ExprRef) -> ExprRef:
    """
    Introduce sharing (Tseitin-style) for repeated subexpressions to emulate SSA.
    Leaves quantifiers untouched.
    """
    counts: Dict[ExprRef, int] = defaultdict(int)

    def count(e: ExprRef):
        counts[e] += 1
        if is_quantifier(e):
            return
        for c in e.children():
            count(c)

    count(expr)

    shared_eqs = []
    cache: Dict[ExprRef, ExprRef] = {}

    def rebuild(e: ExprRef) -> ExprRef:
        if is_quantifier(e):
            return e
        if counts[e] > 1 and e.num_args() > 0:
            if e in cache:
                return cache[e]
            new_children = [rebuild(c) for c in e.children()]
            try:
                rebuilt = e.decl()(*new_children)
            except Exception:
                rebuilt = e
            name = fresh_const(e.sort(), "ssa")
            shared_eqs.append(name == rebuilt)
            cache[e] = name
            return name
        if e.num_args() == 0:
            return e
        new_children = [rebuild(c) for c in e.children()]
        try:
            return e.decl()(*new_children)
        except Exception:
            return e

    new_root = rebuild(expr)
    if shared_eqs:
        return And(new_root, *shared_eqs)
    return new_root
