# sic_smt/taint.py ------------------------------------------------------------
"""
Algorithm 2  (inferSIC) plus § 6 R-absorbing theory plug-ins.
"""
from __future__ import annotations
from functools import lru_cache
from typing import Iterable, List, Sequence, Set
from z3 import *  # type: ignore


# ----------------------------------------------------------------- utilities
def _all(*xs: BoolRef) -> BoolRef:
    if not xs:
        return BoolVal(True)
    return And(*xs)


def _any(*xs: BoolRef) -> BoolRef:
    if not xs:
        return BoolVal(False)
    return Or(*xs)


# ---------------------------------------------------------------- theory SIC
def _sic_bool(op, args, subs):
    a, b = args
    sa, sb = subs
    if op == Z3_OP_AND:
        return Or(And(sa, a == False), And(sb, b == False))
    if op == Z3_OP_OR:
        return Or(And(sa, a == True), And(sb, b == True))
    if op == Z3_OP_IMPLIES:
        return Or(And(sa, a == False), And(sb, b == True))
    return False


_bv_zero = lambda t: t.sort().cast(0)  # noqa: E731


def _sic_bv(op, args, subs):
    if len(args) != 2:
        return False
    a, b = args
    sa, sb = subs
    z_a = _bv_zero(a)
    if op in (Z3_OP_BAND, Z3_OP_BMUL):
        return Or(And(sa, a == z_a), And(sb, b == z_a))
    if op == Z3_OP_BOR:
        ones = ~z_a
        return Or(And(sa, a == ones), And(sb, b == ones))
    return False


def _sic_arith(op, args, subs):
    if op == Z3_OP_MUL and len(args) == 2:
        a, b = args
        sa, sb = subs
        zero = IntVal(0) if a.sort().kind() == Z3_INT_SORT else RealVal(0)
        return Or(And(sa, a == zero), And(sb, b == zero))
    return False


def _sic_ite(op, args, subs):
    if op != Z3_OP_ITE:
        return False
    c, t, e = args
    sc, st, se = subs
    return Or(
        And(sc, _any(st, se), Ite(c, st, se)),
        And(st, se, t == e),
    )


def _sic_array(op, args, subs):
    # (select (store a i v) j)
    if op == Z3_OP_SELECT_AS_ARRAY:
        return False
    if op == Z3_OP_SELECT and args[0].decl().kind() == Z3_OP_STORE:
        store = args[0]
        a, i, v = store.children()
        j = args[1]
        sa, si, sv, sj = (
            subs[0].children()[0],
            subs[0].children()[1],
            subs[0].children()[2],
            subs[1],
        )
        # simplified: independence if i == j OR v/select agree
        return Or(
            And(si, sj, i == j),
            And(sv, Select(a, j) == v),
        )
    return False


# Plug-in list evaluated in precedence order
_THEORY_PLUGINS = (_sic_bool, _sic_bv, _sic_arith, _sic_ite, _sic_array)


def theory_sic(expr: ExprRef, subsic: Sequence[BoolRef]) -> BoolRef:
    """
    Dispatcher returning an operator-specific Ψ_f (can be False).
    """
    op = expr.decl().kind()
    args = list(expr.children())
    for plugin in _THEORY_PLUGINS:
        psi = plugin(op, args, subsic)
        if not is_false(psi):
            return psi
    return BoolVal(False)


# ----------------------------------------------------------- inferSIC engine
def infer_sic(root: ExprRef, targets: Set[ExprRef]) -> BoolRef:
    """
    Compute a **quantifier-free** SIC for *root* wrt *targets* (fresh consts).
    """

    @lru_cache(maxsize=None)
    def go(e: ExprRef) -> BoolRef:
        if e.decl().kind() == Z3_OP_UNINTERPRETED and not e.num_args():
            return BoolVal(False) if e in targets else BoolVal(True)
        if e.num_args() == 0:  # numeral
            return BoolVal(True)

        sub = [go(c) for c in e.children()]
        psi = theory_sic(e, sub)
        return Or(psi, _all(*sub))

    return go(root)
