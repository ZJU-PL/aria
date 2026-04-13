# taintmodel/taint.py ----------------------------------------------------------
"""
Algorithm 2  (inferSIC) plus § 6 R-absorbing theory plug-ins.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Optional, Set, Tuple

from z3 import *  # type: ignore

from aria.utils.z3.expr import get_atoms

from .theories import (
    _all,
    _any,
    _bv_constant_sics,
    _normalize_bv_op,
    _simplify_bv_equality,
    _theory_sic_for_parts,
    theory_sic,
)


# ----------------------------------------------------------------- utilities
def _has_quantifier(expr: ExprRef) -> bool:
    stack = [expr]
    while stack:
        node = stack.pop()
        if is_quantifier(node):
            return True
        stack.extend(node.children())
    return False


def _collect_uninterp_consts(expr: ExprRef) -> Set[ExprRef]:
    syms: Set[ExprRef] = set()
    stack = [expr]
    while stack:
        e = stack.pop()
        if is_quantifier(e):
            stack.append(e.body())
            continue
        if not is_app(e):
            continue
        if e.decl().kind() == Z3_OP_UNINTERPRETED and e.num_args() == 0:
            syms.add(e)
        else:
            stack.extend(e.children())
    return syms


# ----------------------------------------------------------- inferSIC engine
def infer_sic_and_wic(
    root: ExprRef,
    targets: Set[ExprRef],
    do_simplify: bool = True,
    verify_wic: bool = False,
    candidate_guards: Optional[Set[ExprRef]] = None,
) -> Tuple[BoolRef, bool]:
    """
    Compute a **quantifier-free** SIC and whether it is a WIC for *root*
    wrt *targets* (fresh consts).

    When ``candidate_guards`` is provided, the traversal also records
    target-free Boolean guards that can later be used by the refinement loop in
    ``QuantSolver``.
    """

    shadow_cache: Dict[ExprRef, Tuple[ExprRef, bool]] = {}
    shadow_constraints = []
    targets_list = list(targets)

    def _is_target_symbol(e: ExprRef) -> bool:
        return any(e.eq(t) for t in targets_list)

    def _contains_target(e: ExprRef) -> bool:
        return any(_is_target_symbol(s) for s in _collect_uninterp_consts(e))

    def record_guard(guard: ExprRef) -> None:
        """
        Export target-free Boolean structure from the taint pass.

        We keep both the whole guard and its atomic predicates so that the
        solver can choose between coarse regions and smaller feature literals
        during refinement.
        """
        if candidate_guards is None:
            return
        if _has_quantifier(guard):
            return

        simplified = simplify(guard)
        if _contains_target(simplified):
            return

        candidate_guards.add(simplified)
        if is_true(simplified) or is_false(simplified) or not is_bool(simplified):
            return

        try:
            for atom in get_atoms(simplified):
                atom = simplify(atom)
                if _has_quantifier(atom) or _contains_target(atom):
                    continue
                candidate_guards.add(atom)
        except Exception:
            return

    # Collect relevant indices for arrays (select/store occurrences)
    rel_indices: Dict[ExprRef, Set[ExprRef]] = {}

    def collect_indices(e: ExprRef) -> None:
        if is_quantifier(e):
            collect_indices(e.body())
            return
        if not is_app(e):
            return
        k = e.decl().kind()
        if k == Z3_OP_SELECT:
            arr, idx = e.children()
            rel_indices.setdefault(arr, set()).add(idx)
        elif k == Z3_OP_STORE:
            arr, idx, val = e.children()
            rel_indices.setdefault(arr, set()).add(idx)
            collect_indices(arr)
            collect_indices(idx)
            collect_indices(val)
            return
        for c in e.children():
            collect_indices(c)

    collect_indices(root)

    associative_ops = {
        Z3_OP_AND,
        Z3_OP_OR,
        Z3_OP_BAND,
        Z3_OP_BOR,
        Z3_OP_BMUL,
        Z3_OP_BADD,
        Z3_OP_ADD,
        Z3_OP_MUL,
    }

    def flatten_assoc(e: ExprRef) -> List[ExprRef]:
        if not is_app(e):
            return [e]
        kind = e.decl().kind()
        if kind not in associative_ops:
            return [e]
        flat: List[ExprRef] = []
        for child in e.children():
            if is_app(child) and child.decl().kind() == kind:
                flat.extend(flatten_assoc(child))
            else:
                flat.append(child)
        return flat

    @lru_cache(maxsize=None)
    def go(e: ExprRef) -> Tuple[BoolRef, bool]:
        if is_quantifier(e):
            return BoolVal(False), False

        # Variables / constants
        if e.decl().kind() == Z3_OP_UNINTERPRETED and not e.num_args():
            if _is_target_symbol(e):
                return BoolVal(False), False
            return BoolVal(True), True
        if e.num_args() == 0:  # numeral
            return BoolVal(True), True

        # Arrays: select/store handling via shadow arrays
        if e.decl().kind() == Z3_OP_SELECT:
            arr, idx = e.children()
            # Re-express select(store(..)) as an ite to eliminate the store
            if arr.decl().kind() == Z3_OP_STORE:
                base, store_idx, store_val = arr.children()
                rewritten = If(store_idx == idx, store_val, Select(base, idx))
                return go(rewritten)
            if arr.decl().kind() == Z3_OP_ITE:
                cond, then_arr, else_arr = arr.children()
                rewritten = If(cond, Select(then_arr, idx), Select(else_arr, idx))
                return go(rewritten)
            if arr.decl().kind() == Z3_OP_CONST_ARRAY and arr.num_args() == 1:
                (value,) = arr.children()
                return go(value)
            idx_sic, idx_wic = go(idx)
            arr_shadow, arr_wic = get_array_shadow(arr)
            sic = And(idx_sic, Select(arr_shadow, idx))
            return sic, idx_wic and arr_wic

        if e.decl().kind() == Z3_OP_STORE:
            arr, idx, val = e.children()
            arr_sic, arr_wic = go(arr)
            idx_sic, idx_wic = go(idx)
            val_sic, val_wic = go(val)
            sic = _all(arr_sic, idx_sic, val_sic)
            return sic, arr_wic and idx_wic and val_wic

        if e.decl().kind() == Z3_OP_EQ and len(e.children()) == 2:
            lhs, rhs = e.children()
            simplified_eq = _simplify_bv_equality(e)
            if not simplified_eq.eq(e):
                return go(simplified_eq)
            if lhs.sort().kind() == Z3_BV_SORT and rhs.sort().kind() == Z3_BV_SORT:
                conds = []
                all_wic = True
                if is_bv_value(lhs) and is_app(rhs):
                    rhs_sub = [go(c) for c in rhs.children()]
                    rhs_consts = _bv_constant_sics(
                        _normalize_bv_op(rhs.decl().kind(), rhs.decl().name()),
                        list(rhs.children()),
                        [item[0] for item in rhs_sub],
                    )
                    for const, cond, is_wic in rhs_consts:
                        if const.eq(lhs):
                            conds.append(cond)
                            all_wic = all_wic and is_wic and all(
                                item[1] for item in rhs_sub
                            )
                if is_bv_value(rhs) and is_app(lhs):
                    lhs_sub = [go(c) for c in lhs.children()]
                    lhs_consts = _bv_constant_sics(
                        _normalize_bv_op(lhs.decl().kind(), lhs.decl().name()),
                        list(lhs.children()),
                        [item[0] for item in lhs_sub],
                    )
                    for const, cond, is_wic in lhs_consts:
                        if const.eq(rhs):
                            conds.append(cond)
                            all_wic = all_wic and is_wic and all(
                                item[1] for item in lhs_sub
                            )
                if conds:
                    return _any(*conds), all_wic

        if e.decl().kind() in associative_ops:
            flat_children = flatten_assoc(e)
            if len(flat_children) > e.num_args():
                sub = [go(c) for c in flat_children]
                sub_sic = [p[0] for p in sub]
                sub_wic = [p[1] for p in sub]
                psi, psi_wic = _theory_sic_for_parts(
                    e.decl().kind(), flat_children, sub_sic, e.decl().name()
                )
                combined = Or(psi, _all(*sub_sic))
                if not is_false(psi):
                    record_guard(psi)
                record_guard(combined)
                is_wic = psi_wic and all(sub_wic)
                return combined, is_wic

        # Generic recursive case
        sub = [go(c) for c in e.children()]
        sub_sic = [p[0] for p in sub]
        sub_wic = [p[1] for p in sub]
        psi, psi_wic = theory_sic(e, sub_sic)
        combined = Or(psi, _all(*sub_sic))
        if not is_false(psi):
            record_guard(psi)
        record_guard(combined)
        is_wic = psi_wic and all(sub_wic)
        return combined, is_wic

    def get_array_shadow(arr: ExprRef) -> Tuple[ExprRef, bool]:
        """
        Build an array of Boolean taint summaries for ``arr``.

        ``Select(shadow(arr), i)`` means "reading ``arr[i]`` is independent of
        the target symbols under the current path conditions".
        """
        if arr in shadow_cache:
            return shadow_cache[arr]

        k = arr.decl().kind()
        if k == Z3_OP_STORE:
            a, i, v = arr.children()
            base_shadow, base_wic = get_array_shadow(a)
            v_sic, v_wic = go(v)
            sh = Store(base_shadow, i, v_sic)
            shadow_cache[arr] = (sh, base_wic and v_wic)
            return shadow_cache[arr]

        if k == Z3_OP_UNINTERPRETED and arr.num_args() == 0:
            base_taint = BoolVal(False) if arr in targets else BoolVal(True)
            sh = K(arr.domain(), base_taint)
            for ridx in rel_indices.get(arr, set()):
                if _contains_target(ridx):
                    continue
                shadow_constraints.append(Select(sh, ridx) == base_taint)
            shadow_cache[arr] = (sh, True)
            return shadow_cache[arr]

        if k == Z3_OP_CONST_ARRAY and arr.num_args() == 1:
            (value,) = arr.children()
            value_sic, value_wic = go(value)
            sh = K(arr.domain(), value_sic)
            shadow_cache[arr] = (sh, value_wic)
            return shadow_cache[arr]

        if k == Z3_OP_ITE and arr.num_args() == 3:
            cond, then_arr, else_arr = arr.children()
            cond_sic, cond_wic = go(cond)
            then_shadow, then_wic = get_array_shadow(then_arr)
            else_shadow, else_wic = get_array_shadow(else_arr)
            sh = If(cond, then_shadow, else_shadow)
            shadow_cache[arr] = (sh, cond_wic and then_wic and else_wic)
            if not is_true(cond_sic):
                shadow_constraints.append(cond_sic)
            return shadow_cache[arr]

        # Fallback: conservative (assume dependence everywhere)
        sh = K(arr.domain(), BoolVal(False))
        shadow_cache[arr] = (sh, False)
        return shadow_cache[arr]

    res, is_wic = go(root)
    if shadow_constraints:
        res = And(res, *shadow_constraints)
    if do_simplify:
        res = simplify(res)
    record_guard(res)
    if _contains_target(res):
        res = BoolVal(False)
        is_wic = False
    if verify_wic:
        is_wic = _prove_wic(root, res, set(targets_list))
    return res, is_wic


def infer_sic_candidates(
    root: ExprRef,
    targets: Set[ExprRef],
    do_simplify: bool = True,
    verify_wic: bool = False,
) -> Tuple[BoolRef, bool, List[BoolRef]]:
    """
    Return the inferred SIC together with target-free guard candidates that were
    observed during taint propagation.

    The returned list is sorted deterministically so the refinement loop can be
    replayed and tested without depending on hash iteration order.
    """
    candidate_guards: Set[ExprRef] = set()
    sic, is_wic = infer_sic_and_wic(
        root,
        targets,
        do_simplify=do_simplify,
        verify_wic=verify_wic,
        candidate_guards=candidate_guards,
    )

    ordered = sorted(candidate_guards, key=lambda expr: expr.sexpr())
    return sic, is_wic, [expr for expr in ordered if is_bool(expr)]


def infer_sic(root: ExprRef, targets: Set[ExprRef], do_simplify: bool = True) -> BoolRef:
    """Compatibility wrapper returning only the SIC."""
    res, _ = infer_sic_and_wic(root, targets, do_simplify=do_simplify)
    return res


# --------------------------------------------------------------- WIC checker
def _prove_wic(root: ExprRef, sic: BoolRef, targets: Set[ExprRef]) -> bool:
    """
    Check whether SIC is also a WIC by asking Z3:
    ∃t,t'. sic(t) ∧ sic(t') ∧ root(t) ≠ root(t') ?
    If unsat, then WIC holds.
    """
    if not targets:
        return True

    prime_map = {t: Const(f"{t.decl().name()}__prime", t.sort()) for t in targets}

    def subst(expr: ExprRef, mapping: Dict[ExprRef, ExprRef]) -> ExprRef:
        return substitute(expr, *[(k, v) for k, v in mapping.items()])

    root_prime = subst(root, prime_map)
    sic_prime = subst(sic, prime_map)

    diff = Xor(root, root_prime) if is_bool(root) else root != root_prime

    solver = Solver()
    solver.add(sic, sic_prime, diff)
    return solver.check() == unsat


# ------------------------------------------------------ explicit-taint path
def _taint_expr(expr: ExprRef, taint_map: Dict[ExprRef, BoolRef]) -> BoolRef:
    """
    Build taint constraints using explicit taint variables (Algorithm-style).
    """

    @lru_cache(maxsize=None)
    def go(e: ExprRef) -> BoolRef:
        if e.decl().kind() == Z3_OP_UNINTERPRETED and e.num_args() == 0:
            return taint_map.get(e, BoolVal(True))
        if e.num_args() == 0:
            return BoolVal(True)
        sub = [go(c) for c in e.children()]
        psi, _ = theory_sic(e, sub)
        return Or(psi, _all(*sub))

    return go(expr)


def infer_sic_with_taints(
    root: ExprRef,
    targets: Set[ExprRef],
    do_simplify: bool = True,
    verify_wic: bool = False,
) -> Tuple[BoolRef, Dict[ExprRef, BoolRef]]:
    """
    Compute SIC by generating taint variables, constraints, and instantiating them.
    Returns (sic, taint_var_map).
    """
    symbols = _collect_uninterp_consts(root)
    taint_map: Dict[ExprRef, BoolRef] = {s: Bool(f"{s.decl().name()}__taint") for s in symbols}
    taint_constraint = _taint_expr(root, taint_map)

    subs = [
        (tvar, BoolVal(False) if sym in targets else BoolVal(True))
        for sym, tvar in taint_map.items()
    ]
    instantiated = substitute(taint_constraint, *subs)
    if do_simplify:
        instantiated = simplify(instantiated)
    return instantiated, taint_map
