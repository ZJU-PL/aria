# sic_smt/taint.py ------------------------------------------------------------
"""
Algorithm 2  (inferSIC) plus § 6 R-absorbing theory plug-ins.
"""
from __future__ import annotations
from functools import lru_cache
from typing import Dict, Sequence, Set, Tuple
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
def _sic_bool(op, args, subs) -> Tuple[BoolRef, bool]:
    if op == Z3_OP_NOT and len(args) == 1:
        return subs[0], True
    if len(args) == 2:
        a, b = args
        sa, sb = subs
        if op == Z3_OP_AND:
            return Or(And(sa, a == False), And(sb, b == False)), True
        if op == Z3_OP_OR:
            return Or(And(sa, a == True), And(sb, b == True)), True
        if op == Z3_OP_IMPLIES:
            return Or(And(sa, a == False), And(sb, b == True)), True
        if op == Z3_OP_XOR:
            return And(sa, sb), False
        if op == Z3_OP_EQ:
            return Or(And(sa, a == b), And(sb, a == b)), True
    if op == Z3_OP_DISTINCT:
        return And(*subs), False
    return BoolVal(False), False


_bv_zero = lambda t: t.sort().cast(0)  # noqa: E731


def _sic_bv(op, args, subs) -> Tuple[BoolRef, bool]:
    # Unary ops passthrough
    if len(args) == 1:
        a = args[0]
        sa = subs[0]
        if a.sort().kind() != Z3_BV_SORT:
            return BoolVal(False), False
        if op in (Z3_OP_BNOT, Z3_OP_SIGN_EXT, Z3_OP_ZERO_EXT, Z3_OP_EXTRACT, Z3_OP_ROTATE_LEFT, Z3_OP_ROTATE_RIGHT):
            return sa, True
        return BoolVal(False), False

    if len(args) != 2:
        return BoolVal(False), False

    a, b = args
    sa, sb = subs
    if a.sort().kind() != Z3_BV_SORT or b.sort().kind() != Z3_BV_SORT:
        return BoolVal(False), False

    z_a = _bv_zero(a)
    if op in (Z3_OP_BAND, Z3_OP_BMUL):
        return Or(And(sa, a == z_a), And(sb, b == z_a)), True
    if op == Z3_OP_BOR:
        ones = ~z_a
        return Or(And(sa, a == ones), And(sb, b == ones)), True
    if op in (Z3_OP_BXOR,):
        return And(sa, sb), False
    if op in (Z3_OP_BSHL, Z3_OP_BLSHR):
        width = a.size()
        shift_too_large = UGE(b, BitVecVal(width, b.size()))
        return And(sb, shift_too_large), True
    if op in (Z3_OP_BASHR, Z3_OP_BSHR):
        width = a.size()
        shift_too_large = UGE(b, BitVecVal(width, b.size()))
        return And(sb, shift_too_large), True
    if op in (Z3_OP_BUDIV, Z3_OP_BSDIV, Z3_OP_BSMOD, Z3_OP_BUREM):
        zero = _bv_zero(b)
        return And(sb, b != zero), False
    if op in (Z3_OP_BADD, Z3_OP_BSUB):
        return And(sa, sb), False
    if op in (Z3_OP_CONCAT,):
        return And(sa, sb), False
    if op in (
        Z3_OP_BLT,
        Z3_OP_BGT,
        Z3_OP_BLE,
        Z3_OP_BGE,
        Z3_OP_ULEQ,
        Z3_OP_UGEQ,
        Z3_OP_ULT,
        Z3_OP_UGT,
        Z3_OP_EQ,
    ):
        return And(sa, sb), False
    return BoolVal(False), False


def _sic_arith(op, args, subs) -> Tuple[BoolRef, bool]:
    if len(args) == 1:
        a = args[0]
        sa = subs[0]
        if op in (Z3_OP_UMINUS,):
            return sa, True
        if op in (Z3_OP_ABS,):
            return sa, True
        return BoolVal(False), False
    if op == Z3_OP_MUL and len(args) == 2:
        a, b = args
        sa, sb = subs
        zero = IntVal(0) if a.sort().kind() == Z3_INT_SORT else RealVal(0)
        return Or(And(sa, a == zero), And(sb, b == zero)), True
    if op in (Z3_OP_ADD, Z3_OP_SUB):
        return And(*subs), False
    if op in (Z3_OP_DIV, Z3_OP_IDIV, Z3_OP_REM, Z3_OP_MOD) and len(args) == 2:
        a, b = args
        sa, sb = subs
        zero = IntVal(0) if a.sort().kind() == Z3_INT_SORT else RealVal(0)
        return And(sa, sb, b != zero), False
    if op in (Z3_OP_LT, Z3_OP_GT, Z3_OP_LE, Z3_OP_GE):
        return And(*subs), False
    if op == Z3_OP_DISTINCT:
        return And(*subs), False
    return BoolVal(False), False


def _sic_ite(op, args, subs) -> Tuple[BoolRef, bool]:
    if op != Z3_OP_ITE:
        return BoolVal(False), False
    c, t, e = args
    sc, st, se = subs
    return (
        Or(
            And(sc, _any(st, se), If(c, st, se)),
            And(st, se, t == e),
        ),
        False,
    )


def _sic_array(op, args, subs) -> Tuple[BoolRef, bool]:
    # array equality: independent if both sides are
    if op == Z3_OP_EQ and len(args) == 2 and all(is_array(a) for a in args):
        return And(*subs), False
    # select(store(a, i, v), j): handled by rewrite in main engine
    # store(a, i, v): independent if all parts are
    if op == Z3_OP_STORE and len(args) == 3:
        return And(*subs), False
    # select(a, i): independent if both independent
    if op == Z3_OP_SELECT and len(args) == 2:
        return And(*subs), False
    return BoolVal(False), False


# Plug-in list evaluated in precedence order
_THEORY_PLUGINS = (_sic_bool, _sic_bv, _sic_arith, _sic_ite, _sic_array)


def theory_sic(expr: ExprRef, subsic: Sequence[BoolRef]) -> Tuple[BoolRef, bool]:
    """
    Dispatcher returning an operator-specific Ψ_f (can be False).
    """
    op = expr.decl().kind()
    args = list(expr.children())
    # Normalize store/select chains: handled earlier for select, but ensure store is visible here too
    for plugin in _THEORY_PLUGINS:
        psi, is_wic = plugin(op, args, subsic)
        if not is_false(psi):
            return psi, is_wic
    return BoolVal(False), False


# ----------------------------------------------------------- inferSIC engine
def infer_sic_and_wic(
    root: ExprRef,
    targets: Set[ExprRef],
    do_simplify: bool = True,
    verify_wic: bool = False,
) -> Tuple[BoolRef, bool]:
    """
    Compute a **quantifier-free** SIC and whether it is a WIC for *root*
    wrt *targets* (fresh consts).
    """

    shadow_cache: Dict[ExprRef, Tuple[ExprRef, bool]] = {}
    shadow_constraints = []
    targets_set = set(targets)

    def _node_count(e: ExprRef) -> int:
        seen = set()
        stack = [e]
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            stack.extend(n.children())
        return len(seen)

    def _contains_target(e: ExprRef) -> bool:
        return any(s in targets_set for s in _collect_uninterp_consts(e))

    # Collect relevant indices for arrays (select/store occurrences)
    rel_indices: Dict[ExprRef, Set[ExprRef]] = {}

    def collect_indices(e: ExprRef):
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

    @lru_cache(maxsize=None)
    def go(e: ExprRef) -> Tuple[BoolRef, bool]:
        if is_quantifier(e):
            return BoolVal(False), False

        # Variables / constants
        if e.decl().kind() == Z3_OP_UNINTERPRETED and not e.num_args():
            if e in targets:
                return BoolVal(False), False
            return BoolVal(True), True
        if e.num_args() == 0:  # numeral
            return BoolVal(True), True

        # Arrays: select/store handling via shadow arrays
        if e.decl().kind() == Z3_OP_SELECT:
            arr, idx = e.children()
            # Re-express select(store(..)) as an ite to eliminate the store and
            # let generic rules compute a SIC without keeping the target index.
            if arr.decl().kind() == Z3_OP_STORE:
                base, store_idx, store_val = arr.children()
                rewritten = If(store_idx == idx, store_val, Select(base, idx))
                return go(rewritten)
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

        # Generic recursive case
        sub = [go(c) for c in e.children()]
        sub_sic = [p[0] for p in sub]
        sub_wic = [p[1] for p in sub]
        psi, psi_wic = theory_sic(e, sub_sic)
        combined = Or(psi, _all(*sub_sic))
        is_wic = psi_wic and all(sub_wic)
        return combined, is_wic

    def get_array_shadow(arr: ExprRef) -> Tuple[ExprRef, bool]:
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
                # Avoid reintroducing target symbols into the SIC
                if _contains_target(ridx):
                    continue
                shadow_constraints.append(Select(sh, ridx) == base_taint)
            shadow_cache[arr] = (sh, True)
            return shadow_cache[arr]

        # Fallback shadow: conservative taint False everywhere
        sh = K(arr.domain(), BoolVal(False))
        shadow_cache[arr] = (sh, False)
        return shadow_cache[arr]

    res, is_wic = go(root)
    if shadow_constraints:
        res = And(res, *shadow_constraints)
    if do_simplify:
        res = simplify(res)
    if _contains_target(res):
        res = BoolVal(False)
        is_wic = False
    # Bound SIC size to stay close to paper's linear overhead
    orig_size = _node_count(root)
    sic_size = _node_count(res)
    if sic_size > max(orig_size * 10, 50):
        res = simplify(res, flat=True, elim_and=True, elim_ite=True)
        sic_size = _node_count(res)
    if sic_size > max(orig_size * 10, 50):
        res = BoolVal(False)
        is_wic = False
    if verify_wic and is_wic:
        is_wic = _prove_wic(root, res, targets_set)
    return res, is_wic


def infer_sic(root: ExprRef, targets: Set[ExprRef], do_simplify: bool = True) -> BoolRef:
    """Compatibility wrapper returning only the SIC."""
    res, _ = infer_sic_and_wic(root, targets, do_simplify=do_simplify)
    return res


# ---------------------------------------------------------------- taint vars (explicit)
def _collect_uninterp_consts(expr: ExprRef) -> Set[ExprRef]:
    syms = set()
    stack = [expr]
    while stack:
        e = stack.pop()
        if e.decl().kind() == Z3_OP_UNINTERPRETED and e.num_args() == 0:
            syms.add(e)
        else:
            stack.extend(e.children())
    return syms


def _prove_wic(root: ExprRef, sic: BoolRef, targets: Set[ExprRef]) -> bool:
    """
    Check whether SIC is also a WIC by asking Z3:
    ∃t,t'. sic(t) ∧ sic(t') ∧ root(t) ≠ root(t') ?
    If unsat, then WIC holds.
    """
    if not targets:
        return True

    prime_map = {}
    for t in targets:
        nm = f"{t.decl().name()}__prime"
        prime_map[t] = Const(nm, t.sort())

    def subst(expr, mapping):
        subs = [(k, v) for k, v in mapping.items()]
        return substitute(expr, *subs)

    root_prime = subst(root, prime_map)
    sic_prime = subst(sic, prime_map)

    if is_bool(root):
        diff = Xor(root, root_prime)
    else:
        diff = root != root_prime

    solver = Solver()
    solver.add(sic, sic_prime, diff)
    res = solver.check()
    return res == unsat


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

    # Instantiate taint vars: targets -> False, others -> True
    subs = []
    for sym, tvar in taint_map.items():
        val = BoolVal(False) if sym in targets else BoolVal(True)
        subs.append((tvar, val))
    instantiated = substitute(taint_constraint, *subs)
    if do_simplify:
        instantiated = simplify(instantiated)
    if verify_wic:
        _ = verify_wic  # flag ignored here; verification happens in infer_sic_and_wic path
    return instantiated, taint_map
