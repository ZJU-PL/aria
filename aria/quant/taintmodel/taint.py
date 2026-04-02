# sic_smt/taint.py ------------------------------------------------------------
"""
Algorithm 2  (inferSIC) plus § 6 R-absorbing theory plug-ins.
"""
from __future__ import annotations
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Set, Tuple
from z3 import *  # type: ignore

from aria.utils.z3_expr_utils import get_atoms


# ----------------------------------------------------------------- utilities
def _all(*xs: BoolRef) -> BoolRef:
    if not xs:
        return BoolVal(True)
    return And(*xs)


def _any(*xs: BoolRef) -> BoolRef:
    if not xs:
        return BoolVal(False)
    return Or(*xs)


def _has_quantifier(expr: ExprRef) -> bool:
    stack = [expr]
    while stack:
        node = stack.pop()
        if is_quantifier(node):
            return True
        stack.extend(node.children())
    return False


# ---------------------------------------------------------------- theory SIC
def _sic_bool(op, args, subs) -> Tuple[BoolRef, bool]:
    if op == Z3_OP_NOT and len(args) == 1:
        return subs[0], True
    if op == Z3_OP_AND and len(args) >= 2:
        return _any(*[_all(sub, arg == False) for arg, sub in zip(args, subs)]), True
    if op == Z3_OP_OR and len(args) >= 2:
        return _any(*[_all(sub, arg == True) for arg, sub in zip(args, subs)]), True
    if len(args) == 2:
        a, b = args
        sa, sb = subs
        if op == Z3_OP_IMPLIES:
            return _any(_all(sa, a == False), _all(sb, b == True)), True
        # The paper does not define a special equality rule here; falling back
        # to conjunction preserves SIC soundness for mixed dependent terms.
        if op == Z3_OP_XOR:
            return _all(sa, sb), False
    if op == Z3_OP_DISTINCT:
        return _all(*subs), False
    return BoolVal(False), False


_bv_zero = lambda t: t.sort().cast(0)  # noqa: E731


def _bv_ones(t):
    return ~_bv_zero(t)


def _bv_signed_min(t):
    return BitVecVal(1 << (t.size() - 1), t.size())


def _bv_signed_max(t):
    return BitVecVal((1 << (t.size() - 1)) - 1, t.size())


def _bv_one(t):
    return BitVecVal(1, t.size())


def _normalize_bv_op(op: int, op_name: str) -> int:
    prefix_map = (
        ("bvudiv", Z3_OP_BUDIV),
        ("bvurem", Z3_OP_BUREM),
        ("bvsdiv", Z3_OP_BSDIV),
        ("bvsrem", Z3_OP_BSREM),
        ("bvsmod", Z3_OP_BSMOD),
        ("bvashr", Z3_OP_BASHR),
        ("bvlshr", Z3_OP_BLSHR),
        ("bvshl", Z3_OP_BSHL),
        ("bvult", Z3_OP_ULT),
        ("bvule", Z3_OP_ULEQ),
        ("bvugt", Z3_OP_UGT),
        ("bvuge", Z3_OP_UGEQ),
        ("bvslt", Z3_OP_SLT),
        ("bvsle", Z3_OP_SLEQ),
        ("bvsgt", Z3_OP_SGT),
        ("bvsge", Z3_OP_SGEQ),
        ("bvand", Z3_OP_BAND),
        ("bvor", Z3_OP_BOR),
        ("bvxor", Z3_OP_BXOR),
        ("bvadd", Z3_OP_BADD),
        ("bvsub", Z3_OP_BSUB),
        ("bvmul", Z3_OP_BMUL),
        ("concat", Z3_OP_CONCAT),
    )
    for prefix, normalized in prefix_map:
        if op_name.startswith(prefix):
            return normalized
    return op


def _bv_constant_sics(
    op: int, args: Sequence[ExprRef], subs: Sequence[BoolRef]
) -> Sequence[Tuple[ExprRef, BoolRef, bool]]:
    if not args or any(arg.sort().kind() != Z3_BV_SORT for arg in args):
        return []

    if len(args) == 1:
        a = args[0]
        sa = subs[0]
        if op == Z3_OP_EXTRACT:
            hi, lo = a.decl().params() if a.decl().kind() == Z3_OP_EXTRACT else (None, None)
            width = hi - lo + 1 if hi is not None and lo is not None else a.size()
            return [
                (BitVecVal(0, width), _all(sa, a == _bv_zero(a)), True),
                (BitVecVal((1 << width) - 1, width), _all(sa, a == _bv_ones(a)), True),
            ]
        return []

    a, b = args[0], args[1]
    sa, sb = subs[0], subs[1]
    zero = _bv_zero(a)
    ones = _bv_ones(a)
    one = _bv_one(a)
    width = a.size()

    if op in (Z3_OP_BAND, Z3_OP_BMUL):
        return [
            (
                zero,
                _any(*[_all(sub, arg == _bv_zero(arg)) for arg, sub in zip(args, subs)]),
                True,
            )
        ]
    if op == Z3_OP_BOR:
        return [
            (
                ones,
                _any(*[_all(sub, arg == _bv_ones(arg)) for arg, sub in zip(args, subs)]),
                True,
            )
        ]
    if op in (Z3_OP_BSHL, Z3_OP_BLSHR):
        shift_too_large = UGE(b, BitVecVal(width, b.size()))
        return [(zero, _any(_all(sa, a == zero), _all(sb, shift_too_large)), True)]
    if op == Z3_OP_BASHR:
        return [
            (zero, _all(sa, a == zero), True),
            (ones, _all(sa, a == ones), True),
        ]
    if op == Z3_OP_BUDIV:
        return [
            (ones, _all(sb, b == zero), True),
            (zero, _all(sa, sb, a == zero, b != zero), True),
        ]
    if op == Z3_OP_BUREM:
        return [(zero, _any(_all(sa, a == zero), _all(sb, b == one)), True)]
    if op == Z3_OP_BSDIV:
        return [
            (zero, _all(sa, sb, a == zero, b != zero), True),
            (_bv_signed_min(a), _all(sa, sb, a == _bv_signed_min(a), b == ones), True),
        ]
    if op in (Z3_OP_BSREM, Z3_OP_BSMOD):
        return [(zero, _any(_all(sa, a == zero), _all(sb, b == one), _all(sb, b == ones)), True)]
    if op == Z3_OP_CONCAT:
        total_width = sum(arg.size() for arg in args)
        return [
            (BitVecVal(0, total_width), _all(*[_all(sub, arg == _bv_zero(arg)) for arg, sub in zip(args, subs)]), True),
            (
                BitVecVal((1 << total_width) - 1, total_width),
                _all(*[_all(sub, arg == _bv_ones(arg)) for arg, sub in zip(args, subs)]),
                True,
            ),
        ]
    return []


def _simplify_bv_equality(expr: ExprRef) -> ExprRef:
    if expr.decl().kind() != Z3_OP_EQ or len(expr.children()) != 2:
        return expr
    lhs, rhs = expr.children()
    if lhs.sort().kind() != Z3_BV_SORT or rhs.sort().kind() != Z3_BV_SORT:
        return expr
    simplified = simplify(expr)
    return simplified


def _sic_bv(op, args, subs) -> Tuple[BoolRef, bool]:
    # Unary ops passthrough
    if len(args) == 1:
        a = args[0]
        sa = subs[0]
        if a.sort().kind() != Z3_BV_SORT:
            return BoolVal(False), False
        if op in (
            Z3_OP_BNOT,
            Z3_OP_SIGN_EXT,
            Z3_OP_ZERO_EXT,
            Z3_OP_EXTRACT,
            Z3_OP_ROTATE_LEFT,
            Z3_OP_ROTATE_RIGHT,
        ):
            return sa, True
        return BoolVal(False), False

    if len(args) < 2:
        return BoolVal(False), False

    if any(arg.sort().kind() != Z3_BV_SORT for arg in args):
        return BoolVal(False), False

    a, b = args[0], args[1]
    sa, sb = subs[0], subs[1]

    z_a = _bv_zero(a)
    if op in (Z3_OP_BAND, Z3_OP_BMUL):
        return _any(
            *[_all(sub, arg == _bv_zero(arg)) for arg, sub in zip(args, subs)]
        ), True
    if op == Z3_OP_BOR:
        return _any(
            *[_all(sub, arg == _bv_ones(arg)) for arg, sub in zip(args, subs)]
        ), True
    if op in (Z3_OP_BXOR,):
        return _all(sa, sb), False
    if op in (Z3_OP_BSHL, Z3_OP_BLSHR):
        width = a.size()
        shift_too_large = UGE(b, BitVecVal(width, b.size()))
        return _any(_all(sa, a == z_a), _all(sb, shift_too_large)), True
    if op == Z3_OP_BASHR:
        return _any(_all(sa, a == z_a), _all(sa, a == _bv_ones(a))), True
    if op in (
        Z3_OP_BUDIV,
        Z3_OP_BSDIV,
        Z3_OP_BSMOD,
        Z3_OP_BUREM,
        Z3_OP_BSREM,
    ):
        one = _bv_one(a)
        ones = _bv_ones(a)
        if op == Z3_OP_BUDIV:
            return _all(sb, b == z_a), True
        if op == Z3_OP_BUREM:
            return _any(_all(sa, a == z_a), _all(sb, b == one)), True
        if op == Z3_OP_BSDIV:
            return _any(
                _all(sa, sb, a == z_a, b != z_a),
                _all(sa, sb, a == _bv_signed_min(a), b == ones),
            ), True
        if op in (Z3_OP_BSREM, Z3_OP_BSMOD):
            return _any(
                _all(sa, a == z_a),
                _all(sb, b == one),
                _all(sb, b == ones),
            ), True
        return BoolVal(False), False
    if op == Z3_OP_BADD:
        return _all(*subs), False
    if op == Z3_OP_BSUB:
        return _all(sa, sb), False
    if op in (Z3_OP_CONCAT,):
        return _all(*subs), False
    if op == Z3_OP_ULT:
        return _any(_all(sa, a == _bv_ones(a)), _all(sb, b == z_a)), True
    if op == Z3_OP_ULEQ:
        return _any(_all(sa, a == z_a), _all(sb, b == _bv_ones(b))), True
    if op == Z3_OP_UGT:
        return _any(_all(sa, a == z_a), _all(sb, b == _bv_ones(b))), True
    if op == Z3_OP_UGEQ:
        return _any(_all(sa, a == _bv_ones(a)), _all(sb, b == z_a)), True
    if op == Z3_OP_SLT:
        return _any(
            _all(sa, a == _bv_signed_max(a)),
            _all(sb, b == _bv_signed_min(b)),
        ), True
    if op == Z3_OP_SLEQ:
        return _any(
            _all(sa, a == _bv_signed_min(a)),
            _all(sb, b == _bv_signed_max(b)),
        ), True
    if op == Z3_OP_SGT:
        return _any(
            _all(sa, a == _bv_signed_min(a)),
            _all(sb, b == _bv_signed_max(b)),
        ), True
    if op == Z3_OP_SGEQ:
        return _any(
            _all(sa, a == _bv_signed_max(a)),
            _all(sb, b == _bv_signed_min(b)),
        ), True
    return BoolVal(False), False


def _sic_arith(op, args, subs) -> Tuple[BoolRef, bool]:
    if len(args) == 1:
        a = args[0]
        sa = subs[0]
        if op in (Z3_OP_UMINUS,):
            return sa, True
        return BoolVal(False), False
    if op == Z3_OP_MUL and len(args) >= 2:
        a, b = args[0], args[1]
        sa, sb = subs[0], subs[1]
        zero = IntVal(0) if a.sort().kind() == Z3_INT_SORT else RealVal(0)
        return _any(*[_all(sub, arg == zero) for arg, sub in zip(args, subs)]), True
    if op in (Z3_OP_ADD, Z3_OP_SUB):
        return _all(*subs), False
    if op in (Z3_OP_DIV, Z3_OP_IDIV, Z3_OP_REM, Z3_OP_MOD) and len(args) == 2:
        a, b = args
        sa, sb = subs
        zero = IntVal(0) if a.sort().kind() == Z3_INT_SORT else RealVal(0)
        return _all(sa, sb, b != zero), False
    if op in (Z3_OP_LT, Z3_OP_GT, Z3_OP_LE, Z3_OP_GE):
        return _all(*subs), False
    if op == Z3_OP_DISTINCT:
        return _all(*subs), False
    return BoolVal(False), False


def _sic_ite(op, args, subs) -> Tuple[BoolRef, bool]:
    if op != Z3_OP_ITE:
        return BoolVal(False), False
    c, t, e = args
    sc, st, se = subs
    return (
        _any(_all(sc, _any(st, se), If(c, st, se)), _all(st, se, t == e)),
        False,
    )


def _sic_array(op, args, subs) -> Tuple[BoolRef, bool]:
    # array equality: independent if both sides are
    if op == Z3_OP_EQ and len(args) == 2 and all(is_array(a) for a in args):
        return _all(*subs), False
    # select(store(a, i, v), j): handled by rewrite in main engine
    # store(a, i, v): independent if all parts are
    if op == Z3_OP_STORE and len(args) == 3:
        return _all(*subs), False
    # select(a, i): independent if both independent
    if op == Z3_OP_SELECT and len(args) == 2:
        return _all(*subs), False
    return BoolVal(False), False


# Plug-in list evaluated in precedence order
_THEORY_PLUGINS = (_sic_bool, _sic_bv, _sic_arith, _sic_ite, _sic_array)


def _theory_sic_for_parts(
    op: int, args: Sequence[ExprRef], subsic: Sequence[BoolRef], op_name: str = ""
) -> Tuple[BoolRef, bool]:
    op = _normalize_bv_op(op, op_name)
    for plugin in _THEORY_PLUGINS:
        psi, is_wic = plugin(op, args, subsic)
        if not is_false(psi):
            return psi, is_wic
    return BoolVal(False), False


def theory_sic(expr: ExprRef, subsic: Sequence[BoolRef]) -> Tuple[BoolRef, bool]:
    """
    Dispatcher returning an operator-specific Ψ_f (can be False).
    """
    op = expr.decl().kind()
    args = list(expr.children())
    return _theory_sic_for_parts(op, args, subsic, expr.decl().name())


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

    def collect_indices(e: ExprRef):
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

    def flatten_assoc(e: ExprRef) -> Sequence[ExprRef]:
        if not is_app(e):
            return [e]
        kind = e.decl().kind()
        if kind not in associative_ops:
            return [e]
        flat = []
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
            # Re-express select(store(..)) as an ite to eliminate the store and
            # let generic rules compute a SIC without keeping the target index.
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
                value_sic, value_wic = go(value)
                return value_sic, value_wic
            idx_sic, idx_wic = go(idx)
            arr_shadow, arr_wic = get_array_shadow(arr)
            # The shadow array stores per-index independence conditions.
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
                # Avoid reintroducing target symbols into the SIC
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
            # A branch guard can itself be the reason the array read becomes
            # independent, so preserve that obligation separately.
            if not is_true(cond_sic):
                shadow_constraints.append(cond_sic)
            return shadow_cache[arr]

        # Fallback shadow: assume dependence everywhere. This is conservative
        # and ensures we never synthesize a spurious SIC from an unsupported
        # array construct.
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


# ---------------------------------------------------------------- taint vars (explicit)
def _collect_uninterp_consts(expr: ExprRef) -> Set[ExprRef]:
    syms = set()
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
