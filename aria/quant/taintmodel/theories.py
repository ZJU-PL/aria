# taintmodel/theories.py -------------------------------------------------------
"""
Theory-specific SIC rules (operator plug-ins).

Each plug-in maps an operator with its children and their sub-SICs to a
Boolean condition Ψ_f that is True iff evaluating the operator at those
argument values is independent of the tracked target symbols.

Public surface
--------------
theory_sic(expr, subsic)        -- dispatch to the right plug-in
_all(*xs), _any(*xs)            -- And/Or with identity handling
_simplify_bv_equality(expr)     -- canonicalise BV equality nodes
_bv_constant_sics(...)          -- constant-propagation SIC entries
_normalize_bv_op(op, name)      -- normalise parsed BV operator names
"""
from __future__ import annotations

from typing import Sequence, Tuple

from z3 import *  # type: ignore


# --------------------------------------------------------------- Bool helpers
def _all(*xs: BoolRef) -> BoolRef:
    if not xs:
        return BoolVal(True)
    return And(*xs)


def _any(*xs: BoolRef) -> BoolRef:
    if not xs:
        return BoolVal(False)
    return Or(*xs)


# --------------------------------------------------------------- BV helpers
_bv_zero = lambda t: t.sort().cast(0)  # noqa: E731


def _bv_ones(t: ExprRef) -> ExprRef:
    return ~_bv_zero(t)


def _bv_signed_min(t: ExprRef) -> ExprRef:
    return BitVecVal(1 << (t.size() - 1), t.size())


def _bv_signed_max(t: ExprRef) -> ExprRef:
    return BitVecVal((1 << (t.size() - 1)) - 1, t.size())


def _bv_one(t: ExprRef) -> ExprRef:
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


# --------------------------------------------------------- Boolean theory SIC
def _sic_bool(op: int, args: Sequence[ExprRef], subs: Sequence[BoolRef]) -> Tuple[BoolRef, bool]:
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
        if op == Z3_OP_XOR:
            return _all(sa, sb), False
    if op == Z3_OP_DISTINCT:
        return _all(*subs), False
    return BoolVal(False), False


# --------------------------------------------------- BV constant-SIC helpers
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
            (
                BitVecVal(0, total_width),
                _all(*[_all(sub, arg == _bv_zero(arg)) for arg, sub in zip(args, subs)]),
                True,
            ),
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
    return simplify(expr)


# ------------------------------------------------------------ BV theory SIC
def _sic_bv(op: int, args: Sequence[ExprRef], subs: Sequence[BoolRef]) -> Tuple[BoolRef, bool]:
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
    if op in (Z3_OP_BUDIV, Z3_OP_BSDIV, Z3_OP_BSMOD, Z3_OP_BUREM, Z3_OP_BSREM):
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


# --------------------------------------------------------- Arith theory SIC
def _sic_arith(op: int, args: Sequence[ExprRef], subs: Sequence[BoolRef]) -> Tuple[BoolRef, bool]:
    if len(args) == 1:
        if op in (Z3_OP_UMINUS,):
            return subs[0], True
        return BoolVal(False), False
    if op == Z3_OP_MUL and len(args) >= 2:
        a = args[0]
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


# ------------------------------------------------------------- ITE theory SIC
def _sic_ite(op: int, args: Sequence[ExprRef], subs: Sequence[BoolRef]) -> Tuple[BoolRef, bool]:
    if op != Z3_OP_ITE:
        return BoolVal(False), False
    c, t, e = args
    sc, st, se = subs
    return (
        _any(_all(sc, _any(st, se), If(c, st, se)), _all(st, se, t == e)),
        False,
    )


# ---------------------------------------------------------- Array theory SIC
def _sic_array(op: int, args: Sequence[ExprRef], subs: Sequence[BoolRef]) -> Tuple[BoolRef, bool]:
    if op == Z3_OP_EQ and len(args) == 2 and all(is_array(a) for a in args):
        return _all(*subs), False
    if op == Z3_OP_STORE and len(args) == 3:
        return _all(*subs), False
    if op == Z3_OP_SELECT and len(args) == 2:
        return _all(*subs), False
    return BoolVal(False), False


# -------------------------------------------- Plug-in dispatch (public API)
_THEORY_PLUGINS = (_sic_bool, _sic_bv, _sic_arith, _sic_ite, _sic_array)


def _theory_sic_for_parts(
    op: int,
    args: Sequence[ExprRef],
    subsic: Sequence[BoolRef],
    op_name: str = "",
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
