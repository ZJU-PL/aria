from __future__ import annotations

from typing import Any

import z3

from ._dsl_sorts import resolve_sort, DEFAULT_RM
from ._dsl_nodes import (
    Value,
    Variable,
    Unop,
    Binop,
    Call,
    Index,
    StoreNode,
    Builtin,
    Quantifier,
)
from ._dsl_nodes import BIN_OPS, UNARY_OPS


def is_bool(x: Any) -> bool:
    try:
        return hasattr(x, "sort") and x.sort() == z3.BoolSort()
    except Exception:
        return False


def is_bv(x: Any) -> bool:
    try:
        return hasattr(x, "sort") and z3.is_bv_sort(x.sort())
    except Exception:
        return False


def is_fp(x: Any) -> bool:
    try:
        return hasattr(x, "sort") and z3.is_fp_sort(x.sort())
    except Exception:
        return False


def coerce_int_to_bv(val: Any, other: Any):
    if is_bv(other) and isinstance(val, int):
        return z3.BitVecVal(val, other.sort().size())
    return val


def traverse(value: Value, vars: dict[str, Any]):
    if isinstance(value, Variable):
        return vars[value.var]
    elif isinstance(value, Unop):
        if value.op == "__invert__":
            arg = traverse(value.arg, vars)
            if is_bool(arg):
                return z3.Not(arg)
            return eval(f"~(__arg)", {"__arg": arg})
        else:
            return eval(
                f"{UNARY_OPS[value.op]} (__arg)", {"__arg": traverse(value.arg, vars)}
            )
    elif isinstance(value, Binop):
        if value.op == "__pow__":
            base = traverse(value.left, vars)
            exp = traverse(value.right, vars)
            try:
                if isinstance(exp, int):
                    k = exp
                elif isinstance(exp, z3.IntNumRef):
                    k = exp.as_long()
                else:
                    k = None
            except Exception:
                k = None
            if (
                hasattr(base, "sort")
                and base.sort() == z3.IntSort()
                and isinstance(k, int)
                and k >= 0
            ):
                result = z3.IntVal(1)
                for _ in range(k):
                    result = result * base
                return result
            return base**exp
        if value.op in ("__and__", "__or__", "__xor__"):
            left = traverse(value.left, vars)
            right = traverse(value.right, vars)
            if is_bool(left) and is_bool(right):
                if value.op == "__and__":
                    return z3.And(left, right)
                if value.op == "__or__":
                    return z3.Or(left, right)
                return z3.Xor(left, right)
            return eval(
                f"(__left) {BIN_OPS[value.op]} (__right)",
                {
                    "__left": coerce_int_to_bv(left, right),
                    "__right": coerce_int_to_bv(right, left),
                },
            )
        elif value.op in ("__rshift__", "__lshift__"):
            left = traverse(value.left, vars)
            right = traverse(value.right, vars)
            if is_bool(left) and is_bool(right):
                if value.op == "__rshift__":
                    return z3.Implies(left, right)
                else:
                    return z3.Implies(right, left)
            return eval(
                f"(__left) {BIN_OPS[value.op]} (__right)",
                {
                    "__left": coerce_int_to_bv(left, right),
                    "__right": coerce_int_to_bv(right, left),
                },
            )
        else:
            left = traverse(value.left, vars)
            right = traverse(value.right, vars)
            if is_fp(left) and is_fp(right):
                if value.op == "__add__":
                    return z3.fpAdd(DEFAULT_RM, left, right)
                if value.op == "__sub__":
                    return z3.fpSub(DEFAULT_RM, left, right)
                if value.op == "__mul__":
                    return z3.fpMul(DEFAULT_RM, left, right)
                if value.op == "__truediv__":
                    return z3.fpDiv(DEFAULT_RM, left, right)
                if value.op in (
                    "__lt__",
                    "__le__",
                    "__gt__",
                    "__ge__",
                    "__eq__",
                    "__ne__",
                ):
                    return eval(
                        f"(__left) {BIN_OPS[value.op]} (__right)",
                        {"__left": left, "__right": right},
                    )
            left = coerce_int_to_bv(left, right)
            right = coerce_int_to_bv(right, left)
            return eval(
                f"(__left) {BIN_OPS[value.op]} (__right)",
                {"__left": left, "__right": right},
            )
    elif isinstance(value, Call):
        return traverse(value.fn, vars)(*(traverse(arg, vars) for arg in value.args))
    elif isinstance(value, Index):
        base = traverse(value.base, vars)
        key = traverse(value.key, vars)
        try:
            if isinstance(base, z3.ArrayRef):
                dom = base.sort().domain()
                if z3.is_bv_sort(dom) and isinstance(key, int):
                    key = z3.BitVecVal(key, dom.size())
        except Exception:
            pass
        return z3.Select(base, key)
    elif isinstance(value, StoreNode):
        base = traverse(value.base, vars)
        key = traverse(value.key, vars)
        val = traverse(value.value, vars)
        try:
            if isinstance(base, z3.ArrayRef):
                dom = base.sort().domain()
                rng = base.sort().range()
                if z3.is_bv_sort(dom) and isinstance(key, int):
                    key = z3.BitVecVal(key, dom.size())
                if z3.is_bv_sort(rng) and isinstance(val, int):
                    val = z3.BitVecVal(val, rng.size())
        except Exception:
            pass
        return z3.Store(base, key, val)
    elif isinstance(value, Builtin):
        name = value.name
        args = [traverse(a, vars) for a in value.args]
        if name == "concat":
            return z3.Concat(*args)
        if name == "bv_concat":
            return z3.Concat(*args)
        if name == "extract":
            hi, lo, val = args
            return z3.Extract(int(hi), int(lo), val)
        if name == "zeroext":
            ext, val = args
            return z3.ZeroExt(int(ext), val)
        if name == "signext":
            ext, val = args
            return z3.SignExt(int(ext), val)
        if name == "lshr":
            left, shift = args
            return z3.LShR(left, shift)
        if name == "rol":
            val, bits = args
            return z3.RotateLeft(val, int(bits))
        if name == "ror":
            val, bits = args
            return z3.RotateRight(val, int(bits))
        if name == "repeat":
            reps, val = args
            return z3.RepeatBitVec(int(reps), val)
        if name == "bv2int":
            val, signed = args
            return z3.BV2Int(val, is_signed=bool(signed))
        if name == "int2bv":
            width, val = args
            return z3.Int2BV(int(width), val)
        if name == "ite":
            return z3.If(*args)
        if name == "distinct":
            return z3.Distinct(*args)
        if name == "length":
            return z3.Length(*args)
        if name == "contains":
            return z3.Contains(*args)
        if name == "prefixof":
            return z3.PrefixOf(*args)
        if name == "suffixof":
            return z3.SuffixOf(*args)
        if name == "substring":
            return z3.SubString(*args)
        if name == "indexof":
            if len(args) == 3:
                return z3.IndexOf(args[0], args[1], args[2])
            return z3.IndexOf(args[0], args[1])
        if name == "replace":
            return z3.Replace(*args)
        if name == "str_to_int":
            return z3.StrToInt(*args)
        if name == "int_to_str":
            return z3.IntToStr(*args)
        raise NotImplementedError(f"Unknown builtin: {name}")
    elif isinstance(value, Quantifier):
        local = dict(vars)
        qvars = []
        for name, spec in value.bindings:
            srt = resolve_sort(spec)
            v = z3.Const(name, srt)
            local[name] = v
            qvars.append(v)
        body = traverse(value.body, local)
        if value.kind == "forall":
            return z3.ForAll(qvars, body)
        else:
            return z3.Exists(qvars, body)
    else:
        return value
