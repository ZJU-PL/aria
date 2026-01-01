from __future__ import annotations

import inspect
import weakref
from typing import Any, Iterable

import z3

from ._dsl_sorts import resolve_sort, DEFAULT_RM

__all__ = [
    "Value",
    "Variable",
    "Namespace",
    "Z3Wrapper",
    "Unop",
    "Binop",
    "Call",
    "Index",
    "StoreNode",
    "Quantifier",
    "Builtin",
    "ForAll",
    "Exists",
    "Store",
]

COMPARISON_OPS = {
    "__eq__": "==",
    "__ne__": "!=",
    "__lt__": "<",
    "__le__": "<=",
    "__gt__": ">",
    "__ge__": ">=",
}
REVERSIBLE_BIN_OPS = {
    "__add__": "+",
    "__sub__": "-",
    "__mul__": "*",
    "__truediv__": "/",
    "__floordiv__": "//",
    "__pow__": "**",
    "__mod__": "%",
    "__and__": "&",
    "__or__": "|",
    "__xor__": "^",
    "__rshift__": ">>",
    "__lshift__": "<<",
}
BIN_OPS = {**COMPARISON_OPS, **REVERSIBLE_BIN_OPS}
UNARY_OPS = {
    "__pos__": "+",
    "__neg__": "-",
    "__invert__": "~",
}


class ValueMeta(type):
    def __init__(cls, name: str, bases: tuple[type, ...], ns: dict[str, Any]):
        for attr in BIN_OPS:

            def bin_op(left: Value, right: Any, op: str = attr) -> "Binop":
                return Binop(left, right, op, ns=left.ns)

            setattr(cls, attr, bin_op)
        for attr in REVERSIBLE_BIN_OPS:

            def rbin_op(left: Value, right: Any, op: str = attr) -> "Binop":
                return Binop(right, left, op, ns=left.ns)

            setattr(cls, attr.replace("__", "__r", 1), rbin_op)
        for attr in UNARY_OPS:

            def un_op(arg: Value, op: str = attr) -> "Unop":
                return Unop(arg, op, ns=arg.ns)

            setattr(cls, attr, un_op)


class Value(metaclass=ValueMeta):
    def __init__(self, *, ns: weakref.ref["Namespace"]):
        self.ns = ns

    def __bool__(self):
        frame = inspect.currentframe().f_back
        warn = True
        try:
            info = inspect.getframeinfo(frame, context=1)
            if info.code_context:
                line = info.code_context[0].lstrip()
                if line.startswith("assert ") or line == "assert":
                    warn = False
        except Exception:
            pass
        if warn:
            # Suppress warnings in split-out module to keep noise low
            pass
        self.ns().assertion(self)
        return True

    def __getitem__(self, item: Any):
        return Index(self, item, ns=self.ns)


class Unop(Value):
    def __repr__(self):
        return f"({UNARY_OPS[self.op]}{self.arg})"

    def __init__(self, arg: Value, op: str, **kwargs):
        self.arg = arg
        self.op = op
        super().__init__(**kwargs)


class Binop(Value):
    def __repr__(self):
        return f"({self.left} {BIN_OPS[self.op]} {self.right})"

    def __init__(self, left: Value, right: Value, op: str, **kwargs):
        self.op = op
        self.left = left
        self.right = right
        super().__init__(**kwargs)


class Call(Value):
    def __repr__(self):
        return f"({self.fn}({', '.join(map(repr, self.args))}))"

    def __init__(self, fn: "Variable", *args: Value, **kwargs):
        self.fn = fn
        self.args = args
        super().__init__(**kwargs)


class Index(Value):
    def __repr__(self):
        return f"({self.base}[{self.key}])"

    def __init__(self, base: Value, key: Value, **kwargs):
        self.base = base
        self.key = key
        super().__init__(**kwargs)


class StoreNode(Value):
    def __repr__(self):
        return f"(store {self.base}[{self.key}] := {self.value})"

    def __init__(self, base: Value, key: Value, value: Value, **kwargs):
        self.base = base
        self.key = key
        self.value = value
        super().__init__(**kwargs)


class Quantifier(Value):
    def __repr__(self):
        bvars = ", ".join(f"{n}: {s}" for n, s in self.bindings)
        return f"({self.kind} ({bvars}). {self.body})"

    def __init__(
        self, kind: str, bindings: list[tuple[str, Any]], body: Value, **kwargs
    ):
        self.kind = kind
        self.bindings = bindings
        self.body = body
        super().__init__(**kwargs)


class Builtin(Value):
    def __repr__(self):
        args = ", ".join(map(repr, self.args))
        return f"{self.name}({args})"

    def __init__(self, name: str, *args: Value, **kwargs):
        self.name = name
        self.args = args
        super().__init__(**kwargs)


class Variable(Value):
    def __repr__(self):
        return self.var

    def __init__(self, var: str, **kwargs):
        self.var = var
        super().__init__(**kwargs)

    def __call__(self, *args: Value):
        return Call(self, *args, ns=self.ns)


class Namespace(dict):
    def __init__(self, *args, **kwds):
        self.assertions: list[Value] = []
        super().__init__(self, *args, **kwds)

    def __getitem__(self, key: str):
        try:
            x = super().__getitem__(key)
        except KeyError:
            if key in ("__name__", "__annotations__"):
                raise KeyError
            try:
                frame = inspect.currentframe().f_back
                g = frame.f_globals
                if key in g:
                    return g[key]
            except Exception:
                pass
            if key in globals():
                return globals()[key]
            try:
                import builtins

                return getattr(builtins, key)
            except Exception:
                return Variable(key, ns=weakref.ref(self))
        else:
            return x

    def assertion(self, value: Value):
        self.assertions.append(value)


class Z3Wrapper(Value):
    def __repr__(self):
        return f"z3({self.expr})"

    def __init__(self, expr: Any, **kwargs):
        self.expr = expr
        super().__init__(**kwargs)


def Store(base: Value, key: Value, value: Value) -> StoreNode:
    ns = (
        base.ns
        if isinstance(base, Value)
        else (key.ns if isinstance(key, Value) else value.ns)
    )
    return StoreNode(base, key, value, ns=ns)


def ForAll(
    bindings: dict[str, Any] | Iterable[tuple[str, Any]], body: Value
) -> Quantifier:
    if isinstance(bindings, dict):
        bind_list = list(bindings.items())
    else:
        bind_list = list(bindings)
    return Quantifier("forall", bind_list, body, ns=body.ns)


def Exists(
    bindings: dict[str, Any] | Iterable[tuple[str, Any]], body: Value
) -> Quantifier:
    if isinstance(bindings, dict):
        bind_list = list(bindings.items())
    else:
        bind_list = list(bindings)
    return Quantifier("exists", bind_list, body, ns=body.ns)
