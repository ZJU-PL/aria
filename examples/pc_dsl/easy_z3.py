"""
A DSL for simplifying the encoding of path conditions

"""

from __future__ import annotations

import dis
import inspect
import warnings
import weakref
from typing import Any, Iterable

import z3

# Re-export modularized pieces
from ._dsl_sorts import (
    resolve_sort,
    BV,
    FP,
    Array,
    Seq,
    SetSort,
    Tuple,
    Enum,
    U,
    BVVal,
    FPVal,
    DEFAULT_RM,
)
from ._dsl_nodes import (
    Value,
    Variable,
    Namespace,
    Unop,
    Binop,
    Call,
    Index,
    StoreNode,
    Quantifier,
    Builtin,
    ForAll,
    Exists,
    Store,
)
from ._dsl_traverse import traverse
from ._dsl_api import (
    get_constraints,
    get_free_vars,
    rename_free_vars,
    to_z3,
    from_z3,
)

__all__ = (
    "Solver",
    # sort helpers
    "BV",
    "FP",
    "Array",
    "Seq",
    "SetSort",
    "Tuple",
    "Enum",
    "U",
    # array ops
    "Store",
    # quantifiers
    "ForAll",
    "Exists",
    # convenience constructors and ops
    "BVVal",
    "FPVal",
    "Concat",
    "BVConcat",
    "Extract",
    "ZeroExt",
    "SignExt",
    "LShR",
    "RotateLeft",
    "RotateRight",
    "RepeatBV",
    "BV2Int",
    "Int2BV",
    "IfThenElse",
    "Distinct",
    "Length",
    "Contains",
    "PrefixOf",
    "SuffixOf",
    "Substring",
    "IndexOf",
    "Replace",
    "StrToInt",
    "IntToStr",
    # external APIs
    "get_constraints",
    "rename_free_vars",
    "get_free_vars",
    "to_z3",
    "from_z3",
    # env/scoping
    "declare_var",
    "declare_fun",
    "declare_sort",
    "current_env",
    "dsl_vars",
    # context manager
    "Context",
    "context",
)

from ._dsl_nodes import BIN_OPS, UNARY_OPS, REVERSIBLE_BIN_OPS


class ValueMeta(type):
    def __init__(cls, name: str, bases: tuple[type, ...], ns: dict[str, Any]):
        for attr in BIN_OPS:
            # default value needs to be used to ensure the right value of attr is used
            def bin_op(left: Value, right: Any, op: str = attr) -> Binop:
                # left ns needs to be chosen since right arg can be anything
                return Binop(left, right, op, ns=left.ns)

            setattr(cls, attr, bin_op)
        for attr in REVERSIBLE_BIN_OPS:

            def rbin_op(left: Value, right: Any, op: str = attr) -> Binop:
                # left ns needs to be chosen since right arg can be anything
                return Binop(right, left, op, ns=left.ns)

            setattr(cls, attr.replace("__", "__r", 1), rbin_op)
        for attr in UNARY_OPS:

            def un_op(arg: Value, op: str = attr) -> Unop:
                return Unop(arg, op, ns=arg.ns)

            setattr(cls, attr, un_op)


from ._dsl_nodes import Value, Namespace, Index  # noqa: F401 re-export behavior

from ._dsl_nodes import Unop, Binop, Call, StoreNode, Quantifier, Builtin  # noqa: F401

from ._dsl_nodes import Variable  # noqa: F401

from ._dsl_nodes import Namespace  # noqa: F401


class SolverMeta(type):
    @classmethod
    def __prepare__(cls, name: str, bases: tuple[type, ...]) -> Namespace:
        return Namespace()

    def __init__(cls, name: str, bases: tuple[type, ...], ns: Namespace):
        vars = {}
        anns = getattr(cls, "__annotations__", {}) or {}
        # Evaluation context for string annotations: use the module where the class is defined
        import sys

        eval_globals = (
            sys.modules.get(cls.__module__).__dict__
            if hasattr(cls, "__module__")
            else globals()
        )
        for var, ann in anns.items():
            try:
                if isinstance(ann, str):
                    ann = eval(ann, eval_globals, dict(ns))
                # Variable of a given sort
                if not isinstance(ann, dict):
                    sort = resolve_sort(ann)
                    vars[var] = z3.Const(var, sort)
                else:
                    # Uninterpreted function
                    in_sort, out_sort = next(iter(ann.items()))
                    out_sort = resolve_sort(out_sort)
                    if isinstance(in_sort, tuple):
                        in_sorts = tuple(resolve_sort(x) for x in in_sort)
                        vars[var] = z3.Function(var, *in_sorts, out_sort)
                    else:
                        in_sorts = resolve_sort(in_sort)
                        vars[var] = z3.Function(var, in_sorts, out_sort)
            except Exception:
                # Skip invalid annotation entries rather than failing the whole class
                continue
        solver = z3.Solver()
        constraints: list[Any] = []
        for term in ns.assertions:
            c = traverse(term, vars)
            constraints.append(c)
            solver.add(c)
        # Do not auto-check; keep z3-like interface
        cls.__solver = solver
        cls.__model = None
        cls.__vars = vars
        cls.__constraints = tuple(constraints)
        cls.__unsat_core: tuple[z3.ExprRef, ...] | None = None
        # Scoping frames for additional declarations
        cls.__frames: list[dict[str, dict[str, Any]]] = [
            {"vars": {}, "funs": {}, "sorts": {}}
        ]

    def __repr__(self):
        if self.__model is not None:
            return repr(self.__model)
        s = getattr(self, "_SolverMeta__solver", None)
        if s is None:
            return "<unsolved>"
        return f"<unsolved: {len(s.assertions())} constraints>"

    def __iter__(self):
        if self.__model is None:
            raise RuntimeError("No model available. Call check() first.")
        g = inspect.currentframe().f_back.f_globals
        for x in self.__vars:
            g[x] = getattr(self, x)
        return iter(() for () in ())

    def __getattr__(cls, attr):
        try:
            # If no model yet, return a DSL Variable proxy for declared symbols
            if cls.__model is None:
                if attr in cls.__vars:
                    ns = Namespace()
                    return Variable(attr, ns=weakref.ref(ns))
                raise AttributeError("No model available. Call check() first.")
            x = cls.__model[cls.__vars[attr]]
            if isinstance(x, z3.FuncInterp):
                pass
            else:
                if x.sort() == z3.BoolSort():
                    return bool(x)
                if x.sort() == z3.IntSort():
                    return x.as_long()
                if x.sort() == z3.RealSort():
                    if isinstance(x, z3.RatNumRef):
                        return float(x.as_fraction())
                    return float(x.approx().as_fraction())
            return x
        except Exception as e:
            raise AttributeError from e


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


from ._dsl_sorts import DEFAULT_RM  # re-export

from ._dsl_sorts import resolve_sort  # noqa: F401

from ._dsl_sorts import BV, FP, Array, U, BVVal, FPVal  # noqa: F401


def Concat(*args: Any):
    from ._dsl_nodes import Builtin, Value  # local import to avoid cycles

    if not args:
        raise ValueError("Concat requires at least one argument")
    ns = next((a.ns for a in args if isinstance(a, Value)), (lambda: None))
    return Builtin("concat", *args, ns=ns)


def IfThenElse(cond: Any, then_val: Any, else_val: Any):
    from ._dsl_nodes import Builtin, Value

    ns = next(
        (a.ns for a in (cond, then_val, else_val) if isinstance(a, Value)),
        (lambda: None),
    )
    return Builtin("ite", cond, then_val, else_val, ns=ns)


def Distinct(*args: Any):
    from ._dsl_nodes import Builtin, Value

    ns = next((a.ns for a in args if isinstance(a, Value)), (lambda: None))
    return Builtin("distinct", *args, ns=ns)


def Length(s: Any):
    from ._dsl_nodes import Builtin, Value

    ns = s.ns if isinstance(s, Value) else (lambda: None)
    return Builtin("length", s, ns=ns)


def BVConcat(*args: Any):
    from ._dsl_nodes import Builtin, Value

    if not args:
        raise ValueError("BVConcat requires at least one argument")
    ns = next((a.ns for a in args if isinstance(a, Value)), (lambda: None))
    return Builtin("bv_concat", *args, ns=ns)


def Extract(high: int, low: int, value: Any):
    from ._dsl_nodes import Builtin, Value

    ns = value.ns if isinstance(value, Value) else (lambda: None)
    return Builtin("extract", int(high), int(low), value, ns=ns)


def ZeroExt(ext: int, value: Any):
    from ._dsl_nodes import Builtin, Value

    ns = value.ns if isinstance(value, Value) else (lambda: None)
    return Builtin("zeroext", int(ext), value, ns=ns)


def SignExt(ext: int, value: Any):
    from ._dsl_nodes import Builtin, Value

    ns = value.ns if isinstance(value, Value) else (lambda: None)
    return Builtin("signext", int(ext), value, ns=ns)


def LShR(value: Any, shift: Any):
    from ._dsl_nodes import Builtin, Value

    ns = value.ns if isinstance(value, Value) else (
        shift.ns if isinstance(shift, Value) else (lambda: None)
    )
    return Builtin("lshr", value, shift, ns=ns)


def RotateLeft(value: Any, bits: int):
    from ._dsl_nodes import Builtin, Value

    ns = value.ns if isinstance(value, Value) else (lambda: None)
    return Builtin("rol", value, int(bits), ns=ns)


def RotateRight(value: Any, bits: int):
    from ._dsl_nodes import Builtin, Value

    ns = value.ns if isinstance(value, Value) else (lambda: None)
    return Builtin("ror", value, int(bits), ns=ns)


def RepeatBV(count: int, value: Any):
    from ._dsl_nodes import Builtin, Value

    ns = value.ns if isinstance(value, Value) else (lambda: None)
    return Builtin("repeat", int(count), value, ns=ns)


def BV2Int(value: Any, signed: bool = False):
    from ._dsl_nodes import Builtin, Value

    ns = value.ns if isinstance(value, Value) else (lambda: None)
    return Builtin("bv2int", value, bool(signed), ns=ns)


def Int2BV(width: int, value: Any):
    from ._dsl_nodes import Builtin, Value

    ns = value.ns if isinstance(value, Value) else (lambda: None)
    return Builtin("int2bv", int(width), value, ns=ns)


def Contains(s: Any, sub: Any):
    from ._dsl_nodes import Builtin, Value

    ns = s.ns if isinstance(s, Value) else (lambda: None)
    return Builtin("contains", s, sub, ns=ns)


def PrefixOf(prefix: Any, s: Any):
    from ._dsl_nodes import Builtin, Value

    ns = s.ns if isinstance(s, Value) else (
        prefix.ns if isinstance(prefix, Value) else (lambda: None)
    )
    return Builtin("prefixof", prefix, s, ns=ns)


def SuffixOf(suffix: Any, s: Any):
    from ._dsl_nodes import Builtin, Value

    ns = s.ns if isinstance(s, Value) else (
        suffix.ns if isinstance(suffix, Value) else (lambda: None)
    )
    return Builtin("suffixof", suffix, s, ns=ns)


def Substring(s: Any, offset: Any, length: Any):
    from ._dsl_nodes import Builtin, Value

    ns = s.ns if isinstance(s, Value) else (
        offset.ns if isinstance(offset, Value) else (lambda: None)
    )
    return Builtin("substring", s, offset, length, ns=ns)


def IndexOf(s: Any, sub: Any, start: Any | None = None):
    from ._dsl_nodes import Builtin, Value

    ns = s.ns if isinstance(s, Value) else (
        sub.ns if isinstance(sub, Value) else (lambda: None)
    )
    if start is None:
        return Builtin("indexof", s, sub, ns=ns)
    return Builtin("indexof", s, sub, start, ns=ns)


def Replace(s: Any, old: Any, new: Any):
    from ._dsl_nodes import Builtin, Value

    ns = s.ns if isinstance(s, Value) else (
        old.ns if isinstance(old, Value) else (lambda: None)
    )
    return Builtin("replace", s, old, new, ns=ns)


def StrToInt(s: Any):
    from ._dsl_nodes import Builtin, Value

    ns = s.ns if isinstance(s, Value) else (lambda: None)
    return Builtin("str_to_int", s, ns=ns)


def IntToStr(i: Any):
    from ._dsl_nodes import Builtin, Value

    ns = i.ns if isinstance(i, Value) else (lambda: None)
    return Builtin("int_to_str", i, ns=ns)


from ._dsl_nodes import Store, ForAll, Exists  # noqa: F401

from ._dsl_traverse import traverse  # noqa: F401


class Solver(metaclass=SolverMeta):
    @classmethod
    def constraints(cls) -> tuple[Any, ...]:
        return getattr(
            cls, "_SolverMeta__constraints", getattr(cls, "_Solver__constraints", ())
        )

    @classmethod
    def variables(cls) -> dict[str, Any]:
        return dict(
            getattr(cls, "_SolverMeta__vars", getattr(cls, "_Solver__vars", {}))
        )

    @classmethod
    def model(cls) -> z3.ModelRef | None:
        return getattr(cls, "_SolverMeta__model", getattr(cls, "_Solver__model", None))

    @classmethod
    def reset(cls):
        s = z3.Solver()
        for c in cls.constraints():
            s.add(c)
        cls._SolverMeta__solver = s
        cls._SolverMeta__model = None
        cls._SolverMeta__unsat_core = None

    @classmethod
    def set(cls, **options: Any):
        getattr(cls, "_SolverMeta__solver").set(**options)

    @classmethod
    def push(cls, n: int = 1):
        s = getattr(cls, "_SolverMeta__solver")
        for _ in range(max(0, int(n))):
            s.push()
            # new empty frame for scoped declarations
            cls._SolverMeta__frames.append({"vars": {}, "funs": {}, "sorts": {}})
        return None

    @classmethod
    def pop(cls, n: int = 1):
        s = getattr(cls, "_SolverMeta__solver")
        for _ in range(max(0, int(n))):
            s.pop()
            if len(cls._SolverMeta__frames) > 1:
                cls._SolverMeta__frames.pop()
        return None

    @classmethod
    def add(cls, *exprs: Any):
        s = getattr(cls, "_SolverMeta__solver")
        env = cls.current_env()
        z3exprs: list[Any] = []
        for e in exprs:
            if isinstance(e, z3.AstRef):
                z3exprs.append(e)
            else:
                z3exprs.append(to_z3(e, env))
        s.add(*z3exprs)

    @classmethod
    def check(cls, *assumptions: Any) -> z3.CheckSatResult:
        s = getattr(cls, "_SolverMeta__solver")
        env = cls.current_env()
        z3_assumptions = []
        for a in assumptions:
            if isinstance(a, str):
                z3_assumptions.append(z3.Bool(a))
            elif isinstance(a, z3.AstRef):
                z3_assumptions.append(a)
            else:
                z3_assumptions.append(to_z3(a, env))
        res = s.check(*z3_assumptions)
        cls._SolverMeta__unsat_core = None
        if res == z3.sat:
            cls._SolverMeta__model = s.model()
        else:
            cls._SolverMeta__model = None
            if res == z3.unsat:
                try:
                    cls._SolverMeta__unsat_core = tuple(s.unsat_core())
                except Exception:
                    cls._SolverMeta__unsat_core = ()
        return res

    @classmethod
    def current_assertions(cls) -> tuple[Any, ...]:
        return tuple(getattr(cls, "_SolverMeta__solver").assertions())

    @classmethod
    def current_env(cls) -> dict[str, Any]:
        env = dict(cls.variables())
        for frame in cls._SolverMeta__frames:
            env.update(frame.get("sorts", {}))
            env.update(frame.get("vars", {}))
            env.update(frame.get("funs", {}))
        return env

    @classmethod
    def dsl_vars(cls) -> dict[str, Value]:
        ns = Namespace()
        return {
            name: Variable(name, ns=weakref.ref(ns)) for name in cls.variables().keys()
        }

    @classmethod
    def declare_var(cls, name: str, sort: Any) -> Any:
        z3sort = resolve_sort(sort) if not isinstance(sort, z3.SortRef) else sort
        sym = z3.Const(name, z3sort)
        cls._SolverMeta__frames[-1]["vars"][name] = sym
        return sym

    @classmethod
    def declare_fun(
        cls, name: str, domain: Any | tuple[Any, ...], range_sort: Any
    ) -> Any:
        if isinstance(domain, tuple):
            in_sorts = tuple(
                resolve_sort(d) if not isinstance(d, z3.SortRef) else d for d in domain
            )
            out_sort = (
                resolve_sort(range_sort)
                if not isinstance(range_sort, z3.SortRef)
                else range_sort
            )
            fn = z3.Function(name, *in_sorts, out_sort)
        else:
            in_sort = (
                resolve_sort(domain) if not isinstance(domain, z3.SortRef) else domain
            )
            out_sort = (
                resolve_sort(range_sort)
                if not isinstance(range_sort, z3.SortRef)
                else range_sort
            )
            fn = z3.Function(name, in_sort, out_sort)
        cls._SolverMeta__frames[-1]["funs"][name] = fn
        return fn

    @classmethod
    def declare_sort(cls, name: str) -> z3.SortRef:
        srt = U(name)
        cls._SolverMeta__frames[-1]["sorts"][name] = srt
        return srt

    @classmethod
    def unsat_core(cls) -> tuple[z3.ExprRef, ...] | None:
        return getattr(
            cls,
            "_SolverMeta__unsat_core",
            getattr(cls, "_Solver__unsat_core", None),
        )


# Module-level proxies for scoping helpers (default to latest Solver subclass if desired)
def declare_var(name: str, sort: Any) -> Any:
    # Require explicit Solver subclass usage; forward to the most recent subclass if used directly
    # Here we just forward to Solver.declare_var for convenience
    return Solver.declare_var(name, sort)


def declare_fun(name: str, domain: Any | tuple[Any, ...], range_sort: Any) -> Any:
    return Solver.declare_fun(name, domain, range_sort)


def declare_sort(name: str) -> z3.SortRef:
    return Solver.declare_sort(name)


class Context:
    def __init__(self, solver_cls: type["Solver"], frames: int = 1):
        self._solver = solver_cls
        self._frames = int(frames)
        self._entered = False

    def __enter__(self):
        for _ in range(max(1, self._frames)):
            self._solver.push()
        self._entered = True
        return self

    def __exit__(self, exc_type, exc, tb):
        # Pop all frames we pushed, even on exceptions
        for _ in range(max(1, self._frames)):
            self._solver.pop()
        self._entered = False
        # Propagate exception if any
        return False

    # Shortcuts to manage declarations and constraints within the context
    @property
    def env(self) -> dict[str, Any]:
        return self._solver.current_env()

    def var(self, name: str, sort: Any) -> Any:
        return self._solver.declare_var(name, sort)

    def fun(self, name: str, domain: Any | tuple[Any, ...], range_sort: Any) -> Any:
        return self._solver.declare_fun(name, domain, range_sort)

    def sort(self, name: str) -> z3.SortRef:
        return self._solver.declare_sort(name)

    def add(self, *exprs: Any) -> None:
        self._solver.add(*exprs)


def context(solver_cls: type["Solver"], frames: int = 1) -> Context:
    return Context(solver_cls, frames)
