"""
NeuroSym formula AST — supports QF_LIA, QF_BV, QF_ABV.
No external SMT solver dependency.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


# ── Sorts ──────────────────────────────────────────────────────────────────────

class Sort:
    pass


class BoolSort(Sort):
    _inst = None
    def __new__(cls):
        if cls._inst is None:
            cls._inst = super().__new__(cls)
        return cls._inst
    def __repr__(self): return "Bool"
    def __hash__(self): return hash("Bool")
    def __eq__(self, other): return isinstance(other, BoolSort)


class IntSort(Sort):
    _inst = None
    def __new__(cls):
        if cls._inst is None:
            cls._inst = super().__new__(cls)
        return cls._inst
    def __repr__(self): return "Int"
    def __hash__(self): return hash("Int")
    def __eq__(self, other): return isinstance(other, IntSort)


@dataclass(frozen=True)
class BVSort(Sort):
    width: int
    def __repr__(self): return f"(_ BitVec {self.width})"


@dataclass(frozen=True)
class ArraySort(Sort):
    index: Sort
    element: Sort
    def __repr__(self): return f"(Array {self.index} {self.element})"


# Singletons
BOOL = BoolSort()
INT  = IntSort()


def bv(w: int) -> BVSort:
    return BVSort(w)


# ── Terms ──────────────────────────────────────────────────────────────────────

class Term:
    """Base class for all AST terms. Every Term has a .sort attribute."""
    sort: Sort = None


@dataclass(frozen=True)
class BoolLit(Term):
    value: bool
    sort: Sort = field(default_factory=BoolSort, compare=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, 'sort', BOOL)

    def __repr__(self):
        return "true" if self.value else "false"


@dataclass(frozen=True)
class IntLit(Term):
    value: int
    sort: Sort = field(default_factory=IntSort, compare=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, 'sort', INT)

    def __repr__(self):
        return str(self.value)


@dataclass(frozen=True)
class BVLit(Term):
    value: int   # unsigned integer value
    width: int   # bit-width

    @property
    def sort(self):
        return BVSort(self.width)

    def __repr__(self):
        if self.width % 4 == 0:
            return f"#x{self.value:0{self.width // 4}x}"
        return f"#b{self.value:0{self.width}b}"


@dataclass(frozen=True)
class Var(Term):
    name: str
    sort: Sort

    def __repr__(self):
        return self.name


class App(Term):
    """Function application: an operator applied to arguments."""
    __slots__ = ('op', 'args', 'sort', '_params')

    def __init__(self, op: str, args: List[Term], sort: Sort = None,
                 params: tuple = ()):
        self.op     = op
        self.args   = args
        self.sort   = sort
        self._params = params   # extra index params, e.g. (hi, lo) for extract

    def __repr__(self):
        if self._params:
            indexed = f"(_ {self.op} {' '.join(str(p) for p in self._params)})"
            if not self.args:
                return indexed
            return f"({indexed} {' '.join(repr(a) for a in self.args)})"
        if not self.args:
            return self.op
        return f"({self.op} {' '.join(repr(a) for a in self.args)})"

    def __eq__(self, other):
        return (isinstance(other, App) and self.op == other.op
                and self._params == other._params and self.args == other.args)

    def __hash__(self):
        return hash((self.op, self._params, tuple(id(a) for a in self.args)))


# ── Convenience constructors ───────────────────────────────────────────────────

TRUE  = BoolLit(True)
FALSE = BoolLit(False)


def mk_and(args: List[Term]) -> Term:
    flat = []
    for a in args:
        if isinstance(a, BoolLit):
            if not a.value: return FALSE
        elif isinstance(a, App) and a.op == 'and' and not a._params:
            flat.extend(a.args)
        else:
            flat.append(a)
    if not flat: return TRUE
    if len(flat) == 1: return flat[0]
    return App('and', flat, BOOL)


def mk_or(args: List[Term]) -> Term:
    flat = []
    for a in args:
        if isinstance(a, BoolLit):
            if a.value: return TRUE
        elif isinstance(a, App) and a.op == 'or' and not a._params:
            flat.extend(a.args)
        else:
            flat.append(a)
    if not flat: return FALSE
    if len(flat) == 1: return flat[0]
    return App('or', flat, BOOL)


def mk_not(a: Term) -> Term:
    if isinstance(a, BoolLit): return BoolLit(not a.value)
    if isinstance(a, App) and a.op == 'not': return a.args[0]
    return App('not', [a], BOOL)


def mk_eq(a: Term, b: Term) -> Term:
    return App('=', [a, b], BOOL)


# ── NsFormula container ────────────────────────────────────────────────────────

@dataclass
class NsFormula:
    logic:      str
    assertions: List[Term]       # list of Bool terms
    variables:  Dict[str, Var]   # name → Var
    var_names:  List[str]        # sorted variable names
    source:     str              # raw SMT-LIB2 string
