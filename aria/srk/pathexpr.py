"""
Path expressions for graph analysis.

This module provides functionality for working with path expressions,
which are regular expressions over graph edges used to represent
sets of paths in directed graphs.

Path expressions support operations like:
- Edge: individual edges
- Mul: concatenation of path expressions
- Add: union of path expressions
- Star: Kleene star (zero or more repetitions)
- One: empty path
- Zero: empty set of paths
- Omega: infinite repetition (for infinite paths)
- Segment: grouping construct for nested path expressions
"""

from __future__ import annotations
from typing import TypeVar, Generic, Union, Dict, Any, Callable, Optional, Protocol, Set, Tuple, IO
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import hashlib
import weakref

# Type variables
A = TypeVar('A')  # Algebra result type
B = TypeVar('B')  # Omega algebra result type
T = TypeVar('T')  # Path expression type

# Forward declarations for recursive types
PathExpr = 'PathExpr'
OmegaPathExpr = 'OmegaPathExpr'


@dataclass(frozen=True)
class PathExpr:
    """Hash-consed path expression node."""
    tag: int
    obj: Union[
        'Edge', 'Mul', 'Add', 'Star', 'One', 'Zero', 'Omega', 'Segment'
    ]

    def __hash__(self) -> int:
        return hash((type(self.obj), self.obj))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PathExpr):
            return False
        return (self.obj == other.obj and
                type(self.obj) == type(other.obj))


class Edge:
    """Edge constructor for path expressions."""
    def __init__(self, src: int, tgt: int):
        self.src = src
        self.tgt = tgt

    def __repr__(self) -> str:
        return f"Edge({self.src}, {self.tgt})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Edge):
            return False
        return self.src == other.src and self.tgt == other.tgt

    def __hash__(self) -> int:
        return hash((self.src, self.tgt))


class Mul:
    """Multiplication (concatenation) constructor."""
    def __init__(self, left: PathExpr, right: PathExpr):
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"Mul({self.left}, {self.right})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mul):
            return False
        return self.left.tag == other.left.tag and self.right.tag == other.right.tag

    def __hash__(self) -> int:
        return hash((self.left.tag, self.right.tag))


class Add:
    """Addition (union) constructor."""
    def __init__(self, left: PathExpr, right: PathExpr):
        self.left = left
        self.right = right

    def __repr__(self) -> str:
        return f"Add({self.left}, {self.right})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Add):
            return False
        return self.left.tag == other.left.tag and self.right.tag == other.right.tag

    def __hash__(self) -> int:
        return hash((self.left.tag, self.right.tag))


class Star:
    """Kleene star constructor."""
    def __init__(self, expr: PathExpr):
        self.expr = expr

    def __repr__(self) -> str:
        return f"Star({self.expr})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Star):
            return False
        return self.expr.tag == other.expr.tag

    def __hash__(self) -> int:
        return hash(self.expr.tag)


class Omega:
    """Omega (infinite repetition) constructor."""
    def __init__(self, expr: PathExpr):
        self.expr = expr

    def __repr__(self) -> str:
        return f"Omega({self.expr})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Omega):
            return False
        return self.expr.tag == other.expr.tag

    def __hash__(self) -> int:
        return hash(self.expr.tag)


class Segment:
    """Segment (grouping) constructor."""
    def __init__(self, expr: PathExpr):
        self.expr = expr

    def __repr__(self) -> str:
        return f"Segment({self.expr})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Segment):
            return False
        return self.expr.tag == other.expr.tag

    def __hash__(self) -> int:
        return hash(self.expr.tag)


class One:
    """One (empty path) constructor."""
    def __repr__(self) -> str:
        return "One"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, One)

    def __hash__(self) -> int:
        return hash("One")


class Zero:
    """Zero (empty set) constructor."""
    def __repr__(self) -> str:
        return "Zero"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Zero)

    def __hash__(self) -> int:
        return hash("Zero")


class PathExprTable:
    """Hash table for memoizing path expression evaluations."""

    def __init__(self, size: int = 991):
        self.table: Dict[PathExpr, Any] = {}
        self.size = size

    def get(self, expr: PathExpr) -> Optional[Any]:
        return self.table.get(expr)

    def put(self, expr: PathExpr, value: Any) -> None:
        self.table[expr] = value

    def contains(self, expr: PathExpr) -> bool:
        return expr in self.table

    def clear(self) -> None:
        self.table.clear()


class OmegaPathExprTable:
    """Hash table for memoizing omega path expression evaluations."""

    def __init__(self, size: int = 991):
        self.table: Dict[OmegaPathExpr, Any] = {}
        self.size = size

    def get(self, expr: OmegaPathExpr) -> Optional[Any]:
        return self.table.get(expr)

    def put(self, expr: OmegaPathExpr, value: Any) -> None:
        self.table[expr] = value

    def contains(self, expr: OmegaPathExpr) -> bool:
        return expr in self.table

    def clear(self) -> None:
        self.table.clear()


class PathExprContext:
    """Context for managing hash-consed path expressions."""

    def __init__(self, size: int = 991):
        self.table: Dict[Any, PathExpr] = {}
        self.next_tag = 0
        self.size = size

    def hashcons(self, obj: Any) -> PathExpr:
        """Get the hash-consed version of a path expression object."""
        key = obj
        if key in self.table:
            return self.table[key]

        expr = PathExpr(self.next_tag, obj)
        self.table[key] = expr
        self.next_tag += 1
        return expr


# Type definitions for algebras
class OpenPathExpr:
    """Union type for open path expressions in algebras."""
    pass


class OpenNestedPathExpr:
    """Union type for open nested path expressions in algebras."""
    pass


class OpenOmegaPathExpr:
    """Union type for open omega path expressions in algebras."""
    pass


# Algebra type definitions
Algebra = Callable[[Union[
    'EdgeAlg', 'MulAlg', 'AddAlg', 'StarAlg', 'OneAlg', 'ZeroAlg'
]], A]

NestedAlgebra = Callable[[Union[
    'SegmentAlg', OpenPathExpr
]], A]

OmegaAlgebra = Callable[[Union[
    'OmegaAlg', 'OmegaMulAlg', 'OmegaAddAlg'
]], B]


@dataclass
class EdgeAlg:
    """Edge case for algebras."""
    src: int
    tgt: int


@dataclass
class MulAlg:
    """Multiplication case for algebras."""
    left: A
    right: A


@dataclass
class AddAlg:
    """Addition case for algebras."""
    left: A
    right: A


@dataclass
class StarAlg:
    """Star case for algebras."""
    expr: A


@dataclass
class OneAlg:
    """One case for algebras."""
    pass


@dataclass
class ZeroAlg:
    """Zero case for algebras."""
    pass


@dataclass
class SegmentAlg:
    """Segment case for nested algebras."""
    expr: A


@dataclass
class OmegaAlg:
    """Omega case for omega algebras."""
    expr: A


@dataclass
class OmegaMulAlg:
    """Omega multiplication case for omega algebras."""
    left: A
    right: B


@dataclass
class OmegaAddAlg:
    """Omega addition case for omega algebras."""
    left: B
    right: B


# Constructor functions
def mk_one(context: PathExprContext) -> PathExpr:
    """Create the One (empty path) path expression."""
    return context.hashcons(One())


def mk_zero(context: PathExprContext) -> PathExpr:
    """Create the Zero (empty set) path expression."""
    return context.hashcons(Zero())


def mk_edge(context: PathExprContext, src: int, tgt: int) -> PathExpr:
    """Create an Edge path expression."""
    return context.hashcons(Edge(src, tgt))


def mk_mul(context: PathExprContext, left: PathExpr, right: PathExpr) -> PathExpr:
    """Create a multiplication (concatenation) path expression."""
    # Simplification rules
    if isinstance(left.obj, Zero) or isinstance(right.obj, Zero):
        return mk_zero(context)
    if isinstance(left.obj, One):
        return right
    if isinstance(right.obj, One):
        return left

    # Additional simplification rules
    if isinstance(left.obj, Mul) and isinstance(right.obj, Mul):
        # (a * b) * (c * d) = a * b * c * d
        return mk_mul(context,
                     mk_mul(context, left.obj.left, left.obj.right),
                     mk_mul(context, right.obj.left, right.obj.right))

    return context.hashcons(Mul(left, right))


def mk_add(context: PathExprContext, left: PathExpr, right: PathExpr) -> PathExpr:
    """Create an addition (union) path expression."""
    # Simplification rules
    if isinstance(left.obj, Zero):
        return right
    if isinstance(right.obj, Zero):
        return left

    return context.hashcons(Add(left, right))


def mk_star(context: PathExprContext, expr: PathExpr) -> PathExpr:
    """Create a Kleene star path expression."""
    # Simplification rules
    if isinstance(expr.obj, Zero) or isinstance(expr.obj, One):
        return mk_one(context)
    if isinstance(expr.obj, Star):
        return expr

    # Additional simplification rules
    if isinstance(expr.obj, Mul):
        # (a * b)* = a* * b* (distributivity)
        left_star = mk_star(context, expr.obj.left)
        right_star = mk_star(context, expr.obj.right)
        return mk_mul(context, left_star, right_star)

    return context.hashcons(Star(expr))


def mk_omega(context: PathExprContext, expr: PathExpr) -> PathExpr:
    """Create an omega (infinite repetition) path expression."""
    # Simplification rules
    if isinstance(expr.obj, Zero):
        return mk_zero(context)
    if isinstance(expr.obj, Omega):
        return expr

    # Additional simplification rules
    if isinstance(expr.obj, Star):
        # ω* = ω (since star already allows infinite repetition)
        return mk_omega(context, expr.obj.expr)

    return context.hashcons(Omega(expr))


def mk_segment(context: PathExprContext, expr: PathExpr) -> PathExpr:
    """Create a segment (grouping) path expression."""
    # Simplification rules
    if isinstance(expr.obj, Zero):
        return mk_zero(context)
    if isinstance(expr.obj, One):
        return mk_one(context)

    return context.hashcons(Segment(expr))


def mk_omega_add(context: PathExprContext, left: PathExpr, right: PathExpr) -> PathExpr:
    """Create an omega addition path expression."""
    return mk_add(context, left, right)


def mk_omega_mul(context: PathExprContext, left: PathExpr, right: PathExpr) -> PathExpr:
    """Create an omega multiplication path expression."""
    return mk_mul(context, left, right)


def promote(expr: PathExpr) -> PathExpr:
    """Cast a simple path expression to a nested path expression."""
    return expr


def promote_omega(expr: PathExpr) -> PathExpr:
    """Cast a simple omega path expression to a nested omega path expression."""
    return expr


def destruct_flat(expr: PathExpr) -> Union[
    EdgeAlg, MulAlg, AddAlg, StarAlg, OneAlg, ZeroAlg, OmegaAlg, SegmentAlg
]:
    """Destruct a flat path expression for pattern matching."""
    obj = expr.obj

    if isinstance(obj, Edge):
        return EdgeAlg(obj.src, obj.tgt)
    elif isinstance(obj, Mul):
        return MulAlg(destruct_flat(obj.left), destruct_flat(obj.right))
    elif isinstance(obj, Add):
        return AddAlg(destruct_flat(obj.left), destruct_flat(obj.right))
    elif isinstance(obj, Star):
        return StarAlg(destruct_flat(obj.expr))
    elif isinstance(obj, Omega):
        return OmegaAlg(destruct_flat(obj.expr))
    elif isinstance(obj, One):
        return OneAlg()
    elif isinstance(obj, Zero):
        return ZeroAlg()
    elif isinstance(obj, Segment):
        return SegmentAlg(destruct_flat(obj.expr))
    else:
        raise ValueError(f"Unknown path expression type: {type(obj)}")


def pp_expr(expr: PathExpr) -> str:
    """Pretty print a path expression."""
    flat = destruct_flat(expr)

    if isinstance(flat, EdgeAlg):
        return f"{flat.src}->{flat.tgt}"
    elif isinstance(flat, MulAlg):
        left_str = pp_expr(flat.left) if isinstance(flat.left, PathExpr) else str(flat.left)
        right_str = pp_expr(flat.right) if isinstance(flat.right, PathExpr) else str(flat.right)
        return f"({left_str} {right_str})"
    elif isinstance(flat, AddAlg):
        left_str = pp_expr(flat.left) if isinstance(flat.left, PathExpr) else str(flat.left)
        right_str = pp_expr(flat.right) if isinstance(flat.right, PathExpr) else str(flat.right)
        return f"({left_str} + {right_str})"
    elif isinstance(flat, StarAlg):
        inner_str = pp_expr(flat.expr) if isinstance(flat.expr, PathExpr) else str(flat.expr)
        return f"({inner_str})*"
    elif isinstance(flat, OmegaAlg):
        inner_str = pp_expr(flat.expr) if isinstance(flat.expr, PathExpr) else str(flat.expr)
        return f"({inner_str})ω"
    elif isinstance(flat, OneAlg):
        return "1"
    elif isinstance(flat, ZeroAlg):
        return "0"
    elif isinstance(flat, SegmentAlg):
        inner_str = pp_expr(flat.expr) if isinstance(flat.expr, PathExpr) else str(flat.expr)
        return f"[{inner_str}]"
    else:
        return str(flat)


def show(expr: PathExpr) -> str:
    """Convert a path expression to string."""
    return pp_expr(expr)


def show_omega(expr: PathExpr) -> str:
    """Convert an omega path expression to string."""
    return pp_expr(expr)


def mk_table(size: int = 991) -> PathExprTable:
    """Create a new memoization table for path expressions."""
    return PathExprTable(size)


def mk_context(size: int = 991) -> PathExprContext:
    """Create a new path expression context."""
    return PathExprContext(size)


def mk_omega_table(table: PathExprTable, size: int = 991) -> Tuple[PathExprTable, OmegaPathExprTable]:
    """Create a new omega memoization table."""
    return (table, OmegaPathExprTable(size))


def eval(algebra: Algebra[A], expr: PathExpr, table: Optional[PathExprTable] = None) -> A:
    """Evaluate a path expression using the given algebra."""
    if table is None:
        table = mk_table()

    def eval_rec(e: PathExpr) -> A:
        if table.contains(e):
            return table.get(e)

        result = None
        obj = e.obj

        if isinstance(obj, One):
            result = algebra(OneAlg())
        elif isinstance(obj, Zero):
            result = algebra(ZeroAlg())
        elif isinstance(obj, Edge):
            result = algebra(EdgeAlg(obj.src, obj.tgt))
        elif isinstance(obj, Mul):
            left_val = eval_rec(obj.left)
            right_val = eval_rec(obj.right)
            result = algebra(MulAlg(left_val, right_val))
        elif isinstance(obj, Add):
            left_val = eval_rec(obj.left)
            right_val = eval_rec(obj.right)
            result = algebra(AddAlg(left_val, right_val))
        elif isinstance(obj, Star):
            inner_val = eval_rec(obj.expr)
            result = algebra(StarAlg(inner_val))
        elif isinstance(obj, Segment):
            inner_val = eval_rec(obj.expr)
            result = algebra(SegmentAlg(inner_val))
        else:
            raise ValueError(f"Cannot evaluate path expression type {type(e.obj).__name__}: {e}")

        table.put(e, result)
        return result

    return eval_rec(expr)


def eval_nested(nested_algebra: NestedAlgebra[A], expr: PathExpr, table: Optional[PathExprTable] = None) -> A:
    """Evaluate a nested path expression using the given nested algebra."""
    if table is None:
        table = mk_table()

    def eval_rec(e: PathExpr) -> A:
        if table.contains(e):
            return table.get(e)

        result = None
        obj = e.obj

        if isinstance(obj, One):
            result = nested_algebra(OneAlg())
        elif isinstance(obj, Zero):
            result = nested_algebra(ZeroAlg())
        elif isinstance(obj, Edge):
            result = nested_algebra(EdgeAlg(obj.src, obj.tgt))
        elif isinstance(obj, Mul):
            left_val = eval_rec(obj.left)
            right_val = eval_rec(obj.right)
            result = nested_algebra(MulAlg(left_val, right_val))
        elif isinstance(obj, Add):
            left_val = eval_rec(obj.left)
            right_val = eval_rec(obj.right)
            result = nested_algebra(AddAlg(left_val, right_val))
        elif isinstance(obj, Star):
            inner_val = eval_rec(obj.expr)
            result = nested_algebra(StarAlg(inner_val))
        elif isinstance(obj, Segment):
            inner_val = eval_rec(obj.expr)
            result = nested_algebra(SegmentAlg(inner_val))
        else:
            raise ValueError(f"Cannot evaluate nested path expression: {e}")

        table.put(e, result)
        return result

    return eval_rec(expr)


def eval_omega(
    algebra: NestedAlgebra[A],
    omega_algebra: OmegaAlgebra[A, B],
    expr: PathExpr,
    table: Optional[Tuple[PathExprTable, OmegaPathExprTable]] = None
) -> B:
    """Evaluate an omega path expression using the given algebras."""
    if table is None:
        table = mk_omega_table(mk_table())

    path_table, omega_table = table

    def eval_rec(e: PathExpr) -> B:
        if omega_table.contains(e):
            return omega_table.get(e)

        result = None
        obj = e.obj

        if isinstance(obj, Omega):
            inner_val = eval_nested(algebra, obj.expr, path_table)
            result = omega_algebra(OmegaAlg(inner_val))
        elif isinstance(obj, Add):
            left_val = eval_rec(obj.left)
            right_val = eval_rec(obj.right)
            result = omega_algebra(OmegaAddAlg(left_val, right_val))
        elif isinstance(obj, Mul):
            left_val = eval_nested(algebra, obj.left, path_table)
            right_val = eval_rec(obj.right)
            result = omega_algebra(OmegaMulAlg(left_val, right_val))
        elif isinstance(obj, Zero):
            inner_val = eval_nested(algebra, e, path_table)
            result = omega_algebra(OmegaAlg(inner_val))
        else:
            raise ValueError(f"Cannot evaluate omega path expression: {e}")

        omega_table.put(e, result)
        return result

    return eval_rec(expr)


def forget(table: PathExprTable, predicate: Callable[[int, int], bool]) -> None:
    """Forget memoized values for path expressions that involve edges not satisfying the predicate."""
    keys_to_remove = []

    for expr in table.table:
        # Check if this expression involves edges that don't satisfy the predicate
        def check_expr(e: PathExpr) -> bool:
            obj = e.obj
            if isinstance(obj, Edge):
                return predicate(obj.src, obj.tgt)
            elif isinstance(obj, (Mul, Add)):
                return check_expr(obj.left) and check_expr(obj.right)
            elif isinstance(obj, (Star, Omega, Segment)):
                return check_expr(obj.expr)
            elif isinstance(obj, (One, Zero)):
                return True
            else:
                return True

        if not check_expr(expr):
            keys_to_remove.append(expr)

    for key in keys_to_remove:
        del table.table[key]


def accept_epsilon(expr: PathExpr) -> bool:
    """Check if a path expression accepts the empty path (epsilon)."""
    obj = expr.obj

    if isinstance(obj, Zero):
        return False
    elif isinstance(obj, One):
        return True
    elif isinstance(obj, Edge):
        return False
    elif isinstance(obj, Mul):
        return accept_epsilon(obj.left) and accept_epsilon(obj.right)
    elif isinstance(obj, Add):
        return accept_epsilon(obj.left) or accept_epsilon(obj.right)
    elif isinstance(obj, Star):
        return True
    elif isinstance(obj, Omega):
        return False
    elif isinstance(obj, Segment):
        return accept_epsilon(obj.expr)
    else:
        return False


def first(expr: PathExpr) -> Set[Tuple[int, int]]:
    """Compute the set of edges that can start a path matching the expression."""
    obj = expr.obj

    if isinstance(obj, Zero) or isinstance(obj, One):
        return set()
    elif isinstance(obj, Edge):
        return {(obj.src, obj.tgt)}
    elif isinstance(obj, Mul):
        if accept_epsilon(obj.left):
            return first(obj.left) | first(obj.right)
        else:
            return first(obj.left)
    elif isinstance(obj, Add):
        return first(obj.left) | first(obj.right)
    elif isinstance(obj, Star):
        return first(obj.expr)
    elif isinstance(obj, Omega):
        return first(obj.expr)
    elif isinstance(obj, Segment):
        return first(obj.expr)
    else:
        return set()


def derivative(context: PathExprContext, edge: Tuple[int, int], expr: PathExpr) -> PathExpr:
    """Compute the derivative of a path expression with respect to an edge."""
    src, tgt = edge
    obj = expr.obj

    if isinstance(obj, Zero) or isinstance(obj, One):
        return mk_zero(context)
    elif isinstance(obj, Edge):
        if obj.src == src and obj.tgt == tgt:
            return mk_one(context)
        else:
            return mk_zero(context)
    elif isinstance(obj, Mul):
        deriv_left = derivative(context, edge, obj.left)
        result = mk_mul(context, deriv_left, obj.right)

        if accept_epsilon(obj.left):
            deriv_right = derivative(context, edge, obj.right)
            result = mk_add(context, result, deriv_right)

        return result
    elif isinstance(obj, Star):
        deriv_inner = derivative(context, edge, obj.expr)
        return mk_mul(context, deriv_inner, expr)
    elif isinstance(obj, Add):
        deriv_left = derivative(context, edge, obj.left)
        deriv_right = derivative(context, edge, obj.right)
        return mk_add(context, deriv_left, deriv_right)
    elif isinstance(obj, Omega):
        deriv_inner = derivative(context, edge, obj.expr)
        return mk_mul(context, deriv_inner, expr)
    elif isinstance(obj, Segment):
        deriv_inner = derivative(context, edge, obj.expr)
        return mk_segment(context, deriv_inner)
    else:
        return mk_zero(context)


def equiv(context: PathExprContext, expr1: PathExpr, expr2: PathExpr) -> bool:
    """Check if two path expressions are equivalent."""
    # Simple equivalence check based on epsilon acceptance and first sets
    if accept_epsilon(expr1) != accept_epsilon(expr2):
        return False

    first1 = first(expr1)
    first2 = first(expr2)

    if first1 != first2:
        return False

    # Check derivatives for all edges in the first sets
    all_edges = first1 | first2

    for edge in all_edges:
        deriv1 = derivative(context, edge, expr1)
        deriv2 = derivative(context, edge, expr2)

        if not equiv(context, deriv1, deriv2):
            return False

    return True


# Pretty printing functions
def pp(formatter: IO[str], expr: PathExpr) -> None:
    """Pretty print a path expression using a formatter."""
    print(pp_expr(expr), file=formatter)


def pp_omega(formatter: IO[str], expr: PathExpr) -> None:
    """Pretty print an omega path expression using a formatter."""
    print(pp_expr(expr), file=formatter)
