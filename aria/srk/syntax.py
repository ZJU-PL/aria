"""
Core syntax module for symbolic expressions and formulas.

This module implements the fundamental data structures for symbolic reasoning
in the SRK (Symbolic Reasoning Kit) system. It provides the foundation for
representing and manipulating symbolic expressions, formulas, and logical
structures used in program analysis and verification.

Key Components:
- Type system for expressions (integers, reals, booleans, arrays, functions)
- Symbol management with unique identifiers and optional names
- Expression hierarchy (terms, formulas, arithmetic/logical operations)
- Context management for symbol scoping and expression construction
- Substitution and rewriting operations for expression transformation

Example:
    >>> from aria.srk.syntax import Context, Type, mk_symbol
    >>> ctx = Context()
    >>> x = mk_symbol(ctx, 'x', Type.real)
    >>> expr = ctx.mk_add(ctx.mk_const(x), ctx.mk_real(1))
    >>> print(expr)  # x + 1
"""

from __future__ import annotations
from typing import (
    Dict,
    List,
    Set,
    Tuple,
    Optional,
    Union,
    Any,
    TypeVar,
    Generic,
    Callable,
)
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass, field
from fractions import Fraction
import itertools

# Import QQ for rational number operations
from aria.srk.qQ import QQ

# Type variables for generic types
T = TypeVar("T")
U = TypeVar("U")


# Core types
class Type(Enum):
    """Expression types in the SRK type system.

    This enum defines the basic types that expressions can have in SRK:
    - INT: Integer values and arithmetic
    - REAL: Real number values and arithmetic
    - BOOL: Boolean values and logical operations
    - ARRAY: Array types for indexed structures
    - FUN: Function types for higher-order operations

    The type system ensures type safety in symbolic expressions and
    helps guide the application of appropriate operations.
    """

    INT = "Int"
    REAL = "Real"
    BOOL = "Bool"
    ARRAY = "Array"
    FUN = "Fun"

    def __str__(self) -> str:
        return self.value


# Type aliases for cleaner code - these are not enums but just type hints
ArithType = Union[Type.INT, Type.REAL]
TermType = Union[Type.INT, Type.REAL, Type.ARRAY]
FormulaType = Union[Type.INT, Type.REAL, Type.BOOL, Type.ARRAY]


class Symbol:
    """Represents a symbol in symbolic expressions.

    Symbols are the atomic units in SRK expressions, identified by unique
    integer IDs and optionally having human-readable names. They carry type
    information that determines what operations can be performed with them.

    Attributes:
        id (int): Unique integer identifier for the symbol.
        name (Optional[str]): Optional human-readable name for the symbol.
        typ (Type): The type of the symbol (INT, REAL, BOOL, etc.).

    Example:
        >>> sym = Symbol(42, 'x', Type.REAL)
        >>> print(f"Symbol {sym.name} has ID {sym.id} and type {sym.typ}")
    """

    def __init__(self, id: int, name: Optional[str] = None, typ: Type = Type.INT):
        """Initialize a symbol with unique ID, optional name, and type.

        Args:
            id: Unique integer identifier.
            name: Optional human-readable name.
            typ: The type of values this symbol represents.
        """
        self.id = id
        self.name = name
        self.typ = typ

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Symbol):
            return False
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    def __str__(self) -> str:
        if self.name:
            return self.name
        return f"s{self.id}"

    def __repr__(self) -> str:
        if self.name:
            return f"Symbol({self.id}, '{self.name}', {self.typ})"
        return f"Symbol({self.id}, {self.typ})"


class Context:
    """Manages symbols and expressions within a context.

    Contexts ensure expressions don't cross boundaries and provide
    symbol management functionality.
    """

    def __init__(self):
        self._next_id = 0
        self._symbols: Dict[int, Symbol] = {}
        self._named_symbols: Dict[str, Symbol] = {}
        self._expressions: Dict[int, Expression] = {}

    def mk_symbol(self, name: Optional[str] = None, typ: Type = Type.INT) -> Symbol:
        """Create a fresh symbol."""
        if not isinstance(typ, Type):
            raise TypeError(f"typ must be a Type enum value, got {type(typ)}")
        if name is not None and not isinstance(name, str):
            raise TypeError(f"name must be a string or None, got {type(name)}")

        symbol_id = self._next_id
        self._next_id += 1

        symbol = Symbol(symbol_id, name, typ)
        self._symbols[symbol_id] = symbol

        if name:
            # In the original SRK, multiple symbols can have the same name
            # We store the most recently created one
            self._named_symbols[name] = symbol

        return symbol

    def mk_var(self, var_id_or_symbol, typ: Type = None, name: Optional[str] = None):
        """Create a variable expression.

        Can be called in two ways:
        - mk_var(var_id: int, typ: Type) - create variable with given ID and type
        - mk_var(symbol: Symbol) - create variable from symbol (uses symbol.id and symbol.typ)
        """
        if isinstance(var_id_or_symbol, Symbol):
            # Called as mk_var(symbol) - use symbol's id and type
            symbol = var_id_or_symbol
            return Var(symbol.id, symbol.typ, symbol.name)
        elif isinstance(var_id_or_symbol, int) and typ is not None:
            # Called as mk_var(var_id, typ) - use given ID and type
            if not isinstance(typ, Type):
                raise TypeError(f"typ must be a Type enum value, got {type(typ)}")
            return Var(var_id_or_symbol, typ, name)
        else:
            raise TypeError(
                f"mk_var expects (var_id, typ) or (symbol), got ({type(var_id_or_symbol)}, {type(typ)})"
            )

    def mk_const(self, symbol: Symbol):
        """Create a constant expression."""
        return Const(symbol)

    def register_named_symbol(self, name: str, typ: Type) -> None:
        """Register a named symbol.

        Matches OCaml semantics: the name must be unique across *types*.
        Calling register_named_symbol with the same name and the same type
        is idempotent (no-op).  Calling it with the same name but a different
        type raises ValueError.
        """
        if name in self._named_symbols:
            existing = self._named_symbols[name]
            if existing.typ != typ:
                raise ValueError(
                    f"Symbol name '{name}' already registered with type "
                    f"{existing.typ}, cannot re-register with {typ}"
                )
            # Same name, same type → idempotent, nothing to do.
            return
        self.mk_symbol(name, typ)

    def is_registered_name(self, name: str) -> bool:
        """Check if a name is registered."""
        return name in self._named_symbols

    def get_named_symbol(self, name: str) -> Symbol:
        """Get a symbol by name."""
        if name not in self._named_symbols:
            raise KeyError(f"Symbol name '{name}' not found")
        return self._named_symbols[name]

    def symbol_name(self, symbol: Symbol) -> Optional[str]:
        """Get the name of a symbol if it has one."""
        for name, sym in self._named_symbols.items():
            if sym == symbol:
                return name
        return None

    def dup_symbol(self, symbol: Symbol) -> Symbol:
        """Return a fresh symbol with the same name and type as *symbol*.

        Mirrors OCaml's ``dup_symbol``: the new symbol has a distinct id but
        inherits the name (if any) and type of the original.
        """
        return self.mk_symbol(name=symbol.name, typ=symbol.typ)

    @staticmethod
    def compare_symbol(a: Symbol, b: Symbol) -> int:
        """Total order on symbols by id (mirrors OCaml's compare_symbol)."""
        if a.id < b.id:
            return -1
        if a.id > b.id:
            return 1
        return 0

    def typ_symbol(self, symbol: Symbol) -> Type:
        """Get the type of a symbol."""
        return symbol.typ

    def show_symbol(self, symbol: Symbol) -> str:
        """String representation of a symbol."""
        return str(symbol)

    def stats(self) -> Tuple[int, int, int]:
        """Return statistics: (num_expressions, num_symbols, num_named_symbols)."""
        return len(self._expressions), len(self._symbols), len(self._named_symbols)


class Expression(ABC):
    """Abstract base class for all expressions."""

    typ: Type

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Ensure all subclasses have a typ attribute
        if not hasattr(cls, "typ"):
            raise TypeError(f"Expression subclass {cls} must define 'typ' attribute")

    def __eq__(self, other: object) -> bool:
        """Check equality of expressions."""
        if not isinstance(other, Expression):
            return False
        if type(self) != type(other):
            return False
        # For expressions with the same type, compare their structural content
        # This is a basic implementation - subclasses should override for proper equality
        return True

    def __hash__(self) -> int:
        """Hash based on type and attributes."""
        # Simple hash based on type and a few key attributes
        attrs = []
        for attr_name in ["typ"]:
            if hasattr(self, attr_name):
                attrs.append(getattr(self, attr_name))
        return hash((type(self), tuple(attrs)))

    def __str__(self) -> str:
        """String representation of expression."""
        # Provide a more informative default representation
        attrs = []
        for attr_name in dir(self):
            if not attr_name.startswith("_") and attr_name != "typ":
                try:
                    value = getattr(self, attr_name)
                    if not callable(value) and value != self.typ:
                        attrs.append(f"{attr_name}={value}")
                except:
                    pass
        if attrs:
            return f"{type(self).__name__}({', '.join(attrs)})"
        return f"{type(self).__name__}()"

    def accept(self, visitor: ExpressionVisitor[T]) -> T:
        """Accept a visitor."""
        # This should be overridden by subclasses to call the appropriate visit method
        raise NotImplementedError(f"accept method not implemented for {type(self)}")


class ExpressionVisitor(Generic[T]):
    """Visitor pattern for expressions."""

    def visit_var(self, var: Var) -> T:
        """Visit a variable expression."""
        return self._default_visit(var)

    def visit_const(self, const: Const) -> T:
        """Visit a constant expression."""
        return self._default_visit(const)

    def visit_app(self, app: App) -> T:
        """Visit a function application expression."""
        return self._default_visit(app)

    def visit_select(self, select: Select) -> T:
        """Visit a select expression."""
        return self._default_visit(select)

    def visit_store(self, store: Store) -> T:
        """Visit a store expression."""
        return self._default_visit(store)

    def visit_add(self, add: Add) -> T:
        """Visit an addition expression."""
        return self._default_visit(add)

    def visit_mul(self, mul: Mul) -> T:
        """Visit a multiplication expression."""
        return self._default_visit(mul)

    def visit_ite(self, ite: Ite) -> T:
        """Visit an if-then-else expression."""
        return self._default_visit(ite)

    def visit_div(self, div: Div) -> T:
        """Visit a division expression."""
        return self._default_visit(div)

    def visit_mod(self, mod: Mod) -> T:
        """Visit a modulo expression."""
        return self._default_visit(mod)

    def visit_floor(self, floor: Floor) -> T:
        """Visit a floor expression."""
        return self._default_visit(floor)

    def visit_neg(self, neg: Neg) -> T:
        """Visit a negation expression."""
        return self._default_visit(neg)

    def visit_pow(self, pow_expr: Pow) -> T:
        """Visit an exponentiation expression."""
        return self._default_visit(pow_expr)

    def visit_true(self, true_expr: TrueExpr) -> T:
        """Visit a true expression."""
        return self._default_visit(true_expr)

    def visit_false(self, false_expr: FalseExpr) -> T:
        """Visit a false expression."""
        return self._default_visit(false_expr)

    def visit_and(self, and_expr: And) -> T:
        """Visit an and expression."""
        return self._default_visit(and_expr)

    def visit_or(self, or_expr: Or) -> T:
        """Visit an or expression."""
        return self._default_visit(or_expr)

    def visit_not(self, not_expr: Not) -> T:
        """Visit a not expression."""
        return self._default_visit(not_expr)

    def visit_eq(self, eq: Eq) -> T:
        """Visit an equality expression."""
        return self._default_visit(eq)

    def visit_lt(self, lt: Lt) -> T:
        """Visit a less-than expression."""
        return self._default_visit(lt)

    def visit_leq(self, leq: Leq) -> T:
        """Visit a less-than-or-equal expression."""
        return self._default_visit(leq)

    def visit_forall(self, forall: Forall) -> T:
        """Visit a forall expression."""
        return self._default_visit(forall)

    def visit_exists(self, exists: Exists) -> T:
        """Visit an exists expression."""
        return self._default_visit(exists)


class DefaultExpressionVisitor(ExpressionVisitor[T]):
    """Default visitor implementation that provides basic functionality."""

    def visit_var(self, var: Var) -> T:
        """Default visit for variables."""
        return self._default_visit(var)

    def visit_const(self, const: Const) -> T:
        """Default visit for constants."""
        return self._default_visit(const)

    def visit_app(self, app: App) -> T:
        """Default visit for function applications."""
        return self._default_visit(app)

    def visit_select(self, select: Select) -> T:
        """Default visit for select expressions."""
        return self._default_visit(select)

    def visit_store(self, store: Store) -> T:
        """Default visit for store expressions."""
        return self._default_visit(store)

    def visit_add(self, add: Add) -> T:
        """Default visit for addition."""
        return self._default_visit(add)

    def visit_mul(self, mul: Mul) -> T:
        """Default visit for multiplication."""
        return self._default_visit(mul)

    def visit_ite(self, ite: Ite) -> T:
        """Default visit for if-then-else."""
        return self._default_visit(ite)

    def visit_div(self, div: Div) -> T:
        """Default visit for division."""
        return self._default_visit(div)

    def visit_mod(self, mod: Mod) -> T:
        """Default visit for modulo."""
        return self._default_visit(mod)

    def visit_floor(self, floor: Floor) -> T:
        """Default visit for floor."""
        return self._default_visit(floor)

    def visit_neg(self, neg: Neg) -> T:
        """Default visit for negation."""
        return self._default_visit(neg)

    def visit_pow(self, pow_expr: Pow) -> T:
        """Default visit for exponentiation."""
        return self._default_visit(pow_expr)

    def visit_true(self, true_expr: TrueExpr) -> T:
        """Default visit for true."""
        return self._default_visit(true_expr)

    def visit_false(self, false_expr: FalseExpr) -> T:
        """Default visit for false."""
        return self._default_visit(false_expr)

    def visit_and(self, and_expr: And) -> T:
        """Default visit for and."""
        return self._default_visit(and_expr)

    def visit_or(self, or_expr: Or) -> T:
        """Default visit for or."""
        return self._default_visit(or_expr)

    def visit_not(self, not_expr: Not) -> T:
        """Default visit for not."""
        return self._default_visit(not_expr)

    def visit_eq(self, eq: Eq) -> T:
        """Default visit for equality."""
        return self._default_visit(eq)

    def visit_lt(self, lt: Lt) -> T:
        """Default visit for less-than."""
        return self._default_visit(lt)

    def visit_leq(self, leq: Leq) -> T:
        """Default visit for less-than-or-equal."""
        return self._default_visit(leq)

    def visit_forall(self, forall: Forall) -> T:
        """Default visit for forall."""
        return self._default_visit(forall)

    def visit_exists(self, exists: Exists) -> T:
        """Default visit for exists."""
        return self._default_visit(exists)

    def _default_visit(self, expr: Expression) -> T:
        """Default visit implementation that returns the expression unchanged."""
        # For many use cases, the default behavior should be to return the expression
        # This allows visitors to focus only on the cases they need to handle
        return expr  # type: ignore


# Concrete expression types
@dataclass(frozen=True)
class Var(Expression):
    """Variable expression."""

    var_id: int
    var_type: Type
    # Optional human-friendly name carried from the originating symbol.
    name: Optional[str] = None

    typ = Type.INT  # Variables are typed

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Var):
            return False
        return self.var_id == other.var_id and self.var_type == other.var_type

    def __hash__(self) -> int:
        return hash((self.var_id, self.var_type))

    def __str__(self) -> str:
        if self.name:
            return self.name
        return f"v{self.var_id}"

    def accept(self, visitor: ExpressionVisitor[T]) -> T:
        return visitor.visit_var(self)


@dataclass(frozen=True)
class Const(Expression):
    """Constant symbol expression."""

    symbol: Symbol

    @property
    def typ(self) -> Type:
        """Constants take the type of their symbol."""
        return self.symbol.typ

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Const):
            return False
        return self.symbol == other.symbol

    def __hash__(self) -> int:
        return hash(self.symbol)

    def get_real(self) -> float:
        """Get the real value of this constant if it represents a real number."""
        if self.symbol.name and self.symbol.name.startswith("real_"):
            try:
                return float(self.symbol.name[5:])  # Remove "real_" prefix
            except ValueError:
                pass
        raise AttributeError(f"Constant {self.symbol} does not represent a real number")

    def __str__(self) -> str:
        return str(self.symbol)

    def accept(self, visitor: ExpressionVisitor[T]) -> T:
        return visitor.visit_const(self)


@dataclass(frozen=True)
class App(Expression):
    """Function application expression."""

    symbol: Symbol
    args: Tuple[Expression, ...]

    typ = Type.INT  # Function applications are typed

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, App):
            return False
        return self.symbol == other.symbol and self.args == other.args

    def __hash__(self) -> int:
        return hash((self.symbol, self.args))

    def __str__(self) -> str:
        if not self.args:
            return str(self.symbol)
        return f"{self.symbol}({', '.join(str(arg) for arg in self.args)})"

    def accept(self, visitor: ExpressionVisitor[T]) -> T:
        return visitor.visit_app(self)


@dataclass(frozen=True)
class Select(Expression):
    """Array select expression."""

    array: Expression
    index: Expression

    typ = Type.INT  # Array elements are integers

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Select):
            return False
        return self.array == other.array and self.index == other.index

    def __hash__(self) -> int:
        return hash((self.array, self.index))

    def __str__(self) -> str:
        return f"{self.array}[{self.index}]"

    def accept(self, visitor: ExpressionVisitor[T]) -> T:
        return visitor.visit_select(self)


@dataclass(frozen=True)
class Store(Expression):
    """Array store expression."""

    array: Expression
    index: Expression
    value: Expression

    typ = Type.ARRAY  # Store returns an array

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Store):
            return False
        return (
            self.array == other.array
            and self.index == other.index
            and self.value == other.value
        )

    def __hash__(self) -> int:
        return hash((self.array, self.index, self.value))

    def __str__(self) -> str:
        return f"{self.array}[{self.index} := {self.value}]"

    def accept(self, visitor: ExpressionVisitor[T]) -> T:
        return visitor.visit_store(self)


@dataclass(frozen=True)
class Add(Expression):
    """Addition expression."""

    args: Tuple[ArithExpression, ...]

    typ = Type.REAL  # Addition promotes to real

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Add):
            return False
        return self.args == other.args

    def __hash__(self) -> int:
        return hash(self.args)

    def __str__(self) -> str:
        return f"({' + '.join(str(arg) for arg in self.args)})"

    def accept(self, visitor: ExpressionVisitor[T]) -> T:
        return visitor.visit_add(self)


@dataclass(frozen=True)
class Mul(Expression):
    """Multiplication expression."""

    args: Tuple[ArithExpression, ...]

    typ = Type.REAL  # Multiplication promotes to real

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mul):
            return False
        return self.args == other.args

    def __hash__(self) -> int:
        return hash(self.args)

    def __str__(self) -> str:
        return f"({' * '.join(str(arg) for arg in self.args)})"

    def accept(self, visitor: ExpressionVisitor[T]) -> T:
        return visitor.visit_mul(self)


@dataclass(frozen=True)
class Ite(Expression):
    """If-then-else expression."""

    condition: FormulaExpression
    then_branch: Expression
    else_branch: Expression

    typ = Type.BOOL  # ITE takes the type of branches

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Ite):
            return False
        return (
            self.condition == other.condition
            and self.then_branch == other.then_branch
            and self.else_branch == other.else_branch
        )

    def __hash__(self) -> int:
        return hash((self.condition, self.then_branch, self.else_branch))

    def __str__(self) -> str:
        return f"({self.condition} ? {self.then_branch} : {self.else_branch})"

    def accept(self, visitor: ExpressionVisitor[T]) -> T:
        return visitor.visit_ite(self)


@dataclass(frozen=True)
class Div(Expression):
    """Division expression (C99 semantics: truncates toward zero for integers)."""

    left: ArithExpression
    right: ArithExpression

    typ = Type.REAL

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Div):
            return False
        return self.left == other.left and self.right == other.right

    def __hash__(self) -> int:
        return hash((self.left, self.right))

    def __str__(self) -> str:
        return f"({self.left} / {self.right})"

    def accept(self, visitor: ExpressionVisitor[T]) -> T:
        return visitor.visit_div(self)


@dataclass(frozen=True)
class Mod(Expression):
    """Modulo expression."""

    left: ArithExpression
    right: ArithExpression

    typ = Type.INT

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mod):
            return False
        return self.left == other.left and self.right == other.right

    def __hash__(self) -> int:
        return hash((self.left, self.right))

    def __str__(self) -> str:
        return f"({self.left} mod {self.right})"

    def accept(self, visitor: ExpressionVisitor[T]) -> T:
        return visitor.visit_mod(self)


@dataclass(frozen=True)
class Floor(Expression):
    """Floor expression (round toward -infinity)."""

    arg: ArithExpression

    typ = Type.INT

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Floor):
            return False
        return self.arg == other.arg

    def __hash__(self) -> int:
        return hash(self.arg)

    def __str__(self) -> str:
        return f"⌊{self.arg}⌋"

    def accept(self, visitor: ExpressionVisitor[T]) -> T:
        return visitor.visit_floor(self)


@dataclass(frozen=True)
class Neg(Expression):
    """Arithmetic negation expression."""

    arg: ArithExpression

    typ = Type.REAL

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Neg):
            return False
        return self.arg == other.arg

    def __hash__(self) -> int:
        return hash(self.arg)

    def __str__(self) -> str:
        return f"(-{self.arg})"

    def accept(self, visitor: ExpressionVisitor[T]) -> T:
        return visitor.visit_neg(self)


@dataclass(frozen=True)
class Pow(Expression):
    """Exponentiation expression: base^exponent."""

    base: ArithExpression
    exponent: int

    typ = Type.REAL

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Pow):
            return False
        return self.base == other.base and self.exponent == other.exponent

    def __hash__(self) -> int:
        return hash((self.base, self.exponent))

    def __str__(self) -> str:
        return f"({self.base}^{self.exponent})"

    def accept(self, visitor: ExpressionVisitor[T]) -> T:
        return visitor.visit_pow(self)


# Boolean expressions (formulas)
@dataclass(frozen=True)
class TrueExpr(Expression):
    """True formula."""

    typ = Type.BOOL

    def __eq__(self, other: object) -> bool:
        return isinstance(other, TrueExpr)

    def __hash__(self) -> int:
        return hash("true")

    def __str__(self) -> str:
        return "true"

    def accept(self, visitor: ExpressionVisitor[T]) -> T:
        return visitor.visit_true(self)


@dataclass(frozen=True)
class FalseExpr(Expression):
    """False formula."""

    typ = Type.BOOL

    def __eq__(self, other: object) -> bool:
        return isinstance(other, FalseExpr)

    def __hash__(self) -> int:
        return hash("false")

    def __str__(self) -> str:
        return "false"

    def accept(self, visitor: ExpressionVisitor[T]) -> T:
        return visitor.visit_false(self)


@dataclass(frozen=True)
class And(Expression):
    """Conjunction formula."""

    args: Tuple[FormulaExpression, ...]

    typ = Type.BOOL

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, And):
            return False
        return self.args == other.args

    def __hash__(self) -> int:
        return hash(self.args)

    def __str__(self) -> str:
        return f"({' ∧ '.join(str(arg) for arg in self.args)})"

    def accept(self, visitor: ExpressionVisitor[T]) -> T:
        return visitor.visit_and(self)


@dataclass(frozen=True)
class Or(Expression):
    """Disjunction formula."""

    args: Tuple[FormulaExpression, ...]

    typ = Type.BOOL

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Or):
            return False
        return self.args == other.args

    def __hash__(self) -> int:
        return hash(self.args)

    def __str__(self) -> str:
        return f"({' ∨ '.join(str(arg) for arg in self.args)})"

    def accept(self, visitor: ExpressionVisitor[T]) -> T:
        return visitor.visit_or(self)


@dataclass(frozen=True)
class Not(Expression):
    """Negation formula."""

    arg: FormulaExpression

    typ = Type.BOOL

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Not):
            return False
        return self.arg == other.arg

    def __hash__(self) -> int:
        return hash(self.arg)

    def __str__(self) -> str:
        return f"¬{self.arg}"

    def accept(self, visitor: ExpressionVisitor[T]) -> T:
        return visitor.visit_not(self)


@dataclass(frozen=True)
class Eq(Expression):
    """Equality formula."""

    left: ArithExpression
    right: ArithExpression

    typ = Type.BOOL

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Eq):
            return False
        return (self.left == other.left and self.right == other.right) or (
            self.left == other.right and self.right == other.left
        )

    def __hash__(self) -> int:
        # Make hash symmetric
        return hash(
            (min(self.left, self.right, key=hash), max(self.left, self.right, key=hash))
        )

    def __str__(self) -> str:
        return f"({self.left} = {self.right})"

    def accept(self, visitor: ExpressionVisitor[T]) -> T:
        return visitor.visit_eq(self)


@dataclass(frozen=True)
class Lt(Expression):
    """Less-than formula."""

    left: ArithExpression
    right: ArithExpression

    typ = Type.BOOL

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Lt):
            return False
        return self.left == other.left and self.right == other.right

    def __hash__(self) -> int:
        return hash((self.left, self.right))

    def __str__(self) -> str:
        return f"({self.left} < {self.right})"

    def accept(self, visitor: ExpressionVisitor[T]) -> T:
        return visitor.visit_lt(self)


@dataclass(frozen=True)
class Leq(Expression):
    """Less-than-or-equal formula."""

    left: ArithExpression
    right: ArithExpression

    typ = Type.BOOL

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Leq):
            return False
        return self.left == other.left and self.right == other.right

    def __hash__(self) -> int:
        return hash((self.left, self.right))

    def __str__(self) -> str:
        return f"({self.left} ≤ {self.right})"

    def accept(self, visitor: ExpressionVisitor[T]) -> T:
        return visitor.visit_leq(self)


@dataclass(frozen=True)
class Forall(Expression):
    """Universal quantification formula."""

    var_name: str
    var_type: Type
    body: FormulaExpression

    typ = Type.BOOL

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Forall):
            return False
        return (
            self.var_name == other.var_name
            and self.var_type == other.var_type
            and self.body == other.body
        )

    def __hash__(self) -> int:
        return hash((self.var_name, self.var_type, self.body))

    def __str__(self) -> str:
        return f"∀{self.var_name}:{self.var_type}. {self.body}"

    def accept(self, visitor: ExpressionVisitor[T]) -> T:
        return visitor.visit_forall(self)


@dataclass(frozen=True)
class Exists(Expression):
    """Existential quantification formula."""

    var_name: str
    var_type: Type
    body: FormulaExpression

    typ = Type.BOOL

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Exists):
            return False
        return (
            self.var_name == other.var_name
            and self.var_type == other.var_type
            and self.body == other.body
        )

    def __hash__(self) -> int:
        return hash((self.var_name, self.var_type, self.body))

    def __str__(self) -> str:
        return f"∃{self.var_name}:{self.var_type}. {self.body}"

    def accept(self, visitor: ExpressionVisitor[T]) -> T:
        return visitor.visit_exists(self)


# Type aliases for cleaner code
ArithExpression = Union[Var, Const, App, Add, Mul, Div, Mod, Floor, Neg, Pow, Ite, Select, Store]
TermExpression = Union[Var, Const, App, Add, Mul, Div, Mod, Floor, Neg, Pow, Ite, Select, Store]
# Alias for backward compatibility
ArithTerm = TermExpression
FormulaExpression = Union[
    TrueExpr, FalseExpr, And, Or, Not, Eq, Lt, Leq, Forall, Exists
]
# Alias for backward compatibility
Formula = FormulaExpression
AnyExpression = Union[Expression, ArithExpression, FormulaExpression, TermExpression]


class ExpressionBuilder:
    """Builder class for creating expressions in a specific context."""

    def __init__(self, context: Context):
        self.context = context

    def mk_symbol(self, name: Optional[str] = None, typ: Type = Type.INT) -> Symbol:
        """Create a symbol."""
        return self.context.mk_symbol(name, typ)

    def mk_var(self, var_id_or_symbol, typ: Type = None) -> Var:
        """Create a variable expression.

        Can be called in two ways:
        - mk_var(var_id: int, typ: Type) - create variable with given ID and type
        - mk_var(symbol: Symbol) - create variable from symbol (uses symbol.id and symbol.typ)
        """
        if isinstance(var_id_or_symbol, Symbol):
            # Called as mk_var(symbol) - use symbol's id and type
            symbol = var_id_or_symbol
            return Var(symbol.id, symbol.typ, symbol.name)
        elif isinstance(var_id_or_symbol, int) and typ is not None:
            # Called as mk_var(var_id, typ) - use given ID and type
            if not isinstance(typ, Type):
                raise TypeError(f"typ must be a Type enum value, got {type(typ)}")
            return Var(var_id_or_symbol, typ)
        else:
            raise TypeError(
                f"mk_var expects (var_id, typ) or (symbol), got ({type(var_id_or_symbol)}, {type(typ)})"
            )

    def mk_const(self, symbol: Symbol) -> Const:
        """Create a constant expression."""
        return Const(symbol)

    def mk_app(self, symbol: Symbol, args: List[Expression]) -> App:
        """Create a function application expression."""
        return App(symbol, tuple(args))

    def mk_select(self, array: Expression, index: Expression) -> Select:
        """Create a select expression."""
        return Select(array, index)

    def mk_store(
        self, array: Expression, index: Expression, value: Expression
    ) -> Store:
        """Create a store expression."""
        return Store(array, index, value)

    def mk_add(self, args: List[ArithExpression]) -> Add:
        """Create an addition expression."""
        return Add(tuple(args))

    def mk_mul(self, args: List[ArithExpression]) -> Mul:
        """Create a multiplication expression."""
        return Mul(tuple(args))

    def mk_ite(
        self,
        condition: FormulaExpression,
        then_branch: Expression,
        else_branch: Expression,
    ) -> Ite:
        """Create an if-then-else expression."""
        return Ite(condition, then_branch, else_branch)

    def mk_true(self) -> TrueExpr:
        """Create a true formula."""
        return TrueExpr()

    def mk_false(self) -> FalseExpr:
        """Create a false formula."""
        return FalseExpr()

    def mk_and(self, args: List[FormulaExpression]) -> And:
        """Create a conjunction formula."""
        return And(tuple(args))

    def mk_or(self, args: List[FormulaExpression]) -> Or:
        """Create a disjunction formula."""
        return Or(tuple(args))

    def mk_not(self, arg: FormulaExpression) -> Not:
        """Create a negation formula."""
        return Not(arg)

    def mk_eq(self, left: ArithExpression, right: ArithExpression) -> Eq:
        """Create an equality formula."""
        return Eq(left, right)

    def mk_lt(self, left: ArithExpression, right: ArithExpression) -> Lt:
        """Create a less-than formula."""
        return Lt(left, right)

    def mk_leq(self, left: ArithExpression, right: ArithExpression) -> Leq:
        """Create a less-than-or-equal formula."""
        return Leq(left, right)

    def mk_geq(self, left: ArithExpression, right: ArithExpression) -> Leq:
        """Create a greater-than-or-equal formula."""
        # GEQ is equivalent to LEQ with arguments swapped
        return Leq(right, left)

    def mk_div(self, left: ArithExpression, right: ArithExpression) -> Div:
        """Create a division expression."""
        return Div(left, right)

    def mk_mod(self, left: ArithExpression, right: ArithExpression) -> Mod:
        """Create a modulo expression."""
        return Mod(left, right)

    def mk_floor(self, arg: ArithExpression) -> Floor:
        """Create a floor expression."""
        return Floor(arg)

    def mk_neg(self, arg: ArithExpression) -> Neg:
        """Create an arithmetic negation expression."""
        return Neg(arg)

    def mk_pow(self, base: ArithExpression, exponent: int) -> Pow:
        """Create an exponentiation expression: base^exponent."""
        return Pow(base, exponent)

    def mk_idiv(self, left: ArithExpression, right: ArithExpression) -> App:
        """Create C99 integer division (truncate toward zero)."""
        idiv_sym = self.mk_symbol("idiv", Type.FUN([Type.REAL, Type.REAL], Type.INT))
        return App(idiv_sym, (left, right))

    def mk_ceiling(self, arg: ArithExpression) -> App:
        """Create a ceiling expression (round toward +infinity)."""
        ceil_sym = self.mk_symbol("ceiling", Type.FUN([Type.REAL], Type.INT))
        return App(ceil_sym, (arg,))

    def mk_truncate(self, arg: ArithExpression) -> App:
        """Create a truncation toward zero expression."""
        trunc_sym = self.mk_symbol("truncate", Type.FUN([Type.REAL], Type.INT))
        return App(trunc_sym, (arg,))

    def mk_arr_eq(self, left: Expression, right: Expression) -> Eq:
        """Create an array equality formula."""
        return Eq(left, right)

    def mk_compare(
        self, op: str, left: ArithExpression, right: ArithExpression
    ) -> FormulaExpression:
        """Generic comparison factory (dispatches to mk_eq/mk_lt/mk_leq)."""
        if op == "Eq":
            return self.mk_eq(left, right)
        elif op in ("Lt", "Leq"):
            return self.mk_leq(left, right) if op == "Leq" else self.mk_lt(left, right)
        else:
            raise ValueError(f"Unknown comparison operator: {op}")

    def mk_int(self, value: int) -> Const:
        """Create an integer constant expression."""
        int_symbol = self.mk_symbol(f"int_{value}", Type.INT)
        return Const(int_symbol)

    def mk_zz(self, value: int) -> Const:
        """Create an arbitrary-precision integer constant (same as mk_int)."""
        return self.mk_int(value)

    def mk_if(self, cond: FormulaExpression, then_expr: FormulaExpression) -> Or:
        """Create an implication (cond => then_expr), desugared as (!cond or then_expr)."""
        return Or((then_expr, Not(cond)))

    def mk_iff(self, left: FormulaExpression, right: FormulaExpression) -> And:
        """Create if-and-only-if (left <=> right)."""
        return And(
            (
                Or((right, Not(left))),
                Or((left, Not(right))),
            )
        )

    def mk_forall_const(
        self, symbol: Symbol, body: FormulaExpression
    ) -> Forall:
        """Replace constant symbol with universally quantified variable."""
        name = symbol.name or f"s{symbol.id}"
        return Forall(name, symbol.typ, body)

    def mk_exists_const(
        self, symbol: Symbol, body: FormulaExpression
    ) -> Exists:
        """Replace constant symbol with existentially quantified variable."""
        name = symbol.name or f"s{symbol.id}"
        return Exists(name, symbol.typ, body)

    def mk_real(self, value: float) -> Const:
        """Create a real constant."""
        real_symbol = self.mk_symbol(f"real_{value}", Type.REAL)
        return self.mk_const(real_symbol)

    def mk_forall(
        self, var_name: str, var_type: Type, body: FormulaExpression
    ) -> Forall:
        """Create a universal quantification formula."""
        return Forall(var_name, var_type, body)

    def mk_exists(
        self, var_name: str, var_type: Type, body: FormulaExpression
    ) -> Exists:
        """Create an existential quantification formula."""
        return Exists(var_name, var_type, body)


# Convenience functions for creating contexts and expressions
def make_context() -> Context:
    """Create a new context."""
    return Context()


def make_expression_builder(context: Context) -> ExpressionBuilder:
    """Create an expression builder for a context."""
    return ExpressionBuilder(context)


# Default context and builder for convenience
_default_context = make_context()
_default_builder = make_expression_builder(_default_context)


def mk_symbol(*args) -> Symbol:
    """Create a symbol.

    Supports two forms:
    - mk_symbol(name, typ) -> uses default context
    - mk_symbol(context, name, typ) -> uses explicit context
    """
    if len(args) == 2 and not isinstance(args[0], Context):
        name, typ = args
        return _default_context.mk_symbol(name, typ)
    elif len(args) == 3 and isinstance(args[0], Context):
        context, name, typ = args
        return context.mk_symbol(name, typ)
    else:
        raise TypeError(
            f"mk_symbol expects (name, typ) or (context, name, typ), got {len(args)} args"
        )


def mk_var(*args) -> Var:
    """Create a variable expression.

    Can be called in multiple ways:
    - mk_var(var_id, typ) - use default context
    - mk_var(context, var_id, typ) - use specific context
    - mk_var(symbol) - create variable from symbol (uses symbol.id as var_id)
    - mk_var(context, symbol) - create variable from symbol with specific context

    Args:
        *args: Either (var_id, typ), (context, var_id, typ), (symbol), or (context, symbol)

    Returns:
        A Var expression
    """
    if len(args) == 1:
        # Called as mk_var(symbol) or mk_var(context)
        arg = args[0]
        if isinstance(arg, Context):
            # This shouldn't happen in normal usage, but handle gracefully
            raise TypeError("mk_var with single Context argument not supported")
        elif isinstance(arg, Symbol):
            # Called as mk_var(symbol) - use default context and symbol's type
            symbol = arg
            return _default_builder.mk_var(symbol)
        else:
            raise TypeError(
                f"mk_var expects Symbol or Context as single argument, got {type(arg)}"
            )
    elif len(args) == 2:
        # Could be mk_var(var_id, typ) or mk_var(context, symbol)
        first, second = args
        if isinstance(first, Context):
            # Called as mk_var(context, symbol)
            context, symbol = first, second
            if isinstance(symbol, Symbol):
                return context.mk_var(symbol)
            else:
                raise TypeError(
                    f"Second argument must be Symbol when first is Context, got {type(symbol)}"
                )
        else:
            # Called as mk_var(var_id, typ) - use default context
            var_id, typ = first, second
            return _default_builder.mk_var(var_id, typ)
    elif len(args) == 3 and isinstance(args[0], Context):
        # Called as mk_var(context, var_id, typ) - use specific context
        context, var_id, typ = args
        return context.mk_var(var_id, typ)
    else:
        raise TypeError(f"mk_var expects 1, 2, or 3 arguments, got {len(args)}")


def mk_const(*args) -> Const:
    """Create a constant expression.

    Supports:
    - mk_const(symbol) -> uses default context
    - mk_const(context, symbol) -> uses explicit context
    """
    if len(args) == 1 and isinstance(args[0], Symbol):
        (symbol,) = args
        return _default_builder.mk_const(symbol)
    elif (
        len(args) == 2 and isinstance(args[0], Context) and isinstance(args[1], Symbol)
    ):
        context, symbol = args
        builder = make_expression_builder(context)
        return builder.mk_const(symbol)
    else:
        raise TypeError(
            f"mk_const expects (symbol) or (context, symbol), got {len(args)} args"
        )


def mk_add(*args) -> Add:
    """Create an addition expression.

    Supports:
    - mk_add(args) -> uses default context
    - mk_add(context, args) -> uses explicit context
    """
    if len(args) == 1:
        (terms,) = args
        return _default_builder.mk_add(terms)
    elif len(args) == 2 and isinstance(args[0], Context):
        context, terms = args
        builder = make_expression_builder(context)
        return builder.mk_add(terms)
    else:
        raise TypeError(
            f"mk_add expects (args) or (context, args), got {len(args)} args"
        )


def mk_mul(*args) -> Mul:
    """Create a multiplication expression.

    Supports:
    - mk_mul(args) -> uses default context
    - mk_mul(context, args) -> uses explicit context
    """
    if len(args) == 1:
        (terms,) = args
        return _default_builder.mk_mul(terms)
    elif len(args) == 2 and isinstance(args[0], Context):
        context, terms = args
        builder = make_expression_builder(context)
        return builder.mk_mul(terms)
    else:
        raise TypeError(
            f"mk_mul expects (args) or (context, args), got {len(args)} args"
        )


def mk_eq(*args) -> Eq:
    """Create an equality formula.

    Supports:
    - mk_eq(left, right)
    - mk_eq(context, left, right)
    """
    if len(args) == 2:
        left, right = args
        return _default_builder.mk_eq(left, right)
    elif len(args) == 3 and isinstance(args[0], Context):
        context, left, right = args
        builder = make_expression_builder(context)
        return builder.mk_eq(left, right)
    else:
        raise TypeError(
            f"mk_eq expects (left, right) or (context, left, right), got {len(args)} args"
        )


def mk_lt(*args) -> Lt:
    """Create a less-than formula.

    Supports:
    - mk_lt(left, right)
    - mk_lt(context, left, right)
    """
    if len(args) == 2:
        left, right = args
        return _default_builder.mk_lt(left, right)
    elif len(args) == 3 and isinstance(args[0], Context):
        context, left, right = args
        builder = make_expression_builder(context)
        return builder.mk_lt(left, right)
    else:
        raise TypeError(
            f"mk_lt expects (left, right) or (context, left, right), got {len(args)} args"
        )


def mk_leq(*args) -> Leq:
    """Create a less-than-or-equal formula.

    Supports:
    - mk_leq(left, right)
    - mk_leq(context, left, right)
    """
    if len(args) == 2:
        left, right = args
        return _default_builder.mk_leq(left, right)
    elif len(args) == 3 and isinstance(args[0], Context):
        context, left, right = args
        builder = make_expression_builder(context)
        return builder.mk_leq(left, right)
    else:
        raise TypeError(
            f"mk_leq expects (left, right) or (context, left, right), got {len(args)} args"
        )


def mk_geq(*args) -> Leq:
    """Create a greater-than-or-equal formula.

    Supports:
    - mk_geq(left, right)
    - mk_geq(context, left, right)
    """
    if len(args) == 2:
        left, right = args
        return _default_builder.mk_geq(left, right)
    elif len(args) == 3 and isinstance(args[0], Context):
        context, left, right = args
        builder = make_expression_builder(context)
        return builder.mk_geq(left, right)
    else:
        raise TypeError(
            f"mk_geq expects (left, right) or (context, left, right), got {len(args)} args"
        )


def mk_div(*args) -> Div:
    """Create a division expression.

    Supports:
    - mk_div(left, right)
    - mk_div(context, left, right)
    """
    if len(args) == 2:
        left, right = args
        return _default_builder.mk_div(left, right)
    elif len(args) == 3 and isinstance(args[0], Context):
        context, left, right = args
        builder = make_expression_builder(context)
        return builder.mk_div(left, right)
    else:
        raise TypeError(
            f"mk_div expects (left, right) or (context, left, right), got {len(args)} args"
        )


def mk_mod(*args) -> Mod:
    """Create a modulo expression.

    Supports:
    - mk_mod(left, right)
    - mk_mod(context, left, right)
    """
    if len(args) == 2:
        left, right = args
        return _default_builder.mk_mod(left, right)
    elif len(args) == 3 and isinstance(args[0], Context):
        context, left, right = args
        builder = make_expression_builder(context)
        return builder.mk_mod(left, right)
    else:
        raise TypeError(
            f"mk_mod expects (left, right) or (context, left, right), got {len(args)} args"
        )


def mk_floor(*args) -> Floor:
    """Create a floor expression.

    Supports:
    - mk_floor(arg)
    - mk_floor(context, arg)
    """
    if len(args) == 1:
        (arg,) = args
        return _default_builder.mk_floor(arg)
    elif len(args) == 2 and isinstance(args[0], Context):
        context, arg = args
        builder = make_expression_builder(context)
        return builder.mk_floor(arg)
    else:
        raise TypeError(
            f"mk_floor expects (arg) or (context, arg), got {len(args)} args"
        )


def mk_neg(*args) -> Neg:
    """Create an arithmetic negation expression.

    Supports:
    - mk_neg(arg)
    - mk_neg(context, arg)
    """
    if len(args) == 1:
        (arg,) = args
        return _default_builder.mk_neg(arg)
    elif len(args) == 2 and isinstance(args[0], Context):
        context, arg = args
        builder = make_expression_builder(context)
        return builder.mk_neg(arg)
    else:
        raise TypeError(
            f"mk_neg expects (arg) or (context, arg), got {len(args)} args"
        )


def mk_real(*args) -> Const:
    """Create a real constant.

    Supports two forms:
    - mk_real(value) -> uses default context
    - mk_real(context, value) -> uses explicit context
    """
    if len(args) == 1:
        (value,) = args
        # Accept Fraction/int/float, convert QQ helpers
        try:
            # Handle QQ.one()/QQ.zero() by numeric conversion
            from .qQ import QQ as _QQ

            if isinstance(value, _QQ):
                val = float(value)
            else:
                val = float(value)
        except Exception:
            val = value
        return _default_builder.mk_real(val)
    elif len(args) == 2 and isinstance(args[0], Context):
        context, value = args
        builder = make_expression_builder(context)
        try:
            from .qQ import QQ as _QQ

            if isinstance(value, _QQ):
                val = float(value)
            else:
                val = float(value)
        except Exception:
            val = value
        return builder.mk_real(val)
    else:
        raise TypeError(
            f"mk_real expects (value) or (context, value), got {len(args)} args"
        )


def mk_idiv(*args) -> App:
    """Create C99 integer division (truncate toward zero)."""
    if len(args) == 2:
        left, right = args
        return _default_builder.mk_idiv(left, right)
    elif len(args) == 3 and isinstance(args[0], Context):
        context, left, right = args
        builder = make_expression_builder(context)
        return builder.mk_idiv(left, right)
    else:
        raise TypeError(
            f"mk_idiv expects (left, right) or (context, left, right), got {len(args)} args"
        )


def mk_ceiling(*args) -> App:
    """Create a ceiling expression (round toward +infinity)."""
    if len(args) == 1:
        (arg,) = args
        return _default_builder.mk_ceiling(arg)
    elif len(args) == 2 and isinstance(args[0], Context):
        context, arg = args
        builder = make_expression_builder(context)
        return builder.mk_ceiling(arg)
    else:
        raise TypeError(
            f"mk_ceiling expects (arg) or (context, arg), got {len(args)} args"
        )


def mk_truncate(*args) -> App:
    """Create a truncation toward zero expression."""
    if len(args) == 1:
        (arg,) = args
        return _default_builder.mk_truncate(arg)
    elif len(args) == 2 and isinstance(args[0], Context):
        context, arg = args
        builder = make_expression_builder(context)
        return builder.mk_truncate(arg)
    else:
        raise TypeError(
            f"mk_truncate expects (arg) or (context, arg), got {len(args)} args"
        )


def mk_arr_eq(*args) -> Eq:
    """Create an array equality formula."""
    if len(args) == 2:
        left, right = args
        return _default_builder.mk_arr_eq(left, right)
    elif len(args) == 3 and isinstance(args[0], Context):
        context, left, right = args
        builder = make_expression_builder(context)
        return builder.mk_arr_eq(left, right)
    else:
        raise TypeError(
            f"mk_arr_eq expects (left, right) or (context, left, right), got {len(args)} args"
        )


def mk_compare(*args) -> FormulaExpression:
    """Generic comparison factory (dispatches to mk_eq/mk_lt/mk_leq)."""
    if len(args) == 3:
        op, left, right = args
        return _default_builder.mk_compare(op, left, right)
    elif len(args) == 4 and isinstance(args[0], Context):
        context, op, left, right = args
        builder = make_expression_builder(context)
        return builder.mk_compare(op, left, right)
    else:
        raise TypeError(
            f"mk_compare expects (op, left, right) or (context, op, left, right), got {len(args)} args"
        )


def mk_int(*args) -> Const:
    """Create an integer constant expression."""
    if len(args) == 1:
        (value,) = args
        return _default_builder.mk_int(value)
    elif len(args) == 2 and isinstance(args[0], Context):
        context, value = args
        builder = make_expression_builder(context)
        return builder.mk_int(value)
    else:
        raise TypeError(
            f"mk_int expects (value) or (context, value), got {len(args)} args"
        )


def mk_if(*args) -> Or:
    """Create an implication formula (cond => then_expr)."""
    if len(args) == 2:
        cond, then_expr = args
        return _default_builder.mk_if(cond, then_expr)
    elif len(args) == 3 and isinstance(args[0], Context):
        context, cond, then_expr = args
        builder = make_expression_builder(context)
        return builder.mk_if(cond, then_expr)
    else:
        raise TypeError(
            f"mk_if expects (cond, then) or (context, cond, then), got {len(args)} args"
        )


def mk_iff(*args) -> And:
    """Create if-and-only-if formula."""
    if len(args) == 2:
        left, right = args
        return _default_builder.mk_iff(left, right)
    elif len(args) == 3 and isinstance(args[0], Context):
        context, left, right = args
        builder = make_expression_builder(context)
        return builder.mk_iff(left, right)
    else:
        raise TypeError(
            f"mk_iff expects (left, right) or (context, left, right), got {len(args)} args"
        )


def mk_forall_const(*args) -> Forall:
    """Replace constant symbol with universally quantified variable."""
    if len(args) == 2:
        symbol, body = args
        return _default_builder.mk_forall_const(symbol, body)
    elif len(args) == 3 and isinstance(args[0], Context):
        context, symbol, body = args
        builder = make_expression_builder(context)
        return builder.mk_forall_const(symbol, body)
    else:
        raise TypeError(
            f"mk_forall_const expects (symbol, body) or (context, symbol, body), got {len(args)} args"
        )


def mk_exists_const(*args) -> Exists:
    """Replace constant symbol with existentially quantified variable."""
    if len(args) == 2:
        symbol, body = args
        return _default_builder.mk_exists_const(symbol, body)
    elif len(args) == 3 and isinstance(args[0], Context):
        context, symbol, body = args
        builder = make_expression_builder(context)
        return builder.mk_exists_const(symbol, body)
    else:
        raise TypeError(
            f"mk_exists_const expects (symbol, body) or (context, symbol, body), got {len(args)} args"
        )


def mk_forall_consts(
    srk: Context, pred: Callable[[Symbol], bool], body: FormulaExpression
) -> FormulaExpression:
    """Universally quantify all constant symbols matching a predicate."""
    from .srkSimplify import purify

    purify_body = purify(srk, body)
    const_syms = [
        s for s in symbols(purify_body) if pred(s) and s.typ != Type.BOOL
    ]
    result = body
    for s in const_syms:
        result = mk_forall_const(srk, s, result)
    return result


def mk_exists_consts(
    srk: Context, pred: Callable[[Symbol], bool], body: FormulaExpression
) -> FormulaExpression:
    """Existentially quantify all constant symbols matching a predicate."""
    from .srkSimplify import purify

    purify_body = purify(srk, body)
    const_syms = [
        s for s in symbols(purify_body) if pred(s) and s.typ != Type.BOOL
    ]
    result = body
    for s in const_syms:
        result = mk_exists_const(srk, s, result)
    return result


def of_linterm(srk: Context, term: Any) -> ArithExpression:
    """Convert a linear term (QQVector) into a syntax expression.

    This is a small compatibility shim for quantifier-elimination code which
    builds/consumes linear terms as sparse vectors.
    """
    from fractions import Fraction
    from .linear import const_dim as _const_dim

    # Accept both QQVector-like objects and plain dicts.
    entries = getattr(term, "entries", term)
    if not isinstance(entries, dict):
        raise TypeError(f"of_linterm expected QQVector or dict, got {type(term)}")

    parts: List[ArithExpression] = []

    const_coeff = entries.get(_const_dim, Fraction(0))
    try:
        const_coeff = Fraction(const_coeff)
    except Exception:
        const_coeff = Fraction(float(const_coeff))
    if const_coeff != 0:
        parts.append(mk_real(srk, const_coeff))

    # Remaining dimensions encode variables; dimension 0 is reserved for the constant.
    for dim, coeff in sorted(entries.items(), key=lambda kv: kv[0]):
        if dim == _const_dim:
            continue

        try:
            coeff = Fraction(coeff)
        except Exception:
            coeff = Fraction(float(coeff))

        # With const_dim = -1, dimension IS the var_id directly.
        var_id = int(dim)

        # Recover type/name from the context when possible.
        var_type = Type.REAL
        var_name: Optional[str] = None
        try:
            sym = getattr(srk, "_symbols", {}).get(var_id)
            if sym is not None:
                var_type = sym.typ
                var_name = sym.name
        except Exception:
            pass

        var = Var(var_id, var_type, var_name)

        if coeff == 1:
            parts.append(var)
        elif coeff == -1:
            parts.append(mk_mul(srk, [mk_real(srk, Fraction(-1)), var]))
        else:
            parts.append(mk_mul(srk, [mk_real(srk, coeff), var]))

    if not parts:
        return mk_real(srk, Fraction(0))
    if len(parts) == 1:
        return parts[0]
    return mk_add(srk, parts)


def mk_and(*args) -> And:
    """Create a conjunction formula.

    Supports:
    - mk_and(args) -> uses default context
    - mk_and(context, args) -> uses explicit context
    """
    if len(args) == 1:
        (conjuncts,) = args
        return _default_builder.mk_and(conjuncts)
    elif len(args) == 2 and isinstance(args[0], Context):
        context, conjuncts = args
        builder = make_expression_builder(context)
        return builder.mk_and(conjuncts)
    else:
        raise TypeError(
            f"mk_and expects (args) or (context, args), got {len(args)} args"
        )


def mk_or(*args) -> Or:
    """Create a disjunction formula.

    Supports:
    - mk_or(args) -> uses default context
    - mk_or(context, args) -> uses explicit context
    """
    if len(args) == 1:
        (disjuncts,) = args
        return _default_builder.mk_or(disjuncts)
    elif len(args) == 2 and isinstance(args[0], Context):
        context, disjuncts = args
        builder = make_expression_builder(context)
        return builder.mk_or(disjuncts)
    else:
        raise TypeError(
            f"mk_or expects (args) or (context, args), got {len(args)} args"
        )


def mk_true(*args) -> TrueExpr:
    """Create a true formula.

    Supports:
    - mk_true() -> uses default context
    - mk_true(context) -> uses explicit context
    """
    if len(args) == 0:
        return _default_builder.mk_true()
    elif len(args) == 1 and isinstance(args[0], Context):
        context = args[0]
        builder = make_expression_builder(context)
        return builder.mk_true()
    else:
        raise TypeError(f"mk_true expects () or (context), got {len(args)} args")


def mk_false(*args) -> FalseExpr:
    """Create a false formula.

    Supports:
    - mk_false() -> uses default context
    - mk_false(context) -> uses explicit context
    """
    if len(args) == 0:
        return _default_builder.mk_false()
    elif len(args) == 1 and isinstance(args[0], Context):
        context = args[0]
        builder = make_expression_builder(context)
        return builder.mk_false()
    else:
        raise TypeError(f"mk_false expects () or (context), got {len(args)} args")


def mk_exists(*args) -> Exists:
    """Create an Exists expression.

    Supports:
    - mk_exists(var_name, var_type, body)
    - mk_exists(context, var_name, var_type, body)
    """
    if len(args) == 3:
        var_name, var_type, body = args
        return _default_builder.mk_exists(var_name, var_type, body)
    if len(args) == 4 and isinstance(args[0], Context):
        _, var_name, var_type, body = args
        return _default_builder.mk_exists(var_name, var_type, body)
    raise TypeError(
        "mk_exists expects (var_name, var_type, body) or "
        "(context, var_name, var_type, body)"
    )


def mk_forall(*args) -> Forall:
    """Create a Forall expression.

    Supports:
    - mk_forall(var_name, var_type, body)
    - mk_forall(context, var_name, var_type, body)
    """
    if len(args) == 3:
        var_name, var_type, body = args
        return _default_builder.mk_forall(var_name, var_type, body)
    if len(args) == 4 and isinstance(args[0], Context):
        _, var_name, var_type, body = args
        return _default_builder.mk_forall(var_name, var_type, body)
    raise TypeError(
        "mk_forall expects (var_name, var_type, body) or "
        "(context, var_name, var_type, body)"
    )


def mk_ite(
    condition: FormulaExpression, then_branch: Expression, else_branch: Expression
) -> Ite:
    """Create an if-then-else expression."""
    return _default_builder.mk_ite(condition, then_branch, else_branch)


def mk_not(*args) -> Not:
    """Create a negation expression.

    Supports:
    - mk_not(arg)
    - mk_not(context, arg)
    """
    if len(args) == 1:
        (arg,) = args
        return _default_builder.mk_not(arg)
    elif len(args) == 2 and isinstance(args[0], Context):
        context, arg = args
        builder = make_expression_builder(context)
        return builder.mk_not(arg)
    else:
        raise TypeError(f"mk_not expects (arg) or (context, arg), got {len(args)} args")


def mk_iff(*args) -> FormulaExpression:
    """Create Boolean equivalence.

    Supports:
    - mk_iff(left, right)
    - mk_iff(context, left, right)
    """
    if len(args) == 2:
        left, right = args
        return mk_and([mk_or([mk_not(left), right]), mk_or([mk_not(right), left])])
    if len(args) == 3 and isinstance(args[0], Context):
        context, left, right = args
        return mk_and(
            context,
            [
                mk_or(context, [mk_not(context, left), right]),
                mk_or(context, [mk_not(context, right), left]),
            ],
        )
    raise TypeError(
        f"mk_iff expects (left, right) or (context, left, right), got {len(args)} args"
    )


def mk_app(
    context_or_symbol: Union[Context, Symbol],
    symbol_or_args: Union[Symbol, List[Expression]],
    args: List[Expression] = None,
) -> App:
    """Create an application expression.

    Can be called in two ways:
    - mk_app(symbol, args) - use default context
    - mk_app(context, symbol, args) - use specific context

    Args:
        context_or_symbol: Either a Context or a Symbol
        symbol_or_args: Either a Symbol (if first arg is Context) or List[Expression]
        args: List of arguments (if first arg is Context)

    Returns:
        An App expression
    """
    if isinstance(context_or_symbol, Context) and args is not None:
        # Called as mk_app(context, symbol, args)
        context, symbol = context_or_symbol, symbol_or_args
        builder = make_expression_builder(context)
        return builder.mk_app(symbol, args)
    else:
        # Called as mk_app(symbol, args) - use default context
        symbol, args = context_or_symbol, symbol_or_args
        return _default_builder.mk_app(symbol, args)


def mk_pow(*args) -> Pow:
    """Create an exponentiation expression.

    Supports:
    - mk_pow(base, exponent)
    - mk_pow(context, base, exponent)
    """
    if len(args) == 2:
        base, exponent = args
        return _default_builder.mk_pow(base, exponent)
    elif len(args) == 3 and isinstance(args[0], Context):
        context, base, exponent = args
        builder = make_expression_builder(context)
        return builder.mk_pow(base, exponent)
    else:
        raise TypeError(
            f"mk_pow expects (base, exponent) or (context, base, exponent), got {len(args)} args"
        )


def mk_select(
    context_or_array: Union[Context, Expression],
    array_or_index: Union[Expression, Expression],
    index: Expression = None,
) -> Select:
    """Create a select expression.

    Can be called in two ways:
    - mk_select(array, index) - use default context
    - mk_select(context, array, index) - use specific context

    Args:
        context_or_array: Either a Context or an Expression (array)
        array_or_index: Either an Expression (array) or Expression (index)
        index: Expression (index) if first arg is Context

    Returns:
        A Select expression
    """
    if isinstance(context_or_array, Context) and index is not None:
        # Called as mk_select(context, array, index)
        context, array = context_or_array, array_or_index
        return context.mk_select(array, index)
    else:
        # Called as mk_select(array, index) - use default context
        array, index = context_or_array, array_or_index
        return _default_builder.mk_select(array, index)


def mk_store(
    context_or_array: Union[Context, Expression],
    array_or_index: Union[Expression, Expression],
    index_or_value: Union[Expression, Expression] = None,
    value: Expression = None,
) -> Store:
    """Create a store expression.

    Can be called in two ways:
    - mk_store(array, index, value) - use default context
    - mk_store(context, array, index, value) - use specific context

    Args:
        context_or_array: Either a Context or an Expression (array)
        array_or_index: Either an Expression (array) or Expression (index)
        index_or_value: Either an Expression (index) or Expression (value)
        value: Expression (value) if first arg is Context

    Returns:
        A Store expression
    """
    if isinstance(context_or_array, Context) and value is not None:
        # Called as mk_store(context, array, index, value)
        context, array, index = context_or_array, array_or_index, index_or_value
        return context.mk_store(array, index, value)
    else:
        # Called as mk_store(array, index, value) - use default context
        array, index, value = context_or_array, array_or_index, index_or_value
        return _default_builder.mk_store(array, index, value)


def mk_int(context_or_value: Union[Context, int], value: int = None) -> Const:
    """Create an integer constant.

    Can be called in two ways:
    - mk_int(value) - use default context
    - mk_int(context, value) - use specific context

    Args:
        context_or_value: Either a Context or an int value
        value: int value if first arg is Context

    Returns:
        A Const expression
    """
    if isinstance(context_or_value, Context) and value is not None:
        # Called as mk_int(context, value)
        context, value = context_or_value, value
        return context.mk_const(context.mk_symbol(str(value), Type.INT))
    else:
        # Called as mk_int(value) - use default context
        value = context_or_value
        return _default_builder.mk_const(
            _default_builder.mk_symbol(str(value), Type.INT)
        )


def mk_zero(context: Optional[Context] = None, typ: Type = Type.INT) -> Const:
    """Create the zero constant for the requested type."""
    if typ == Type.REAL:
        return mk_real(context, 0) if context is not None else mk_real(0)
    if typ == Type.INT:
        return mk_int(context, 0) if context is not None else mk_int(0)
    raise TypeError(f"mk_zero only supports INT and REAL, got {typ}")


def mk_one(context: Optional[Context] = None, typ: Type = Type.INT) -> Const:
    """Create the one constant for the requested type."""
    if typ == Type.REAL:
        return mk_real(context, 1) if context is not None else mk_real(1)
    if typ == Type.INT:
        return mk_int(context, 1) if context is not None else mk_int(1)
    raise TypeError(f"mk_one only supports INT and REAL, got {typ}")


def mk_sub(
    context_or_left: Union[Context, ArithExpression],
    left_or_right: Union[ArithExpression, ArithExpression],
    right: ArithExpression = None,
) -> ArithExpression:
    """Create a subtraction expression.

    Can be called in two ways:
    - mk_sub(left, right) - use default context
    - mk_sub(context, left, right) - use specific context

    Args:
        context_or_left: Either a Context or an ArithExpression (left operand)
        left_or_right: Either an ArithExpression (left) or ArithExpression (right)
        right: ArithExpression (right) if first arg is Context

    Returns:
        An ArithExpression representing left - right
    """
    if isinstance(context_or_left, Context) and right is not None:
        # Called as mk_sub(context, left, right)
        context, left = context_or_left, left_or_right
        return context.mk_add([left, context.mk_neg(right)])
    else:
        # Called as mk_sub(left, right) - use default context
        left, right = context_or_left, left_or_right
        return _default_builder.mk_add([left, _default_builder.mk_neg(right)])


def mk_if(
    context_or_condition: Union[Context, FormulaExpression],
    condition_or_then: Union[FormulaExpression, Expression],
    then_branch: Expression = None,
    else_branch: Expression = None,
) -> Expression:
    """Create an if-then-else expression.

    Can be called in two ways:
    - mk_if(condition, then_branch, else_branch) - use default context
    - mk_if(context, condition, then_branch, else_branch) - use specific context

    Args:
        context_or_condition: Either a Context or a FormulaExpression (condition)
        condition_or_then: Either a FormulaExpression (condition) or Expression (then_branch)
        then_branch: Expression for then branch if first arg is Context
        else_branch: Expression for else branch if first arg is Context

    Returns:
        An Expression representing if condition then then_branch else else_branch
    """
    if (
        isinstance(context_or_condition, Context)
        and then_branch is not None
        and else_branch is not None
    ):
        # Called as mk_if(context, condition, then_branch, else_branch)
        context, condition = context_or_condition, condition_or_then
        return context.mk_ite(condition, then_branch, else_branch)
    else:
        # Called as mk_if(condition, then_branch, else_branch) - use default context
        condition, then_branch, else_branch = (
            context_or_condition,
            condition_or_then,
            then_branch,
        )
        return _default_builder.mk_ite(condition, then_branch, else_branch)


# Type aliases
Term = TermExpression


# Utility functions that need to be implemented
def substitute(
    expr: Expression,
    subst_map: Union[Dict[Symbol, Expression], Callable[[int, Type], Expression]],
) -> Expression:
    """Substitute in an expression.

    Two modes:
    1. **Constant substitution** (dict): replaces each `Const(s)` with the
       mapped expression. Symbols are tracked by identity.
    2. **De Bruijn substitution** (callable (int, Type) -> expr): replaces
       each `Var(i, ty)` by calling the function with (i, ty). De Bruijn
       indices are shifted when going under a quantifier (indices increase
       by 1 for each additional enclosing binder).

    De Bruijn semantics:
    - Inside `Forall(n, t, body)`, Var(0, t) refers to the bound variable n
    - Inside the body, free Var indices are shifted by +1 compared to outside
    - `substitute(body, fn)` handles this by incrementing depth under quantifiers
    """
    if callable(subst_map):
        return _substitute_de_bruijn(expr, subst_map, 0)
    if isinstance(subst_map, dict):
        return _substitute_const(expr, subst_map)
    raise TypeError(f"substitute expects dict or callable, got {type(subst_map)}")


def shift_db_indices(expr: Expression, delta: int) -> Expression:
    """Shift all De Bruijn Var indices in an expression by delta.

    Positive delta pushes variables deeper (adding more enclosing
    quantifiers). Negative delta brings them out.

    This is used when embedding expressions inside quantifiers.
    """
    if delta == 0:
        return expr
    return _shift_db(expr, delta)


def _shift_db(expr: Expression, delta: int) -> Expression:
    if isinstance(expr, Var):
        return Var(expr.var_id + delta, expr.var_type, expr.name)
    if isinstance(expr, (Const, TrueExpr, FalseExpr)):
        return expr
    if isinstance(expr, Add):
        return Add(tuple(_shift_db(a, delta) for a in expr.args))
    if isinstance(expr, Mul):
        return Mul(tuple(_shift_db(a, delta) for a in expr.args))
    if isinstance(expr, Div):
        return Div(_shift_db(expr.left, delta), _shift_db(expr.right, delta))
    if isinstance(expr, Mod):
        return Mod(_shift_db(expr.left, delta), _shift_db(expr.right, delta))
    if isinstance(expr, Floor):
        return Floor(_shift_db(expr.arg, delta))
    if isinstance(expr, Neg):
        return Neg(_shift_db(expr.arg, delta))
    if isinstance(expr, Pow):
        return Pow(_shift_db(expr.base, delta), expr.exponent)
    if isinstance(expr, Ite):
        return Ite(
            _shift_db(expr.condition, delta),
            _shift_db(expr.then_branch, delta),
            _shift_db(expr.else_branch, delta),
        )
    if isinstance(expr, App):
        return App(expr.symbol, tuple(_shift_db(a, delta) for a in expr.args))
    if isinstance(expr, Select):
        return Select(_shift_db(expr.array, delta), _shift_db(expr.index, delta))
    if isinstance(expr, Store):
        return Store(
            _shift_db(expr.array, delta),
            _shift_db(expr.index, delta),
            _shift_db(expr.value, delta),
        )
    if isinstance(expr, And):
        return And(tuple(_shift_db(a, delta) for a in expr.args))
    if isinstance(expr, Or):
        return Or(tuple(_shift_db(a, delta) for a in expr.args))
    if isinstance(expr, Not):
        return Not(_shift_db(expr.arg, delta))
    if isinstance(expr, Eq):
        return Eq(_shift_db(expr.left, delta), _shift_db(expr.right, delta))
    if isinstance(expr, Lt):
        return Lt(_shift_db(expr.left, delta), _shift_db(expr.right, delta))
    if isinstance(expr, Leq):
        return Leq(_shift_db(expr.left, delta), _shift_db(expr.right, delta))
    if isinstance(expr, Forall):
        # The bound variable itself (index 0 in body) stays unchanged.
        # Only apply delta to indices that refer to outer binders
        # (which are at positions >= 1 in the body relative to the outer scope).
        # But since Forall already shifts body indices by +1 internally,
        # we only shift the body by delta here.
        new_body = _shift_db(expr.body, delta)
        return Forall(expr.var_name, expr.var_type, new_body)
    if isinstance(expr, Exists):
        new_body = _shift_db(expr.body, delta)
        return Exists(expr.var_name, expr.var_type, new_body)
    return expr


def open_forall(srk: Context, name: str, typ: Type, body: FormulaExpression) -> Forall:
    """Create a universally quantified formula with De Bruijn indexing.

    Var(0, typ) inside body refers to the newly bound variable.
    Existing Var(n, ...) indices in body that refer to outer binders
    are already at their correct positions (shifted externally if needed).

    Use `shift_db_indices(body, 1)` before calling this to push outer
    variables one level deeper.
    """
    return Forall(name, typ, body)


def open_exists(srk: Context, name: str, typ: Type, body: FormulaExpression) -> Exists:
    """Create an existentially quantified formula with De Bruijn indexing.

    Var(0, typ) inside body refers to the newly bound variable.
    """
    return Exists(name, typ, body)


def _substitute_de_bruijn(
    expr: Expression,
    subst_fn: Callable[[int, Type], Expression],
    depth: int,
) -> Expression:
    """De Bruijn substitution: replaces Var(i, ty) with subst_fn(i + depth, ty).

    The depth parameter tracks how many quantifiers we've passed through.
    Each quantifier increases depth by 1, shifting indices in the body.
    """
    if isinstance(expr, Var):
        return subst_fn(expr.var_id + depth, expr.var_type)
    if isinstance(expr, Const):
        return expr
    if isinstance(expr, TrueExpr):
        return expr
    if isinstance(expr, FalseExpr):
        return expr
    if isinstance(expr, Add):
        new_args = tuple(
            _substitute_de_bruijn(a, subst_fn, depth) for a in expr.args
        )
        return Add(new_args) if new_args != expr.args else expr
    if isinstance(expr, Mul):
        new_args = tuple(
            _substitute_de_bruijn(a, subst_fn, depth) for a in expr.args
        )
        return Mul(new_args) if new_args != expr.args else expr
    if isinstance(expr, Div):
        new_left = _substitute_de_bruijn(expr.left, subst_fn, depth)
        new_right = _substitute_de_bruijn(expr.right, subst_fn, depth)
        if new_left is not expr.left or new_right is not expr.right:
            return Div(new_left, new_right)
        return expr
    if isinstance(expr, Mod):
        new_left = _substitute_de_bruijn(expr.left, subst_fn, depth)
        new_right = _substitute_de_bruijn(expr.right, subst_fn, depth)
        if new_left is not expr.left or new_right is not expr.right:
            return Mod(new_left, new_right)
        return expr
    if isinstance(expr, Floor):
        new_arg = _substitute_de_bruijn(expr.arg, subst_fn, depth)
        return Floor(new_arg) if new_arg is not expr.arg else expr
    if isinstance(expr, Neg):
        new_arg = _substitute_de_bruijn(expr.arg, subst_fn, depth)
        return Neg(new_arg) if new_arg is not expr.arg else expr
    if isinstance(expr, Pow):
        new_base = _substitute_de_bruijn(expr.base, subst_fn, depth)
        return Pow(new_base, expr.exponent) if new_base is not expr.base else expr
    if isinstance(expr, Ite):
        new_cond = _substitute_de_bruijn(expr.condition, subst_fn, depth)
        new_then = _substitute_de_bruijn(expr.then_branch, subst_fn, depth)
        new_else = _substitute_de_bruijn(expr.else_branch, subst_fn, depth)
        if (
            new_cond is not expr.condition
            or new_then is not expr.then_branch
            or new_else is not expr.else_branch
        ):
            return Ite(new_cond, new_then, new_else)
        return expr
    if isinstance(expr, App):
        new_args = tuple(
            _substitute_de_bruijn(a, subst_fn, depth) for a in expr.args
        )
        return App(expr.symbol, new_args) if new_args != expr.args else expr
    if isinstance(expr, Select):
        new_arr = _substitute_de_bruijn(expr.array, subst_fn, depth)
        new_idx = _substitute_de_bruijn(expr.index, subst_fn, depth)
        if new_arr is not expr.array or new_idx is not expr.index:
            return Select(new_arr, new_idx)
        return expr
    if isinstance(expr, Store):
        new_arr = _substitute_de_bruijn(expr.array, subst_fn, depth)
        new_idx = _substitute_de_bruijn(expr.index, subst_fn, depth)
        new_val = _substitute_de_bruijn(expr.value, subst_fn, depth)
        if (
            new_arr is not expr.array
            or new_idx is not expr.index
            or new_val is not expr.value
        ):
            return Store(new_arr, new_idx, new_val)
        return expr
    if isinstance(expr, And):
        new_args = tuple(
            _substitute_de_bruijn(a, subst_fn, depth) for a in expr.args
        )
        return And(new_args) if new_args != expr.args else expr
    if isinstance(expr, Or):
        new_args = tuple(
            _substitute_de_bruijn(a, subst_fn, depth) for a in expr.args
        )
        return Or(new_args) if new_args != expr.args else expr
    if isinstance(expr, Not):
        new_arg = _substitute_de_bruijn(expr.arg, subst_fn, depth)
        return Not(new_arg) if new_arg is not expr.arg else expr
    if isinstance(expr, Eq):
        new_left = _substitute_de_bruijn(expr.left, subst_fn, depth)
        new_right = _substitute_de_bruijn(expr.right, subst_fn, depth)
        if new_left is not expr.left or new_right is not expr.right:
            return Eq(new_left, new_right)
        return expr
    if isinstance(expr, Lt):
        new_left = _substitute_de_bruijn(expr.left, subst_fn, depth)
        new_right = _substitute_de_bruijn(expr.right, subst_fn, depth)
        if new_left is not expr.left or new_right is not expr.right:
            return Lt(new_left, new_right)
        return expr
    if isinstance(expr, Leq):
        new_left = _substitute_de_bruijn(expr.left, subst_fn, depth)
        new_right = _substitute_de_bruijn(expr.right, subst_fn, depth)
        if new_left is not expr.left or new_right is not expr.right:
            return Leq(new_left, new_right)
        return expr
    if isinstance(expr, Forall):
        # Under a quantifier, existing de Bruijn indices shift by 1
        # because the new bound variable occupies index 0
        new_body = _substitute_de_bruijn(expr.body, subst_fn, depth + 1)
        if new_body is not expr.body:
            return Forall(expr.var_name, expr.var_type, new_body)
        return expr
    if isinstance(expr, Exists):
        new_body = _substitute_de_bruijn(expr.body, subst_fn, depth + 1)
        if new_body is not expr.body:
            return Exists(expr.var_name, expr.var_type, new_body)
        return expr
    return expr


def _substitute_const(expr: Expression, subst_map: Dict[Symbol, Expression]) -> Expression:
    """Constant-based substitution: replaces Const(s) by subst_map[s]."""

    class ConstSubstVisitor(ExpressionVisitor[Expression]):
        def __init__(self, subst_map: Dict[Symbol, Expression]):
            self.subst_map = subst_map
            self._by_id = {s.id: r for s, r in subst_map.items()}

        def visit_var(self, var: Var) -> Expression:
            if var.var_id in self._by_id:
                return self._by_id[var.var_id]
            return var

        def visit_const(self, const: Const) -> Expression:
            if const.symbol in self.subst_map:
                return self.subst_map[const.symbol]
            return const

        def visit_app(self, app: App) -> Expression:
            new_sym = self.subst_map.get(app.symbol, app.symbol)
            new_args = tuple(a.accept(self) for a in app.args)
            return App(new_sym, new_args)

        def visit_select(self, select: Select) -> Expression:
            return Select(select.array.accept(self), select.index.accept(self))

        def visit_store(self, store: Store) -> Expression:
            return Store(store.array.accept(self), store.index.accept(self), store.value.accept(self))

        def visit_add(self, add: Add) -> Expression:
            return Add(tuple(a.accept(self) for a in add.args))

        def visit_mul(self, mul: Mul) -> Expression:
            return Mul(tuple(a.accept(self) for a in mul.args))

        def visit_div(self, div: Div) -> Expression:
            return Div(div.left.accept(self), div.right.accept(self))

        def visit_mod(self, mod: Mod) -> Expression:
            return Mod(mod.left.accept(self), mod.right.accept(self))

        def visit_floor(self, floor: Floor) -> Expression:
            return Floor(floor.arg.accept(self))

        def visit_neg(self, neg: Neg) -> Expression:
            return Neg(neg.arg.accept(self))

        def visit_pow(self, pow_expr: Pow) -> Expression:
            return Pow(pow_expr.base.accept(self), pow_expr.exponent)

        def visit_ite(self, ite: Ite) -> Expression:
            return Ite(ite.condition.accept(self), ite.then_branch.accept(self), ite.else_branch.accept(self))

        def visit_true(self, t: TrueExpr) -> Expression:
            return t

        def visit_false(self, f: FalseExpr) -> Expression:
            return f

        def visit_and(self, a: And) -> Expression:
            return And(tuple(x.accept(self) for x in a.args))

        def visit_or(self, o: Or) -> Expression:
            return Or(tuple(x.accept(self) for x in o.args))

        def visit_not(self, n: Not) -> Expression:
            return Not(n.arg.accept(self))

        def visit_eq(self, eq: Eq) -> Expression:
            return Eq(eq.left.accept(self), eq.right.accept(self))

        def visit_lt(self, lt: Lt) -> Expression:
            return Lt(lt.left.accept(self), lt.right.accept(self))

        def visit_leq(self, leq: Leq) -> Expression:
            return Leq(leq.left.accept(self), leq.right.accept(self))

        def visit_forall(self, f: Forall) -> Expression:
            active = {s: r for s, r in self.subst_map.items() if s.name != f.var_name}
            new_body = _substitute_const(f.body, active) if active else f.body
            return Forall(f.var_name, f.var_type, new_body) if new_body is not f.body else f

        def visit_exists(self, e: Exists) -> Expression:
            active = {s: r for s, r in self.subst_map.items() if s.name != e.var_name}
            new_body = _substitute_const(e.body, active) if active else e.body
            return Exists(e.var_name, e.var_type, new_body) if new_body is not e.body else e

        def _default_visit(self, expr: Expression) -> Expression:
            return expr

    return expr.accept(ConstSubstVisitor(subst_map))


def substitute_map(expr: Expression, subst_map: Dict[Symbol, Expression]) -> Expression:
    """Compatibility alias for substituting a symbol-to-expression map."""
    return substitute(expr, subst_map)


def substitute_sym(srk: Context, subst: Callable[[Symbol], Expression], expr: Expression) -> Expression:
    """Replace each application f(e0,...,en) with subst(f)[e0/0, ..., en/n].

    Mirrors OCaml's substitute_sym: constant symbols are treated as
    nullary function applications.
    """
    env: List[Expression] = []

    def go(e: Expression, bound: Set[str]) -> Expression:
        if isinstance(e, Const):
            fn_body = subst(e.symbol)
            return substitute_by_name(fn_body, {}, bound)
        if isinstance(e, App):
            fn_body = subst(e.symbol)
            subst_map_local = {i: go(a, bound) for i, a in enumerate(e.args)}
            return substitute_by_name(fn_body, subst_map_local, bound)
        if isinstance(e, Add):
            new_args = tuple(go(a, bound) for a in e.args)
            return Add(new_args) if new_args != e.args else e
        if isinstance(e, Mul):
            new_args = tuple(go(a, bound) for a in e.args)
            return Mul(new_args) if new_args != e.args else e
        if isinstance(e, Div):
            new_left = go(e.left, bound)
            new_right = go(e.right, bound)
            return Div(new_left, new_right) if new_left is not e.left or new_right is not e.right else e
        if isinstance(e, Mod):
            new_left = go(e.left, bound)
            new_right = go(e.right, bound)
            return Mod(new_left, new_right) if new_left is not e.left or new_right is not e.right else e
        if isinstance(e, Floor):
            new_arg = go(e.arg, bound)
            return Floor(new_arg) if new_arg is not e.arg else e
        if isinstance(e, Neg):
            new_arg = go(e.arg, bound)
            return Neg(new_arg) if new_arg is not e.arg else e
        if isinstance(e, Pow):
            new_base = go(e.base, bound)
            return Pow(new_base, e.exponent) if new_base is not e.base else e
        if isinstance(e, Ite):
            return Ite(go(e.condition, bound), go(e.then_branch, bound), go(e.else_branch, bound))
        if isinstance(e, Select):
            return Select(go(e.array, bound), go(e.index, bound))
        if isinstance(e, Store):
            return Store(go(e.array, bound), go(e.index, bound), go(e.value, bound))
        if isinstance(e, And):
            new_args = tuple(go(a, bound) for a in e.args)
            return And(new_args) if new_args != e.args else e
        if isinstance(e, Or):
            new_args = tuple(go(a, bound) for a in e.args)
            return Or(new_args) if new_args != e.args else e
        if isinstance(e, Not):
            new_arg = go(e.arg, bound)
            return Not(new_arg) if new_arg is not e.arg else e
        if isinstance(e, Eq):
            return Eq(go(e.left, bound), go(e.right, bound))
        if isinstance(e, Lt):
            return Lt(go(e.left, bound), go(e.right, bound))
        if isinstance(e, Leq):
            return Leq(go(e.left, bound), go(e.right, bound))
        if isinstance(e, Forall):
            return Forall(e.var_name, e.var_type, go(e.body, bound | {e.var_name}))
        if isinstance(e, Exists):
            return Exists(e.var_name, e.var_type, go(e.body, bound | {e.var_name}))
        return e

    def substitute_by_name(expr: Expression, var_map: Dict[int, Expression], bound: Set[str]) -> Expression:
        if isinstance(expr, Var) and expr.var_id in var_map:
            return var_map[expr.var_id]
        return go(expr, bound)

    return go(expr, set())


def fold_constants(
    f: Callable[[Symbol, Any], Any], expr: Expression, init: Any
) -> Any:
    """Fold over all constant symbols in an expression.

    f receives (symbol, accumulator) for each constant occurrence.
    """
    acc = init

    def walk(e: Expression) -> None:
        nonlocal acc
        if isinstance(e, Const):
            acc = f(e.symbol, acc)
        elif isinstance(e, App):
            for arg in e.args:
                walk(arg)
        elif isinstance(e, Select):
            walk(e.array)
            walk(e.index)
        elif isinstance(e, Store):
            walk(e.array)
            walk(e.index)
            walk(e.value)
        elif isinstance(e, Add):
            for arg in e.args:
                walk(arg)
        elif isinstance(e, Mul):
            for arg in e.args:
                walk(arg)
        elif isinstance(e, Div):
            walk(e.left)
            walk(e.right)
        elif isinstance(e, Mod):
            walk(e.left)
            walk(e.right)
        elif isinstance(e, Floor):
            walk(e.arg)
        elif isinstance(e, Neg):
            walk(e.arg)
        elif isinstance(e, Pow):
            walk(e.base)
        elif isinstance(e, Ite):
            walk(e.condition)
            walk(e.then_branch)
            walk(e.else_branch)
        elif isinstance(e, And):
            for arg in e.args:
                walk(arg)
        elif isinstance(e, Or):
            for arg in e.args:
                walk(arg)
        elif isinstance(e, Not):
            walk(e.arg)
        elif isinstance(e, Eq):
            walk(e.left)
            walk(e.right)
        elif isinstance(e, Lt):
            walk(e.left)
            walk(e.right)
        elif isinstance(e, Leq):
            walk(e.left)
            walk(e.right)
        elif isinstance(e, Forall):
            walk(e.body)
        elif isinstance(e, Exists):
            walk(e.body)

    walk(expr)
    return acc


def free_vars(expr: Expression) -> Set[Symbol]:
    """Extract free constant/variable symbols from an expression.

    Bound quantifier names are removed by name, matching this port's quantifier
    representation.  Function symbols are not counted as free variables.
    """

    def go(e: Expression, bound_names: Set[str]) -> Set[Symbol]:
        if isinstance(e, Const):
            if e.symbol.name in bound_names:
                return set()
            return {e.symbol}
        if isinstance(e, Var):
            if e.name in bound_names:
                return set()
            return {Symbol(e.var_id, e.name, e.var_type)}
        if isinstance(e, App):
            result: Set[Symbol] = set()
            for arg in e.args:
                result.update(go(arg, bound_names))
            return result
        if isinstance(e, Select):
            return go(e.array, bound_names) | go(e.index, bound_names)
        if isinstance(e, Store):
            return (
                go(e.array, bound_names)
                | go(e.index, bound_names)
                | go(e.value, bound_names)
            )
        if isinstance(e, (Add, Mul, And, Or)):
            result: Set[Symbol] = set()
            for arg in e.args:
                result.update(go(arg, bound_names))
            return result
        if isinstance(e, Ite):
            return (
                go(e.condition, bound_names)
                | go(e.then_branch, bound_names)
                | go(e.else_branch, bound_names)
            )
        if isinstance(e, Not):
            return go(e.arg, bound_names)
        if isinstance(e, (Eq, Lt, Leq, Div, Mod)):
            return go(e.left, bound_names) | go(e.right, bound_names)
        if isinstance(e, (Neg, Floor)):
            return go(e.arg, bound_names)
        if isinstance(e, Pow):
            return go(e.base, bound_names)
        if isinstance(e, Forall):
            return go(e.body, bound_names | {e.var_name})
        if isinstance(e, Exists):
            return go(e.body, bound_names | {e.var_name})
        return set()

    return go(expr, set())


def vars(expr: Expression) -> Set[Symbol]:
    """Compatibility alias for free variables."""
    return free_vars(expr)


def size(expr: Expression) -> int:
    """Return the number of AST nodes in an expression."""
    if isinstance(expr, (Var, Const, TrueExpr, FalseExpr)):
        return 1
    if isinstance(expr, App):
        return 1 + sum(size(arg) for arg in expr.args)
    if isinstance(expr, Select):
        return 1 + size(expr.array) + size(expr.index)
    if isinstance(expr, Store):
        return 1 + size(expr.array) + size(expr.index) + size(expr.value)
    if isinstance(expr, (Add, Mul, And, Or)):
        return 1 + sum(size(arg) for arg in expr.args)
    if isinstance(expr, Ite):
        return 1 + size(expr.condition) + size(expr.then_branch) + size(expr.else_branch)
    if isinstance(expr, (Neg, Not, Floor)):
        return 1 + size(expr.arg)
    if isinstance(expr, Pow):
        return 1 + size(expr.base)
    if isinstance(expr, (Eq, Lt, Leq, Div, Mod)):
        return 1 + size(expr.left) + size(expr.right)
    if isinstance(expr, (Forall, Exists)):
        return 1 + size(expr.body)
    return 1


def eliminate_ite(expr: Expression) -> Expression:
    """Eliminate Boolean ITEs by rewriting them to disjunctions."""
    if isinstance(expr, Ite) and expr.typ == Type.BOOL:
        condition = eliminate_ite(expr.condition)
        then_branch = eliminate_ite(expr.then_branch)
        else_branch = eliminate_ite(expr.else_branch)
        return mk_or(
            [
                mk_and([condition, then_branch]),
                mk_and([mk_not(condition), else_branch]),
            ]
        )
    if isinstance(expr, And):
        return And(tuple(eliminate_ite(arg) for arg in expr.args))
    if isinstance(expr, Or):
        return Or(tuple(eliminate_ite(arg) for arg in expr.args))
    if isinstance(expr, Not):
        return Not(eliminate_ite(expr.arg))
    if isinstance(expr, Eq):
        return Eq(eliminate_ite(expr.left), eliminate_ite(expr.right))
    if isinstance(expr, Lt):
        return Lt(eliminate_ite(expr.left), eliminate_ite(expr.right))
    if isinstance(expr, Leq):
        return Leq(eliminate_ite(expr.left), eliminate_ite(expr.right))
    if isinstance(expr, Forall):
        return Forall(expr.var_name, expr.var_type, eliminate_ite(expr.body))
    if isinstance(expr, Exists):
        return Exists(expr.var_name, expr.var_type, eliminate_ite(expr.body))
    return expr


def eliminate_arr_eq(expr: Expression) -> Expression:
    """Eliminate array equality through extensionality.

    Replaces `Eq(a, b)` where a or b are arrays with:
      forall i. select(a, i) = select(b, i)

    Only eliminates when both sides are array-typed expressions.
    """
    if isinstance(expr, Eq):
        left, right = expr.left, expr.right
        if _is_arr_type(left) or _is_arr_type(right):
            idx_var_name = "_i"
            a_access = Select(left, Var(0, Type.INT))
            b_access = Select(right, Var(0, Type.INT))
            body = Eq(a_access, b_access)
            return Forall(idx_var_name, Type.INT, body)
        return expr
    if isinstance(expr, Not):
        new_arg = eliminate_arr_eq(expr.arg)
        return Not(new_arg) if new_arg is not expr.arg else expr
    if isinstance(expr, And):
        new_args = tuple(eliminate_arr_eq(a) for a in expr.args)
        return And(new_args) if new_args != expr.args else expr
    if isinstance(expr, Or):
        new_args = tuple(eliminate_arr_eq(a) for a in expr.args)
        return Or(new_args) if new_args != expr.args else expr
    if isinstance(expr, Forall):
        new_body = eliminate_arr_eq(expr.body)
        return Forall(expr.var_name, expr.var_type, new_body) if new_body is not expr.body else expr
    if isinstance(expr, Exists):
        new_body = eliminate_arr_eq(expr.body)
        return Exists(expr.var_name, expr.var_type, new_body) if new_body is not expr.body else expr
    return expr


def _is_arr_type(expr: Expression) -> bool:
    """Check if an expression is array-typed."""
    if isinstance(expr, Store):
        return True
    if isinstance(expr, Select):
        return False  # select returns element, not array
    if isinstance(expr, Const) and expr.symbol.typ == Type.ARRAY:
        return True
    if isinstance(expr, App) and expr.symbol.typ == Type.ARRAY:
        return True
    return False


def rewrite(
    *args, down: Optional[Callable[[Expression], Expression]] = None, up: Optional[Callable[[Expression], Expression]] = None
) -> Expression:
    """Rewrite an expression using rewrite rules.

    Performs a two-pass traversal: the *down* rewriter is applied to each
    sub-expression on the way down the tree (pre-order), then the *up*
    rewriter is applied on the way back up (post-order).

    Accepts multiple calling conventions for compatibility:
    - rewrite(expr, down=..., up=...)
    - rewrite(srk, expr, down=..., up=...)
    - rewrite(srk, down_fn, expr)
    """
    # Normalize arguments to (expr, down, up)
    if not args:
        raise TypeError("rewrite expects at least 1 positional argument")

    if isinstance(args[0], Context):
        if len(args) == 2:
            _, expr = args
        elif len(args) == 3 and callable(args[1]):
            _, down_fn, expr = args
            down = down_fn
        elif len(args) == 3:
            _, expr, up_fn = args
            if callable(up_fn) and up is None:
                up = up_fn
        else:
            raise TypeError(
                f"rewrite expects (srk, expr, ...) or (expr, ...), got {len(args)} args"
            )
    else:
        expr = args[0]
        if len(args) == 2 and down is None:
            down = args[1]
        elif len(args) == 3 and down is None and up is None:
            down, up = args[1], args[2]
        elif len(args) > 3:
            raise TypeError(
                f"rewrite expects at most 3 positional args, got {len(args)}"
            )

    def _rewrite_node(e: Expression) -> Expression:
        # Down pass: apply down rewriter before recursing
        if down:
            e = down(e)
        # Recurse into children and rebuild
        e = _rewrite_children(e)
        # Up pass: apply up rewriter after recursing
        if up:
            e = up(e)
        return e

    def _rewrite_children(e: Expression) -> Expression:
        if isinstance(e, Var):
            return e
        if isinstance(e, Const):
            return e
        if isinstance(e, TrueExpr):
            return e
        if isinstance(e, FalseExpr):
            return e
        if isinstance(e, Add):
            new_args = tuple(_rewrite_node(a) for a in e.args)
            return Add(new_args) if new_args != e.args else e
        if isinstance(e, Mul):
            new_args = tuple(_rewrite_node(a) for a in e.args)
            return Mul(new_args) if new_args != e.args else e
        if isinstance(e, Div):
            new_left = _rewrite_node(e.left)
            new_right = _rewrite_node(e.right)
            if new_left is not e.left or new_right is not e.right:
                return Div(new_left, new_right)
            return e
        if isinstance(e, Mod):
            new_left = _rewrite_node(e.left)
            new_right = _rewrite_node(e.right)
            if new_left is not e.left or new_right is not e.right:
                return Mod(new_left, new_right)
            return e
        if isinstance(e, Floor):
            new_arg = _rewrite_node(e.arg)
            return Floor(new_arg) if new_arg is not e.arg else e
        if isinstance(e, Neg):
            new_arg = _rewrite_node(e.arg)
            return Neg(new_arg) if new_arg is not e.arg else e
        if isinstance(e, Pow):
            new_base = _rewrite_node(e.base)
            if new_base is not e.base:
                return Pow(new_base, e.exponent)
            return e
        if isinstance(e, Ite):
            new_cond = _rewrite_node(e.condition)
            new_then = _rewrite_node(e.then_branch)
            new_else = _rewrite_node(e.else_branch)
            if new_cond is not e.condition or new_then is not e.then_branch or new_else is not e.else_branch:
                return Ite(new_cond, new_then, new_else)
            return e
        if isinstance(e, App):
            new_args = tuple(_rewrite_node(a) for a in e.args)
            return App(e.symbol, new_args) if new_args != e.args else e
        if isinstance(e, Select):
            new_arr = _rewrite_node(e.array)
            new_idx = _rewrite_node(e.index)
            if new_arr is not e.array or new_idx is not e.index:
                return Select(new_arr, new_idx)
            return e
        if isinstance(e, Store):
            new_arr = _rewrite_node(e.array)
            new_idx = _rewrite_node(e.index)
            new_val = _rewrite_node(e.value)
            if new_arr is not e.array or new_idx is not e.index or new_val is not e.value:
                return Store(new_arr, new_idx, new_val)
            return e
        if isinstance(e, And):
            new_args = tuple(_rewrite_node(a) for a in e.args)
            return And(new_args) if new_args != e.args else e
        if isinstance(e, Or):
            new_args = tuple(_rewrite_node(a) for a in e.args)
            return Or(new_args) if new_args != e.args else e
        if isinstance(e, Not):
            new_arg = _rewrite_node(e.arg)
            return Not(new_arg) if new_arg is not e.arg else e
        if isinstance(e, Eq):
            new_left = _rewrite_node(e.left)
            new_right = _rewrite_node(e.right)
            if new_left is not e.left or new_right is not e.right:
                return Eq(new_left, new_right)
            return e
        if isinstance(e, Lt):
            new_left = _rewrite_node(e.left)
            new_right = _rewrite_node(e.right)
            if new_left is not e.left or new_right is not e.right:
                return Lt(new_left, new_right)
            return e
        if isinstance(e, Leq):
            new_left = _rewrite_node(e.left)
            new_right = _rewrite_node(e.right)
            if new_left is not e.left or new_right is not e.right:
                return Leq(new_left, new_right)
            return e
        if isinstance(e, Forall):
            new_body = _rewrite_node(e.body)
            return Forall(e.var_name, e.var_type, new_body) if new_body is not e.body else e
        if isinstance(e, Exists):
            new_body = _rewrite_node(e.body)
            return Exists(e.var_name, e.var_type, new_body) if new_body is not e.body else e
        return e

    return _rewrite_node(expr)


def nnf_rewriter(expr: Expression) -> Expression:
    """Convert formula to negation normal form (NNF).

    Pushes negations inward using De Morgan's laws and
    eliminates double negations, implications, and iff.
    """
    return _nnf_rewriter(expr)


def _nnf_rewriter(expr: Expression) -> Expression:
    if isinstance(expr, Not):
        inner = expr.arg
        if isinstance(inner, Not):
            # Double negation: !!p => p
            return _nnf_rewriter(inner.arg)
        if isinstance(inner, TrueExpr):
            return FalseExpr()
        if isinstance(inner, FalseExpr):
            return TrueExpr()
        if isinstance(inner, And):
            # !(a && b) => !a || !b
            return Or(tuple(_nnf_rewriter(Not(a)) for a in inner.args))
        if isinstance(inner, Or):
            # !(a || b) => !a && !b
            return And(tuple(_nnf_rewriter(Not(a)) for a in inner.args))
        if isinstance(inner, Eq):
            # !(a = b) => a < b || b < a
            return Or((Lt(inner.left, inner.right), Lt(inner.right, inner.left)))
        if isinstance(inner, Lt):
            # !(a < b) => b <= a
            return Leq(inner.right, inner.left)
        if isinstance(inner, Leq):
            # !(a <= b) => b < a
            return Lt(inner.right, inner.left)
        if isinstance(inner, Forall):
            # !(forall x. p) => exists x. !p
            return Exists(inner.var_name, inner.var_type, _nnf_rewriter(Not(inner.body)))
        if isinstance(inner, Exists):
            # !(exists x. p) => forall x. !p
            return Forall(inner.var_name, inner.var_type, _nnf_rewriter(Not(inner.body)))
        if isinstance(inner, Ite) and inner.typ == Type.BOOL:
            # !(ite c t e) => ite c (!t) (!e)
            return Ite(
                _nnf_rewriter(inner.condition),
                _nnf_rewriter(Not(inner.then_branch)),
                _nnf_rewriter(Not(inner.else_branch)),
            )
        # Negation of atomic formula or term case - keep as is
        return Not(_nnf_rewriter(inner))
    # Recurse into compound expressions
    return _nnf_recurse(expr)


def _nnf_recurse(expr: Expression) -> Expression:
    if isinstance(expr, (Var, Const, TrueExpr, FalseExpr)):
        return expr
    if isinstance(expr, Add):
        return Add(tuple(_nnf_recurse(a) for a in expr.args))
    if isinstance(expr, Mul):
        return Mul(tuple(_nnf_recurse(a) for a in expr.args))
    if isinstance(expr, Div):
        return Div(_nnf_recurse(expr.left), _nnf_recurse(expr.right))
    if isinstance(expr, Mod):
        return Mod(_nnf_recurse(expr.left), _nnf_recurse(expr.right))
    if isinstance(expr, Floor):
        return Floor(_nnf_recurse(expr.arg))
    if isinstance(expr, Neg):
        return Neg(_nnf_recurse(expr.arg))
    if isinstance(expr, Pow):
        return Pow(_nnf_recurse(expr.base), expr.exponent)
    if isinstance(expr, Ite):
        return Ite(
            _nnf_recurse(expr.condition),
            _nnf_recurse(expr.then_branch),
            _nnf_recurse(expr.else_branch),
        )
    if isinstance(expr, App):
        return App(expr.symbol, tuple(_nnf_recurse(a) for a in expr.args))
    if isinstance(expr, Select):
        return Select(_nnf_recurse(expr.array), _nnf_recurse(expr.index))
    if isinstance(expr, Store):
        return Store(
            _nnf_recurse(expr.array),
            _nnf_recurse(expr.index),
            _nnf_recurse(expr.value),
        )
    if isinstance(expr, And):
        return And(tuple(_nnf_recurse(a) for a in expr.args))
    if isinstance(expr, Or):
        return Or(tuple(_nnf_recurse(a) for a in expr.args))
    if isinstance(expr, Not):
        return _nnf_rewriter(Not(_nnf_recurse(expr.arg)))
    if isinstance(expr, Eq):
        return Eq(_nnf_recurse(expr.left), _nnf_recurse(expr.right))
    if isinstance(expr, Lt):
        return Lt(_nnf_recurse(expr.left), _nnf_recurse(expr.right))
    if isinstance(expr, Leq):
        return Leq(_nnf_recurse(expr.left), _nnf_recurse(expr.right))
    if isinstance(expr, Forall):
        return Forall(expr.var_name, expr.var_type, _nnf_recurse(expr.body))
    if isinstance(expr, Exists):
        return Exists(expr.var_name, expr.var_type, _nnf_recurse(expr.body))
    return expr


def destruct(expr: Expression) -> Tuple[str, Any]:
    """Destruct an expression into its constructor and components.

    Returns a tuple where the first element is a string indicating the
    constructor type, and the second element contains the components.

    Mirrors the OCaml polymorphic-variant-based destruct.
    """
    if isinstance(expr, Var):
        return ("Var", (expr.var_id, expr.var_type))
    elif isinstance(expr, Const):
        symbol = expr.symbol
        try:
            if symbol.name and symbol.name.startswith("real_"):
                val = float(symbol.name[5:])
                return ("Real", val)
        except (ValueError, AttributeError):
            pass
        return ("Const", symbol)
    elif isinstance(expr, App):
        return ("App", (expr.symbol, expr.args))
    elif isinstance(expr, Select):
        return ("Select", (expr.array, expr.index))
    elif isinstance(expr, Store):
        return ("Store", (expr.array, expr.index, expr.value))
    elif isinstance(expr, Add):
        return ("Add", expr.args)
    elif isinstance(expr, Mul):
        return ("Mul", expr.args)
    elif isinstance(expr, Div):
        return ("Binop", ("Div", expr.left, expr.right))
    elif isinstance(expr, Mod):
        return ("Binop", ("Mod", expr.left, expr.right))
    elif isinstance(expr, Floor):
        return ("Unop", ("Floor", expr.arg))
    elif isinstance(expr, Neg):
        return ("Unop", ("Neg", expr.arg))
    elif isinstance(expr, Pow):
        return ("Pow", (expr.base, expr.exponent))
    elif isinstance(expr, Ite):
        return ("Ite", (expr.condition, expr.then_branch, expr.else_branch))
    elif isinstance(expr, TrueExpr):
        return ("Tru", ())
    elif isinstance(expr, FalseExpr):
        return ("Fls", ())
    elif isinstance(expr, And):
        return ("And", expr.args)
    elif isinstance(expr, Or):
        return ("Or", expr.args)
    elif isinstance(expr, Not):
        return ("Not", expr.arg)
    elif isinstance(expr, Eq):
        return ("Atom", ("Arith", "Eq", expr.left, expr.right))
    elif isinstance(expr, Lt):
        return ("Atom", ("Arith", "Lt", expr.left, expr.right))
    elif isinstance(expr, Leq):
        return ("Atom", ("Arith", "Leq", expr.left, expr.right))
    elif isinstance(expr, Forall):
        return ("Quantify", ("Forall", expr.var_name, expr.var_type, expr.body))
    elif isinstance(expr, Exists):
        return ("Quantify", ("Exists", expr.var_name, expr.var_type, expr.body))
    else:
        return ("Unknown", expr)


def symbols(expr: Expression) -> Set[Symbol]:
    """Extract all symbols from an expression."""

    class SymbolExtractor(ExpressionVisitor[Set[Symbol]]):
        def visit_var(self, var: Var) -> Set[Symbol]:
            return set()

        def visit_const(self, const: Const) -> Set[Symbol]:
            return {const.symbol}

        def visit_app(self, app: App) -> Set[Symbol]:
            result = {app.symbol}
            for arg in app.args:
                result.update(arg.accept(self))
            return result

        def visit_select(self, select: Select) -> Set[Symbol]:
            result = select.array.accept(self)
            result.update(select.index.accept(self))
            return result

        def visit_store(self, store: Store) -> Set[Symbol]:
            result = store.array.accept(self)
            result.update(store.index.accept(self))
            result.update(store.value.accept(self))
            return result

        def visit_add(self, add: Add) -> Set[Symbol]:
            result = set()
            for arg in add.args:
                result.update(arg.accept(self))
            return result

        def visit_mul(self, mul: Mul) -> Set[Symbol]:
            result = set()
            for arg in mul.args:
                result.update(arg.accept(self))
            return result

        def visit_ite(self, ite: Ite) -> Set[Symbol]:
            result = ite.condition.accept(self)
            result.update(ite.then_branch.accept(self))
            result.update(ite.else_branch.accept(self))
            return result

        def visit_true(self, true_expr: TrueExpr) -> Set[Symbol]:
            return set()

        def visit_false(self, false_expr: FalseExpr) -> Set[Symbol]:
            return set()

        def visit_and(self, and_expr: And) -> Set[Symbol]:
            result = set()
            for arg in and_expr.args:
                result.update(arg.accept(self))
            return result

        def visit_or(self, or_expr: Or) -> Set[Symbol]:
            result = set()
            for arg in or_expr.args:
                result.update(arg.accept(self))
            return result

        def visit_not(self, not_expr: Not) -> Set[Symbol]:
            return not_expr.arg.accept(self)

        def visit_eq(self, eq: Eq) -> Set[Symbol]:
            result = eq.left.accept(self)
            result.update(eq.right.accept(self))
            return result

        def visit_lt(self, lt: Lt) -> Set[Symbol]:
            result = lt.left.accept(self)
            result.update(lt.right.accept(self))
            return result

        def visit_leq(self, leq: Leq) -> Set[Symbol]:
            result = leq.left.accept(self)
            result.update(leq.right.accept(self))
            return result

        def visit_forall(self, forall: Forall) -> Set[Symbol]:
            return forall.body.accept(self)

        def visit_exists(self, exists: Exists) -> Set[Symbol]:
            return exists.body.accept(self)

        def _default_visit(self, expr: Expression) -> Set[Symbol]:
            return set()

    extractor = SymbolExtractor()
    return expr.accept(extractor)


def typ_symbol(symbol: Symbol) -> Type:
    """Get the type of a symbol."""
    return symbol.typ


class Env:
    """Environment for expression evaluation."""

    def __init__(self):
        pass


def expr_typ(expr: Expression) -> Type:
    """Get the type of an expression."""
    # Most expression types have their type as a class attribute
    if hasattr(expr, "typ"):
        return expr.typ
    # For more complex expressions, we need to infer the type
    if isinstance(expr, Var):
        return expr.var_type
    elif isinstance(expr, Const):
        return expr.symbol.typ
    elif isinstance(expr, App):
        # Function applications take the return type of the function
        # For now, assume they return the same type as their first argument
        if expr.args:
            return expr_typ(expr.args[0])
        return Type.INT  # Default fallback
    elif isinstance(expr, Add) or isinstance(expr, Mul):
        # Arithmetic operations promote to real if any operand is real
        for arg in expr.args:
            if expr_typ(arg) == Type.REAL:
                return Type.REAL
        return Type.INT
    elif isinstance(expr, Select):
        # Array select returns the element type (assumed INT for now)
        return Type.INT
    elif isinstance(expr, Ite):
        # ITE takes the type of its branches
        then_type = expr_typ(expr.then_branch)
        else_type = expr_typ(expr.else_branch)
        if then_type == else_type:
            return then_type
        # If types differ, promote to more general type
        if then_type == Type.REAL or else_type == Type.REAL:
            return Type.REAL
        return Type.INT
    else:
        # For formulas and other expressions, return BOOL
        return Type.BOOL


def int_of_symbol(symbol: Symbol) -> int:
    """Convert a symbol to its integer ID."""
    return symbol.id


def symbol_of_int(id: int) -> Symbol:
    """Create a symbol from an integer ID."""
    # This is a simplified implementation - in a full implementation,
    # this would need to handle type information properly
    return Symbol(id, f"s{id}", Type.INT)


def dup_symbol(context: Context, symbol: Symbol) -> Symbol:
    """Return a fresh symbol with the same name and type as *symbol*.

    Mirrors OCaml's ``dup_symbol``.
    """
    return context.dup_symbol(symbol)


def compare_symbol(a: Symbol, b: Symbol) -> int:
    """Total order on symbols by id (mirrors OCaml's compare_symbol)."""
    return Context.compare_symbol(a, b)


def substitute_const(*args) -> Expression:
    """Substitute constants in an expression.

    Supported calling conventions:
    - substitute_const(subst_map: Dict[Symbol, Expression], expr: Expression)
    - substitute_const(srk: Context, f: Callable[[Symbol], Expression], expr: Expression)

    The second form is used by some quantifier-elimination routines which pass a
    context and a function deciding how to rewrite each constant symbol.
    """
    # Dict-based form (current simplified API)
    if len(args) == 2 and isinstance(args[0], dict):
        subst_map, expr = args
        return substitute(expr, subst_map)

    # Legacy SRK-style form
    if len(args) == 3 and isinstance(args[0], Context) and callable(args[1]):
        srk, f, expr = args

        class ConstSubVisitor(ExpressionVisitor[Expression]):
            def visit_const(self, const: Const) -> Expression:
                sym = const.symbol
                replacement = f(sym)
                # Treat "identity replacement" as no-op to avoid needless churn.
                try:
                    if replacement == mk_const(srk, sym):
                        return const
                except Exception:
                    pass
                return replacement

            def _default_visit(self, e: Expression) -> Expression:
                return e

            def visit_var(self, var: Var) -> Expression:
                return var

            def visit_app(self, app: App) -> Expression:
                return App(app.symbol, tuple(arg.accept(self) for arg in app.args))

            def visit_select(self, select: Select) -> Expression:
                return Select(select.array.accept(self), select.index.accept(self))

            def visit_store(self, store: Store) -> Expression:
                return Store(
                    store.array.accept(self),
                    store.index.accept(self),
                    store.value.accept(self),
                )

            def visit_add(self, add: Add) -> Expression:
                return Add(tuple(arg.accept(self) for arg in add.args))

            def visit_mul(self, mul: Mul) -> Expression:
                return Mul(tuple(arg.accept(self) for arg in mul.args))

            def visit_ite(self, ite: Ite) -> Expression:
                return Ite(
                    ite.condition.accept(self),
                    ite.then_branch.accept(self),
                    ite.else_branch.accept(self),
                )

            def visit_true(self, true_expr: TrueExpr) -> Expression:
                return true_expr

            def visit_false(self, false_expr: FalseExpr) -> Expression:
                return false_expr

            def visit_and(self, and_expr: And) -> Expression:
                return And(tuple(arg.accept(self) for arg in and_expr.args))

            def visit_or(self, or_expr: Or) -> Expression:
                return Or(tuple(arg.accept(self) for arg in or_expr.args))

            def visit_not(self, not_expr: Not) -> Expression:
                return Not(not_expr.arg.accept(self))

            def visit_eq(self, eq: Eq) -> Expression:
                return Eq(eq.left.accept(self), eq.right.accept(self))

            def visit_lt(self, lt: Lt) -> Expression:
                return Lt(lt.left.accept(self), lt.right.accept(self))

            def visit_leq(self, leq: Leq) -> Expression:
                return Leq(leq.left.accept(self), leq.right.accept(self))

            def visit_forall(self, forall: Forall) -> Expression:
                return Forall(
                    forall.var_name, forall.var_type, forall.body.accept(self)
                )

            def visit_exists(self, exists: Exists) -> Expression:
                return Exists(
                    exists.var_name, exists.var_type, exists.body.accept(self)
                )

        return expr.accept(ConstSubVisitor())

    raise TypeError(
        f"substitute_const expects (dict, expr) or (context, f, expr), got {len(args)} args"
    )


def prenex(srk: Context, phi: Expression) -> Expression:
    """
    Convert a formula to prenex normal form.

    Moves all quantifiers to the front of the formula, preserving logical equivalence.

    Args:
        srk: Context
        phi: Formula to convert to prenex form

    Returns:
        Formula in prenex normal form
    """

    def negate_prefix(prefix):
        """Negate quantifier prefix (exists <-> forall)."""
        return [
            (("Forall" if q[0] == "Exists" else "Exists"), name, typ)
            for q, name, typ in prefix
        ]

    def combine(phis):
        """Combine multiple formulas with their quantifier prefixes."""
        if not phis:
            return ([], [])

        result_prefix = []
        result_phis = []

        for qf_pre, phi in phis:
            depth = len(result_prefix)
            depth0 = len(qf_pre)
            # Adjust variable indices to avoid conflicts
            adjusted_phi = _adjust_variable_indices(phi, depth, depth0)
            result_prefix.extend(qf_pre)
            result_phis.append(adjusted_phi)

        return (result_prefix, result_phis)

    def _adjust_variable_indices(expr, old_depth, new_depth):
        """Adjust variable indices to avoid conflicts when combining formulas."""
        # This is a simplified implementation - in practice, this would need
        # to properly handle variable renaming to avoid capture
        return expr

    def process(expr):
        """Process expression to extract quantifier prefix and matrix."""
        match = destruct(expr)

        if not match:
            return ([], expr)

        op, *args = match

        if op == "True":
            return ([], mk_true())
        elif op == "False":
            return ([], mk_false())
        elif op == "Atom":
            return ([], expr)
        elif op == "And":
            conjuncts = [process(arg) for arg in args]
            qf_pre, conjuncts = combine(conjuncts)
            return (qf_pre, mk_and(conjuncts))
        elif op == "Or":
            disjuncts = [process(arg) for arg in args]
            qf_pre, disjuncts = combine(disjuncts)
            return (qf_pre, mk_or(disjuncts))
        elif op == "Quantify":
            qt, name, typ, body = args
            qf_pre, matrix = process(body)
            return ([(qt, name, typ)] + qf_pre, matrix)
        elif op == "Not":
            qf_pre, matrix = process(args[0])
            return (negate_prefix(qf_pre), mk_not(matrix))
        elif op == "Ite":
            cond, then_branch, else_branch = args
            cond_prefix, cond_matrix = process(cond)
            then_prefix, then_matrix = process(then_branch)
            else_prefix, else_matrix = process(else_branch)

            # Combine all three branches
            all_prefixes = [cond_prefix, then_prefix, else_prefix]
            qf_pre, matrices = combine(all_prefixes)

            if len(matrices) == 3:
                cond_m, then_m, else_m = matrices
                return (qf_pre, mk_ite(cond_m, then_m, else_m))
            else:
                # Fallback if combination fails
                return ([], expr)
        else:
            # For other cases, return as-is
            return ([], expr)

    # Process the formula
    qf_pre, matrix = process(phi)

    # Reconstruct the formula with quantifiers at the front
    result = matrix
    for qf in reversed(qf_pre):  # Process in reverse order
        qt, name, typ = qf
        if qt == "Exists":
            result = mk_exists(name, typ, result)
        elif qt == "Forall":
            result = mk_forall(name, typ, result)

    return result


# ---------------------------------------------------------------------------
# Compare functions
# ---------------------------------------------------------------------------

def compare_expr(a: Expression, b: Expression) -> int:
    """Total order on expressions by structural hash then type-tag."""
    if a is b:
        return 0
    tag_order = {
        TrueExpr: 0, FalseExpr: 1, Var: 2, Const: 3, App: 4,
        Add: 5, Mul: 6, Div: 7, Mod: 8, Floor: 9, Neg: 10,
        Ite: 11, Select: 12, Store: 13,
        And: 14, Or: 15, Not: 16, Eq: 17, Lt: 18, Leq: 19,
        Forall: 20, Exists: 21,
    }
    ta = tag_order.get(type(a), 99)
    tb = tag_order.get(type(b), 99)
    if ta != tb:
        return -1 if ta < tb else 1
    ha = hash(a)
    hb = hash(b)
    if ha < hb:
        return -1
    if ha > hb:
        return 1
    return 0


compare_formula = compare_expr
compare_term = compare_expr


# ---------------------------------------------------------------------------
# Symbol.Set and Symbol.Map
# ---------------------------------------------------------------------------

class SymbolSet:
    """Set of unique symbols (mirrors OCaml Symbol.Set)."""

    def __init__(self, elements: Optional[Iterable[Symbol]] = None):
        self._data: Set[Symbol] = set(elements) if elements else set()

    def add(self, sym: Symbol) -> None:
        self._data.add(sym)

    def mem(self, sym: Symbol) -> bool:
        return sym in self._data

    def union(self, other: "SymbolSet") -> "SymbolSet":
        result = SymbolSet()
        result._data = self._data | other._data
        return result

    def inter(self, other: "SymbolSet") -> "SymbolSet":
        result = SymbolSet()
        result._data = self._data & other._data
        return result

    def equal(self, other: "SymbolSet") -> bool:
        return self._data == other._data

    def filter(self, pred: Callable[[Symbol], bool]) -> "SymbolSet":
        return SymbolSet(s for s in self._data if pred(s))

    def elements(self) -> List[Symbol]:
        return list(self._data)

    def enum(self) -> Iterator[Symbol]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __iter__(self) -> Iterator[Symbol]:
        return iter(self._data)

    def __contains__(self, sym: Symbol) -> bool:
        return sym in self._data


class SymbolMap:
    """Map keyed by symbols (mirrors OCaml Symbol.Map)."""

    def __init__(self, init: Optional[Dict[Symbol, Any]] = None):
        self._data: Dict[Symbol, Any] = dict(init) if init else {}

    def add(self, key: Symbol, value: Any) -> None:
        self._data[key] = value

    def find(self, key: Symbol) -> Any:
        return self._data[key]

    def mem(self, key: Symbol) -> bool:
        return key in self._data

    def remove(self, key: Symbol) -> None:
        self._data.pop(key, None)

    def keys(self) -> Iterator[Symbol]:
        return iter(self._data.keys())

    def values(self) -> Iterator[Any]:
        return iter(self._data.values())

    def enum(self) -> Iterator[Tuple[Symbol, Any]]:
        return iter(self._data.items())

    def __getitem__(self, key: Symbol) -> Any:
        return self._data[key]

    def __setitem__(self, key: Symbol, value: Any) -> None:
        self._data[key] = value

    def __contains__(self, key: Symbol) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)


# ---------------------------------------------------------------------------
# Expr submodule (expression utilities)
# ---------------------------------------------------------------------------

class _ExprModule:
    """Utilities for generic expressions (mirrors OCaml Expr module)."""

    @staticmethod
    def equal(a: Expression, b: Expression) -> bool:
        return a == b

    @staticmethod
    def compare(a: Expression, b: Expression) -> int:
        return compare_expr(a, b)

    @staticmethod
    def hash(e: Expression) -> int:
        return hash(e)

    @staticmethod
    def term_of(srk: Context, e: Expression) -> "TermExpression":
        if isinstance(e, (TrueExpr, FalseExpr, And, Or, Not, Eq, Lt, Leq, Forall, Exists)):
            raise ValueError("Expression is not a term")
        return e

    @staticmethod
    def formula_of(srk: Context, e: Expression) -> "FormulaExpression":
        if not isinstance(e, (TrueExpr, FalseExpr, And, Or, Not, Eq, Lt, Leq, Forall, Exists)):
            raise ValueError("Expression is not a formula")
        return e


Expr = _ExprModule()


# ---------------------------------------------------------------------------
# Env module (De Bruijn environments)
# ---------------------------------------------------------------------------

class Env:
    """De Bruijn environment for traversing quantified formulas.
    Mirrors OCaml's Env module: empty, push, find, enum.
    """

    def __init__(self):
        self._stack: List[Any] = []

    @staticmethod
    def empty() -> "Env":
        return Env()

    def push(self, value: Any) -> "Env":
        new_env = Env()
        new_env._stack = list(self._stack)
        new_env._stack.append(value)
        return new_env

    def find(self, idx: int) -> Any:
        if idx < 0 or idx >= len(self._stack):
            raise IndexError(f"De Bruijn index {idx} out of bounds")
        return self._stack[idx]

    def enum(self):
        return iter(self._stack)

    def __len__(self) -> int:
        return len(self._stack)


# ---------------------------------------------------------------------------
# ContextTable (weak-keyed map from contexts to values)
# ---------------------------------------------------------------------------

class ContextTable:
    """Map from Context objects to values (mirrors OCaml ContextTable)."""

    def __init__(self):
        self._data: Dict[int, Any] = {}

    def add(self, ctx: Context, value: Any) -> None:
        self._data[id(ctx)] = value

    def find(self, ctx: Context) -> Any:
        return self._data[id(ctx)]

    def mem(self, ctx: Context) -> bool:
        return id(ctx) in self._data

    def remove(self, ctx: Context) -> None:
        self._data.pop(id(ctx), None)


# ---------------------------------------------------------------------------
# Infix module (operator-style expression building)
# ---------------------------------------------------------------------------

class Infix:
    """Operator-style expression building for a given Context.

    Usage:
        inf = Infix(ctx)
        expr = inf.const(x) + inf.const(y)  # => x + y
        formula = (inf.const(x) < inf.const(y)) & ~inf.const(z).eq(0)
    """

    def __init__(self, ctx: Context):
        self.ctx = ctx

    class _InfixTerm:
        def __init__(self, expr: ArithExpression, infix: "Infix"):
            self.expr = expr
            self._infix = infix

        def __add__(self, other):
            if isinstance(other, Infix._InfixTerm):
                return Infix._InfixTerm(Add((self.expr, other.expr)), self._infix)
            return Infix._InfixTerm(
                Add((self.expr, self._infix.real(other))), self._infix
            )

        def __sub__(self, other):
            if isinstance(other, Infix._InfixTerm):
                rhs = other.expr
            else:
                rhs = self._infix.real(other)
            return Infix._InfixTerm(
                Add((self.expr, Neg(rhs))), self._infix
            )

        def __mul__(self, other):
            if isinstance(other, Infix._InfixTerm):
                return Infix._InfixTerm(Mul((self.expr, other.expr)), self._infix)
            return Infix._InfixTerm(
                Mul((self.expr, self._infix.real(other))), self._infix
            )

        def __truediv__(self, other):
            if isinstance(other, Infix._InfixTerm):
                return Infix._InfixTerm(Div(self.expr, other.expr), self._infix)
            return Infix._InfixTerm(Div(self.expr, self._infix.real(other)), self._infix)

        def __neg__(self):
            return Infix._InfixTerm(Neg(self.expr), self._infix)

        def __lt__(self, other) -> FormulaExpression:
            if isinstance(other, Infix._InfixTerm):
                return Lt(self.expr, other.expr)
            return Lt(self.expr, self._infix.real(other))

        def __le__(self, other) -> FormulaExpression:
            if isinstance(other, Infix._InfixTerm):
                return Leq(self.expr, other.expr)
            return Leq(self.expr, self._infix.real(other))

        def __gt__(self, other) -> FormulaExpression:
            if isinstance(other, Infix._InfixTerm):
                return Lt(other.expr, self.expr)
            return Lt(self._infix.real(other), self.expr)

        def __ge__(self, other) -> FormulaExpression:
            if isinstance(other, Infix._InfixTerm):
                return Leq(other.expr, self.expr)
            return Leq(self._infix.real(other), self.expr)

        def __mod__(self, other) -> "Infix._InfixTerm":
            if isinstance(other, Infix._InfixTerm):
                return Infix._InfixTerm(Mod(self.expr, other.expr), self._infix)
            return Infix._InfixTerm(Mod(self.expr, self._infix.real(other)), self._infix)

    class _InfixFormula:
        def __init__(self, expr: FormulaExpression, infix: "Infix"):
            self.expr = expr
            self._infix = infix

        def __and__(self, other: FormulaExpression) -> FormulaExpression:
            if isinstance(other, Infix._InfixFormula):
                other = other.expr
            return And((self.expr, other))

        def __or__(self, other: FormulaExpression) -> FormulaExpression:
            if isinstance(other, Infix._InfixFormula):
                other = other.expr
            return Or((self.expr, other))

        def __invert__(self) -> FormulaExpression:
            return Not(self.expr)

    def var(self, var_id: int, typ: Type = Type.REAL) -> _InfixTerm:
        return self._InfixTerm(Var(var_id, typ), self)

    def const(self, sym: Symbol) -> _InfixTerm:
        return self._InfixTerm(Const(sym), self)

    def real(self, value: float) -> Const:
        real_sym = self.ctx.mk_symbol(f"real_{value}", Type.REAL)
        return Const(real_sym)

    def tru(self) -> FormulaExpression:
        return TrueExpr()

    def fls(self) -> FormulaExpression:
        return FalseExpr()

    def formula(self, expr: FormulaExpression) -> _InfixFormula:
        return self._InfixFormula(expr, self)

    def __mod__(self, base: ArithExpression, modulus: ArithExpression) -> _InfixTerm:
        return self._InfixTerm(Mod(base, modulus), self)

    def select(self, array: ArithExpression, index: ArithExpression) -> _InfixTerm:
        return self._InfixTerm(Select(array, index), self)

    def store(self, array: ArithExpression, index: ArithExpression, value: ArithExpression) -> _InfixTerm:
        return self._InfixTerm(Store(array, index, value), self)

    def forall(self, var_name: str, var_type: Type, body: FormulaExpression) -> FormulaExpression:
        return Forall(var_name, var_type, body)

    def exists(self, var_name: str, var_type: Type, body: FormulaExpression) -> FormulaExpression:
        return Exists(var_name, var_type, body)


# ---------------------------------------------------------------------------
# Term / ArithTerm / ArrTerm — namespace modules (mirrors OCaml)
# ---------------------------------------------------------------------------

class _TermModule:
    """Namespace for term operations (mirrors OCaml ``Term`` module)."""

    @staticmethod
    def destruct(e: Expression) -> Tuple[str, Any]:
        return destruct(e)

    @staticmethod
    def eval(e: Expression) -> "Fraction":
        return eval_term(e)

    @staticmethod
    def equal(a: Expression, b: Expression) -> bool:
        return a == b

    @staticmethod
    def hash(e: Expression) -> int:
        return hash(e)

    def __str__(self) -> str:
        return "<Term>"


Term = _TermModule()


class _ArithTermModule:
    """Namespace for arithmetic term operations (mirrors OCaml ``ArithTerm``)."""

    @staticmethod
    def destruct(e: Expression) -> Tuple[str, Any]:
        return destruct(e)

    @staticmethod
    def eval(e: Expression) -> "Fraction":
        return eval_term(e)

    @staticmethod
    def equal(a: Expression, b: Expression) -> bool:
        return a == b

    def __str__(self) -> str:
        return "<ArithTerm>"


ArithTerm = _ArithTermModule()


class _ArrTermModule:
    """Namespace for array term operations (mirrors OCaml ``ArrTerm``)."""

    @staticmethod
    def destruct(e: Expression) -> Tuple[str, Any]:
        return destruct(e)

    @staticmethod
    def equal(a: Expression, b: Expression) -> bool:
        return a == b

    def __str__(self) -> str:
        return "<ArrTerm>"


ArrTerm = _ArrTermModule()


class _FormulaModule:
    """Namespace for formula operations (mirrors OCaml ``Formula`` module)."""

    @staticmethod
    def equal(a: Expression, b: Expression) -> bool:
        return a == b

    @staticmethod
    def hash(e: Expression) -> int:
        return hash(e)

    @staticmethod
    def destruct(e: Expression) -> Tuple[str, Any]:
        return destruct(e)

    @staticmethod
    def eval(algebra: Callable, formula: Expression) -> Any:
        return eval_formula(algebra, formula)

    def __str__(self) -> str:
        return "<Formula>"


Formula = _FormulaModule()


# ---------------------------------------------------------------------------
# Formula submodule operations
# ---------------------------------------------------------------------------

def existential_closure(srk: Context, phi: FormulaExpression, pred: Optional[Callable[[Symbol], bool]] = None) -> FormulaExpression:
    """Existentially quantify all free constant symbols in the formula.

    If pred is given, only symbols matching the predicate are quantified.
    """
    syms = symbols(phi)
    result = phi
    for sym in sorted(syms, key=lambda s: s.id, reverse=True):
        if pred is None or pred(sym):
            if sym.typ != Type.BOOL:
                result = mk_exists_const(srk, sym, result)
    return result


def universal_closure(srk: Context, phi: FormulaExpression, pred: Optional[Callable[[Symbol], bool]] = None) -> FormulaExpression:
    """Universally quantify all free constant symbols in the formula.

    If pred is given, only symbols matching the predicate are quantified.
    """
    syms = symbols(phi)
    result = phi
    for sym in sorted(syms, key=lambda s: s.id, reverse=True):
        if pred is None or pred(sym):
            if sym.typ != Type.BOOL:
                result = mk_forall_const(srk, sym, result)
    return result


def skolemize_free(srk: Context, phi: FormulaExpression) -> FormulaExpression:
    """Skolemize free variables: replace existentials with fresh symbols."""
    def go(e: Expression) -> Expression:
        if isinstance(e, Exists):
            sk_sym = mk_symbol(srk, f"sk_{e.var_name}", e.var_type)
            body_sub = substitute_const(
                srk,
                lambda s: mk_const(srk, sk_sym) if s == e.var_name else mk_const(srk, s),
                e.body,
            )
            return go(body_sub)
        if isinstance(e, Add): return Add(tuple(go(a) for a in e.args))
        if isinstance(e, Mul): return Mul(tuple(go(a) for a in e.args))
        if isinstance(e, Div): return Div(go(e.left), go(e.right))
        if isinstance(e, Mod): return Mod(go(e.left), go(e.right))
        if isinstance(e, Floor): return Floor(go(e.arg))
        if isinstance(e, Neg): return Neg(go(e.arg))
        if isinstance(e, Pow): return Pow(go(e.base), e.exponent)
        if isinstance(e, Ite): return Ite(go(e.condition), go(e.then_branch), go(e.else_branch))
        if isinstance(e, App): return App(e.symbol, tuple(go(a) for a in e.args))
        if isinstance(e, Select): return Select(go(e.array), go(e.index))
        if isinstance(e, Store): return Store(go(e.array), go(e.index), go(e.value))
        if isinstance(e, And): return And(tuple(go(a) for a in e.args))
        if isinstance(e, Or): return Or(tuple(go(a) for a in e.args))
        if isinstance(e, Not): return Not(go(e.arg))
        if isinstance(e, Eq): return Eq(go(e.left), go(e.right))
        if isinstance(e, Lt): return Lt(go(e.left), go(e.right))
        if isinstance(e, Leq): return Leq(go(e.left), go(e.right))
        if isinstance(e, Forall): return Forall(e.var_name, e.var_type, go(e.body))
        return e
    return go(phi)


# ---------------------------------------------------------------------------
# Formula.eval and eval_memo — structural formula evaluation
# ---------------------------------------------------------------------------

def eval_formula(
    algebra: Callable[[str, Any], Any], formula: FormulaExpression
) -> Any:
    """Evaluate a formula using an algebra for structural recursion.

    The algebra is called with (label, components) for each expression
    constructor and should return the semantic interpretation.

    Mirrors OCaml `Formula.eval`.
    """
    return _eval_formula_go(algebra, formula)


def _eval_formula_go(algebra: Callable, expr: Expression) -> Any:
    if isinstance(expr, TrueExpr):
        return algebra("Tru", expr)
    if isinstance(expr, FalseExpr):
        return algebra("Fls", expr)
    if isinstance(expr, And):
        children = tuple(_eval_formula_go(algebra, a) for a in expr.args)
        return algebra("And", children)
    if isinstance(expr, Or):
        children = tuple(_eval_formula_go(algebra, a) for a in expr.args)
        return algebra("Or", children)
    if isinstance(expr, Not):
        child = _eval_formula_go(algebra, expr.arg)
        return algebra("Not", child)
    if isinstance(expr, Eq):
        left = _eval_formula_go(algebra, expr.left)
        right = _eval_formula_go(algebra, expr.right)
        return algebra("Atom", ("Eq", left, right))
    if isinstance(expr, Lt):
        left = _eval_formula_go(algebra, expr.left)
        right = _eval_formula_go(algebra, expr.right)
        return algebra("Atom", ("Lt", left, right))
    if isinstance(expr, Leq):
        left = _eval_formula_go(algebra, expr.left)
        right = _eval_formula_go(algebra, expr.right)
        return algebra("Atom", ("Leq", left, right))
    if isinstance(expr, Ite):
        cond = _eval_formula_go(algebra, expr.condition)
        then_b = _eval_formula_go(algebra, expr.then_branch)
        else_b = _eval_formula_go(algebra, expr.else_branch)
        return algebra("Ite", (cond, then_b, else_b))
    if isinstance(expr, Forall):
        return algebra("Forall", expr)
    if isinstance(expr, Exists):
        return algebra("Exists", expr)
    if isinstance(expr, Var):
        return algebra("Var", expr)
    if isinstance(expr, Const):
        return algebra("Const", expr)
    if isinstance(expr, Add):
        children = tuple(_eval_formula_go(algebra, a) for a in expr.args)
        return algebra("Add", children)
    if isinstance(expr, Mul):
        children = tuple(_eval_formula_go(algebra, a) for a in expr.args)
        return algebra("Mul", children)
    if isinstance(expr, Div):
        left = _eval_formula_go(algebra, expr.left)
        right = _eval_formula_go(algebra, expr.right)
        return algebra("Div", (left, right))
    if isinstance(expr, Mod):
        left = _eval_formula_go(algebra, expr.left)
        right = _eval_formula_go(algebra, expr.right)
        return algebra("Mod", (left, right))
    if isinstance(expr, Floor):
        child = _eval_formula_go(algebra, expr.arg)
        return algebra("Floor", child)
    if isinstance(expr, Neg):
        child = _eval_formula_go(algebra, expr.arg)
        return algebra("Neg", child)
    if isinstance(expr, Pow):
        child = _eval_formula_go(algebra, expr.base)
        return algebra("Pow", (child, expr.exponent))
    if isinstance(expr, App):
        children = tuple(_eval_formula_go(algebra, a) for a in expr.args)
        return algebra("App", (expr.symbol, children))
    if isinstance(expr, Select):
        arr = _eval_formula_go(algebra, expr.array)
        idx = _eval_formula_go(algebra, expr.index)
        return algebra("Select", (arr, idx))
    if isinstance(expr, Store):
        arr = _eval_formula_go(algebra, expr.array)
        idx = _eval_formula_go(algebra, expr.index)
        val = _eval_formula_go(algebra, expr.value)
        return algebra("Store", (arr, idx, val))
    return algebra("Unknown", expr)


def eval_formula_memo(
    algebra: Callable[[str, Any], Any], formula: FormulaExpression
) -> Any:
    """Memoized version of eval_formula.

    Caches results by expression identity to avoid recomputation.
    """
    cache: Dict[int, Any] = {}

    def go(expr: Expression) -> Any:
        key = id(expr)
        if key in cache:
            return cache[key]
        result = _eval_formula_go(algebra, expr)
        cache[key] = result
        return result

    return go(formula)


def eval_term(term: ArithExpression) -> "Fraction":
    """Evaluate a constant arithmetic term to a rational value.

    Raises ValueError if the term contains variables or unsupported operations.

    Mirrors OCaml ``Term.eval`` / ``ArithTerm.eval``.
    """
    from fractions import Fraction as Frac

    if isinstance(term, Const):
        sym = term.symbol
        if sym.name and sym.name.startswith("real_"):
            try:
                return Frac(sym.name[5:])
            except ValueError:
                pass
        if sym.name and sym.name.startswith("int_"):
            try:
                return Frac(int(sym.name[4:]), 1)
            except ValueError:
                pass
        raise ValueError(f"Cannot evaluate symbol {sym} as a constant")

    if isinstance(term, Add):
        result = Frac(0)
        for a in term.args:
            result += eval_term(a)
        return result

    if isinstance(term, Mul):
        result = Frac(1)
        for a in term.args:
            result *= eval_term(a)
        return result

    if isinstance(term, Div):
        left = eval_term(term.left)
        right = eval_term(term.right)
        if right == 0:
            raise ZeroDivisionError("Division by zero in term evaluation")
        return left / right

    if isinstance(term, Neg):
        return -eval_term(term.arg)

    if isinstance(term, Pow):
        base = eval_term(term.base)
        return base ** term.exponent

    if isinstance(term, Floor):
        val = eval_term(term.arg)
        import math
        return Frac(math.floor(float(val)), 1)

    raise ValueError(f"Cannot evaluate expression of type {type(term).__name__}")


# ---------------------------------------------------------------------------
# Expr submodule — expanded operations
# ---------------------------------------------------------------------------

class _ExprModule:
    """Utilities for generic expressions (mirrors OCaml Expr module)."""

    @staticmethod
    def equal(a: Expression, b: Expression) -> bool:
        return a == b

    @staticmethod
    def compare(a: Expression, b: Expression) -> int:
        return compare_expr(a, b)

    @staticmethod
    def hash(e: Expression) -> int:
        return hash(e)

    @staticmethod
    def term_of(srk: Context, e: Expression) -> "TermExpression":
        if isinstance(e, (TrueExpr, FalseExpr, And, Or, Not, Eq, Lt, Leq, Forall, Exists)):
            raise ValueError("Expression is not a term")
        return e

    @staticmethod
    def formula_of(srk: Context, e: Expression) -> "FormulaExpression":
        if not isinstance(e, (TrueExpr, FalseExpr, And, Or, Not, Eq, Lt, Leq, Forall, Exists)):
            raise ValueError("Expression is not a formula")
        return e

    @staticmethod
    def refine(srk: Context, e: Expression):
        """Refine an expression into ArithTerm, ArrTerm, or Formula."""
        if isinstance(e, (TrueExpr, FalseExpr, And, Or, Not, Eq, Lt, Leq, Forall, Exists)):
            return ("Formula", e)
        if isinstance(e, Store):
            return ("ArrTerm", e)
        return ("ArithTerm", e)

    @staticmethod
    def arith_term_of(srk: Context, e: Expression) -> "ArithExpression":
        if isinstance(e, (TrueExpr, FalseExpr, And, Or, Not, Eq, Lt, Leq, Forall, Exists)):
            raise ValueError("Expression is not an arith term")
        return e

    @staticmethod
    def arr_term_of(srk: Context, e: Expression) -> "Expression":
        if not isinstance(e, Store):
            raise ValueError("Expression is not an arr term")
        return e

    @staticmethod
    def destruct_sexpr(srk: Context, e: Expression) -> Tuple[str, List[Expression]]:
        """Destruct an expression as an s-expression."""
        tag, comps = destruct(e)
        if tag == "Var":
            return ("Var", [Var(comps[0], comps[1])])
        if tag == "Const":
            return ("Const", [Const(comps)])
        if tag == "App":
            sym, args = comps
            return ("App", list(args))
        if tag == "Add":
            return ("Add", list(comps))
        if tag == "Mul":
            return ("Mul", list(comps))
        if tag == "Binop":
            op, left, right = comps
            return (op, [left, right])
        if tag == "Unop":
            op, arg = comps
            return (op, [arg])
        if tag == "Ite":
            return ("Ite", list(comps))
        if tag == "Tru":
            return ("True", [])
        if tag == "Fls":
            return ("False", [])
        if tag == "And":
            return ("And", list(comps))
        if tag == "Or":
            return ("Or", list(comps))
        if tag == "Not":
            return ("Not", [comps])
        if tag == "Atom":
            kind, op, left, right = comps
            return (op, [left, right])
        if tag == "Quantify":
            qt, name, typ, body = comps
            return (qt, [body])
        return (tag, [])

    @staticmethod
    def construct_sexpr(srk: Context, label: str, children: List[Expression]) -> Expression:
        """Construct an expression from a label and children (inverse of destruct_sexpr)."""
        if label == "Var" and isinstance(children[0], Var):
            v = children[0]
            return Var(v.var_id, v.var_type, v.name)
        if label == "Const" and children:
            return children[0]
        if label == "App" and len(children) >= 1:
            return App(children[0].symbol if isinstance(children[0], App) else mk_symbol(srk, None, Type.INT), tuple(children[1:]) if len(children) > 1 else ())
        if label == "Add":
            return Add(tuple(children))
        if label == "Mul":
            return Mul(tuple(children))
        if label == "Div" and len(children) >= 2:
            return Div(children[0], children[1])
        if label == "Mod" and len(children) >= 2:
            return Mod(children[0], children[1])
        if label == "Floor" and children:
            return Floor(children[0])
        if label == "Neg" and children:
            return Neg(children[0])
        if label == "Ite" and len(children) >= 3:
            return Ite(children[0], children[1], children[2])
        if label == "True":
            return TrueExpr()
        if label == "False":
            return FalseExpr()
        if label == "And":
            return And(tuple(children))
        if label == "Or":
            return Or(tuple(children))
        if label == "Not" and children:
            return Not(children[0])
        if label == "Eq" and len(children) >= 2:
            return Eq(children[0], children[1])
        if label == "Lt" and len(children) >= 2:
            return Lt(children[0], children[1])
        if label == "Leq" and len(children) >= 2:
            return Leq(children[0], children[1])
        if label == "Forall":
            return Forall("x", Type.INT, children[0] if children else TrueExpr())
        if label == "Exists":
            return Exists("x", Type.INT, children[0] if children else TrueExpr())
        if label == "Store" and len(children) >= 3:
            return Store(children[0], children[1], children[2])
        if label == "Select" and len(children) >= 2:
            return Select(children[0], children[1])
        return TrueExpr()

    class HT(dict):
        """Hashtable from expressions to values."""
        def add(self, key: Expression, value: Any) -> None:
            self[id(key)] = value

        def replace(self, key: Expression, value: Any) -> None:
            self[id(key)] = value

        def remove(self, key: Expression) -> None:
            self.pop(id(key), None)

        def find(self, key: Expression) -> Any:
            return self[id(key)]

        def mem(self, key: Expression) -> bool:
            return id(key) in self

        def keys(self):
            return iter(self)

        def values(self):
            return iter(super().values())

        def enum(self):
            return self.items()

    class Set:
        """Set of expressions by structural equality."""
        def __init__(self):
            self._data: Dict[int, Expression] = {}

        def add(self, e: Expression) -> None:
            self._data[hash(e)] = e

        def mem(self, e: Expression) -> bool:
            return hash(e) in self._data and self._data[hash(e)] == e

        def enum(self):
            return iter(self._data.values())

        def elements(self):
            return list(self._data.values())

        def filter(self, pred: Callable[[Expression], bool]):
            result = _ExprModule.Set()
            for e in self._data.values():
                if pred(e):
                    result.add(e)
            return result

    class Map:
        """Map keyed by expressions."""
        def __init__(self):
            self._data: Dict[int, Tuple[Expression, Any]] = {}

        def add(self, key: Expression, value: Any) -> None:
            self._data[hash(key)] = (key, value)

        def find(self, key: Expression) -> Any:
            return self._data[hash(key)][1]

        def remove(self, key: Expression) -> None:
            self._data.pop(hash(key), None)

        def mem(self, key: Expression) -> bool:
            return hash(key) in self._data

        def keys(self):
            for k, _ in self._data.values():
                yield k

        def values(self):
            for _, v in self._data.values():
                yield v

        def enum(self):
            for k, v in self._data.values():
                yield (k, v)


Expr = _ExprModule()


# ---------------------------------------------------------------------------
# pp_smtlib2 — SMTLIB2 serialization
# ---------------------------------------------------------------------------

def pp_smtlib2_gen(
    srk: Context,
    env: Optional[Env] = None,
    named: bool = False,
    strings: bool = False,
) -> Callable[[Any, Union[Expression, List[Expression]]], None]:
    """Create a function that prints expressions in SMTLIB2 format.

    Mirrors OCaml ``pp_smtlib2_gen``. Accepts optional ``named`` (add names
    to bound variables) and ``strings`` (encode symbols as strings) flags.

    Args:
        srk: The SRK context for symbol name resolution.
        env: Optional De Bruijn environment for resolving bound variables.
        named: If True, annotate bound variables with their names.
        strings: If True, encode symbols using SMTLIB2 string encoding.

    Returns:
        A function taking (file-like, expression|list) and printing to the file.
    """
    def printer(out, expr) -> None:
        if isinstance(expr, list):
            for e in expr:
                s = _smtlib2_str(e, env or Env.empty())
                out.write(s)
                out.write("\n")
        else:
            s = _smtlib2_str(expr, env or Env.empty())
            out.write(s)

    return printer


def _smtlib2_str(expr: Expression, env: Env) -> str:
    """Convert an expression to an SMTLIB2 string, with De Bruijn env support."""
    if isinstance(expr, TrueExpr):
        return "true"
    if isinstance(expr, FalseExpr):
        return "false"
    if isinstance(expr, Var):
        try:
            name = env.find(expr.var_id)
            if isinstance(name, str):
                return name
            return str(name)
        except IndexError:
            return f"x{expr.var_id}"
    if isinstance(expr, Const):
        sym = expr.symbol
        if sym.name:
            return sym.name
        return f"c{sym.id}"
    if isinstance(expr, Add):
        inner = " ".join(_smtlib2_str(a, env) for a in expr.args)
        return f"(+ {inner})"
    if isinstance(expr, Mul):
        inner = " ".join(_smtlib2_str(a, env) for a in expr.args)
        return f"(* {inner})"
    if isinstance(expr, Div):
        return f"(/ {_smtlib2_str(expr.left, env)} {_smtlib2_str(expr.right, env)})"
    if isinstance(expr, Mod):
        return f"(mod {_smtlib2_str(expr.left, env)} {_smtlib2_str(expr.right, env)})"
    if isinstance(expr, Floor):
        return f"(to_int {_smtlib2_str(expr.arg, env)})"
    if isinstance(expr, Neg):
        return f"(- {_smtlib2_str(expr.arg, env)})"
    if isinstance(expr, Pow):
        return f"(pow {_smtlib2_str(expr.base, env)} {expr.exponent})"
    if isinstance(expr, And):
        inner = " ".join(_smtlib2_str(a, env) for a in expr.args)
        return f"(and {inner})"
    if isinstance(expr, Or):
        inner = " ".join(_smtlib2_str(a, env) for a in expr.args)
        return f"(or {inner})"
    if isinstance(expr, Not):
        return f"(not {_smtlib2_str(expr.arg, env)})"
    if isinstance(expr, Eq):
        return f"(= {_smtlib2_str(expr.left, env)} {_smtlib2_str(expr.right, env)})"
    if isinstance(expr, Lt):
        return f"(< {_smtlib2_str(expr.left, env)} {_smtlib2_str(expr.right, env)})"
    if isinstance(expr, Leq):
        return f"(<= {_smtlib2_str(expr.left, env)} {_smtlib2_str(expr.right, env)})"
    if isinstance(expr, Forall):
        inner_env = env.push(expr.var_name)
        inner_name = f"({expr.var_name} {_smtlib2_type(expr.var_type)})"
        return f"(forall ({inner_name}) {_smtlib2_str(expr.body, inner_env)})"
    if isinstance(expr, Exists):
        inner_env = env.push(expr.var_name)
        inner_name = f"({expr.var_name} {_smtlib2_type(expr.var_type)})"
        return f"(exists ({inner_name}) {_smtlib2_str(expr.body, inner_env)})"
    if isinstance(expr, Ite):
        return f"(ite {_smtlib2_str(expr.condition, env)} {_smtlib2_str(expr.then_branch, env)} {_smtlib2_str(expr.else_branch, env)})"
    if isinstance(expr, App):
        sym_name = expr.symbol.name or f"fn{expr.symbol.id}"
        args = " ".join(_smtlib2_str(a, env) for a in expr.args)
        return f"({sym_name} {args})" if args else sym_name
    if isinstance(expr, Select):
        return f"(select {_smtlib2_str(expr.array, env)} {_smtlib2_str(expr.index, env)})"
    if isinstance(expr, Store):
        return f"(store {_smtlib2_str(expr.array, env)} {_smtlib2_str(expr.index, env)} {_smtlib2_str(expr.value, env)})"
    return str(expr)


def _smtlib2_type(typ: Type) -> str:
    if typ == Type.INT:
        return "Int"
    if typ == Type.REAL:
        return "Real"
    if typ == Type.BOOL:
        return "Bool"
    return "Int"


def pp_smtlib2(srk: Context, out: Any, expr: Expression) -> None:
    """Print an expression in SMTLIB2 format to the given output.

    Convenience wrapper around pp_smtlib2_gen. Matches OCaml's
    ``Formula.pp_smtlib2``.
    """
    printer = pp_smtlib2_gen(srk)
    printer(out, expr)


# ---------------------------------------------------------------------------
# Standalone top-level functions (mirror OCaml standalone API)
# ---------------------------------------------------------------------------

def int_of_symbol(sym: Symbol) -> int:
    """Get the integer representation of a symbol."""
    return sym.id


def symbol_of_int(i: int) -> Symbol:
    """Create a symbol from its integer representation."""
    return Symbol(i, None, Type.INT)


def context_stats(ctx: Context) -> Tuple[int, int, int]:
    """Return (num_sexprs, num_symbols, num_named_symbols) for a context."""
    return ctx.stats()


def pp_typ(out: Any, typ: Type) -> None:
    out.write(str(typ))


def pp_symbol(ctx: Context, out: Any, sym: Symbol) -> None:
    out.write(ctx.show_symbol(sym))


def pp_expr_unnumbered(ctx: Context, out: Any, expr: Expression, env: Optional[Env] = None) -> None:
    printer = pp_smtlib2_gen(ctx, env)
    printer(out, expr)


class _ExprMemoClass:
    """Expression memoization wrapper (mirrors OCaml ``Expr.ExprMemo.memo``)."""

    @staticmethod
    def memo(size: Optional[int] = None):
        from functools import lru_cache

        def decorator(f):
            return lru_cache(maxsize=size)(f) if size else lru_cache(maxsize=None)(f)
        return decorator


ExprMemo = _ExprMemoClass()


# Nest Symbol.Set/Map inside Symbol class
Symbol.Set = SymbolSet
Symbol.Map = SymbolMap


def eval_partial_term(
    term: ArithExpression, env: Dict[int, "Fraction"]
) -> "Fraction":
    """Evaluate a term with a partial assignment of variables.

    Mirrors OCaml ``Term.eval_partial``. Variables not in env
    raise ValueError.
    """
    from fractions import Fraction as Frac

    if isinstance(term, Var):
        if term.var_id in env:
            return env[term.var_id]
        raise ValueError(f"Variable {term.var_id} not in partial assignment")
    if isinstance(term, Const):
        return eval_term(term)
    if isinstance(term, Add):
        return sum(eval_partial_term(a, env) for a in term.args)
    if isinstance(term, Mul):
        result = Frac(1)
        for a in term.args:
            result *= eval_partial_term(a, env)
        return result
    if isinstance(term, Div):
        return eval_partial_term(term.left, env) / eval_partial_term(term.right, env)
    if isinstance(term, Neg):
        return -eval_partial_term(term.arg, env)
    if isinstance(term, Pow):
        base = eval_partial_term(term.base, env)
        return base ** term.exponent
    return eval_term(term)


def destruct_formula(f: FormulaExpression) -> Tuple[str, Any]:
    """Destruct a formula (mirrors OCaml ``Formula.destruct``)."""
    return destruct(f)


def construct_formula(srk: Context, label: str, children: List[Expression]) -> FormulaExpression:
    """Construct a formula from label and children (mirrors OCaml ``Formula.construct``)."""
    return Expr.construct_sexpr(srk, label, children)


def make_context() -> Context:
    return Context()


def make_simplifying_context() -> "SimplifyingContext":
    from .srkAst import SimplifyingContext
    return SimplifyingContext()
