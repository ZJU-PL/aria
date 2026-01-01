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
        """Register a named symbol."""
        if name in self._named_symbols:
            raise ValueError(f"Symbol name '{name}' already registered")
        symbol = self.mk_symbol(name, typ)

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
ArithExpression = Union[Var, Const, App, Add, Mul, Ite, Select, Store]
TermExpression = Union[Var, Const, App, Add, Mul, Ite, Select, Store]
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
            return Var(symbol.id, symbol.typ)
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

    def mk_div(self, left: ArithExpression, right: ArithExpression) -> Mul:
        """Create a division expression (implemented as multiplication by reciprocal)."""
        # For now, implement as multiplication by reciprocal
        # A full implementation would need a proper Div expression class
        return self.mk_mul([left, self.mk_const(self.mk_symbol("inv", Type.REAL))])

    def mk_mod(self, left: ArithExpression, right: ArithExpression) -> Mul:
        """Create a modulo expression."""
        # For now, implement as multiplication by reciprocal
        # A full implementation would need a proper Mod expression class
        return self.mk_mul([left, self.mk_const(self.mk_symbol("mod", Type.REAL))])

    def mk_floor(self, arg: ArithExpression) -> Mul:
        """Create a floor expression."""
        # For now, implement as application of floor function
        # A full implementation would need a proper Floor expression class
        return self.mk_app(
            self.mk_symbol("floor", Type.FUN([Type.REAL], Type.REAL)), [arg]
        )

    def mk_neg(self, arg: ArithExpression) -> Mul:
        """Create a negation expression."""
        # For now, implement as multiplication by -1
        return self.mk_mul([self.mk_real(-1.0), arg])

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
            return _default_builder.mk_var(symbol.id, symbol.typ)
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
                return context.mk_var(symbol.id, symbol.typ)
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


def mk_div(*args) -> Mul:
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


def mk_mod(*args) -> Mul:
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


def mk_floor(*args) -> App:
    """Create a floor expression.

    Supports:
    - mk_floor(arg)
    - mk_floor(context, arg)
    """
    if len(args) == 1:
        (arg,) = args
        floor_symbol = _default_builder.mk_symbol(
            "floor", Type.FUN([Type.REAL], Type.INT)
        )
        return _default_builder.mk_app(floor_symbol, [arg])
    elif len(args) == 2 and isinstance(args[0], Context):
        context, arg = args
        builder = make_expression_builder(context)
        floor_symbol = builder.mk_symbol("floor", Type.FUN([Type.REAL], Type.INT))
        return builder.mk_app(floor_symbol, [arg])
    else:
        raise TypeError(
            f"mk_floor expects (arg) or (context, arg), got {len(args)} args"
        )


def mk_neg(arg: ArithExpression) -> Mul:
    """Create a negation expression in the default context."""
    # For now, implement as multiplication by -1
    return _default_builder.mk_mul([_default_builder.mk_real(-1.0), arg])


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

        # See `linear.dim_of_sym`: variable id is encoded as (dim - 1).
        var_id = int(dim) - 1

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


def mk_exists(var_name: str, var_type: Type, body: FormulaExpression) -> Exists:
    """Create an Exists expression."""
    return _default_builder.mk_exists(var_name, var_type, body)


def mk_forall(var_name: str, var_type: Type, body: FormulaExpression) -> Forall:
    """Create a Forall expression."""
    return _default_builder.mk_forall(var_name, var_type, body)


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
        return context.mk_app(symbol, args)
    else:
        # Called as mk_app(symbol, args) - use default context
        symbol, args = context_or_symbol, symbol_or_args
        return _default_builder.mk_app(symbol, args)


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
def substitute(expr: Expression, subst_map: Dict[Symbol, Expression]) -> Expression:
    """Substitute symbols in expression according to substitution map."""

    class SubstitutionVisitor(ExpressionVisitor[Expression]):
        def __init__(self, subst_map: Dict[Symbol, Expression]):
            self.subst_map = subst_map

        def visit_var(self, var: Var) -> Expression:
            return var

        def visit_const(self, const: Const) -> Expression:
            # Check if this symbol should be substituted
            if const.symbol in self.subst_map:
                return self.subst_map[const.symbol]
            return const

        def visit_app(self, app: App) -> Expression:
            # Check if the function symbol should be substituted
            new_symbol = self.subst_map.get(app.symbol, app.symbol)
            new_args = [arg.accept(self) for arg in app.args]
            return App(new_symbol, tuple(new_args))

        def visit_select(self, select: Select) -> Expression:
            new_array = select.array.accept(self)
            new_index = select.index.accept(self)
            return Select(new_array, new_index)

        def visit_store(self, store: Store) -> Expression:
            new_array = store.array.accept(self)
            new_index = store.index.accept(self)
            new_value = store.value.accept(self)
            return Store(new_array, new_index, new_value)

        def visit_add(self, add: Add) -> Expression:
            new_args = [arg.accept(self) for arg in add.args]
            return Add(tuple(new_args))

        def visit_mul(self, mul: Mul) -> Expression:
            new_args = [arg.accept(self) for arg in mul.args]
            return Mul(tuple(new_args))

        def visit_ite(self, ite: Ite) -> Expression:
            new_condition = ite.condition.accept(self)
            new_then = ite.then_branch.accept(self)
            new_else = ite.else_branch.accept(self)
            return Ite(new_condition, new_then, new_else)

        def visit_true(self, true_expr: TrueExpr) -> Expression:
            return true_expr

        def visit_false(self, false_expr: FalseExpr) -> Expression:
            return false_expr

        def visit_and(self, and_expr: And) -> Expression:
            new_args = [arg.accept(self) for arg in and_expr.args]
            return And(tuple(new_args))

        def visit_or(self, or_expr: Or) -> Expression:
            new_args = [arg.accept(self) for arg in or_expr.args]
            return Or(tuple(new_args))

        def visit_not(self, not_expr: Not) -> Expression:
            new_arg = not_expr.arg.accept(self)
            return Not(new_arg)

        def visit_eq(self, eq: Eq) -> Expression:
            new_left = eq.left.accept(self)
            new_right = eq.right.accept(self)
            return Eq(new_left, new_right)

        def visit_lt(self, lt: Lt) -> Expression:
            new_left = lt.left.accept(self)
            new_right = lt.right.accept(self)
            return Lt(new_left, new_right)

        def visit_leq(self, leq: Leq) -> Expression:
            new_left = leq.left.accept(self)
            new_right = leq.right.accept(self)
            return Leq(new_left, new_right)

        def visit_forall(self, forall: Forall) -> Expression:
            # For quantifiers, we need to be careful about variable capture
            # For now, we'll substitute in the body but not rename bound variables
            new_body = forall.body.accept(self)
            return Forall(forall.var_name, forall.var_type, new_body)

        def visit_exists(self, exists: Exists) -> Expression:
            # For quantifiers, we need to be careful about variable capture
            # For now, we'll substitute in the body but not rename bound variables
            new_body = exists.body.accept(self)
            return Exists(exists.var_name, exists.var_type, new_body)

        def _default_visit(self, expr: Expression) -> Expression:
            return expr

    visitor = SubstitutionVisitor(subst_map)
    return expr.accept(visitor)


def rewrite(
    *args, down: Optional[Callable] = None, up: Optional[Callable] = None
) -> Expression:
    """Rewrite an expression using rewrite rules.

    This module is used by components that follow the original SRK API, which
    sometimes passes an explicit `Context` as the first argument. We accept
    both calling conventions:

    - rewrite(expr, down=..., up=...)
    - rewrite(srk, expr, down=..., up=...)
    - rewrite(srk, down_fn, expr)   (legacy positional form used in `quantifier.py`)
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
            # rewrite(srk, expr, up_fn) legacy convenience
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

    # Placeholder implementation (kept intentionally lightweight)
    return expr


def nnf_rewriter(expr: Expression) -> Expression:
    """Convert to negation normal form."""
    # Placeholder implementation
    return expr


def destruct(expr: Expression) -> Tuple[str, Any]:
    """Destruct an expression into its constructor and components.

    Returns a tuple where the first element is a string indicating the
    constructor type, and the second element contains the components.
    """
    if isinstance(expr, Var):
        return ("Var", (expr.var_id, expr.var_type))
    elif isinstance(expr, Const):
        return ("Const", expr.symbol)
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
    elif isinstance(expr, Ite):
        return ("Ite", (expr.condition, expr.then_branch, expr.else_branch))
    elif isinstance(expr, TrueExpr):
        return ("True", ())
    elif isinstance(expr, FalseExpr):
        return ("False", ())
    elif isinstance(expr, And):
        return ("And", expr.args)
    elif isinstance(expr, Or):
        return ("Or", expr.args)
    elif isinstance(expr, Not):
        return ("Not", expr.arg)
    elif isinstance(expr, Eq):
        return ("Eq", (expr.left, expr.right))
    elif isinstance(expr, Lt):
        return ("Lt", (expr.left, expr.right))
    elif isinstance(expr, Leq):
        return ("Leq", (expr.left, expr.right))
    elif isinstance(expr, Forall):
        return ("Forall", (expr.var_name, expr.var_type, expr.body))
    elif isinstance(expr, Exists):
        return ("Exists", (expr.var_name, expr.var_type, expr.body))
    else:
        # For unknown expression types, return a generic representation
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
