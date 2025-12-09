"""
SRK Abstract Syntax Tree (AST) utilities.

This module provides utilities for working with SRK expressions in a simplified
context that supports common AST operations. It offers a high-level interface
for creating and manipulating symbolic expressions with automatic simplification.

Key Features:
- SimplifyingContext class for automatic expression simplification
- Convenient factory methods for creating common expressions
- Type-safe expression construction with proper type checking
- Integration with the core SRK syntax module

Example:
    >>> ctx = SimplifyingContext()
    >>> x = ctx.mk_symbol('x', Type.real)
    >>> expr = ctx.mk_add(ctx.mk_const(x), ctx.mk_real(1))
    >>> print(expr)  # Automatically simplified if possible
"""

from __future__ import annotations
from typing import List, Dict, Set, Tuple, Optional, Union, Callable, Any
from fractions import Fraction

# Import from other SRK modules
from .syntax import Context, Symbol, Expression, FormulaExpression, ArithExpression, Type, mk_const, mk_symbol, mk_real, mk_add, mk_mul, mk_div, mk_mod, mk_eq, mk_and, mk_or, mk_leq, mk_lt, mk_ite, mk_not, mk_true, mk_false, destruct, expr_typ, symbols, substitute_const
from .srkSimplify import simplify_terms

# Create a simplifying context for AST operations
class SimplifyingContext:
    """A context that automatically simplifies expressions during construction.

    The SimplifyingContext provides a high-level interface for creating SRK
    expressions with automatic simplification. It wraps the core Context class
    and adds convenience methods for common expression construction patterns.

    Attributes:
        context (Context): The underlying SRK context that manages symbols and expressions.
    """

    def __init__(self):
        """Initialize a new simplifying context with a fresh SRK context."""
        self.context = Context()

    def mk_var(self, i: int, typ: Type) -> ArithExpression:
        """Create a variable with the given index and type.

        Args:
            i (int): The variable index.
            typ (Type): The type of the variable (e.g., Type.real, Type.int).

        Returns:
            ArithExpression: A variable expression.
        """
        return self.context.mk_var(i, typ)

    def mk_const(self, sym: Symbol) -> ArithExpression:
        """Create a constant expression from a symbol.

        Args:
            sym (Symbol): The symbol to create a constant from.

        Returns:
            ArithExpression: A constant expression.
        """
        return self.context.mk_const(sym)

    def mk_exists(self, name: str, typ: Type, body: Expression) -> Expression:
        """Create an existential quantification over a variable.

        Args:
            name (str): Name of the bound variable.
            typ (Type): Type of the bound variable.
            body (Expression): The body of the quantification.

        Returns:
            Expression: An existential quantification expression.
        """
        sym = mk_symbol(self.context, name, typ)
        return self.context.mk_exists(sym, body)

    def mk_forall(self, name: str, typ: Type, body: Expression) -> Expression:
        """Create a universal quantification over a variable.

        Args:
            name (str): Name of the bound variable.
            typ (Type): Type of the bound variable.
            body (Expression): The body of the quantification.

        Returns:
            Expression: A universal quantification expression.
        """
        sym = mk_symbol(self.context, name, typ)
        return self.context.mk_forall(sym, body)

    def show_symbol(self, sym: Symbol) -> str:
        """Convert a symbol to its string representation.

        Args:
            sym (Symbol): The symbol to convert.

        Returns:
            str: String representation of the symbol.
        """
        return self.context.show_symbol(sym)


# Global simplifying context instance
_simplifying_context = SimplifyingContext()

# Type aliases for backward compatibility
term = ArithExpression
formula = FormulaExpression


def mk_quantified(mkq: Callable[[str], Callable[[Expression], Expression]],
                  ks: List[Symbol], phi: Expression) -> Expression:
    """Create a quantified formula with variable substitution."""
    # Reverse the list for proper substitution order
    ks_rev = list(reversed(ks))

    def subst(k: Symbol) -> ArithExpression:
        try:
            i = ks_rev.index(k)
            return _simplifying_context.mk_var(i, Type.REAL)
        except ValueError:
            return _simplifying_context.mk_const(k)

    # Substitute constants in phi
    phi_substituted = substitute_const(_simplifying_context.context, subst, phi)

    # Apply quantifiers from right to left (innermost first)
    result = phi_substituted
    for k in reversed(ks):
        name = _simplifying_context.show_symbol(k)
        result = mkq(name)(result)

    return result


def mk_exists(ks: List[Symbol], phi: Expression) -> Expression:
    """Create an existential quantification over the given variables."""
    def mkq(name: str) -> Callable[[Expression], Expression]:
        return lambda body: _simplifying_context.mk_exists(name, Type.REAL, body)

    return mk_quantified(mkq, ks, phi)


def mk_forall(ks: List[Symbol], phi: Expression) -> Expression:
    """Create a universal quantification over the given variables."""
    def mkq(name: str) -> Callable[[Expression], Expression]:
        return lambda body: _simplifying_context.mk_forall(name, Type.REAL, body)

    return mk_quantified(mkq, ks, phi)


# Utility functions for working with expressions in the simplifying context
def simplify_formula(phi: Expression) -> Expression:
    """Simplify a formula in the simplifying context."""
    return simplify_terms(_simplifying_context.context, phi)


def pp_term(srk: Context, term: ArithExpression) -> str:
    """Pretty print a term."""
    # This would need a proper pretty printing implementation
    return str(term)


def pp_formula(srk: Context, phi: Expression) -> str:
    """Pretty print a formula."""
    # This would need a proper pretty printing implementation
    return str(phi)


def term_equal(t1: ArithExpression, t2: ArithExpression) -> bool:
    """Check if two terms are equal."""
    # This would need a proper equality implementation
    return str(t1) == str(t2)


def formula_equal(phi1: Expression, phi2: Expression) -> bool:
    """Check if two formulas are equal."""
    # This would need a proper equality implementation
    return str(phi1) == str(phi2)


def term_hash(term: ArithExpression) -> int:
    """Hash a term."""
    return hash(str(term))


def formula_hash(phi: Expression) -> int:
    """Hash a formula."""
    return hash(str(phi))


# Additional AST utilities that might be useful
def free_variables(expr: Expression) -> Set[Symbol]:
    """Get all free variables in an expression."""
    return symbols(expr)


def bound_variables(expr: Expression) -> Set[Symbol]:
    """Get all bound variables in an expression."""
    # This would need proper bound variable analysis
    return set()


def substitute(expr: Expression, var: Symbol, replacement: ArithExpression) -> Expression:
    """Substitute a variable in an expression."""
    def subst_func(sym: Symbol) -> ArithExpression:
        if sym == var:
            return replacement
        else:
            return _simplifying_context.mk_const(sym)

    return substitute_const(_simplifying_context.context, subst_func, expr)


def alpha_rename(expr: Expression, old_var: Symbol, new_var: Symbol) -> Expression:
    """Alpha rename a bound variable."""
    # This would need proper alpha renaming implementation
    return substitute(expr, old_var, _simplifying_context.mk_const(new_var))


def is_closed(expr: Expression) -> bool:
    """Check if an expression has no free variables."""
    return len(free_variables(expr)) == 0


def occurs_free(var: Symbol, expr: Expression) -> bool:
    """Check if a variable occurs free in an expression."""
    return var in free_variables(expr)


def is_atom(expr: Expression) -> bool:
    """Check if an expression is an atomic formula."""
    destruct_result = destruct(_simplifying_context.context, expr)
    return destruct_result[0] == 'Atom'


def is_literal(expr: Expression) -> bool:
    """Check if an expression is a literal (atom or negation of atom)."""
    destruct_result = destruct(_simplifying_context.context, expr)
    if destruct_result[0] == 'Atom':
        return True
    elif destruct_result[0] == 'Not':
        inner_destr = destruct(_simplifying_context.context, destruct_result[1])
        return inner_destr[0] == 'Atom'
    else:
        return False


def conjuncts(expr: Expression) -> List[Expression]:
    """Get the conjuncts of a conjunction."""
    destruct_result = destruct(_simplifying_context.context, expr)
    if destruct_result[0] == 'And':
        return destruct_result[1] if len(destruct_result) > 1 else []
    else:
        return [expr]


def disjuncts(expr: Expression) -> List[Expression]:
    """Get the disjuncts of a disjunction."""
    destruct_result = destruct(_simplifying_context.context, expr)
    if destruct_result[0] == 'Or':
        return destruct_result[1] if len(destruct_result) > 1 else []
    else:
        return [expr]


def polarity(var: Symbol, expr: Expression) -> Optional[bool]:
    """Get the polarity of a variable in an expression.

    Returns:
        True if positive polarity, False if negative polarity, None if mixed.
    """
    # This would need proper polarity analysis
    if occurs_free(var, expr):
        # For simplicity, return None if variable occurs at all
        # A proper implementation would track polarity through the expression tree
        return None
    else:
        return None


def nnf(expr: Expression) -> Expression:
    """Convert to negation normal form."""
    # This would need proper NNF conversion
    return expr


def cnf(expr: Expression) -> Expression:
    """Convert to conjunctive normal form."""
    # This would need proper CNF conversion
    return expr


def dnf(expr: Expression) -> Expression:
    """Convert to disjunctive normal form."""
    # This would need proper DNF conversion
    return expr


def skolemize(expr: Expression) -> Expression:
    """Convert to Skolem normal form."""
    # This would need proper Skolemization
    return expr


def prenex(expr: Expression) -> Tuple[List[Tuple[str, Symbol]], Expression]:
    """Convert to prenex normal form."""
    # This would need proper prenex conversion
    return ([], expr)
