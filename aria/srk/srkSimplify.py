"""
Expression simplification algorithms.

This module implements algorithms for simplifying symbolic expressions,
including constant folding, algebraic simplifications, and logical simplifications.
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple, Optional, Union, Any, Callable
from fractions import Fraction
from dataclasses import dataclass, field

from aria.srk.syntax import (
    Context, Symbol, Type, Expression, FormulaExpression, ArithExpression,
    ExpressionBuilder, Eq, Lt, Leq, And, Or, Not, TrueExpr, FalseExpr,
    Add, Mul, Var, Const, Ite
)
from aria.srk.cache import LRUCache

__all__ = [
    # Main classes
    'Simplifier',
    'NNFConverter',
    'CNFConverter',
    'ExpressionSimplifier',
    'RationalTermContext',
    'RationalTerm',
    'Nonlinear',

    # Factory functions
    'make_simplifier',
    'make_nnf_converter',
    'make_cnf_converter',
    'make_expression_simplifier',

    # Utility functions
    'simplify_expression',
    'to_negation_normal_form',
    'to_conjunctive_normal_form',
    'eliminate_ite_expressions',
    'of_term',
    'term_of',
    'simplify_term',
    'simplify_terms_rewriter',
    'simplify_terms',
    'purify_rewriter',
    'purify',
    'partition_implicant',
    'simplify_conjunction',
    'isolate_linear',
    'simplify_dda',
    'destruct_idiv',
    'idiv_to_ite',
    'eliminate_idiv',
    'purify_floor',
    'eliminate_floor',
    'simplify_integer_atom',

    # Re-export syntax types for convenience
    'Context',
    'Symbol',
    'Type',
    'Expression',
    'FormulaExpression',
    'ArithExpression',
    'ExpressionBuilder',
    'Eq',
    'Lt',
    'Leq',
    'And',
    'Or',
    'Not',
    'TrueExpr',
    'FalseExpr',
    'Add',
    'Mul',
    'Var',
    'Const',
]


class Simplifier:
    """Expression simplifier."""

    def __init__(self, context: Context):
        self.context = context
        self.cache = LRUCache(max_size=1000)

    def simplify(self, expression: Expression) -> Expression:
        """Simplify an expression."""
        # Check cache first
        cache_key = id(expression)
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        # Simplify recursively
        simplified = self._simplify_recursive(expression)

        # Cache result
        self.cache.put(cache_key, simplified)

        return simplified

    def _simplify_recursive(self, expression: Expression) -> Expression:
        """Recursive simplification."""
        if isinstance(expression, (Var, Const)):
            return expression

        elif isinstance(expression, Add):
            return self._simplify_add(expression)

        elif isinstance(expression, Mul):
            return self._simplify_mul(expression)

        elif isinstance(expression, Eq):
            return self._simplify_eq(expression)

        elif isinstance(expression, Lt):
            return self._simplify_lt(expression)

        elif isinstance(expression, Leq):
            return self._simplify_leq(expression)

        elif isinstance(expression, And):
            return self._simplify_and(expression)

        elif isinstance(expression, Or):
            return self._simplify_or(expression)

        elif isinstance(expression, Not):
            return self._simplify_not(expression)

        elif isinstance(expression, TrueExpr):
            return expression

        elif isinstance(expression, FalseExpr):
            return expression

        else:
            return expression

    def _simplify_add(self, add_expr: Add) -> Expression:
        """Simplify addition expression."""
        builder = ExpressionBuilder(self.context)

        # Simplify arguments
        simplified_args = [self._simplify_recursive(arg) for arg in add_expr.args]

        # Filter out zero terms
        non_zero_args = [arg for arg in simplified_args if not self._is_zero(arg)]

        if not non_zero_args:
            # All terms are zero, result is zero
            return builder.mk_var(0, Type.INT)  # Placeholder zero

        elif len(non_zero_args) == 1:
            # Only one term, return it
            return non_zero_args[0]

        else:
            # Multiple terms, check for constant folding
            constants = []
            variables = []

            for arg in non_zero_args:
                if self._is_constant(arg):
                    constants.append(self._get_constant_value(arg))
                else:
                    variables.append(arg)

            # Combine constants
            if constants:
                constant_sum = sum(constants)
                if constant_sum != 0:
                    if variables:
                        # Add constant and variables
                        all_args = variables + [self._make_constant(constant_sum)]
                        return builder.mk_add(all_args)
                    else:
                        # Only constant
                        return self._make_constant(constant_sum)
                else:
                    # Constant sum is zero, just return variables
                    if variables:
                        return builder.mk_add(variables)
                    else:
                        return builder.mk_var(0, Type.INT)  # Zero
            else:
                # No constants, just return simplified variables
                return builder.mk_add(variables)

    def _simplify_mul(self, mul_expr: Mul) -> Expression:
        """Simplify multiplication expression."""
        builder = ExpressionBuilder(self.context)

        # Simplify arguments
        simplified_args = [self._simplify_recursive(arg) for arg in mul_expr.args]

        # Filter out 1 and 0 terms
        filtered_args = []
        for arg in simplified_args:
            if self._is_one(arg):
                continue  # Skip 1
            elif self._is_zero(arg):
                # If any term is zero, result is zero
                return builder.mk_var(0, Type.INT)  # Zero
            else:
                filtered_args.append(arg)

        if not filtered_args:
            # All terms were 1 or 0
            return builder.mk_var(1, Type.INT)  # One

        elif len(filtered_args) == 1:
            # Only one term
            return filtered_args[0]

        else:
            # Multiple terms, check for constant folding
            constants = []
            variables = []

            for arg in filtered_args:
                if self._is_constant(arg):
                    constants.append(self._get_constant_value(arg))
                else:
                    variables.append(arg)

            # Combine constants
            if constants:
                constant_product = 1
                for c in constants:
                    constant_product *= c

                if constant_product != 1:
                    if variables:
                        # Multiply constant and variables
                        all_args = variables + [self._make_constant(constant_product)]
                        return builder.mk_mul(all_args)
                    else:
                        # Only constant
                        return self._make_constant(constant_product)
                else:
                    # Constant product is 1, just return variables
                    if variables:
                        return builder.mk_mul(variables)
                    else:
                        return builder.mk_var(1, Type.INT)  # One
            else:
                # No constants
                return builder.mk_mul(filtered_args)

    def _simplify_eq(self, eq_expr: Eq) -> Expression:
        """Simplify equality expression."""
        builder = ExpressionBuilder(self.context)

        left = self._simplify_recursive(eq_expr.left)
        right = self._simplify_recursive(eq_expr.right)

        # Check if both sides are the same constant
        if self._is_constant(left) and self._is_constant(right):
            left_val = self._get_constant_value(left)
            right_val = self._get_constant_value(right)
            if left_val == right_val:
                return builder.mk_true()
            else:
                return builder.mk_false()

        # Check if both sides are the same expression
        if left == right:
            return builder.mk_true()

        return builder.mk_eq(left, right)

    def _simplify_lt(self, lt_expr: Lt) -> Expression:
        """Simplify less-than expression."""
        builder = ExpressionBuilder(self.context)

        left = self._simplify_recursive(lt_expr.left)
        right = self._simplify_recursive(lt_expr.right)

        # Check if both sides are constants
        if self._is_constant(left) and self._is_constant(right):
            left_val = self._get_constant_value(left)
            right_val = self._get_constant_value(right)
            if left_val < right_val:
                return builder.mk_true()
            else:
                return builder.mk_false()

        return builder.mk_lt(left, right)

    def _simplify_leq(self, leq_expr: Leq) -> Expression:
        """Simplify less-than-or-equal expression."""
        builder = ExpressionBuilder(self.context)

        left = self._simplify_recursive(leq_expr.left)
        right = self._simplify_recursive(leq_expr.right)

        # Check if both sides are constants
        if self._is_constant(left) and self._is_constant(right):
            left_val = self._get_constant_value(left)
            right_val = self._get_constant_value(right)
            if left_val <= right_val:
                return builder.mk_true()
            else:
                return builder.mk_false()

        return builder.mk_leq(left, right)

    def _simplify_and(self, and_expr: And) -> Expression:
        """Simplify conjunction."""
        builder = ExpressionBuilder(self.context)

        # Simplify arguments
        simplified_args = [self._simplify_recursive(arg) for arg in and_expr.args]

        # Filter out true terms and check for false
        filtered_args = []
        for arg in simplified_args:
            if isinstance(arg, FalseExpr):
                # If any argument is false, result is false
                return builder.mk_false()
            elif not isinstance(arg, TrueExpr):
                # Keep non-true arguments
                filtered_args.append(arg)

        if not filtered_args:
            # All arguments were true
            return builder.mk_true()
        elif len(filtered_args) == 1:
            # Only one argument
            return filtered_args[0]
        else:
            # Multiple arguments
            return builder.mk_and(filtered_args)

    def _simplify_or(self, or_expr: Or) -> Expression:
        """Simplify disjunction."""
        builder = ExpressionBuilder(self.context)

        # Simplify arguments
        simplified_args = [self._simplify_recursive(arg) for arg in or_expr.args]

        # Filter out false terms and check for true
        filtered_args = []
        for arg in simplified_args:
            if isinstance(arg, TrueExpr):
                # If any argument is true, result is true
                return builder.mk_true()
            elif not isinstance(arg, FalseExpr):
                # Keep non-false arguments
                filtered_args.append(arg)

        if not filtered_args:
            # All arguments were false
            return builder.mk_false()
        elif len(filtered_args) == 1:
            # Only one argument
            return filtered_args[0]
        else:
            # Multiple arguments
            return builder.mk_or(filtered_args)

    def _simplify_not(self, not_expr: Not) -> Expression:
        """Simplify negation."""
        builder = ExpressionBuilder(self.context)

        operand = self._simplify_recursive(not_expr.arg)

        if isinstance(operand, TrueExpr):
            return builder.mk_false()
        elif isinstance(operand, FalseExpr):
            return builder.mk_true()
        elif isinstance(operand, Not):
            # Double negation
            return operand.arg
        else:
            return builder.mk_not(operand)

    def _is_zero(self, expression: Expression) -> bool:
        """Check if expression represents zero."""
        if isinstance(expression, Const):
            return expression.symbol.typ == Type.INT and expression.value == 0
        return False

    def _is_one(self, expression: Expression) -> bool:
        """Check if expression represents one."""
        if isinstance(expression, Const):
            return expression.symbol.typ == Type.INT and expression.value == 1
        return False

    def _is_constant(self, expression: Expression) -> bool:
        """Check if expression is a constant."""
        return isinstance(expression, Const)

    def _get_constant_value(self, expression: Expression) -> Fraction:
        """Get the constant value of an expression."""
        if isinstance(expression, Const):
            return Fraction(expression.value)
        else:
            raise ValueError("Expression is not a constant")

    def _make_constant(self, value: Fraction) -> Expression:
        """Create a constant expression."""
        builder = ExpressionBuilder(self.context)
        # This is a simplified implementation
        # In practice, we'd need to create actual constant symbols
        return builder.mk_var(0, Type.INT)  # Placeholder


class NNFConverter:
    """Convert expressions to Negation Normal Form (NNF)."""

    def __init__(self, context: Context):
        self.context = context

    def to_nnf(self, expression: Expression) -> Expression:
        """Convert expression to NNF."""
        return self._to_nnf_recursive(expression)

    def _to_nnf_recursive(self, expression: Expression) -> Expression:
        """Recursive NNF conversion."""
        builder = ExpressionBuilder(self.context)

        if isinstance(expression, Not):
            return self._push_not_down(expression.arg)

        elif isinstance(expression, And):
            # Simplify arguments and reconstruct
            simplified_args = [self._to_nnf_recursive(arg) for arg in expression.args]
            return builder.mk_and(simplified_args)

        elif isinstance(expression, Or):
            # Simplify arguments and reconstruct
            simplified_args = [self._to_nnf_recursive(arg) for arg in expression.args]
            return builder.mk_or(simplified_args)

        else:
            return expression

    def _push_not_down(self, expression: Expression) -> Expression:
        """Push negation down through the expression."""
        builder = ExpressionBuilder(self.context)

        if isinstance(expression, Not):
            # Double negation
            return self._to_nnf_recursive(expression.arg)

        elif isinstance(expression, And):
            # ¬(A ∧ B) = ¬A ∨ ¬B
            negated_args = [self._push_not_down(builder.mk_not(arg)) for arg in expression.args]
            return builder.mk_or(negated_args)

        elif isinstance(expression, Or):
            # ¬(A ∨ B) = ¬A ∧ ¬B
            negated_args = [self._push_not_down(builder.mk_not(arg)) for arg in expression.args]
            return builder.mk_and(negated_args)

        elif isinstance(expression, TrueExpr):
            return builder.mk_false()

        elif isinstance(expression, FalseExpr):
            return builder.mk_true()

        else:
            # For atomic formulas, just add negation
            return builder.mk_not(expression)


class CNFConverter:
    """Convert expressions to Conjunctive Normal Form (CNF)."""

    def __init__(self, context: Context):
        self.context = context
        self.nnf_converter = NNFConverter(context)

    def to_cnf(self, expression: Expression) -> Expression:
        """Convert expression to CNF."""
        # First convert to NNF
        nnf_expr = self.nnf_converter.to_nnf(expression)

        # Then convert to CNF
        return self._to_cnf_recursive(nnf_expr)

    def _to_cnf_recursive(self, expression: Expression) -> Expression:
        """Recursive CNF conversion."""
        builder = ExpressionBuilder(self.context)

        if isinstance(expression, Or):
            # Distribute over conjunctions
            # (A ∨ B) ∧ C = (A ∧ C) ∨ (B ∧ C)
            # This is a simplified implementation

            # For now, just return the expression as-is
            # A full implementation would need to handle distribution properly
            return expression

        elif isinstance(expression, And):
            # Convert each conjunct to CNF
            cnf_args = [self._to_cnf_recursive(arg) for arg in expression.args]
            return builder.mk_and(cnf_args)

        else:
            return expression


class ExpressionSimplifier:
    """Comprehensive expression simplifier."""

    def __init__(self, context: Context):
        self.context = context
        self.simplifier = Simplifier(context)
        self.nnf_converter = NNFConverter(context)
        self.cnf_converter = CNFConverter(context)

    def simplify(self, expression: Expression) -> Expression:
        """Fully simplify an expression."""
        return self.simplifier.simplify(expression)

    def to_nnf(self, expression: Expression) -> Expression:
        """Convert to Negation Normal Form."""
        return self.nnf_converter.to_nnf(expression)

    def to_cnf(self, expression: Expression) -> Expression:
        """Convert to Conjunctive Normal Form."""
        return self.cnf_converter.to_cnf(expression)

    def eliminate_ite(self, expression: Expression) -> Expression:
        """Eliminate if-then-else expressions."""
        # ITE elimination: ite(c, t, e) = (c ∧ t) ∨ (¬c ∧ e)
        if isinstance(expression, Ite):
            builder = ExpressionBuilder(self.context)

            c, t, e = expression.condition, expression.then_branch, expression.else_branch

            # Recursively eliminate ITE in subexpressions
            c_simplified = self.eliminate_ite(c)
            t_simplified = self.eliminate_ite(t)
            e_simplified = self.eliminate_ite(e)

            # Create the expanded form
            c_and_t = builder.mk_and([c_simplified, t_simplified])
            not_c_and_e = builder.mk_and([builder.mk_not(c_simplified), e_simplified])

            return builder.mk_or([c_and_t, not_c_and_e])

        elif isinstance(expression, (And, Or)):
            # Recursively eliminate in arguments
            builder = ExpressionBuilder(self.context)
            simplified_args = [self.eliminate_ite(arg) for arg in expression.args]

            if isinstance(expression, And):
                return builder.mk_and(simplified_args)
            else:
                return builder.mk_or(simplified_args)

        elif isinstance(expression, Not):
            builder = ExpressionBuilder(self.context)
            simplified_operand = self.eliminate_ite(expression.arg)
            return builder.mk_not(simplified_operand)

        else:
            return expression


# Factory functions
def make_simplifier(context: Context) -> Simplifier:
    """Create an expression simplifier."""
    return Simplifier(context)


def make_nnf_converter(context: Context) -> NNFConverter:
    """Create an NNF converter."""
    return NNFConverter(context)


def make_cnf_converter(context: Context) -> CNFConverter:
    """Create a CNF converter."""
    return CNFConverter(context)


def make_expression_simplifier(context: Context) -> ExpressionSimplifier:
    """Create a comprehensive expression simplifier."""
    return ExpressionSimplifier(context)


# Utility functions
def simplify_expression(expression: Expression, context: Optional[Context] = None) -> Expression:
    """Simplify an expression."""
    ctx = context or Context()
    simplifier = Simplifier(ctx)
    return simplifier.simplify(expression)


def to_negation_normal_form(expression: Expression, context: Optional[Context] = None) -> Expression:
    """Convert expression to NNF."""
    ctx = context or Context()
    converter = NNFConverter(ctx)
    return converter.to_nnf(expression)


def to_conjunctive_normal_form(expression: Expression, context: Optional[Context] = None) -> Expression:
    """Convert expression to CNF."""
    ctx = context or Context()
    converter = CNFConverter(ctx)
    return converter.to_cnf(expression)


def eliminate_ite_expressions(expression: Expression, context: Optional[Context] = None) -> Expression:
    """Eliminate ITE expressions from an expression."""
    ctx = context or Context()
    simplifier = ExpressionSimplifier(ctx)
    return simplifier.eliminate_ite(expression)
"""
SRK simplification and term manipulation utilities.

This module provides comprehensive simplification functionality for SRK expressions,
including term simplification, purification, and specialized simplification routines.
"""
from typing import List, Dict, Set, Tuple, Optional, Union, Callable, Any
from fractions import Fraction
from dataclasses import dataclass
import logging

# Import from other SRK modules
from .syntax import Context, Symbol, Expression, FormulaExpression, ArithExpression, Type, Term, mk_const, mk_symbol, mk_real, mk_add, mk_mul, mk_div, mk_mod, mk_eq, mk_and, mk_or, mk_leq, mk_lt, mk_sub, mk_floor, mk_ite, mk_not, mk_true, mk_false, mk_if, mk_neg, mk_app, mk_select, mk_int, mk_var, destruct, expr_typ, symbols, rewrite
from .interval import Interval
from .polynomial import Polynomial as QQXs, Monomial
from .linear import QQVector, QQMatrix
from .linear_utils import linterm_of
from .srkZ3 import SrkZ3, mk_solver, Solver
from .quantifier import is_presburger_atom, mbp
# Nonlinear is defined in this file below
from .util import ZZ
from .qQ import QQ

# Import BatPervasives-style functionality
from .util import BatDynArray, BatEnum, BatList, BatSet, BatMap, BatHashtbl, BatArray

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class RationalTermContext:
    """Context for rational term operations."""
    srk: Context
    table: Dict[ArithExpression, int]  # Expression -> ID mapping
    enum: BatDynArray[ArithExpression]  # ID -> Expression mapping

    @classmethod
    def mk_context(cls, srk: Context) -> 'RationalTermContext':
        """Create a new rational term context."""
        table = {}
        enum = BatDynArray()
        return cls(srk, table, enum)

    def of_int(self, expr_id: int) -> ArithExpression:
        """Get expression by ID."""
        return self.enum[expr_id]

    def int_of(self, term: ArithExpression) -> int:
        """Get ID for expression, creating if necessary."""
        if term in self.table:
            return self.table[term]

        expr_id = len(self.enum)
        self.enum.append(term)
        self.table[term] = expr_id
        return expr_id


@dataclass
class RationalTerm:
    """Represents a rational function as numerator/denominator polynomials."""
    num: QQXs  # Numerator polynomial
    den: Monomial  # Denominator monomial

    @staticmethod
    def scalar(k: Fraction) -> 'RationalTerm':
        """Create a scalar rational term."""
        return RationalTerm(QQXs.scalar(k), Monomial.one)

    @staticmethod
    def zero() -> 'RationalTerm':
        """Create zero rational term."""
        return RationalTerm.scalar(Fraction(0))

    @staticmethod
    def one() -> 'RationalTerm':
        """Create one rational term."""
        return RationalTerm.scalar(Fraction(1))

    def add(self, other: 'RationalTerm') -> 'RationalTerm':
        """Add two rational terms."""
        den = Monomial.lcm(self.den, other.den)
        # Create multiplier terms for common denominator
        f_mul = QQXs.add_term(Fraction(1), self.den, QQXs.zero)
        g_mul = QQXs.add_term(Fraction(1), other.den, QQXs.zero)

        num = QQXs.add(QQXs.mul(f_mul, self.num), QQXs.mul(g_mul, other.num))
        return RationalTerm(num, den)

    def negate(self) -> 'RationalTerm':
        """Negate a rational term."""
        return RationalTerm(QQXs.negate(self.num), self.den)

    def mul(self, other: 'RationalTerm') -> 'RationalTerm':
        """Multiply two rational terms."""
        num = QQXs.mul(self.num, other.num)
        den = Monomial.mul(self.den, other.den)
        return RationalTerm(num, den)


def of_term(ctx: RationalTermContext, term: ArithExpression) -> RationalTerm:
    """Convert an arithmetic term to a rational term representation."""
    srk = ctx.srk

    def rat_term(term: ArithExpression) -> RationalTerm:
        return RationalTerm(QQXs.of_dim(ctx.int_of(term)), Monomial.one)

    def alg(op) -> RationalTerm:
        if op == 'Add':
            return BatList.fold_left(lambda acc, x: acc.add(x), RationalTerm.zero(), op.args)
        elif op == 'Mul':
            return BatList.fold_left(lambda acc, x: acc.mul(x), RationalTerm.one(), op.args)
        elif op == 'Real':
            return RationalTerm.scalar(op.value)
        elif op == 'Neg':
            return op.arg.negate()
        elif op == 'Floor':
            return rat_term(mk_floor(srk, term_of(ctx, op.arg)))
        elif op == 'App':
            return rat_term(mk_app(srk, op.func, op.args))
        elif op == 'Div':
            return RationalTerm(
                op.left.num,
                Monomial.mul_term(ctx.int_of(term_of(ctx, op.right)), 1, op.left.den)
            )
        elif op == 'Mod':
            return rat_term(mk_mod(srk, term_of(ctx, op.left), term_of(ctx, op.right)))
        elif op == 'Ite':
            return rat_term(mk_ite(srk, op.condition, term_of(ctx, op.then_branch), term_of(ctx, op.else_branch)))
        elif op == 'Var':
            return rat_term(mk_var(srk, op.var_id, op.typ))
        elif op == 'Select':
            return rat_term(mk_select(srk, op.array, term_of(ctx, op.index)))
        else:
            raise ValueError(f"Unknown operation: {op}")

    # This would need the actual destruct function from syntax module
    # For now, returning a placeholder
    return RationalTerm.zero()


def term_of(ctx: RationalTermContext, p: RationalTerm) -> ArithExpression:
    """Convert a rational term back to an arithmetic expression."""
    srk = ctx.srk
    num = QQXs.term_of(srk, ctx.of_int, p.num)
    den = Monomial.term_of(srk, ctx.of_int, p.den)
    return mk_div(srk, num, den)


def simplify_term(srk: Context, term: ArithExpression) -> ArithExpression:
    """Simplify an arithmetic term using rational function representation."""
    ctx = RationalTermContext.mk_context(srk)
    rt = of_term(ctx, term)

    c = QQXs.content(rt.num)
    if c == Fraction(0):
        result = rt
    else:
        # Normalize by dividing out content
        normalized_num = QQXs.scalar_mul(Fraction(1) / abs(c), rt.num)
        result = RationalTerm(normalized_num, rt.den)

    return term_of(ctx, result)


def simplify_terms_rewriter(srk: Context) -> Callable[[Expression], Expression]:
    """Create a rewriter for simplifying terms in formulas."""
    ctx = RationalTermContext.mk_context(srk)

    def rewriter(expr: Expression) -> Expression:
        destruct_result = destruct(srk, expr)
        if destruct_result[0] == 'Atom' and len(destruct_result) > 1:
            atom_type, atom_data = destruct_result[1]
            if atom_type == 'Arith' and len(atom_data) == 3:
                op, s, t = atom_data
                # s - t as a rational function
                rf = of_term(ctx, mk_sub(srk, s, t))

                c = QQXs.content(rf.num)
                if c == Fraction(0):
                    rf = rf
                else:
                    normalized_num = QQXs.scalar_mul(Fraction(1) / abs(c), rf.num)
                    rf = RationalTerm(normalized_num, rf.den)

                # Calculate denominator for scaling
                def calc_denominator():
                    result = ZZ.one
                    for coeff, _ in QQXs.enum(rf.num):
                        result = ZZ.lcm(result, coeff.denominator)
                    return result
                denominator = calc_denominator()

                num_term = QQXs.scalar_mul(Fraction(denominator), rf.num)
                num_term = QQXs.term_of(srk, ctx.of_int, num_term)
                den_term = Monomial.term_of(srk, ctx.of_int, rf.den)

                zero = mk_real(srk, Fraction(0))
                if op == 'Leq':
                    result = mk_or(srk, [
                        mk_and(srk, [mk_leq(srk, num_term, zero), mk_lt(srk, zero, den_term)]),
                        mk_and(srk, [mk_leq(srk, zero, num_term), mk_lt(srk, den_term, zero)])
                    ])
                elif op == 'Lt':
                    result = mk_or(srk, [
                        mk_and(srk, [mk_lt(srk, num_term, zero), mk_lt(srk, zero, den_term)]),
                        mk_and(srk, [mk_lt(srk, zero, num_term), mk_lt(srk, den_term, zero)])
                    ])
                elif op == 'Eq':
                    result = mk_and(srk, [
                        mk_eq(srk, num_term, zero),
                        mk_or(srk, [mk_lt(srk, zero, den_term), mk_lt(srk, den_term, zero)])
                    ])
                else:
                    result = expr
                return result
        return expr

    return rewriter


def simplify_terms(srk: Context, expr: Expression) -> Expression:
    """Simplify terms in an expression."""
    rewriter = simplify_terms_rewriter(srk)
    return rewrite(srk, expr, up=rewriter)


def purify_rewriter(srk: Context, table: Dict[Expression, Symbol]) -> Callable[[Expression], Expression]:
    """Create a rewriter for purifying uninterpreted function applications."""
    def rewriter(expr: Expression) -> Expression:
        destruct_result = destruct(srk, expr)
        if destruct_result[0] == 'Quantify':
            raise ValueError("purify: free variable")
        elif destruct_result[0] == 'App':
            if len(destruct_result) > 1 and destruct_result[1] == []:
                return expr
            else:
                if expr in table:
                    sym = table[expr]
                else:
                    sym = mk_symbol(srk, "uninterp", expr_typ(srk, expr))
                    table[expr] = sym
                return mk_const(srk, sym)
        else:
            return expr

    return rewriter


def purify(srk: Context, expr: Expression) -> Tuple[Expression, Dict[Symbol, Expression]]:
    """Purify uninterpreted function applications in an expression."""
    table = {}
    expr_purified = rewrite(srk, expr, up=purify_rewriter(srk, table))

    symbol_map = {}
    for (term, sym) in table.items():
        symbol_map[sym] = term

    return expr_purified, symbol_map


def partition_implicant(implicant: List[Expression]) -> List[List[Expression]]:
    """Partition an implicant into independent groups."""
    # Separate atoms without symbols
    zero_group = [atom for atom in implicant if not symbols(atom)]

    remaining = [atom for atom in implicant if symbols(atom)]

    if not remaining:
        return [zero_group] if zero_group else [[]]

    # Use disjoint set to group atoms by shared symbols
    from .disjointSet import DisjointSet

    ds = DisjointSet()
    symbol_to_atoms = {}

    for atom in remaining:
        atom_symbols = symbols(atom)
        if atom_symbols:
            # Get representative symbol for this atom
            rep_symbol = next(iter(atom_symbols))
            if rep_symbol not in symbol_to_atoms:
                symbol_to_atoms[rep_symbol] = []
            symbol_to_atoms[rep_symbol].append(atom)

            # Union all symbols in this atom
            for sym in atom_symbols:
                ds.union(rep_symbol, sym)

    # Group atoms by their representative symbols
    groups = {}
    for atom in remaining:
        atom_symbols = symbols(atom)
        if atom_symbols:
            rep = ds.find(next(iter(atom_symbols)))
            if rep not in groups:
                groups[rep] = []
            groups[rep].append(atom)

    partitioned = list(groups.values())

    if zero_group:
        partitioned.append(zero_group)

    return partitioned


def simplify_conjunction(srk: Context, cube: List[Expression]) -> List[Expression]:
    """Simplify a conjunction of formulas using Z3."""
    cube = [simplify_terms(srk, prop) for prop in cube]

    solver = mk_solver(srk)
    indicator_map = {}

    # Create indicator variables for each proposition
    for prop in cube:
        indicator = mk_symbol(srk, 'TyBool')
        indicator_map[indicator] = prop

    # Add negation of conjunction to solver
    solver.add([mk_not(srk, mk_and(srk, cube))])

    # Add conditional constraints for each indicator
    for indicator, prop in indicator_map.items():
        solver.add([mk_if(srk, mk_const(srk, indicator), prop)])

    # Get unsatisfiable core
    assumptions = [mk_const(srk, indicator) for indicator in indicator_map.keys()]

    match solver.get_unsat_core(assumptions):
        case 'Sat':
            assert False, "Should be unsat"
        case 'Unknown':
            return cube
        case ('Unsat', core):
            # Extract corresponding propositions from core
            simplified_cube = []
            for ind in core:
                match destruct(srk, ind):
                    case ('Proposition', ('App', sym, [])):
                        if sym in indicator_map:
                            simplified_cube.append(indicator_map[sym])
                    case _:
                        assert False, "Unexpected unsat core element"
            return simplified_cube


class Nonlinear(Exception):
    """Exception raised for nonlinear operations."""
    pass


def isolate_linear(srk: Context, x: Symbol, term: ArithExpression) -> Optional[Tuple[Fraction, ArithExpression]]:
    """Isolate linear term in x from a nonlinear term."""
    def go(term: ArithExpression) -> Union[str, Tuple[Fraction, List[ArithExpression]]]:
        destruct_result = destruct(srk, term)
        if destruct_result[0] == 'Real':
            return ('Real', destruct_result[1])
        elif destruct_result[0] == 'App' and len(destruct_result) > 1:
            if destruct_result[1] == [x] and len(destruct_result) > 2 and destruct_result[2] == []:
                return ('Lin', (Fraction(1), []))
        elif destruct_result[0] == 'Add':
            xs = destruct_result[1] if len(destruct_result) > 1 else []
            result = ('Real', Fraction(0))
            for t in xs:
                go_t = go(t)
                if result[0] == 'Real' and go_t[0] == 'Real':
                    result = ('Real', result[1] + go_t[1])
                elif result[0] == 'Real' and go_t[0] == 'Lin':
                    result = ('Lin', (go_t[1][0], [mk_real(srk, result[1])] + go_t[1][1]))
                elif result[0] == 'Lin' and go_t[0] == 'Real':
                    result = ('Lin', (result[1][0], result[1][1] + [mk_real(srk, go_t[1])]))
                elif result[0] == 'Lin' and go_t[0] == 'Lin':
                    result = ('Lin', (result[1][0] + go_t[1][0], result[1][1] + go_t[1][1]))
            return result
        elif destruct_result[0] == 'Mul':
            xs = destruct_result[1] if len(destruct_result) > 1 else []
            result = ('Real', Fraction(1))
            for t in xs:
                go_t = go(t)
                if result[0] == 'Real' and go_t[0] == 'Real':
                    result = ('Real', result[1] * go_t[1])
                elif result[0] == 'Real' and result[1] == Fraction(0):
                    result = ('Real', Fraction(0))
                elif go_t[0] == 'Real' and go_t[1] == Fraction(0):
                    result = ('Real', Fraction(0))
                elif result[0] == 'Real' and go_t[0] == 'Lin':
                    result = ('Lin', (result[1] * go_t[1][0], [mk_mul(srk, [mk_real(srk, result[1]), mk_add(srk, go_t[1][1])])]))
                elif result[0] == 'Lin' and go_t[0] == 'Real':
                    result = ('Lin', (result[1][0] * go_t[1], [mk_mul(srk, [mk_real(srk, go_t[1]), mk_add(srk, result[1][1])])]))
                elif result[0] == 'Lin' and go_t[0] == 'Lin':
                    raise Nonlinear()
            return result
        elif destruct_result[0] == 'Binop' and len(destruct_result) > 1 and destruct_result[1] == 'Div':
            s, t = destruct_result[2], destruct_result[3]
            go_s, go_t = go(s), go(t)
            if go_s[0] == 'Real' and go_t[0] == 'Real' and go_t[1] != Fraction(0):
                return ('Real', go_s[1] / go_t[1])
            elif go_s[0] == 'Lin' and go_t[0] == 'Real' and go_t[1] != Fraction(0):
                return ('Lin', (go_s[1][0] / go_t[1], [mk_div(srk, mk_add(srk, go_s[1][1]), mk_real(srk, go_t[1]))]))
            else:
                if x in symbols(term):
                    raise Nonlinear()
                else:
                    return ('Lin', (Fraction(0), [term]))
        elif destruct_result[0] == 'Unop' and len(destruct_result) > 1 and destruct_result[1] == 'Neg':
            t = destruct_result[2]
            go_t = go(t)
            if go_t[0] == 'Real':
                return ('Real', -go_t[1])
            elif go_t[0] == 'Lin':
                return ('Lin', (-go_t[1][0], [mk_neg(srk, mk_add(srk, go_t[1][1]))]))
        else:
            if x in symbols(term):
                raise Nonlinear()
            else:
                return ('Lin', (Fraction(0), [term]))

    try:
        go_result = go(term)
        if go_result[0] == 'Lin':
            return (go_result[1][0], mk_add(srk, go_result[1][1]))
        elif go_result[0] == 'Real':
            return (Fraction(0), mk_real(srk, go_result[1]))
    except Nonlinear:
        return None


def simplify_dda(srk: Context, phi: Expression) -> Expression:
    """Simplify using DDA (Dynamic Dependency Analysis)."""
    solver = mk_solver(srk)

    def simplify_children(star: Callable, children: List[Expression]) -> List[Expression]:
        changed = False

        def go(simplified: List[Expression], remaining: List[Expression]) -> List[Expression]:
            if not remaining:
                return list(reversed(simplified))

            phi_head = remaining[0]
            phi_tail = remaining[1:]
            solver.push()
            solver.add([star(s) for s in simplified])
            solver.add([star(p) for p in phi_tail])
            simple_phi = simplify_dda_impl(phi_head)
            solver.pop()

            if not phi_head.equal(simple_phi):
                changed = True

            return go([simple_phi] + simplified, phi_tail)

        def fix(children: List[Expression]) -> List[Expression]:
            simplified = go([], children)
            if changed:
                changed = False
                return fix(simplified)
            return simplified

        return fix(children)

    def simplify_dda_impl(phi: Expression) -> Expression:
        destruct_result = destruct(srk, phi)
        if destruct_result[0] == 'Or':
            xs = destruct_result[1] if len(destruct_result) > 1 else []
            return mk_or(srk, simplify_children(lambda x: mk_not(srk, x), xs))
        elif destruct_result[0] == 'And':
            xs = destruct_result[1] if len(destruct_result) > 1 else []
            return mk_and(srk, simplify_children(lambda x: x, xs))
        else:
            solver.push()
            solver.add([phi])

            check_result = solver.check([])
            if check_result == 'Unknown':
                simplified = phi
            elif check_result == 'Unsat':
                simplified = mk_false(srk)
            elif check_result == 'Sat':
                solver.pop()
                solver.push()
                solver.add([mk_not(srk, phi)])

                check_result2 = solver.check([])
                if check_result2 == 'Unknown':
                    simplified = phi
                elif check_result2 == 'Unsat':
                    simplified = mk_true(srk)
                elif check_result2 == 'Sat':
                    simplified = phi
            else:
                simplified = phi

            solver.pop()
            return simplified

    return simplify_dda_impl(phi)


def destruct_idiv(srk: Context, t: ArithExpression) -> Optional[Tuple[ArithExpression, int]]:
    """Extract integer division from floor expression."""
    destruct_result = destruct(srk, t)
    if destruct_result[0] == 'Unop' and len(destruct_result) > 1 and destruct_result[1] == 'Floor':
        inner_t = destruct_result[2]
        inner_destr = destruct(srk, inner_t)
        if inner_destr[0] == 'Binop' and len(inner_destr) > 1 and inner_destr[1] == 'Div':
            num, den = inner_destr[2], inner_destr[3]
            den_destr = destruct(srk, den)
            if den_destr[0] == 'Real':
                den_val = den_destr[1]
                if isinstance(den_val, int) and den_val > 0:
                    return (num, den_val)
                else:
                    return None
            else:
                return None
        else:
            return None
    else:
        return None


def idiv_to_ite(srk: Context, expr: Expression, max_val: int = 1000) -> Expression:
    """Convert integer division to if-then-else."""
    if isinstance(expr, tuple) and len(expr) >= 2 and expr[0] == 'ArithTerm':
        t = expr[1]
        destruct_result = destruct_idiv(srk, t)
        if destruct_result is not None:
            num, den = destruct_result
            if den < max_val:
                den_term = mk_real(srk, Fraction(den))
                num_over_den = mk_mul(srk, [mk_real(srk, Fraction(1, den)), num])

                def fold_func(else_branch, r):
                    return mk_ite(
                        srk,
                        mk_eq(srk,
                              mk_mod(srk, mk_sub(srk, num, mk_real(srk, Fraction(r))), den_term),
                              mk_real(srk, Fraction(0))),
                        mk_real(srk, Fraction(-r, den)),
                        else_branch
                    )

                offset = BatEnum.fold(
                    fold_func,
                    mk_real(srk, Fraction(0)),
                    range(1, den)
                )

                return mk_add(srk, [num_over_den, offset])
        return expr
    else:
        return expr


def eliminate_idiv(srk: Context, formula: Expression, max_val: int = 1000) -> Expression:
    """Eliminate integer division from a formula."""
    formula = rewrite(srk, formula, up=lambda e: idiv_to_ite(srk, e, max_val))
    # This would need eliminate_ite function from syntax module
    # return eliminate_ite(srk, formula)
    return formula


def purify_floor(srk: Context, expr: Expression) -> Tuple[Expression, Dict[Symbol, Expression]]:
    """Purify floor operations by introducing fresh symbols."""
    table = {}

    def rewriter(expr: Expression) -> Expression:
        destruct_result = destruct(srk, expr)
        if destruct_result[0] == 'Quantify':
            raise ValueError("purify_floor: free variable")
        elif destruct_result[0] == 'Unop' and len(destruct_result) > 1 and destruct_result[1] == 'Floor':
            t = destruct_result[2]
            if expr_typ(srk, t) == 'TyInt':
                return t
            else:
                if t not in table:
                    sym = mk_symbol(srk, "floor", 'TyInt')
                    table[t] = sym
                return mk_const(srk, table[t])
        else:
            return expr

    expr_purified = rewrite(srk, expr, up=rewriter)

    symbol_map = {}
    for (term, sym) in table.items():
        symbol_map[sym] = term

    return expr_purified, symbol_map


def eliminate_floor(srk: Context, formula: Expression) -> Expression:
    """Eliminate floor operations from a formula."""
    formula = eliminate_idiv(srk, formula, max_val=10)
    formula, symbol_map = purify_floor(srk, formula)

    one = mk_int(srk, 1)
    floor_constraints = []

    for sym, term in symbol_map.items():
        s = mk_const(srk, sym)
        floor_constraints.extend([
            mk_leq(srk, s, term),
            mk_lt(srk, mk_sub(srk, term, one), s)
        ])

    return mk_and(srk, [formula] + floor_constraints)


def simplify_integer_atom(srk: Context, op: str, s: ArithExpression, t: ArithExpression) -> Expression:
    """Simplify integer atoms with divisibility constraints."""
    zero = mk_real(srk, Fraction(0))

    def destruct_int(term: ArithExpression) -> int:
        destruct_result = destruct(srk, term)
        if destruct_result[0] == 'Real':
            q = destruct_result[1]
            if isinstance(q, int):
                return q
            else:
                raise ValueError("simplify_atom: non-integral value")
        else:
            raise ValueError("simplify_atom: non-constant")

    # Normalize: s - t
    if Term.equal(t, zero):
        s_norm = s
    else:
        s_norm = mk_sub(srk, s, t)

    if op == 'Lt' and expr_typ(srk, s_norm) == 'TyInt':
        s_norm = simplify_term(srk, mk_add(srk, [s_norm, mk_real(srk, Fraction(1))]))
        op = 'Leq'
    else:
        s_norm = simplify_term(srk, s_norm)
        op = op

    # Scale linear terms to have integer coefficients
    def zz_linterm(term: ArithExpression) -> Tuple[int, QQVector]:
        qq_linterm = linterm_of(srk, term)
        def calc_multiplier():
            result = ZZ.one
            for qq, _ in QQVector.enum(qq_linterm):
                result = ZZ.lcm(result, qq.denominator)
            return result
        multiplier = calc_multiplier()
        return (multiplier, QQVector.scalar_mul(Fraction(multiplier), qq_linterm))

    if op in ['Eq', 'Leq']:
        destruct_result = destruct(srk, s_norm)
        if destruct_result[0] == 'Binop' and len(destruct_result) > 1 and destruct_result[1] == 'Mod':
            dividend, modulus = destruct_result[2], destruct_result[3]
            modulus_val = destruct_int(modulus)
            multiplier, lt = zz_linterm(dividend)
            return f'Divides({ZZ.mul(multiplier, modulus_val)}, {lt})'
        elif destruct_result[0] == 'Unop' and len(destruct_result) > 1 and destruct_result[1] == 'Neg':
            s_prime = destruct_result[2]
            s_prime_destr = destruct(srk, s_prime)
            if s_prime_destr[0] == 'Binop' and len(s_prime_destr) > 1 and s_prime_destr[1] == 'Mod':
                dividend, modulus = s_prime_destr[2], s_prime_destr[3]
                if op == 'Leq':
                    return 'CompareZero(`Leq, QQVector.zero)'
                else:
                    modulus_val = destruct_int(modulus)
                    multiplier, lt = zz_linterm(dividend)
                    return f'Divides({ZZ.mul(multiplier, modulus_val)}, {lt})'
            else:
                return f'CompareZero({op}, {zz_linterm(s_norm)[1]})'
        elif destruct_result[0] == 'Add' and len(destruct_result) > 1:
            xs = destruct_result[1]
            if len(xs) == 2:
                x, y = xs
                x_destr = destruct(srk, x)
                y_destr = destruct(srk, y)
                if x_destr[0] == 'Real' and y_destr[0] == 'Binop' and len(y_destr) > 1 and y_destr[1] == 'Mod':
                    k, dividend, modulus = x_destr[1], y_destr[2], y_destr[3]
                    if k < Fraction(0) and op == 'Eq':
                        multiplier, lt = zz_linterm(dividend)
                        modulus_val = destruct_int(modulus)
                        if multiplier == 1 and k < Fraction(modulus_val):
                            lt = QQVector.add_term(k, 0, lt)
                            return f'Divides({modulus_val}, {lt})'
                        else:
                            return f'CompareZero({op}, {zz_linterm(s_norm)[1]})'
                    elif QQ.equal(k, Fraction(1)) and op == 'Leq':
                        multiplier, lt = zz_linterm(dividend)
                        modulus_val = destruct_int(modulus)
                        if multiplier == 1:
                            return f'NotDivides({modulus_val}, {lt})'
                        else:
                            return f'CompareZero({op}, {zz_linterm(s_norm)[1]})'
                    else:
                        return f'CompareZero({op}, {zz_linterm(s_norm)[1]})'
                elif y_destr[0] == 'Real' and x_destr[0] == 'Binop' and len(x_destr) > 1 and x_destr[1] == 'Mod':
                    k, dividend, modulus = y_destr[1], x_destr[2], x_destr[3]
                    if k < Fraction(0) and op == 'Eq':
                        multiplier, lt = zz_linterm(dividend)
                        modulus_val = destruct_int(modulus)
                        if multiplier == 1 and k < Fraction(modulus_val):
                            lt = QQVector.add_term(k, 0, lt)
                            return f'Divides({modulus_val}, {lt})'
                        else:
                            return f'CompareZero({op}, {zz_linterm(s_norm)[1]})'
                    elif QQ.equal(k, Fraction(1)) and op == 'Leq':
                        multiplier, lt = zz_linterm(dividend)
                        modulus_val = destruct_int(modulus)
                        if multiplier == 1:
                            return f'NotDivides({modulus_val}, {lt})'
                        else:
                            return f'CompareZero({op}, {zz_linterm(s_norm)[1]})'
                    else:
                        return f'CompareZero({op}, {zz_linterm(s_norm)[1]})'
                elif (x_destr[0] == 'Real' and y_destr[0] == 'Unop' and len(y_destr) > 1 and y_destr[1] == 'Neg') or \
                     (y_destr[0] == 'Real' and x_destr[0] == 'Unop' and len(x_destr) > 1 and x_destr[1] == 'Neg'):
                    k = x_destr[1] if x_destr[0] == 'Real' else y_destr[1]
                    z = y_destr[2] if y_destr[0] == 'Unop' else x_destr[2]
                    if QQ.equal(k, Fraction(1)) and op == 'Leq':
                        z_destr = destruct(srk, z)
                        if z_destr[0] == 'Binop' and len(z_destr) > 1 and z_destr[1] == 'Mod':
                            dividend, modulus = z_destr[2], z_destr[3]
                            modulus_val = destruct_int(modulus)
                            multiplier, lt = zz_linterm(dividend)
                            return f'NotDivides({ZZ.mul(multiplier, modulus_val)}, {lt})'
                        else:
                            return f'CompareZero({op}, {zz_linterm(s_norm)[1]})'
                    else:
                        return f'CompareZero({op}, {zz_linterm(s_norm)[1]})'
                else:
                    return f'CompareZero({op}, {zz_linterm(s_norm)[1]})'
            else:
                return f'CompareZero({op}, {zz_linterm(s_norm)[1]})'
        else:
            return f'CompareZero({op}, {zz_linterm(s_norm)[1]})'
    elif op == 'Lt':
        destruct_result = destruct(srk, s_norm)
        if destruct_result[0] == 'Binop' and len(destruct_result) > 1 and destruct_result[1] == 'Mod':
            dividend, modulus = destruct_result[2], destruct_result[3]
            modulus_val = destruct_int(modulus)
            multiplier, lt = zz_linterm(dividend)
            return f'NotDivides({ZZ.mul(multiplier, modulus_val)}, {lt})'
        elif destruct_result[0] == 'Unop' and len(destruct_result) > 1 and destruct_result[1] == 'Neg':
            s_prime = destruct_result[2]
            s_prime_destr = destruct(srk, s_prime)
            if s_prime_destr[0] == 'Binop' and len(s_prime_destr) > 1 and s_prime_destr[1] == 'Mod':
                dividend, modulus = s_prime_destr[2], s_prime_destr[3]
                modulus_val = destruct_int(modulus)
                multiplier, lt = zz_linterm(dividend)
                return f'NotDivides({ZZ.mul(multiplier, modulus_val)}, {lt})'
            else:
                return f'CompareZero(`Lt, {zz_linterm(s_norm)[1]})'
        else:
            return f'CompareZero(`Lt, {zz_linterm(s_norm)[1]})'
    else:
        return f'CompareZero({op}, {zz_linterm(s_norm)[1]})'


# Note: The functions below are already implemented in the syntax module
# and imported at the top of this file. The placeholders were removed.
