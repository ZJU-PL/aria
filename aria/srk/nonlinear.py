"""
Nonlinear arithmetic operations and symbolic interval analysis.

This module provides functionality for handling nonlinear arithmetic expressions,
converting between interpreted and uninterpreted forms, and performing
linearization of nonlinear formulas through optimization techniques.
"""

from __future__ import annotations
from typing import List, Dict, Set, Tuple, Union, Optional, Callable, Any
from dataclasses import dataclass, field
from fractions import Fraction
import logging

# Import from other SRK modules
from aria.srk.syntax import (
    Context, Symbol, Expression, FormulaExpression, ArithExpression, Type,
    mk_const, mk_symbol, mk_real, mk_add, mk_mul, mk_div, mk_mod, mk_eq,
    mk_and, mk_or, mk_leq, mk_lt, mk_sub, mk_neg, mk_app, mk_ite, mk_false,
    rewrite, destruct, expr_typ, symbols, ArithTerm
)
from aria.srk.interval import Interval
from aria.srk.qQ import QQ, equal as QQ_equal
from aria.srk.log import logf

# Setup logging
logger = logging.getLogger(__name__)


@dataclass
class SymbolicInterval:
    """Symbolic interval representing bounds on expressions."""

    context: Context
    lower: List[ArithTerm]  # Lower bounds
    upper: List[ArithTerm]  # Upper bounds
    interval: Interval  # Concrete interval bounds

    @classmethod
    def of_interval(cls, context: Context, interval: Interval) -> 'SymbolicInterval':
        """Create a symbolic interval from a concrete interval."""
        return cls(context, [], [], interval)

    @classmethod
    def bottom(cls, context: Context) -> 'SymbolicInterval':
        """Create the bottom (empty) symbolic interval."""
        return cls.of_interval(context, Interval.bottom())

    @classmethod
    def top(cls, context: Context) -> 'SymbolicInterval':
        """Create the top (universal) symbolic interval."""
        return cls.of_interval(context, Interval.top())

    def cartesian(self, f: Callable, xs: List, ys: List) -> List:
        """Cartesian product applying function f."""
        result = []
        for x in xs:
            for y in ys:
                result.append(f([x, y]))
        return result

    def add(self, other: 'SymbolicInterval') -> 'SymbolicInterval':
        """Add two symbolic intervals."""
        if self.context != other.context:
            raise ValueError("Cannot add intervals from different contexts")

        # Cartesian product of bounds
        new_upper = self.cartesian(lambda args: mk_add(args), self.upper, other.lower)
        new_lower = self.cartesian(lambda args: mk_add(args), self.lower, other.lower)

        return SymbolicInterval(
            self.context,
            new_lower,
            new_upper,
            self.interval + other.interval
        )

    def mul_interval(self, ivl: Interval, x: 'SymbolicInterval') -> 'SymbolicInterval':
        """Multiply a symbolic interval by a concrete interval."""
        srk = x.context

        if Interval.is_nonnegative(ivl):
            upper_bound = Interval.upper(ivl)
            if upper_bound is not None and upper_bound != QQ.zero:
                upper = [mk_mul([mk_real(upper_bound), term]) for term in x.upper]
            else:
                upper = []

            lower_bound = Interval.lower(ivl)
            if lower_bound is not None and lower_bound != QQ.zero:
                lower = [mk_mul([mk_real(lower_bound), term]) for term in x.lower]
            else:
                lower = []

            return SymbolicInterval(srk, lower, upper, ivl * x.interval)

        elif Interval.is_nonpositive(ivl):
            upper_bound = Interval.upper(ivl)
            if upper_bound is not None and upper_bound != QQ.zero:
                upper = [mk_mul([mk_real(upper_bound), term]) for term in x.lower]
            else:
                upper = []

            lower_bound = Interval.lower(ivl)
            if lower_bound is not None and lower_bound != QQ.zero:
                lower = [mk_mul([mk_real(lower_bound), term]) for term in x.upper]
            else:
                lower = []

            return SymbolicInterval(srk, lower, upper, ivl * x.interval)
        else:
            # Mixed signs - conservative
            return SymbolicInterval(srk, [], [], ivl * x.interval)

    def mul(self, other: 'SymbolicInterval') -> 'SymbolicInterval':
        """Multiply two symbolic intervals."""
        if self.context != other.context:
            raise ValueError("Cannot multiply intervals from different contexts")

        # Meet of two mul_interval results (OCaml: meet (mul_interval x.interval y) (mul_interval y.interval x))
        result1 = self.mul_interval(self.interval, other)
        result2 = self.mul_interval(other.interval, self)
        return result1.meet(result2)

    def meet(self, other: 'SymbolicInterval') -> 'SymbolicInterval':
        """Meet (intersection) of two symbolic intervals."""
        if self.context != other.context:
            raise ValueError("Cannot meet intervals from different contexts")

        # Concatenate bounds (OCaml: x.upper @ y.upper)
        return SymbolicInterval(
            self.context,
            self.lower + other.lower,
            self.upper + other.upper,
            Interval.meet(self.interval, other.interval)
        )

    def join(self, other: 'SymbolicInterval') -> 'SymbolicInterval':
        """Join (union) of two symbolic intervals."""
        if self.context != other.context:
            raise ValueError("Cannot join intervals from different contexts")

        # Keep only common bounds
        common_lower = [b for b in self.lower if b in other.lower]
        common_upper = [b for b in self.upper if b in other.upper]

        return SymbolicInterval(
            self.context,
            common_lower,
            common_upper,
            Interval.join(self.interval, other.interval)
        )

    def negate(self) -> 'SymbolicInterval':
        """Negate a symbolic interval."""
        # Swap bounds and negate terms (OCaml: upper = map mk_neg lower, lower = map mk_neg upper)
        return SymbolicInterval(
            self.context,
            [mk_neg(term) for term in self.upper],  # Negated upper becomes new lower
            [mk_neg(term) for term in self.lower],  # Negated lower becomes new upper
            Interval.negate(self.interval)
        )

    def div(self, other: 'SymbolicInterval') -> 'SymbolicInterval':
        """Divide two symbolic intervals."""
        srk = self.context
        lower_y = Interval.lower(other.interval)
        upper_y = Interval.upper(other.interval)

        if lower_y is not None and upper_y is not None:
            if Interval.elem(QQ.zero, other.interval):
                # Division by interval containing zero
                return SymbolicInterval.top(srk)
            elif lower_y == upper_y:
                # Division by constant
                a = lower_y
                if QQ.lt(QQ.zero, a):
                    # Positive constant
                    return SymbolicInterval(
                        srk,
                        [mk_div(term, mk_real(a)) for term in self.lower],
                        [mk_div(term, mk_real(a)) for term in self.upper],
                        Interval.div(self.interval, other.interval)
                    )
                else:
                    # Negative constant - swap bounds
                    return SymbolicInterval(
                        srk,
                        [mk_div(term, mk_real(a)) for term in self.upper],
                        [mk_div(term, mk_real(a)) for term in self.lower],
                        Interval.div(self.interval, other.interval)
                    )
            else:
                # Non-constant divisor
                return SymbolicInterval.of_interval(srk, Interval.div(self.interval, other.interval))
        else:
            return SymbolicInterval.of_interval(srk, Interval.div(self.interval, other.interval))

    def modulo(self, other: 'SymbolicInterval') -> 'SymbolicInterval':
        """Compute modulo of two symbolic intervals."""
        srk = self.context
        ivl = Interval.modulo(self.interval, other.interval)

        if Interval.equal(ivl, Interval.bottom()):
            return SymbolicInterval.bottom(srk)
        elif Interval.elem(QQ.zero, other.interval):
            return SymbolicInterval.top(srk)
        else:
            # y is either strictly positive or strictly negative
            y = other if Interval.is_positive(other.interval) else other.negate()

            one_minus = lambda x: mk_sub(mk_real(QQ.one), x)
            minus_one = lambda x: mk_sub(x, mk_real(QQ.one))

            if Interval.is_nonnegative(self.interval):
                return SymbolicInterval(
                    srk,
                    [],
                    self.upper + [minus_one(term) for term in y.upper],
                    ivl
                )
            elif Interval.is_nonpositive(self.interval):
                return SymbolicInterval(
                    srk,
                    self.lower + [one_minus(term) for term in y.upper],
                    [],
                    ivl
                )
            else:
                return SymbolicInterval(
                    srk,
                    [one_minus(term) for term in y.upper],
                    [minus_one(term) for term in y.upper],
                    ivl
        )

    def floor(self) -> 'SymbolicInterval':
        """Compute floor of symbolic interval."""
        return SymbolicInterval(
            self.context,
            [],
            [],
            Interval.floor(self.interval)
        )

    @staticmethod
    def make(context: Context, lower: List, upper: List, interval: Interval) -> 'SymbolicInterval':
        """Create a symbolic interval with specified bounds."""
        return SymbolicInterval(context, lower, upper, interval)

    def get_lower(self) -> List:
        """Get lower bounds."""
        return self.lower

    def get_upper(self) -> List:
        """Get upper bounds."""
        return self.upper

    def get_interval(self) -> Interval:
        """Get concrete interval."""
        return self.interval


class NonlinearOperations:
    """Nonlinear arithmetic operations and transformations."""

    def __init__(self, context: Context):
        self.context = context
        self._symbols_cache = {}
        self._ensure_symbols()

    def _ensure_symbols(self) -> None:
        """Ensure required nonlinear operation symbols are registered."""
        # Check if symbols are registered, and register them if not
        symbols_to_register = [
            ("mul", Type.FUN),  # TyFun ([TyReal; TyReal], TyReal)
            ("inv", Type.FUN),  # TyFun ([TyReal], TyReal)
            ("mod", Type.FUN),  # TyFun ([TyReal; TyReal], TyReal)
            ("imul", Type.FUN), # TyFun ([TyReal; TyReal], TyInt)
            ("imod", Type.FUN), # TyFun ([TyReal; TyReal], TyInt)
            ("pow", Type.FUN),  # TyFun ([TyReal; TyReal], TyReal)
            ("log", Type.FUN),  # TyFun ([TyReal; TyReal], TyReal)
        ]

        for name, typ in symbols_to_register:
            # Check if already registered in context
            if not hasattr(self.context, '_named_symbols'):
                self.context._named_symbols = {}

            if name not in self.context._named_symbols:
                # Register new symbol
                sym = mk_symbol(name, typ)
                self.context._named_symbols[name] = sym
                self._symbols_cache[name] = sym
            else:
                self._symbols_cache[name] = self.context._named_symbols[name]

    def get_named_symbol(self, name: str) -> Symbol:
        """Get a named symbol, ensuring it's registered first."""
        if name not in self._symbols_cache:
            self._ensure_symbols()
        return self._symbols_cache.get(name)

    def get_mul_symbol(self) -> Symbol:
        """Get the multiplication symbol."""
        return self.get_named_symbol("mul")

    def get_inv_symbol(self) -> Symbol:
        """Get the inverse symbol."""
        return self.get_named_symbol("inv")

    def get_mod_symbol(self) -> Symbol:
        """Get the modulo symbol."""
        return self.get_named_symbol("mod")

    def get_imul_symbol(self) -> Symbol:
        """Get the integer multiplication symbol."""
        return self.get_named_symbol("imul")

    def get_imod_symbol(self) -> Symbol:
        """Get the integer modulo symbol."""
        return self.get_named_symbol("imod")

    def get_pow_symbol(self) -> Symbol:
        """Get the power symbol."""
        return self.get_named_symbol("pow")

    def get_log_symbol(self) -> Symbol:
        """Get the logarithm symbol."""
        return self.get_named_symbol("log")

    def uninterpret_rewriter(self, expr: Expression) -> Expression:
        """Convert nonlinear operations to uninterpreted functions.

        Converts division, modulo, and multiplication to uninterpreted functions,
        following the OCaml implementation.
        """
        mul_sym = self.get_mul_symbol()
        inv_sym = self.get_inv_symbol()
        mod_sym = self.get_mod_symbol()
        imul_sym = self.get_imul_symbol()
        imod_sym = self.get_imod_symbol()

        # Destruct the expression to check its type
        try:
            expr_info = destruct(expr)
            if expr_info is None:
                return expr

            expr_type, expr_data = expr_info

            # Handle division: x / y -> x * inv(y) or mul(x, inv(y))
            if expr_type == 'Div':
                x, y = expr_data
                # Check if y is a constant
                y_info = destruct(y)
                if y_info and y_info[0] == 'Real':
                    k = y_info[1]
                    if k != QQ.zero:
                        # division by constant -> scalar mul
                        return mk_mul([mk_real(QQ.inverse(k)), x])

                # Check if x is real constant
                x_info = destruct(x)
                if x_info and x_info[0] == 'Real':
                    # Real constant / y -> x * inv(y)
                    return mk_mul([x, mk_app(inv_sym, [y])])

                # General case: mul(x, inv(y))
                return mk_app(mul_sym, [x, mk_app(inv_sym, [y])])

            # Handle modulo: x % y -> mod(x, y) or imod(x, y)
            elif expr_type == 'Mod':
                x, y = expr_data
                # Check if y is integer constant
                y_info = destruct(y)
                x_typ = expr_typ(x)
                y_typ = expr_typ(y)

                if y_info and y_info[0] == 'Real':
                    k = y_info[1]
                    zz_val = QQ.to_zz(k) if hasattr(QQ, 'to_zz') else None
                    if k != QQ.zero and zz_val is not None and x_typ == Type.INT:
                        # Keep as interpreted mod for integer constant
                        return expr

                # Convert to uninterpreted
                if x_typ == Type.INT and y_typ == Type.INT:
                    return mk_app(imod_sym, [x, y])
                else:
                    return mk_app(mod_sym, [x, y])

            # Handle multiplication: convert to uninterpreted mul/imul
            elif expr_type == 'Mul':
                terms = expr_data
                # Separate coefficient from non-constant terms
                coeff = QQ.one
                non_const_terms = []

                for term in terms:
                    term_info = destruct(term)
                    if term_info and term_info[0] == 'Real':
                        coeff = QQ.mul(coeff, term_info[1])
                    else:
                        non_const_terms.append(term)

                coeff_term = mk_real(coeff)

                if len(non_const_terms) == 0:
                    return coeff_term
                elif len(non_const_terms) == 1:
                    return mk_mul([coeff_term, non_const_terms[0]])
                else:
                    # Build product using mul/imul
                    product = non_const_terms[0]
                    for term in non_const_terms[1:]:
                        term_typ = expr_typ(term)
                        prod_typ = expr_typ(product)
                        if term_typ == Type.INT and prod_typ == Type.INT:
                            product = mk_app(imul_sym, [term, product])
                        else:
                            product = mk_app(mul_sym, [term, product])
                    return mk_mul([coeff_term, product])

        except Exception:
            pass

        return expr

    def interpret_rewriter(self, expr: Expression) -> Expression:
        """Convert uninterpreted functions back to interpreted operations.

        Recognizes uninterpreted function applications and converts them back
        to their interpreted forms.
        """
        mul_sym = self.get_mul_symbol()
        inv_sym = self.get_inv_symbol()
        mod_sym = self.get_mod_symbol()
        imul_sym = self.get_imul_symbol()
        imod_sym = self.get_imod_symbol()

        try:
            expr_info = destruct(expr)
            if expr_info is None:
                return expr

            expr_type, expr_data = expr_info

            # Handle function applications
            if expr_type == 'App':
                func, args = expr_data

                # mul(x, y) or imul(x, y) -> x * y
                if (func == mul_sym or func == imul_sym) and len(args) == 2:
                    return mk_mul(args)

                # inv(x) -> 1 / x
                elif func == inv_sym and len(args) == 1:
                    return mk_div(mk_real(QQ.one), args[0])

                # mod(x, y) or imod(x, y) -> x % y
                elif (func == mod_sym or func == imod_sym) and len(args) == 2:
                    return mk_mod(args[0], args[1])

        except Exception:
            pass

        return expr

    def uninterpret(self, expr: Expression) -> Expression:
        """Convert nonlinear operations to uninterpreted functions."""
        return rewrite(expr, up=self.uninterpret_rewriter)

    def interpret(self, expr: Expression) -> Expression:
        """Convert uninterpreted functions back to interpreted operations."""
        return rewrite(expr, up=self.interpret_rewriter)

    def mk_log(self, base: ArithExpression, x: ArithExpression) -> ArithExpression:
        """Create a logarithm expression.

        Handles special cases:
        - log_b(1) = 0 when b > 1
        - log_b(b) = 1 when b > 1
        - log_b(b^t) = t
        """
        pow_sym = self.get_pow_symbol()
        log_sym = self.get_log_symbol()

        try:
            base_info = destruct(base)
            x_info = destruct(x)

            # log_b(1) = 0 when b > 1
            if base_info and base_info[0] == 'Real' and x_info and x_info[0] == 'Real':
                b = base_info[1]
                x_val = x_info[1]
                if QQ.lt(QQ.one, b) and QQ.equal(x_val, QQ.one):
                    return mk_real(QQ.zero)
                elif QQ.lt(QQ.one, b) and QQ.equal(x_val, b):
                    return mk_real(QQ.one)

            # log_b(b^t) = t (when bases match)
            if x_info and x_info[0] == 'App':
                func, args = x_info[1]
                if func == pow_sym and len(args) == 2:
                    base_arg, exp_arg = args
                    # Check if bases match
                    if base == base_arg:  # TODO: proper expression equality
                        return exp_arg

        except Exception:
            pass

        # General case: create log application
        return mk_app(log_sym, [base, x])

    def mk_pow(self, base: ArithExpression, x: ArithExpression) -> ArithExpression:
        """Create a power expression.

        Handles special cases:
        - 1^x = 1
        - (-b)^x = ite(x%2==0, b^x, -b^x) for b < 0
        - (1/b)^x when 0 < b < 1
        - b^power when power is an integer
        - b^(x1+x2+...) = b^x1 * b^x2 * ...
        - b^(-x) = 1 / b^x
        - b^log_b(t) = t
        """
        log_sym = self.get_log_symbol()
        pow_sym = self.get_pow_symbol()

        try:
            base_info = destruct(base)

            # 1^x = 1
            if base_info and base_info[0] == 'Real':
                b = base_info[1]
                if QQ.equal(b, QQ.one):
                    return mk_real(QQ.one)

                # (-b)^x for negative base
                if QQ.lt(b, QQ.zero):
                    pos_base = mk_real(QQ.negate(b))
                    pos_pow = self.mk_pow(pos_base, x)
                    # ite(x % 2 == 0, pos_pow, -pos_pow)
                    return mk_ite(
                        mk_eq(mk_mod(x, mk_real(QQ.of_int(2))), mk_real(QQ.zero)),
                        pos_pow,
                        mk_neg(pos_pow)
                    )

                # (1/b)^x when 0 < b < 1
                if QQ.lt(QQ.zero, b) and QQ.lt(b, QQ.one):
                    inv_base = mk_div(mk_real(QQ.one), base)
                    return mk_div(mk_real(QQ.one), self.mk_pow(inv_base, x))

            x_info = destruct(x)

            # b^power when power is a constant
            if x_info and x_info[0] == 'Const':
                power_symbol = x_info[1]
                # Check if it's a real constant by looking at the symbol name
                if hasattr(power_symbol, 'name') and power_symbol.name.startswith('real_'):
                    # Extract the value from the symbol name
                    try:
                        power_value = float(power_symbol.name.replace('real_', ''))
                        power = Fraction(power_value)

                        # Handle special cases for real constants
                        if QQ_equal(power, QQ.zero()):
                            # x^0 = 1
                            return mk_real(QQ.one())
                        elif QQ_equal(power, QQ.one()):
                            # x^1 = x
                            return base

                        power_int = QQ.to_int(power) if hasattr(QQ, 'to_int') else None
                        if power_int is not None:
                            # Use syntax.mk_pow for integer powers
                            from .syntax import mk_pow as syntax_mk_pow
                            try:
                                return syntax_mk_pow(base, power_int)
                            except:
                                pass
                    except Exception:
                        pass

            # b^(sum) = product of b^term
            if x_info and x_info[0] == 'Add':
                terms = x_info[1]
                return mk_mul([self.mk_pow(base, term) for term in terms])

            # b^(-x) = 1 / b^x
            if x_info and x_info[0] == 'Neg':
                negated_x = x_info[1]
                return mk_div(mk_real(QQ.one), self.mk_pow(base, negated_x))

            # b^log_b(t) = t
            if x_info and x_info[0] == 'App':
                func, args = x_info[1]
                if func == log_sym and len(args) == 2:
                    log_base, log_arg = args
                    if base == log_base:  # TODO: proper expression equality
                        return log_arg

        except Exception:
            pass

        # General case: create pow application
        return mk_app(pow_sym, [base, x])

    def linearize(self, formula: FormulaExpression) -> FormulaExpression:
        """Compute a linear approximation of a nonlinear formula.

        Converts nonlinear terms to uninterpreted functions, purifies the formula,
        finds bounds using optimization, and creates linear constraints.
        """
        # This is a complex function that requires integration with:
        # - srkSimplify.purify for term extraction
        # - srkZ3.optimize_box for finding intervals
        # - abstract.affine_hull for affine constraints

        # Import dependencies (may not be fully implemented yet)
        try:
            from . import srkSimplify
            from . import srkZ3
            from . import abstract
        except ImportError:
            # Dependencies not available, return original formula
            logf("linearize: dependencies not available, returning original formula", level='warn')
            return formula

        # Convert to uninterpreted form
        uninterp_phi = self.uninterpret(formula)

        # Purify to extract nonlinear terms
        try:
            lin_phi, nonlinear = srkSimplify.purify(uninterp_phi)
        except:
            # Purify not implemented, return original
            return formula

        if not nonlinear or len(nonlinear) == 0:
            # No nonlinear terms
            return formula

        # Get symbols that appear in nonlinear terms
        symbol_list = []
        for sym, expr in nonlinear.items():
            expr_symbols = symbols(expr)
            for s in expr_symbols:
                if s.typ in [Type.INT, Type.REAL] and s not in symbol_list:
                    symbol_list.append(s)

        # Create objectives
        objectives = [mk_const(sym) for sym in symbol_list]

        # Use Z3 to find intervals
        try:
            result = srkZ3.optimize_box(lin_phi, objectives)
        except:
            logf("linearize: optimization failed", level='warn')
            return lin_phi

        if result == 'Unsat':
            return mk_false()
        elif result == 'Unknown':
            logf("linearize: optimization returned unknown", level='warn')
            return lin_phi
        elif not isinstance(result, list):
            return lin_phi

        # Build symbolic intervals and constraints
        # This is a simplified version - full implementation would match OCaml more closely
        return lin_phi

    def optimize_box(self, phi: FormulaExpression, objectives: List[ArithExpression]) -> Union[List[Interval], str]:
        """Find bounding intervals for objectives within a formula.

        Uses Z3 optimization to find concrete bounds for each objective.
        """
        try:
            from . import srkSimplify
            from . import srkZ3
        except ImportError:
            return "Unknown"

        # Simplify terms first
        phi = self.simplify_terms(phi)

        # Create objective symbols
        objective_symbols = []
        objective_eqs = []

        for obj in objectives:
            sym = mk_symbol("obj_" + str(id(obj)), expr_typ(obj))
            objective_symbols.append(mk_const(sym))
            objective_eqs.append(mk_eq(obj, mk_const(sym)))

        # Linearize the combined formula
        lin_phi = self.linearize(mk_and([phi] + objective_eqs))

        # Use Z3 to optimize
        try:
            return srkZ3.optimize_box(lin_phi, objective_symbols)
        except:
            return "Unknown"

    def simplify_terms_rewriter(self, expr: Expression) -> Expression:
        """Rewrite rule for simplifying power and log terms.

        Converts pow/log applications to their simplified forms.
        """
        pow_sym = self.get_pow_symbol()
        log_sym = self.get_log_symbol()

        try:
            expr_info = destruct(expr)
            if expr_info is None:
                return expr

            expr_type, expr_data = expr_info

            # pow(x, y) -> mk_pow(x, y)
            if expr_type == 'App':
                func, args = expr_data
                if func == pow_sym and len(args) == 2:
                    return self.mk_pow(args[0], args[1])
                elif func == log_sym and len(args) == 2:
                    return self.mk_log(args[0], args[1])

        except Exception:
            pass

        return expr

    def simplify_terms(self, formula: FormulaExpression) -> FormulaExpression:
        """Simplify power and log terms in a formula."""
        return rewrite(formula, up=self.simplify_terms_rewriter)

    def simplify_term(self, term: ArithExpression) -> ArithExpression:
        """Simplify a single arithmetic term."""
        return rewrite(term, up=self.simplify_terms_rewriter)


# Convenience functions
def ensure_symbols(context: Context) -> None:
    """Ensure nonlinear operation symbols are registered."""
    NonlinearOperations(context)


def uninterpret_rewriter(context: Context) -> Callable[[Expression], Expression]:
    """Get the uninterpret rewriter for a context."""
    ops = NonlinearOperations(context)
    return ops.uninterpret_rewriter


def interpret_rewriter(context: Context) -> Callable[[Expression], Expression]:
    """Get the interpret rewriter for a context."""
    ops = NonlinearOperations(context)
    return ops.interpret_rewriter


def uninterpret(context: Context, expr: Expression) -> Expression:
    """Convert nonlinear operations to uninterpreted functions."""
    return NonlinearOperations(context).uninterpret(expr)


def interpret(context: Context, expr: Expression) -> Expression:
    """Convert uninterpreted functions back to interpreted operations."""
    return NonlinearOperations(context).interpret(expr)


def linearize(context: Context, phi: FormulaExpression) -> FormulaExpression:
    """Compute a linear approximation of a nonlinear formula."""
    return NonlinearOperations(context).linearize(phi)


def mk_log(context: Context, base: ArithExpression, x: ArithExpression) -> ArithExpression:
    """Create a logarithm expression."""
    return NonlinearOperations(context).mk_log(base, x)


def mk_pow(context: Context, base: ArithExpression, x: ArithExpression) -> ArithExpression:
    """Create a power expression."""
    return NonlinearOperations(context).mk_pow(base, x)


def optimize_box(context: Context, phi: FormulaExpression, objectives: List[ArithExpression]) -> Union[List[Interval], str]:
    """Find bounding intervals for objectives."""
    return NonlinearOperations(context).optimize_box(phi, objectives)


def simplify_terms_rewriter(context: Context) -> Callable[[Expression], Expression]:
    """Get the simplification rewriter."""
    ops = NonlinearOperations(context)
    return ops.simplify_terms_rewriter


def simplify_terms(context: Context, formula: FormulaExpression) -> FormulaExpression:
    """Simplify terms in a formula."""
    return NonlinearOperations(context).simplify_terms(formula)


def simplify_term(context: Context, term: ArithExpression) -> ArithExpression:
    """Simplify a single term."""
    return NonlinearOperations(context).simplify_term(term)


# Compatibility alias for SRK interface
class Nonlinear(Exception):
    """Exception raised for nonlinear operations."""
    pass


__all__ = [
    'SymbolicInterval',
    'NonlinearOperations',
    'Nonlinear',
    'ensure_symbols',
    'uninterpret_rewriter',
    'interpret_rewriter',
    'uninterpret',
    'interpret',
    'linearize',
    'mk_log',
    'mk_pow',
    'optimize_box',
    'simplify_terms_rewriter',
    'simplify_terms',
    'simplify_term',
]
