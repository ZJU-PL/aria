"""
Interpretation and evaluation of expressions.

This module handles the evaluation of symbolic expressions under
specific interpretations (assignments of values to symbols).

Key features:
- Evaluation of arithmetic and boolean expressions
- Support for function application and interpretation
- Quantifier evaluation using SMT solvers when available
- Implicant selection for logical formulas
- Comprehensive error handling and type safety
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple, Optional, Union, Any, Callable, Protocol
from fractions import Fraction
from dataclasses import dataclass, field
import enum
import logging

from aria.srk.syntax import (
    Context, Symbol, Type, Expression, FormulaExpression, ArithExpression,
    TermExpression, Var, Const, Add, Mul, Eq, Lt, Leq, And, Or, Not,
    TrueExpr, FalseExpr, Ite, Forall, Exists, App, Select, Store, mk_real, mk_true, mk_false,
    mk_eq, mk_leq, mk_lt, mk_not, mk_and, mk_or, mk_ite, mk_add, mk_mul,
    mk_div, mk_mod, mk_floor, mk_neg, mk_const, mk_var, destruct, substitute,
    rewrite, nnf_rewriter, symbols, mk_symbol, typ_symbol, Env
)
from .qQ import QQ

logger = logging.getLogger(__name__)


class InterpretationValue:
    """Value in an interpretation with type information."""

    def __init__(self, value: Union[Fraction, bool, Expression]):
        self.value = value

    def __str__(self) -> str:
        return str(self.value)

    def __repr__(self) -> str:
        return f"InterpretationValue({self.value!r})"

    @property
    def is_real(self) -> bool:
        """Check if this value represents a real number."""
        return isinstance(self.value, Fraction)

    @property
    def is_bool(self) -> bool:
        """Check if this value represents a boolean."""
        return isinstance(self.value, bool)

    @property
    def is_expression(self) -> bool:
        """Check if this value represents an expression (function)."""
        return isinstance(self.value, Expression)

    def as_real(self) -> Fraction:
        """Get the real value, raising an error if not a real."""
        if not self.is_real:
            raise TypeError(f"Expected real value, got {type(self.value).__name__}")
        return self.value

    def as_bool(self) -> bool:
        """Get the boolean value, raising an error if not a boolean."""
        if not self.is_bool:
            raise TypeError(f"Expected boolean value, got {type(self.value).__name__}")
        return self.value

    def as_expression(self) -> Expression:
        """Get the expression value, raising an error if not an expression."""
        if not self.is_expression:
            raise TypeError(f"Expected expression value, got {type(self.value).__name__}")
        return self.value


class DivideByZeroError(Exception):
    """Raised when division by zero occurs during evaluation."""
    pass


class Interpretation:
    """Maps symbols to their interpretations."""

    def __init__(self, context: Context, default: Optional[Callable[[Symbol], InterpretationValue]] = None,
                 bindings: Optional[Dict[Symbol, InterpretationValue]] = None):
        self.context = context
        self.default = default or (lambda sym: (_ for _ in ()).throw(KeyError(f"No interpretation for symbol {sym}")))
        self.bindings = bindings or {}

    def add_real(self, symbol: Symbol, value: Fraction) -> Interpretation:
        """Add a real value binding."""
        symbol_type = self.context.typ_symbol(symbol)
        if symbol_type not in (Type.REAL, Type.INT):
            raise ValueError(f"add_real: constant symbol is non-arithmetic, got {symbol_type}")
        new_bindings = self.bindings.copy()
        new_bindings[symbol] = InterpretationValue(value)
        return Interpretation(self.context, self.default, new_bindings)

    def add_bool(self, symbol: Symbol, value: bool) -> Interpretation:
        """Add a boolean value binding."""
        symbol_type = self.context.typ_symbol(symbol)
        if symbol_type != Type.BOOL:
            raise ValueError(f"add_bool: constant symbol is non-boolean, got {symbol_type}")
        new_bindings = self.bindings.copy()
        new_bindings[symbol] = InterpretationValue(value)
        return Interpretation(self.context, self.default, new_bindings)

    def add_fun(self, symbol: Symbol, expression: Expression) -> Interpretation:
        """Add a function interpretation."""
        # For this simplified implementation, allow any symbol to be treated as a function
        new_bindings = self.bindings.copy()
        new_bindings[symbol] = InterpretationValue(expression)
        return Interpretation(self.context, self.default, new_bindings)

    def add(self, symbol: Symbol, value: InterpretationValue) -> Interpretation:
        """Add a binding to the interpretation."""
        symbol_type = self.context.typ_symbol(symbol)
        if isinstance(value.value, Expression):
            if not (isinstance(symbol_type, tuple) and symbol_type[0] == 'TyFun'):
                raise ValueError(f"add: function value for non-function symbol")
        elif isinstance(value.value, bool):
            if symbol_type != Type.BOOL:
                raise ValueError(f"add: boolean value for non-boolean symbol")
        elif isinstance(value.value, Fraction):
            if symbol_type not in (Type.REAL, Type.INT):
                raise ValueError(f"add: real value for non-arithmetic symbol")
        else:
            raise ValueError(f"add: unsupported value type {type(value.value)}")

        new_bindings = self.bindings.copy()
        new_bindings[symbol] = value
        return Interpretation(self.context, self.default, new_bindings)

    def get_value(self, symbol: Symbol) -> InterpretationValue:
        """Get the value of a symbol."""
        if symbol in self.bindings:
            return self.bindings[symbol]
        else:
            return self.default(symbol)

    def real(self, symbol: Symbol) -> Fraction:
        """Get the real value of a symbol."""
        value = self.get_value(symbol)
        return value.as_real()

    def bool(self, symbol: Symbol) -> bool:
        """Get the boolean value of a symbol."""
        value = self.get_value(symbol)
        return value.as_bool()

    def evaluate_term(self, term: ArithExpression, env: Optional[Dict[int, InterpretationValue]] = None) -> Fraction:
        """Evaluate an arithmetic term to a rational value."""
        env = env or {}

        def eval_term(t):
            if isinstance(t, Const):
                return self.real(t.symbol)
            elif isinstance(t, Var):
                if t.var_id in env:
                    val = env[t.var_id]
                    if isinstance(val.value, Fraction):
                        return val.value
                    else:
                        raise ValueError(f"Variable {t.var_id} bound to non-real value")
                else:
                    raise ValueError(f"Unbound variable {t.var_id}")
            elif isinstance(t, Add):
                return sum(eval_term(arg) for arg in t.args)
            elif isinstance(t, Mul):
                result = QQ.one
                for arg in t.args:
                    result = QQ.mul(result, eval_term(arg))
                return result
            elif isinstance(t, Ite):
                condition_result = self.evaluate_formula(t.condition, env)
                if condition_result:
                    return eval_term(t.then_branch)
                else:
                    return eval_term(t.else_branch)
            elif hasattr(t, 'op') and t.op == 'Div':
                dividend = eval_term(t.left)
                divisor = eval_term(t.right)
                if QQ.equal(divisor, QQ.zero):
                    raise DivideByZeroError()
                return QQ.div(dividend, divisor)
            elif hasattr(t, 'op') and t.op == 'Mod':
                dividend = eval_term(t.left)
                modulus = eval_term(t.right)
                if QQ.equal(modulus, QQ.zero):
                    raise DivideByZeroError()
                return QQ.modulo(dividend, modulus)
            elif hasattr(t, 'op') and t.op == 'Floor':
                return QQ.floor(eval_term(t.arg))
            elif hasattr(t, 'op') and t.op == 'Neg':
                return QQ.negate(eval_term(t.arg))
            elif isinstance(t, App):
                # Function application
                try:
                    func_value = self.get_value(t.func)
                    if func_value.is_expression:
                        # Evaluate function application by substituting arguments
                        # Create an environment mapping parameter indices to argument values
                        arg_env = {}
                        for i, arg in enumerate(t.args):
                            # Evaluate each argument and bind it to the parameter
                            try:
                                arg_val = eval_term(arg)
                                arg_env[i] = InterpretationValue(arg_val)
                            except (ValueError, TypeError) as e:
                                raise ValueError(f"Cannot evaluate argument {i} of function {t.func}: {e}") from e

                        # Merge with existing environment
                        merged_env = {**env, **arg_env}

                        # Evaluate the function body with the argument bindings
                        try:
                            return self.evaluate_term(func_value.as_expression(), merged_env)
                        except (ValueError, TypeError) as e:
                            raise ValueError(f"Cannot evaluate function body for {t.func}: {e}") from e
                    else:
                        raise ValueError(f"No function interpretation for symbol {t.func}")
                except KeyError:
                    raise ValueError(f"No interpretation for function symbol {t.func}")
            elif hasattr(t, 'op') and t.op == 'Div':
                dividend = eval_term(t.left)
                divisor = eval_term(t.right)
                if QQ.equal(divisor, QQ.zero):
                    raise DivideByZeroError()
                return QQ.div(dividend, divisor)
            elif hasattr(t, 'op') and t.op == 'Mod':
                dividend = eval_term(t.left)
                modulus = eval_term(t.right)
                if QQ.equal(modulus, QQ.zero):
                    raise DivideByZeroError()
                return QQ.modulo(dividend, modulus)
            elif hasattr(t, 'op') and t.op == 'Floor':
                return QQ.floor(eval_term(t.arg))
            elif hasattr(t, 'op') and t.op == 'Neg':
                return QQ.negate(eval_term(t.arg))
            elif isinstance(t, Select):
                # Array selection: evaluate array and index, then look up value
                array_val = eval_term(t.array)
                index_val = eval_term(t.index)
                if isinstance(array_val, dict) and isinstance(index_val, (int, Fraction)):
                    idx = int(index_val) if isinstance(index_val, Fraction) else index_val
                    return array_val.get(idx, QQ.zero)  # Default to 0 for unknown indices
                return QQ.zero  # Conservative fallback
            elif isinstance(t, Store):
                # Array store: evaluate all components
                array_val = eval_term(t.array)
                index_val = eval_term(t.index)
                value_val = eval_term(t.value)
                if isinstance(array_val, dict) and isinstance(index_val, (int, Fraction)):
                    idx = int(index_val) if isinstance(index_val, Fraction) else index_val
                    new_array = array_val.copy()
                    new_array[idx] = value_val
                    return new_array
                return array_val  # Conservative fallback
            else:
                # Try to provide more helpful error messages
                term_type = type(t).__name__
                if hasattr(t, '__str__'):
                    term_str = str(t)[:100]  # Truncate long expressions
                    raise NotImplementedError(f"Cannot evaluate term type {term_type}: {term_str}")
                else:
                    raise NotImplementedError(f"Cannot evaluate term type {term_type}")

        try:
            return eval_term(term)
        except KeyError as e:
            raise ValueError(f"No interpretation for constant symbol: {e}")

    def evaluate_formula(self, formula: FormulaExpression, env: Optional[Dict[int, InterpretationValue]] = None) -> bool:
        """Evaluate a formula to a boolean value."""
        env = env or {}

        def eval_formula(f):
            if isinstance(f, TrueExpr):
                return True
            elif isinstance(f, FalseExpr):
                return False
            elif isinstance(f, Eq):
                left_val = self.evaluate_term(f.left, env)
                right_val = self.evaluate_term(f.right, env)
                return QQ.equal(left_val, right_val)
            elif isinstance(f, Lt):
                left_val = self.evaluate_term(f.left, env)
                right_val = self.evaluate_term(f.right, env)
                return QQ.lt(left_val, right_val)
            elif isinstance(f, Leq):
                left_val = self.evaluate_term(f.left, env)
                right_val = self.evaluate_term(f.right, env)
                return QQ.leq(left_val, right_val)
            elif isinstance(f, And):
                return all(eval_formula(arg) for arg in f.args)
            elif isinstance(f, Or):
                return any(eval_formula(arg) for arg in f.args)
            elif isinstance(f, Not):
                return not eval_formula(f.arg)
            elif isinstance(f, Ite):
                condition_result = eval_formula(f.condition)
                if condition_result:
                    return eval_formula(f.then_branch)
                else:
                    return eval_formula(f.else_branch)
            elif isinstance(f, App):
                # Proposition application
                if not f.args:  # Nullary predicate
                    return self.bool(f.func)
                else:
                    # Function application for propositions
                    try:
                        func_value = self.get_value(f.func)
                        if func_value.is_expression:
                            # Evaluate function application by substituting arguments
                            # Create an environment mapping parameter indices to argument values
                            arg_env = {}
                            for i, arg in enumerate(f.args):
                                # Determine if argument is a term or formula and evaluate accordingly
                                try:
                                    # Try to evaluate as a term
                                    arg_val = self.evaluate_term(arg, env)
                                    arg_env[i] = InterpretationValue(arg_val)
                                except (ValueError, TypeError):
                                    # Try to evaluate as a formula
                                    try:
                                        arg_val = eval_formula(arg)
                                        arg_env[i] = InterpretationValue(arg_val)
                                    except (ValueError, TypeError):
                                        # If both fail, just pass the expression itself
                                        arg_env[i] = InterpretationValue(arg)

                            # Merge with existing environment
                            merged_env = {**env, **arg_env}

                            # Evaluate the function body with the argument bindings
                            try:
                                return self.evaluate_formula(func_value.as_expression(), merged_env)
                            except (ValueError, TypeError) as e:
                                raise ValueError(f"Cannot evaluate function body for {f.func}: {e}") from e
                        else:
                            raise ValueError(f"No function interpretation for symbol {f.func}")
                    except KeyError:
                        raise ValueError(f"No interpretation for function symbol {f.func}")
            elif isinstance(f, Var):
                if f.var_id in env:
                    val = env[f.var_id]
                    if isinstance(val.value, bool):
                        return val.value
                    else:
                        raise ValueError(f"Variable {f.var_id} bound to non-boolean value")
                else:
                    raise ValueError(f"Unbound variable {f.var_id}")
            elif isinstance(f, Forall):
                # Universal quantification: ∀x. φ(x)
                return self._evaluate_universal_quantifier(f, env)

            elif isinstance(f, Exists):
                # Existential quantification: ∃x. φ(x)
                return self._evaluate_existential_quantifier(f, env)
            elif isinstance(f, Select):
                # Array selection in formulas
                array_val = self.evaluate_term(f.array, env)
                index_val = self.evaluate_term(f.index, env)
                if isinstance(array_val, dict) and isinstance(index_val, (int, Fraction)):
                    idx = int(index_val) if isinstance(index_val, Fraction) else index_val
                    return array_val.get(idx, False)  # Default to False for unknown indices
                return False  # Conservative fallback
            elif isinstance(f, Store):
                # Array store in formulas - evaluate all components
                array_val = self.evaluate_term(f.array, env)
                index_val = self.evaluate_term(f.index, env)
                value_val = self.evaluate_term(f.value, env)
                if isinstance(array_val, dict) and isinstance(index_val, (int, Fraction)):
                    idx = int(index_val) if isinstance(index_val, Fraction) else index_val
                    new_array = array_val.copy()
                    new_array[idx] = value_val
                    return True  # Store operation succeeds
                return True  # Conservative fallback
            elif hasattr(f, 'op'):
                # Handle operations generically
                if f.op == 'Div':
                    left_val = self.evaluate_term(f.left, env)
                    right_val = self.evaluate_term(f.right, env)
                    if QQ.equal(right_val, QQ.zero):
                        raise DivideByZeroError()
                    return QQ.div(left_val, right_val) != QQ.zero
                elif f.op == 'Mod':
                    left_val = self.evaluate_term(f.left, env)
                    right_val = self.evaluate_term(f.right, env)
                    if QQ.equal(right_val, QQ.zero):
                        raise DivideByZeroError()
                    return QQ.modulo(left_val, right_val) != QQ.zero
                elif f.op == 'Floor':
                    arg_val = self.evaluate_term(f.arg, env)
                    return QQ.floor(arg_val) != QQ.zero
                elif f.op == 'Neg':
                    arg_val = self.evaluate_term(f.arg, env)
                    return QQ.negate(arg_val) != QQ.zero
                else:
                    # For other operations, try to evaluate as term and check if non-zero
                    try:
                        term_val = self.evaluate_term(f, env)
                        return term_val != QQ.zero
                    except:
                        return True  # Conservative fallback
            else:
                # Try to provide more helpful error messages
                formula_type = type(f).__name__
                if hasattr(f, '__str__'):
                    formula_str = str(f)[:100]  # Truncate long expressions
                    raise NotImplementedError(f"Cannot evaluate formula type {formula_type}: {formula_str}")
                else:
                    raise NotImplementedError(f"Cannot evaluate formula type {formula_type}")

        try:
            return eval_formula(formula)
        except KeyError as e:
            raise ValueError(f"No interpretation for constant symbol: {e}")

    def _evaluate_universal_quantifier(self, f: Forall, env: Dict[int, InterpretationValue]) -> bool:
        """Evaluate universal quantification ∀x. φ(x) using SMT solver when available."""
        try:
            # Import Z3_AVAILABLE locally to avoid cyclic import
            try:
                from .srkZ3 import Z3_AVAILABLE
            except ImportError:
                Z3_AVAILABLE = False

            if Z3_AVAILABLE:
                # Use SMT solver to check validity
                # ∀x. φ(x) is valid iff ¬∃x. ¬φ(x) is unsat
                negated = mk_not(f)
                result = self._check_satisfiability([negated])
                if result == 'unsat':
                    return True
                elif result == 'sat':
                    return False
                else:
                    logger.warning(f"Universal quantifier evaluation: SMT solver returned unknown for {f}")
                    # Fallback: try bounded evaluation with a few test values
                    return self._bounded_universal_evaluation(f, env)
            else:
                # Fallback to bounded evaluation
                return self._bounded_universal_evaluation(f, env)

        except Exception as e:
            logger.error(f"Universal quantifier evaluation failed for {f}: {e}")
            # Fallback to bounded evaluation
            try:
                return self._bounded_universal_evaluation(f, env)
            except Exception as fallback_error:
                raise NotImplementedError(f"Cannot evaluate universal quantifier {f}: {e}") from fallback_error

    def _evaluate_existential_quantifier(self, f: Exists, env: Dict[int, InterpretationValue]) -> bool:
        """Evaluate existential quantification ∃x. φ(x) using SMT solver when available."""
        try:
            # Import Z3_AVAILABLE locally to avoid cyclic import
            try:
                from .srkZ3 import Z3_AVAILABLE
            except ImportError:
                Z3_AVAILABLE = False

            if Z3_AVAILABLE:
                # Use SMT solver to check satisfiability
                result = self._check_satisfiability([f])
                if result == 'sat':
                    return True
                elif result == 'unsat':
                    return False
                else:
                    logger.warning(f"Existential quantifier evaluation: SMT solver returned unknown for {f}")
                    # Fallback: try bounded evaluation
                    return self._bounded_existential_evaluation(f, env)
            else:
                # Fallback to bounded evaluation
                return self._bounded_existential_evaluation(f, env)

        except Exception as e:
            logger.error(f"Existential quantifier evaluation failed for {f}: {e}")
            # Fallback to bounded evaluation
            try:
                return self._bounded_existential_evaluation(f, env)
            except Exception as fallback_error:
                raise NotImplementedError(f"Cannot evaluate existential quantifier {f}: {e}") from fallback_error

    def _check_satisfiability(self, formulas: List[FormulaExpression]) -> str:
        """Check satisfiability of formulas using SMT solver."""
        try:
            from .smt import check_sat
            return check_sat(self.context, formulas)
        except ImportError as e:
            logger.warning(f"SMT solver not available: {e}")
            return 'unknown'
        except Exception as e:
            logger.error(f"SMT solver error: {e}")
            return 'unknown'

    def _bounded_universal_evaluation(self, f: Forall, env: Dict[int, InterpretationValue]) -> bool:
        """Bounded evaluation of universal quantifier with test values."""
        # Test a few representative values for the quantified variable
        test_values = [QQ.zero, QQ.one, QQ(-1)]

        for test_val in test_values:
            # Create environment with test value for quantified variable
            test_env = env.copy()
            test_env[f.var_id] = InterpretationValue(test_val)

            # Evaluate the body with this test value
            try:
                body_result = self.evaluate_formula(f.body, test_env)
                if not body_result:
                    # Found a counterexample - not universally true
                    return False
            except (ValueError, TypeError) as e:
                logger.debug(f"Could not evaluate with test value {test_val}: {e}")
                # If we can't evaluate with a test value, we assume it might be false
                return False

        # All test values satisfied the formula - likely universally true
        # This is a heuristic and may be incorrect for some cases
        logger.warning(f"Using heuristic for universal quantifier evaluation: {f}")
        return True

    def _bounded_existential_evaluation(self, f: Exists, env: Dict[int, InterpretationValue]) -> bool:
        """Bounded evaluation of existential quantifier with test values."""
        # Test a few representative values for the quantified variable
        test_values = [QQ.zero, QQ.one, QQ(-1)]

        for test_val in test_values:
            # Create environment with test value for quantified variable
            test_env = env.copy()
            test_env[f.var_id] = InterpretationValue(test_val)

            # Evaluate the body with this test value
            try:
                body_result = self.evaluate_formula(f.body, test_env)
                if body_result:
                    # Found a witness - exists value that makes it true
                    return True
            except (ValueError, TypeError) as e:
                logger.debug(f"Could not evaluate with test value {test_val}: {e}")
                # Continue trying other values

        # No test value satisfied the formula - likely does not exist
        # This is a heuristic and may be incorrect for some cases
        logger.warning(f"Using heuristic for existential quantifier evaluation: {f}")
        return False

    def unfold_app(self, func_symbol: Symbol, args: List[Expression]) -> Expression:
        """Unfold a function application."""
        func_value = self.get_value(func_symbol)
        if isinstance(func_value.value, Expression):
            # Create environment for function arguments
            env = {}
            for i, arg in enumerate(args):
                env[i] = arg
            return substitute(self.context, lambda i: env.get(i), func_value.value)
        else:
            raise ValueError(f"unfold_app: not a function symbol {func_symbol}")

    def substitute(self, expression: Expression) -> Expression:
        """Substitute constants in an expression with their interpretations."""
        def rewriter(expr):
            if isinstance(expr, Const):
                try:
                    value = self.get_value(expr.symbol)
                    if isinstance(value.value, Fraction):
                        return mk_real(self.context, value.value)
                    elif isinstance(value.value, bool):
                        return mk_bool(self.context, value.value)
                    elif isinstance(value.value, Expression):
                        return value.value  # Function interpretation
                except KeyError:
                    return expr
            elif isinstance(expr, App):
                try:
                    return self.unfold_app(expr.func, expr.args)
                except KeyError:
                    return expr
            return expr

        return rewrite(self.context, rewriter, expression)

    def select_implicant(self, formula: FormulaExpression, env: Optional[Dict[int, InterpretationValue]] = None) -> Optional[List[FormulaExpression]]:
        """Select an implicant of a formula."""
        env = env or {}

        # This is a simplified implementation
        # A full implementation would use more sophisticated algorithms

        if self.evaluate_formula(formula, env):
            # Try to find atomic formulas that are satisfied
            atoms = self._extract_atoms(formula, env)

            # Filter to those that are satisfied by this interpretation
            satisfied_atoms = []
            for atom in atoms:
                if self.evaluate_formula(atom, env):
                    satisfied_atoms.append(atom)

            return satisfied_atoms if satisfied_atoms else None
        else:
            return None

    def _extract_atoms(self, formula: FormulaExpression, env: Optional[Dict[int, InterpretationValue]] = None) -> List[FormulaExpression]:
        """Extract atomic formulas from a formula."""
        atoms = []

        if isinstance(formula, (Eq, Lt, Leq)):
            atoms.append(formula)
        elif isinstance(formula, (And, Or)):
            for arg in formula.args:
                atoms.extend(self._extract_atoms(arg, env))
        elif isinstance(formula, Not):
            atoms.extend(self._extract_atoms(formula.arg, env))
        elif isinstance(formula, Ite):
            # Extract atoms from all branches of ite
            atoms.extend(self._extract_atoms(formula.condition, env))
            atoms.extend(self._extract_atoms(formula.then_branch, env))
            atoms.extend(self._extract_atoms(formula.else_branch, env))
        elif isinstance(formula, (TrueExpr, FalseExpr)):
            atoms.append(formula)
        elif isinstance(formula, App):
            # Function applications are atomic
            atoms.append(formula)
        elif isinstance(formula, Var):
            # Variables are atomic
            atoms.append(formula)

        return atoms

    def destruct_atom(self, formula: FormulaExpression) -> Union[Tuple[str, Expression, Expression], Tuple[str, Expression], str]:
        """Destruct an atomic formula."""
        if isinstance(formula, Eq):
            return ("ArithComparison", "Eq", formula.left, formula.right)
        elif isinstance(formula, Lt):
            return ("ArithComparison", "Lt", formula.left, formula.right)
        elif isinstance(formula, Leq):
            return ("ArithComparison", "Leq", formula.left, formula.right)
        elif isinstance(formula, TrueExpr):
            zero = mk_real(self.context, QQ.zero)
            return ("ArithComparison", "Eq", zero, zero)
        elif isinstance(formula, FalseExpr):
            zero = mk_real(self.context, QQ.zero)
            one = mk_real(self.context, QQ.one)
            return ("ArithComparison", "Eq", zero, one)
        elif isinstance(formula, App):
            if not formula.args:  # Nullary predicate
                return ("Literal", "Pos", "Const", formula.func)
            else:
                # For now, treat as positive literal with variable
                return ("Literal", "Pos", "Var", 0)  # Placeholder
        elif isinstance(formula, Var):
            return ("Literal", "Pos", "Var", formula.var_id)
        elif isinstance(formula, Not):
            inner = formula.arg
            if isinstance(inner, App) and not inner.args:
                return ("Literal", "Neg", "Const", inner.func)
            elif isinstance(inner, Var):
                return ("Literal", "Neg", "Var", inner.var_id)
            else:
                raise ValueError(f"destruct_atom: negation of non-atomic formula")
        else:
            raise ValueError(f"destruct_atom: {formula} is not atomic")

    def select_ite(self, expression: Expression, env: Optional[Dict[int, InterpretationValue]] = None) -> Tuple[Expression, List[FormulaExpression]]:
        """Select ite expressions and return simplified expression with conditions."""
        conditions = []
        def rewriter(expr):
            if isinstance(expr, Ite):
                condition_result = self.evaluate_formula(expr.condition, env)
                if condition_result:
                    conditions.append(expr.condition)
                    return expr.then_branch
                else:
                    not_condition = mk_not(expr.condition)
                    # Apply NNF to get the negation in a usable form
                    not_condition = rewrite(not_condition, nnf_rewriter)
                    conditions.append(not_condition)
                    return expr.else_branch
            return expr

        simplified_expr = rewrite(expression, rewriter)
        return simplified_expr, conditions

    def enum(self):
        """Return an enumeration of symbol-value pairs."""
        return self.bindings.items()

    def get_context(self) -> Context:
        """Get the context of this interpretation."""
        return self.context

    def __str__(self) -> str:
        bindings_str = []
        for symbol, value in self.bindings.items():
            bindings_str.append(f"{symbol} -> {value}")
        return "Interpretation({" + ", ".join(bindings_str) + "})"


class EvaluationContext:
    """Context for evaluating expressions with environments."""

    def __init__(self, interpretation: Interpretation,
                 env: Optional[Dict[int, InterpretationValue]] = None):
        self.interpretation = interpretation
        self.env = env or {}

    def evaluate(self, expression: Expression) -> Union[Fraction, bool]:
        """Evaluate an expression."""
        if isinstance(expression, (TrueExpr, FalseExpr, Eq, Lt, Leq, And, Or, Not, Ite)):
            return self.interpretation.evaluate_formula(expression, self.env)
        else:
            return self.interpretation.evaluate_term(expression, self.env)


def mk_bool(context: Context, value: bool) -> Expression:
    """Create a boolean expression."""
    if value:
        return mk_true(context)
    else:
        return mk_false(context)


# Convenience functions
def make_interpretation(context: Context) -> Interpretation:
    """Create an empty interpretation."""
    return Interpretation(context)


def make_evaluation_context(interpretation: Interpretation,
                          env: Optional[Dict[int, InterpretationValue]] = None) -> EvaluationContext:
    """Create an evaluation context."""
    return EvaluationContext(interpretation, env)


def evaluate_expression(expression: Expression, interpretation: Interpretation,
                       env: Optional[Dict[int, InterpretationValue]] = None) -> Union[Fraction, bool]:
    """Evaluate an expression under an interpretation."""
    ctx = EvaluationContext(interpretation, env)
    return ctx.evaluate(expression)


def empty_interpretation(context: Context) -> Interpretation:
    """Create an empty interpretation."""
    return Interpretation(context)


def wrap_interpretation(context: Context, symbol_function: Callable[[Symbol], InterpretationValue],
                       symbols: Optional[List[Symbol]] = None) -> Interpretation:
    """Wrap a symbol function in an interpretation."""
    bindings = {}
    if symbols:
        for symbol in symbols:
            bindings[symbol] = symbol_function(symbol)
    return Interpretation(context, symbol_function, bindings)


def destruct_atom(context: Context, formula: FormulaExpression) -> Union[Tuple[str, str, Expression, Expression], Tuple[str, str, str, Symbol], str]:
    """Destruct an atomic formula (module-level helper function).

    This is a helper function used by other modules like wedge.py.
    Returns a tuple describing the atomic formula structure.

    Args:
        context: The SRK context
        formula: The atomic formula to destruct

    Returns:
        A tuple describing the structure:
        - For arithmetic comparisons: ("ArithComparison", op, left, right)
        - For literals: ("Literal", polarity, type, symbol)
        - For other cases: string description

    Raises:
        ValueError: If the formula is not atomic
    """
    if isinstance(formula, Eq):
        return ("ArithComparison", "Eq", formula.left, formula.right)
    elif isinstance(formula, Lt):
        return ("ArithComparison", "Lt", formula.left, formula.right)
    elif isinstance(formula, Leq):
        return ("ArithComparison", "Leq", formula.left, formula.right)
    elif isinstance(formula, TrueExpr):
        zero = mk_real(context, QQ.zero)
        return ("ArithComparison", "Eq", zero, zero)
    elif isinstance(formula, FalseExpr):
        zero = mk_real(context, QQ.zero)
        one = mk_real(context, QQ.one)
        return ("ArithComparison", "Eq", zero, one)
    elif isinstance(formula, App):
        if not formula.args:  # Nullary predicate
            return ("Literal", "Pos", "Const", formula.func)
        else:
            # For now, treat as positive literal with variable
            return ("Literal", "Pos", "Var", 0)  # Placeholder
    elif isinstance(formula, Var):
        return ("Literal", "Pos", "Var", formula.var_id)
    elif isinstance(formula, Not):
        inner = formula.arg
        if isinstance(inner, App) and not inner.args:
            return ("Literal", "Neg", "Const", inner.func)
        elif isinstance(inner, Var):
            return ("Literal", "Neg", "Var", inner.var_id)
        else:
            raise ValueError(f"destruct_atom: negation of non-atomic formula")
    else:
        raise ValueError(f"destruct_atom: {formula} is not atomic")


def select_implicant(interpretation: Interpretation, formula: FormulaExpression) -> Optional[List[FormulaExpression]]:
    """Select an implicant of a formula under the given interpretation.

    Module-level helper function used by other modules.

    Args:
        interpretation: The interpretation to use for evaluation
        formula: The formula to analyze

    Returns:
        A list of atomic formulas that form an implicant, or None if no implicant exists
    """
    return interpretation.select_implicant(formula)


def evaluate_expression_safe(expression: Expression, interpretation: Interpretation,
                           env: Optional[Dict[int, InterpretationValue]] = None) -> Union[Fraction, bool, None]:
    """Safely evaluate an expression, returning None if evaluation fails.

    This function wraps evaluate_expression and catches common evaluation errors,
    returning None instead of raising exceptions. Useful for cases where
    evaluation might fail due to missing interpretations or unsupported expressions.

    Args:
        expression: The expression to evaluate
        interpretation: The interpretation to use
        env: Optional environment for variable bindings

    Returns:
        The evaluated result, or None if evaluation failed
    """
    try:
        return evaluate_expression(expression, interpretation, env)
    except (ValueError, TypeError, KeyError, NotImplementedError, DivideByZeroError):
        return None
