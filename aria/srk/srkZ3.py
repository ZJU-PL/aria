"""
Z3 SMT solver integration for SRK.

This module provides integration with the Z3 SMT solver, allowing SRK
expressions to be translated to Z3 and solved using Z3's capabilities.
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple, Optional, Union, Any, Callable
from fractions import Fraction
from enum import Enum

try:
    import z3
    Z3_AVAILABLE = True
except ImportError:
    z3 = None
    Z3_AVAILABLE = False

from aria.srk.syntax import (
    Context, Symbol, Type, Expression, Var, Const, Add, Mul, Eq, Lt, Leq, And, Or, Not,
    TrueExpr, FalseExpr, Ite, Forall, Exists, App, Select, Store, mk_real, mk_true, mk_false,
    mk_eq, mk_leq, mk_lt, mk_not, mk_and, mk_or, mk_ite, mk_add, mk_mul,
    mk_div, mk_mod, mk_floor, mk_neg, mk_const, mk_var, mk_symbol, typ_symbol,
    int_of_symbol, symbol_of_int, symbols, mk_exists, mk_forall
)
from aria.srk.interpretation import Interpretation, InterpretationValue
from aria.srk.qQ import QQ
from aria.srk.log import logf


class Z3Result(Enum):
    """Result of Z3 query."""
    SAT = "sat"
    UNSAT = "unsat"
    UNKNOWN = "unknown"


def bool_val(z3_expr) -> bool:
    """Convert Z3 boolean value to Python boolean."""
    if not Z3_AVAILABLE:
        raise ImportError("Z3 is not available")

    try:
        result = z3.Boolean.get_bool_value(z3_expr)
        if result == z3.L_TRUE:
            return True
        elif result == z3.L_FALSE:
            return False
        else:
            raise ValueError("bool_val: not a Boolean")
    except Exception as e:
        raise ValueError(f"bool_val: {e}")


def qq_val(z3_expr) -> Fraction:
    """Convert Z3 rational value to QQ fraction."""
    if not Z3_AVAILABLE:
        raise ImportError("Z3 is not available")
    if (hasattr(z3_expr, 'is_int') and z3_expr.is_int()) or \
       (hasattr(z3_expr, 'is_rational') and z3_expr.is_rational()):
        return QQ.of_string(str(z3_expr))
    raise ValueError("qq_val: not a numeral")


def typ_of_sort(sort) -> Type:
    """Convert Z3 sort to SRK type."""
    if not Z3_AVAILABLE:
        raise ImportError("Z3 is not available")

    if str(sort) == 'Int':
        return Type.INT
    elif str(sort) == 'Real':
        return Type.REAL
    elif str(sort) == 'Bool':
        return Type.BOOL
    elif 'Array' in str(sort):
        return Type.ARRAY
    else:
        raise ValueError(f"typ_of_sort: unsupported sort {sort}")


def sort_of_typ(z3_ctx, typ: Type) -> Any:
    """Convert SRK type to Z3 sort."""
    if not Z3_AVAILABLE:
        raise ImportError("Z3 is not available")

    type_map = {
        Type.INT: z3.IntSort,
        Type.REAL: z3.RealSort,
        Type.BOOL: z3.BoolSort,
    }

    if typ in type_map:
        return type_map[typ](z3_ctx)
    elif typ == Type.ARRAY:
        return z3.ArraySort(z3_ctx, sort_of_typ(z3_ctx, Type.INT), sort_of_typ(z3_ctx, Type.INT))
    raise ValueError(f"sort_of_typ: unsupported type {typ}")


class Z3Model:
    """Represents a model from Z3."""

    def __init__(self, z3_model, srk_ctx: Context):
        if not Z3_AVAILABLE:
            raise ImportError("Z3 is not available")
        self.z3_model = z3_model
        self.srk_ctx = srk_ctx

    def get_value(self, symbol: Symbol) -> InterpretationValue:
        """Get the value of a symbol in the model."""
        if not Z3_AVAILABLE:
            raise ImportError("Z3 is not available")

        symbol_type = typ_symbol(self.srk_ctx, symbol)
        if symbol_type in (Type.REAL, Type.INT):
            # Create a Z3 expression for the symbol and evaluate it
            sort = sort_of_typ(self.z3_model.ctx, symbol_type)
            z3_symbol = z3.Const(int_of_symbol(symbol), sort)
            z3_expr = z3_symbol
            try:
                result = self.z3_model.eval(z3_expr, True)
                if result is not None:
                    return InterpretationValue(qq_val(result))
                else:
                    raise ValueError(f"No value for symbol {symbol}")
            except Exception:
                raise ValueError(f"Cannot evaluate symbol {symbol}")
        elif symbol_type == Type.BOOL:
            sort = sort_of_typ(self.z3_model.ctx, symbol_type)
            z3_symbol = z3.FuncDecl.mk_const_decl(self.z3_model.ctx,
                                                  z3.Symbol.mk_int(self.z3_model.ctx, int_of_symbol(symbol)),
                                                  sort)
            z3_expr = z3.Expr.mk_const_f(self.z3_model.ctx, z3_symbol)
            try:
                result = self.z3_model.eval(z3_expr, True)
                if result is not None:
                    return InterpretationValue(bool_val(result))
                else:
                    raise ValueError(f"No value for symbol {symbol}")
            except Exception:
                raise ValueError(f"Cannot evaluate symbol {symbol}")
        else:
            raise ValueError(f"Unsupported symbol type: {symbol_type}")


class SrkZ3:
    """Z3 context for SRK integration."""

    def __init__(self, context: Context):
        """Initialize Z3 context."""
        if not Z3_AVAILABLE:
            raise ImportError("Z3 is not available")
        self.srk_context = context
        self.z3_ctx = z3.Context()
        self.solver = z3.Solver(ctx=self.z3_ctx)

        # Mapping from SRK symbols to Z3 expressions
        self.symbol_map: Dict[Symbol, z3.ExprRef] = {}

    def z3_of_symbol(self, symbol: Symbol) -> z3.Symbol:
        """Convert SRK symbol to Z3 symbol."""
        return z3.Symbol.mk_int(self.z3_ctx, int_of_symbol(symbol))

    def decl_of_symbol(self, symbol: Symbol) -> z3.FuncDecl:
        """Create Z3 function declaration for SRK symbol."""
        symbol_type = typ_symbol(symbol)
        if symbol_type in (Type.INT, Type.REAL, Type.BOOL, Type.ARRAY):
            raise ValueError(f"decl_of_symbol: not a function symbol, got {symbol_type}")

        if isinstance(symbol_type, tuple) and symbol_type[0] == 'TyFun':
            params, ret_type = symbol_type[1], symbol_type[2]
            param_sorts = [sort_of_typ(self.z3_ctx, p) for p in params]
            return_sort = sort_of_typ(self.z3_ctx, ret_type)
            return z3.FuncDecl.mk_func_decl(self.z3_ctx,
                                           self.z3_of_symbol(symbol),
                                           param_sorts,
                                           return_sort)
        else:
            raise ValueError(f"decl_of_symbol: invalid function type {symbol_type}")

    def get_z3_symbol(self, symbol: Symbol) -> z3.ExprRef:
        """Get or create Z3 expression for SRK symbol."""
        if symbol in self.symbol_map:
            return self.symbol_map[symbol]

        # Create Z3 expression based on symbol type
        symbol_type = typ_symbol(symbol)
        if symbol_type not in (Type.INT, Type.REAL, Type.BOOL):
            raise ValueError(f"Unsupported symbol type: {symbol_type}")

        sort = sort_of_typ(self.z3_ctx, symbol_type)
        z3_expr = z3.Const(int_of_symbol(symbol), sort)
        self.symbol_map[symbol] = z3_expr
        return z3_expr

    def z3_of_expr(self, expr: Expression) -> z3.ExprRef:
        """Convert SRK expression to Z3 expression."""
        return self._z3_of_expr(expr)

    def _z3_of_expr(self, expr: Expression) -> z3.ExprRef:
        """Internal conversion function."""
        if isinstance(expr, Const):
            return self.get_z3_symbol(expr.symbol)
        elif isinstance(expr, Symbol):
            # Direct symbol reference - convert to Z3 expression
            return self.get_z3_symbol(expr)
        elif isinstance(expr, Var):
            # Variables need special handling - they should be bound variables
            # For now, treat as integer variables
            return z3.Quantifier.mk_bound(self.z3_ctx, expr.var_id,
                                         sort_of_typ(self.z3_ctx, expr.var_type))
        elif isinstance(expr, Add):
            if not expr.args:
                return z3.IntVal(0, self.z3_ctx)
            result = self._z3_of_expr(expr.args[0])
            for arg in expr.args[1:]:
                result = result + self._z3_of_expr(arg)
            return result
        elif isinstance(expr, Mul):
            if not expr.args:
                return z3.IntVal(1, self.z3_ctx)
            result = self._z3_of_expr(expr.args[0])
            for arg in expr.args[1:]:
                result = result * self._z3_of_expr(arg)
            return result
        elif isinstance(expr, Ite):
            condition = self.z3_of_formula(expr.condition)
            then_branch = self._z3_of_expr(expr.then_branch)
            else_branch = self._z3_of_expr(expr.else_branch)
            return z3.If(condition, then_branch, else_branch)
        elif isinstance(expr, App):
            if not expr.args:  # Nullary function
                return self.get_z3_symbol(expr.symbol)
            else:
                decl = self.decl_of_symbol(expr.symbol)
                args = [self._z3_of_expr(arg) for arg in expr.args]
                return z3.Expr.mk_app(self.z3_ctx, decl, args)
        elif isinstance(expr, Select):
            # Array selection: select(array, index)
            array_expr = self._z3_of_expr(expr.array)
            index_expr = self._z3_of_expr(expr.index)
            return z3.Select(array_expr, index_expr)
        elif isinstance(expr, (TrueExpr, FalseExpr, And, Or, Not, Eq, Lt, Leq)):
            # These are formula expressions - convert using z3_of_formula
            return self.z3_of_formula(expr)
        elif hasattr(expr, 'left') and hasattr(expr, 'right'):
            # Handle binary operations generically
            left = self._z3_of_expr(expr.left)
            right = self._z3_of_expr(expr.right)
            if isinstance(expr, (Eq,)):
                return z3.Boolean.mk_eq(self.z3_ctx, left, right)
            elif isinstance(expr, (Lt,)):
                return z3.Arithmetic.mk_lt(self.z3_ctx, left, right)
            elif isinstance(expr, (Leq,)):
                return z3.Arithmetic.mk_le(self.z3_ctx, left, right)
            else:
                # Try to convert as formula
                return self.z3_of_formula(expr)
        elif hasattr(expr, 'args') and hasattr(expr, 'symbol'):
            # Handle n-ary operations generically
            if not expr.args:
                return self.get_z3_symbol(expr.symbol)
            else:
                # Try to determine operation type from symbol name or type
                symbol_name = str(expr.symbol) if hasattr(expr.symbol, '__str__') else ""
                if 'add' in symbol_name.lower() or '+' in symbol_name:
                    args_z3 = [self._z3_of_expr(arg) for arg in expr.args]
                    return z3.Arithmetic.mk_add(self.z3_ctx, args_z3)
                elif 'mul' in symbol_name.lower() or '*' in symbol_name:
                    args_z3 = [self._z3_of_expr(arg) for arg in expr.args]
                    return z3.Arithmetic.mk_mul(self.z3_ctx, args_z3)
                elif 'and' in symbol_name.lower():
                    args_z3 = [self.z3_of_formula(arg) for arg in expr.args]
                    return z3.Boolean.mk_and(self.z3_ctx, args_z3)
                elif 'or' in symbol_name.lower():
                    args_z3 = [self.z3_of_formula(arg) for arg in expr.args]
                    return z3.Boolean.mk_or(self.z3_ctx, args_z3)
                else:
                    # Default to function application
                    decl = self.decl_of_symbol(expr.symbol)
                    args_z3 = [self._z3_of_expr(arg) for arg in expr.args]
                    return z3.Expr.mk_app(self.z3_ctx, decl, args_z3)
        else:
            # Try to provide more helpful error messages
            expr_type = type(expr).__name__
            if hasattr(expr, '__str__'):
                expr_str = str(expr)[:100]  # Truncate long expressions
                raise NotImplementedError(f"Unsupported expression type {expr_type}: {expr_str}")
            else:
                raise NotImplementedError(f"Unsupported expression type {expr_type}")

    def z3_of_formula(self, formula: Expression) -> z3.ExprRef:
        """Convert SRK formula to Z3 formula."""
        if isinstance(formula, TrueExpr):
            return z3.BoolVal(True, self.z3_ctx)
        elif isinstance(formula, FalseExpr):
            return z3.BoolVal(False, self.z3_ctx)
        elif isinstance(formula, Eq):
            left = self._z3_of_expr(formula.left)
            right = self._z3_of_expr(formula.right)
            return left == right
        elif isinstance(formula, Lt):
            left = self._z3_of_expr(formula.left)
            right = self._z3_of_expr(formula.right)
            return left < right
        elif isinstance(formula, Leq):
            left = self._z3_of_expr(formula.left)
            right = self._z3_of_expr(formula.right)
            return left <= right
        elif isinstance(formula, And):
            if not formula.args:
                return z3.BoolVal(True, self.z3_ctx)
            conjuncts = [self.z3_of_formula(arg) for arg in formula.args]
            return z3.And(conjuncts)
        elif isinstance(formula, Or):
            if not formula.args:
                return z3.BoolVal(False, self.z3_ctx)
            disjuncts = [self.z3_of_formula(arg) for arg in formula.args]
            return z3.Or(disjuncts)
        elif isinstance(formula, Not):
            return z3.Not(self.z3_of_formula(formula.arg))
        elif isinstance(formula, Ite):
            condition = self.z3_of_formula(formula.condition)
            then_branch = self.z3_of_formula(formula.then_branch)
            else_branch = self.z3_of_formula(formula.else_branch)
            return z3.If(condition, then_branch, else_branch)
        elif isinstance(formula, Forall):
            # Universal quantification
            body = self.z3_of_formula(formula.body)
            # Create bound variable with appropriate sort
            var_sort = sort_of_typ(self.z3_ctx, formula.var_type)
            var_name = formula.var_name if formula.var_name else "_"
            # Create quantifier with proper bound variable
            return z3.ForAll([z3.Const(var_name, var_sort)], body)
        elif isinstance(formula, Exists):
            # Existential quantification
            body = self.z3_of_formula(formula.body)
            # Create bound variable with appropriate sort
            var_sort = sort_of_typ(self.z3_ctx, formula.var_type)
            var_name = formula.var_name if formula.var_name else "_"
            # Create quantifier with proper bound variable
            return z3.Exists([z3.Const(var_name, var_sort)], body)
        elif isinstance(formula, App):
            if not formula.args:  # Nullary predicate
                return self.get_z3_symbol(formula.symbol)
            else:
                decl = self.decl_of_symbol(formula.symbol)
                args = [self._z3_of_expr(arg) for arg in formula.args]
                return z3.Expr.mk_app(self.z3_ctx, decl, args)
        elif isinstance(formula, Var):
            # Variables in formulas are boolean variables
            return z3.Quantifier.mk_bound(self.z3_ctx, formula.var_id,
                                         sort_of_typ(self.z3_ctx, Type.BOOL))
        else:
            # Try to provide more helpful error messages
            formula_type = type(formula).__name__
            if hasattr(formula, '__str__'):
                formula_str = str(formula)[:100]  # Truncate long expressions
                raise NotImplementedError(f"Unsupported formula type {formula_type}: {formula_str}")
            else:
                raise NotImplementedError(f"Unsupported formula type {formula_type}")

    def add_formula(self, formula: Expression) -> None:
        """Add a formula to the Z3 solver."""
        z3_formula = self.z3_of_formula(formula)
        self.solver.add(z3_formula)

    def check_sat(self) -> Z3Result:
        """Check satisfiability of the current formulas."""
        try:
            result = self.solver.check()
            if result == z3.sat:
                return Z3Result.SAT
            elif result == z3.unsat:
                return Z3Result.UNSAT
            else:
                return Z3Result.UNKNOWN
        except Exception as e:
            logf(f"srkZ3.check_sat: caught exception: {e}", level="warn")
            return Z3Result.UNKNOWN

    def get_model(self) -> Optional[Z3Model]:
        """Get a model if the formulas are satisfiable."""
        if self.check_sat() == Z3Result.SAT:
            z3_model = self.solver.model()
            return Z3Model(z3_model, self.srk_context)
        return None

    def push(self) -> None:
        """Push a backtracking point."""
        self.solver.push()

    def pop(self, num: int = 1) -> None:
        """Pop backtracking points."""
        self.solver.pop(num)

    def reset(self) -> None:
        """Reset the solver state."""
        self.solver.reset()

    def get_unsat_core(self) -> List[Expression]:
        """Get unsat core if the formulas are unsatisfiable."""
        if self.check_sat() == Z3Result.UNSAT:
            try:
                z3_core = self.solver.unsat_core()
                # Convert Z3 expressions back to SRK expressions
                core = []
                for z3_expr in z3_core:
                    try:
                        srk_expr = self._expr_of_z3(z3_expr)
                        core.append(srk_expr)
                    except Exception as e:
                        logf(f"get_unsat_core: failed to convert expression: {e}", level="warn")
                return core
            except Exception as e:
                logf(f"get_unsat_core: failed to get unsat core: {e}", level="warn")
                return []
        return []

    def quantifier_elimination(self, formula: Expression) -> Expression:
        """Eliminate quantifiers using Z3 tactics."""
        if not Z3_AVAILABLE:
            raise ImportError("Z3 is not available")

        try:
            # Check if formula has quantifiers
            if not self._has_quantifiers_srk(formula):
                return formula

            # Convert formula to Z3
            z3_formula = self.z3_of_formula(formula)

            # Apply simplify tactic first to clean up the formula
            try:
                simplify_tactic = z3.Tactic('simplify')
                simplified = simplify_tactic(z3_formula)
                if simplified and len(simplified) > 0:
                    z3_formula = simplified[0]
            except Exception as e:
                logf(f"simplify tactic failed: {e}", level="debug")

            # Try qe2 first (more modern and efficient)
            try:
                qe2_tactic = z3.Tactic('qe2')
                result = qe2_tactic(z3_formula)
                if result and len(result) > 0:
                    # Check if result is valid and different from input
                    if len(result) == 1:
                        result_expr = result[0]
                        if not result_expr.is_false() and str(result_expr) != str(z3_formula):
                            return self._expr_of_z3(result_expr)
            except Exception as e:
                logf(f"qe2 tactic failed, trying qe: {e}", level="debug")

            # Fallback to qe tactic
            try:
                qe_tactic = z3.Tactic('qe')
                result = qe_tactic(z3_formula)
                if result and len(result) > 0:
                    # Check if result is valid and different from input
                    if len(result) == 1:
                        result_expr = result[0]
                        if not result_expr.is_false() and str(result_expr) != str(z3_formula):
                            return self._expr_of_z3(result_expr)
            except Exception as e:
                logf(f"qe tactic failed: {e}", level="debug")

            # If both tactics fail, return original formula
            return formula

        except Exception as e:
            logf(f"quantifier_elimination: failed: {e}", level="warn")
            return formula

    def _has_quantifiers_srk(self, formula: Expression) -> bool:
        """Check if an SRK expression has quantifiers."""
        if isinstance(formula, (Forall, Exists)):
            return True
        elif hasattr(formula, 'args'):
            # Check recursively in arguments
            for arg in formula.args:
                if self._has_quantifiers_srk(arg):
                    return True
        elif hasattr(formula, 'body'):
            # Check body of quantified formulas
            return self._has_quantifiers_srk(formula.body)
        return False


    def get_array_model(self, array_symbol: Symbol) -> Optional[Dict[Expression, Expression]]:
        """Get array model for a given array symbol."""
        if not Z3_AVAILABLE or self.check_sat() != Z3Result.SAT:
            return None

        try:
            if typ_symbol(array_symbol) != Type.ARRAY:
                raise ValueError(f"Symbol {array_symbol} is not an array")

            # Get the Z3 model
            z3_model = self.solver.model()

            # Get the Z3 array expression
            z3_array = self.get_z3_symbol(array_symbol)
            z3_array_val = z3_model.eval(z3_array, model_completion=True)

            # Extract array model as a dictionary
            array_model = {}

            # Try to extract concrete values from the array
            # This is a simplified approach - a full implementation would
            # need to handle symbolic array models more sophisticatedly
            for i in range(100):  # Sample first 100 indices
                try:
                    idx_val = z3.IntVal(i, self.z3_ctx)
                    elem_val = z3_model.eval(z3.Select(z3_array_val, idx_val), model_completion=True)

                    # Convert Z3 values back to SRK expressions
                    idx_expr = self._expr_of_z3(idx_val)
                    elem_expr = self._expr_of_z3(elem_val)
                    array_model[idx_expr] = elem_expr

                except Exception:
                    # Stop if we can't evaluate more indices
                    break

            return array_model
        except Exception as e:
            logf(f"get_array_model: failed: {e}", level="warn")
            return None

    def get_function_model(self, func_symbol: Symbol) -> Optional[Dict[Tuple[Expression, ...], Expression]]:
        """Get function model for a given function symbol."""
        if not Z3_AVAILABLE or self.check_sat() != Z3Result.SAT:
            return None

        try:
            symbol_type = typ_symbol(func_symbol)
            if not isinstance(symbol_type, tuple) or symbol_type[0] != 'TyFun':
                raise ValueError(f"Symbol {func_symbol} is not a function")

            # Get the Z3 model
            z3_model = self.solver.model()

            # Get the Z3 function declaration
            z3_func_decl = self.decl_of_symbol(func_symbol)

            # Extract function model as a dictionary
            func_model = {}

            # Try to extract concrete values from the function
            # This is a simplified approach for unary functions
            if len(symbol_type[1]) == 1:  # Unary function
                for i in range(50):  # Sample first 50 arguments
                    try:
                        # Create argument value
                        arg_type = symbol_type[1][0]
                        if arg_type == Type.INT:
                            arg_val = z3.IntVal(i, self.z3_ctx)
                        elif arg_type == Type.REAL:
                            arg_val = z3.RealVal(i, self.z3_ctx)
                        elif arg_type == Type.BOOL:
                            arg_val = z3.BoolVal(i % 2 == 0, self.z3_ctx)
                        else:
                            continue  # Skip unsupported types

                        # Evaluate function application
                        func_app = z3.Expr.mk_app(self.z3_ctx, z3_func_decl, [arg_val])
                        result_val = z3_model.eval(func_app, model_completion=True)

                        # Convert Z3 values back to SRK expressions
                        arg_expr = self._expr_of_z3(arg_val)
                        result_expr = self._expr_of_z3(result_val)
                        func_model[(arg_expr,)] = result_expr

                    except Exception:
                        # Stop if we can't evaluate more arguments
                        break

            return func_model
        except Exception as e:
            logf(f"get_function_model: failed: {e}", level="warn")
            return None


    def _expr_of_z3(self, z3_expr) -> Expression:
        """Internal conversion from Z3 to SRK."""
        ast_kind = z3.AST.get_ast_kind(z3_expr.ast)

        if ast_kind == z3.APP_AST:
            decl = z3_expr.get_func_decl()
            args = [self._expr_of_z3(arg) for arg in z3_expr.get_args()]
            decl_kind = z3.FuncDecl.get_decl_kind(decl)

            if decl_kind == z3.OP_UNINTERPRETED:
                # Uninterpreted function symbol
                sym = symbol_of_int(z3.Symbol.get_int(z3.FuncDecl.get_name(decl)))
                return App(self.srk_context, sym, args)
            elif decl_kind == z3.OP_ADD:
                return mk_add(self.srk_context, args)
            elif decl_kind == z3.OP_MUL:
                return mk_mul(self.srk_context, args)
            elif decl_kind == z3.OP_SUB:
                if len(args) != 2:
                    raise ValueError("Unsupported subtraction")
                neg_arg = mk_neg(self.srk_context, args[1])
                return mk_add(self.srk_context, [args[0], neg_arg])
            elif decl_kind == z3.OP_UMINUS:
                if len(args) != 1:
                    raise ValueError("Unsupported unary minus")
                return mk_neg(self.srk_context, args[0])
            elif decl_kind == z3.OP_DIV:
                if len(args) != 2:
                    raise ValueError("Unsupported division")
                return mk_div(self.srk_context, args[0], args[1])
            elif decl_kind == z3.OP_MOD:
                if len(args) != 2:
                    raise ValueError("Unsupported modulo")
                return mk_mod(self.srk_context, args[0], args[1])
            elif decl_kind == z3.OP_TRUE:
                return mk_true(self.srk_context)
            elif decl_kind == z3.OP_FALSE:
                return mk_false(self.srk_context)
            elif decl_kind == z3.OP_AND:
                return mk_and(self.srk_context, args)
            elif decl_kind == z3.OP_OR:
                return mk_or(self.srk_context, args)
            elif decl_kind == z3.OP_NOT:
                if len(args) != 1:
                    raise ValueError("Unsupported negation")
                return mk_not(self.srk_context, args[0])
            elif decl_kind == z3.OP_EQ:
                if len(args) != 2:
                    raise ValueError("Unsupported equality")
                return mk_eq(self.srk_context, args[0], args[1])
            elif decl_kind == z3.OP_LE:
                if len(args) != 2:
                    raise ValueError("Unsupported leq")
                return mk_leq(self.srk_context, args[0], args[1])
            elif decl_kind == z3.OP_LT:
                if len(args) != 2:
                    raise ValueError("Unsupported lt")
                return mk_lt(self.srk_context, args[0], args[1])
            elif decl_kind == z3.OP_ITE:
                if len(args) != 3:
                    raise ValueError("Unsupported ite")
                return mk_ite(self.srk_context, args[0], args[1], args[2])
            elif decl_kind == z3.OP_GE:
                if len(args) != 2:
                    raise ValueError("Unsupported ge")
                # a >= b is equivalent to b <= a
                return mk_leq(self.srk_context, args[1], args[0])
            elif decl_kind == z3.OP_GT:
                if len(args) != 2:
                    raise ValueError("Unsupported gt")
                # a > b is equivalent to b < a
                return mk_lt(self.srk_context, args[1], args[0])
            elif decl_kind == z3.OP_IMPLIES:
                if len(args) != 2:
                    raise ValueError("Unsupported implies")
                # a => b is equivalent to (not a) or b
                not_a = mk_not(self.srk_context, args[0])
                return mk_or(self.srk_context, [not_a, args[1]])
            elif decl_kind == z3.OP_XOR:
                if len(args) != 2:
                    raise ValueError("Unsupported xor")
                # a xor b is (a or b) and not (a and b)
                or_expr = mk_or(self.srk_context, args)
                and_expr = mk_and(self.srk_context, args)
                not_and = mk_not(self.srk_context, and_expr)
                return mk_and(self.srk_context, [or_expr, not_and])
            elif decl_kind == z3.OP_DISTINCT:
                # Distinct means all arguments are pairwise different
                # Convert to conjunction of inequalities
                if len(args) >= 2:
                    clauses = []
                    for i in range(len(args)):
                        for j in range(i + 1, len(args)):
                            eq = mk_eq(self.srk_context, args[i], args[j])
                            clauses.append(mk_not(self.srk_context, eq))
                    return mk_and(self.srk_context, clauses)
                else:
                    return mk_true(self.srk_context)
            elif decl_kind == z3.OP_POWER:
                # Power operation: a^b
                if len(args) == 2:
                    # For now, treat as uninterpreted function
                    # In a full implementation, this would handle specific cases
                    power_sym = mk_symbol(self.srk_context, "pow", Type.REAL)
                    return App(self.srk_context, power_sym, args)
                else:
                    raise ValueError("Unsupported power operation")
            elif decl_kind == z3.OP_ABS:
                # Absolute value - treat as uninterpreted function
                if len(args) != 1:
                    raise ValueError("_expr_of_z3: unsupported abs operation")
                abs_sym = mk_symbol(self.srk_context, "abs", Type.REAL)
                return App(self.srk_context, abs_sym, args)
            elif decl_kind == z3.OP_TO_REAL:
                # Integer to real conversion - identity
                if len(args) != 1:
                    raise ValueError("_expr_of_z3: unsupported to_real operation")
                return args[0]
            elif decl_kind == z3.OP_TO_INT:
                # Real to integer conversion (floor) - treat as uninterpreted function
                if len(args) != 1:
                    raise ValueError("_expr_of_z3: unsupported to_int operation")
                floor_sym = mk_symbol(self.srk_context, "floor", Type.INT)
                return App(self.srk_context, floor_sym, args)
            elif decl_kind == z3.OP_IS_INT:
                # Check if real is integer - return true (conservative approximation)
                if len(args) != 1:
                    raise ValueError("_expr_of_z3: unsupported is_int operation")
                return mk_true(self.srk_context)
            elif decl_kind == z3.OP_AS_ARRAY:
                # Array as array operation - identity
                if len(args) != 1:
                    raise ValueError("Unsupported as_array operation")
                return args[0]
            elif decl_kind == z3.OP_STORE:
                # Array store operation
                if len(args) != 3:
                    raise ValueError("Unsupported store operation")
                return Store(self.srk_context, args[0], args[1], args[2])
            elif decl_kind == z3.OP_SELECT:
                # Array select operation
                if len(args) != 2:
                    raise ValueError("Unsupported select operation")
                return Select(self.srk_context, args[0], args[1])
            elif decl_kind == z3.OP_BNOT:
                # Bitwise not
                if len(args) == 1:
                    # For now, treat as uninterpreted function
                    bnot_sym = mk_symbol(self.srk_context, "bnot", Type.INT)
                    return App(self.srk_context, bnot_sym, args)
                else:
                    raise ValueError("Unsupported bnot operation")
            elif decl_kind == z3.OP_BAND:
                # Bitwise and
                if len(args) >= 2:
                    # For now, treat as uninterpreted function
                    band_sym = mk_symbol(self.srk_context, "band", Type.INT)
                    return App(self.srk_context, band_sym, args)
                else:
                    raise ValueError("Unsupported band operation")
            elif decl_kind == z3.OP_BOR:
                # Bitwise or
                if len(args) >= 2:
                    # For now, treat as uninterpreted function
                    bor_sym = mk_symbol(self.srk_context, "bor", Type.INT)
                    return App(self.srk_context, bor_sym, args)
                else:
                    raise ValueError("Unsupported bor operation")
            elif decl_kind == z3.OP_BXOR:
                # Bitwise xor
                if len(args) >= 2:
                    # For now, treat as uninterpreted function
                    bxor_sym = mk_symbol(self.srk_context, "bxor", Type.INT)
                    return App(self.srk_context, bxor_sym, args)
                else:
                    raise ValueError("Unsupported bxor operation")
            elif decl_kind == z3.OP_BNEG:
                # Bitwise negation
                if len(args) == 1:
                    # For now, treat as uninterpreted function
                    bneg_sym = mk_symbol(self.srk_context, "bneg", Type.INT)
                    return App(self.srk_context, bneg_sym, args)
                else:
                    raise ValueError("Unsupported bneg operation")
            elif decl_kind == z3.OP_BADD:
                # Bitwise addition
                if len(args) >= 2:
                    # For now, treat as uninterpreted function
                    badd_sym = mk_symbol(self.srk_context, "badd", Type.INT)
                    return App(self.srk_context, badd_sym, args)
                else:
                    raise ValueError("Unsupported badd operation")
            elif decl_kind == z3.OP_BMUL:
                # Bitwise multiplication
                if len(args) >= 2:
                    # For now, treat as uninterpreted function
                    bmul_sym = mk_symbol(self.srk_context, "bmul", Type.INT)
                    return App(self.srk_context, bmul_sym, args)
                else:
                    raise ValueError("Unsupported bmul operation")
            elif decl_kind in (z3.OP_BSDIV, z3.OP_BUDIV):
                # Signed/unsigned division - treat as regular division
                if len(args) != 2:
                    raise ValueError(f"Unsupported bitvector division operation")
                return mk_div(self.srk_context, args[0], args[1])
            elif decl_kind in (z3.OP_BSREM, z3.OP_BUREM, z3.OP_BSMOD):
                # Signed/unsigned remainder/modulo - treat as regular modulo
                if len(args) != 2:
                    raise ValueError(f"Unsupported bitvector modulo operation")
                return mk_mod(self.srk_context, args[0], args[1])
            elif decl_kind == z3.OP_BSHL:
                # Left shift
                if len(args) == 2:
                    # For now, treat as uninterpreted function
                    shl_sym = mk_symbol(self.srk_context, "shl", Type.INT)
                    return App(self.srk_context, shl_sym, args)
                else:
                    raise ValueError("Unsupported bshl operation")
            elif decl_kind == z3.OP_BLSHR:
                # Logical right shift
                if len(args) == 2:
                    # For now, treat as uninterpreted function
                    lshr_sym = mk_symbol(self.srk_context, "lshr", Type.INT)
                    return App(self.srk_context, lshr_sym, args)
                else:
                    raise ValueError("Unsupported blshr operation")
            elif decl_kind == z3.OP_BASHR:
                # Arithmetic right shift
                if len(args) == 2:
                    # For now, treat as uninterpreted function
                    ashr_sym = mk_symbol(self.srk_context, "ashr", Type.INT)
                    return App(self.srk_context, ashr_sym, args)
                else:
                    raise ValueError("Unsupported bashr operation")
            elif decl_kind == z3.OP_ULEQ:
                # Unsigned less than or equal - treat as regular comparison
                if len(args) != 2:
                    raise ValueError("Unsupported uleq operation")
                return mk_leq(self.srk_context, args[0], args[1])
            elif decl_kind == z3.OP_ULT:
                # Unsigned less than - treat as regular comparison
                if len(args) != 2:
                    raise ValueError("Unsupported ult operation")
                return mk_lt(self.srk_context, args[0], args[1])
            elif decl_kind == z3.OP_UGEQ:
                # Unsigned greater than or equal - treat as regular comparison
                if len(args) != 2:
                    raise ValueError("Unsupported ugeq operation")
                return mk_leq(self.srk_context, args[1], args[0])
            elif decl_kind == z3.OP_UGT:
                # Unsigned greater than - treat as regular comparison
                if len(args) != 2:
                    raise ValueError("Unsupported ugt operation")
                return mk_lt(self.srk_context, args[1], args[0])
            else:
                # Create uninterpreted function for unsupported operators
                try:
                    op_name = f"z3_op_{decl_kind}"
                    op_sym = mk_symbol(self.srk_context, op_name, Type.REAL)
                    return App(self.srk_context, op_sym, args)
                except Exception as e:
                    raise NotImplementedError(f"Unsupported Z3 operator {decl_kind}: {e}")

        elif ast_kind == z3.NUMERAL_AST:
            # Rational constant
            qq_val = QQ.of_string(z3.Arithmetic.Real.numeral_to_string(z3_expr))
            return mk_real(self.srk_context, qq_val)

        elif ast_kind == z3.VAR_AST:
            # Bound variable
            index = z3.Quantifier.get_index(z3_expr)
            typ = typ_of_sort(z3_expr.get_sort())
            return mk_var(self.srk_context, index, typ)

        elif ast_kind == z3.QUANTIFIER_AST:
            # Quantified formula
            quantifier = z3.Quantifier.quantifier_of_expr(z3_expr)
            body = self._expr_of_z3(z3.Quantifier.get_body(quantifier))
            bound_names = z3.Quantifier.get_bound_variable_names(quantifier)
            bound_sorts = z3.Quantifier.get_bound_variable_sorts(quantifier)

            if z3.Quantifier.is_existential(quantifier):
                qt = 'Exists'
            else:
                qt = 'Forall'

            # For simplicity, handle only single variable quantification
            if len(bound_names) == 1 and len(bound_sorts) == 1:
                name = z3.Symbol.to_string(bound_names[0])
                typ = typ_of_sort(bound_sorts[0])
                if qt == 'Exists':
                    return mk_exists(self.srk_context, name, typ, body)
                else:
                    return mk_forall(self.srk_context, name, typ, body)
            else:
                # Multi-variable quantification: convert to nested quantifiers
                # Q x1, x2, ..., xn . body  =>  Q x1 . Q x2 . ... Q xn . body
                result = body
                for name, sort in reversed(list(zip(bound_names, bound_sorts))):
                    var_name = z3.Symbol.to_string(name)
                    var_typ = typ_of_sort(sort)
                    if qt == 'Exists':
                        result = mk_exists(self.srk_context, var_name, var_typ, result)
                    else:
                        result = mk_forall(self.srk_context, var_name, var_typ, result)
                return result

        elif ast_kind == z3.SORT_AST:
            # Sort expression - convert to type
            return typ_of_sort(z3_expr)

        elif ast_kind == z3.FUNC_DECL_AST:
            # Function declaration - convert to symbol
            name = z3.FuncDecl.get_name(z3_expr)
            if z3.Symbol.is_int(name):
                return symbol_of_int(z3.Symbol.get_int(name))
            else:
                # Create a new symbol for string names
                return mk_symbol(self.srk_context, z3.Symbol.to_string(name), Type.REAL)

        elif ast_kind == z3.UNKNOWN_AST:
            # Unknown AST - try to handle gracefully
            logf(f"_expr_of_z3: encountered unknown AST, returning conservative approximation", level="warn")
            return mk_true(self.srk_context)

        else:
            # For unsupported AST kinds, try conservative conversion
            try:
                if z3.Expr.is_const(z3_expr):
                    decl = z3_expr.get_func_decl()
                    name = z3.FuncDecl.get_name(decl)
                    if z3.Symbol.is_int(name):
                        return symbol_of_int(z3.Symbol.get_int(name))
                    else:
                        return mk_symbol(self.srk_context, z3.Symbol.to_string(name), Type.REAL)
                else:
                    return mk_true(self.srk_context)
            except Exception as e:
                # Try to provide more context about the Z3 expression
                try:
                    expr_str = str(z3_expr)[:100]  # Truncate long expressions
                    raise NotImplementedError(f"Unsupported Z3 AST kind {ast_kind} ({expr_str}): {e}")
                except:
                    raise NotImplementedError(f"Unsupported Z3 AST kind {ast_kind}: {e}")


def z3_of_formula(srk_ctx: Context, z3_ctx: SrkZ3, formula: Expression) -> z3.ExprRef:
    """Convert SRK formula to Z3 expression."""
    return z3_ctx.z3_of_formula(formula)


def formula_of_z3(srk_ctx: Context, z3_ctx: SrkZ3, z3_expr: z3.ExprRef) -> Expression:
    """Convert Z3 expression back to SRK formula."""
    return z3_ctx._expr_of_z3(z3_expr)


def load_smtlib2(srk_ctx: Context, smtlib2_content: str) -> None:
    """Load SMT-LIB2 format into SRK context."""
    if not Z3_AVAILABLE:
        raise ImportError("Z3 is not available")

    # Parse SMT-LIB2 content and convert to SRK expressions
    try:
        ast = z3.SMT.parse_smtlib2_string(srk_ctx.z3_ctx if hasattr(srk_ctx, 'z3_ctx') else z3.Context(),
                                         smtlib2_content, [], [], [], [])
        for expr in z3.AST.ASTVector.to_expr_list(ast):
            srk_expr = formula_of_z3(srk_ctx, None, expr)
            # Add the expression to the context (this would need proper implementation)
            # For now, this is a placeholder
    except Exception as e:
        logf(f"load_smtlib2: failed to parse SMT-LIB2 content: {e}", level="warn")


# Utility functions for working with Z3 contexts
def make_z3_context(srk_ctx: Context) -> SrkZ3:
    """Create a Z3 context for an SRK context."""
    return SrkZ3(srk_ctx)


def check_satisfiability(srk_ctx: Context, formulas: List[Expression]) -> Z3Result:
    """Check satisfiability of a list of formulas."""
    z3_ctx = make_z3_context(srk_ctx)

    for formula in formulas:
        z3_ctx.add_formula(formula)

    return z3_ctx.check_sat()


def get_model(srk_ctx: Context, formulas: List[Expression]) -> Optional[Z3Model]:
    """Get a model for satisfiable formulas."""
    z3_ctx = make_z3_context(srk_ctx)

    for formula in formulas:
        z3_ctx.add_formula(formula)

    if z3_ctx.check_sat() == Z3Result.SAT:
        return z3_ctx.get_model()
    return None


def get_concrete_model(srk_ctx: Context, formulas: List[Expression]) -> Optional[Interpretation]:
    """Get a concrete model for satisfiable formulas."""
    z3_ctx = make_z3_context(srk_ctx)

    for formula in formulas:
        z3_ctx.add_formula(formula)

    if z3_ctx.check_sat() == Z3Result.SAT:
        model = z3_ctx.get_model()
        if model:
            # Convert Z3 model to SRK interpretation
            # This is a simplified implementation
            symbols = []
            def get_value(symbol):
                try:
                    z3_model_value = model.get_value(symbol)
                    return z3_model_value
                except:
                    return None

            return Interpretation(srk_ctx, get_value)
    return None


def optimize_box(srk_ctx: Context, formula: Expression, objectives: List[Expression]) -> Tuple[Z3Result, List[Expression]]:
    """Optimize a box (interval) for given objectives."""
    if not Z3_AVAILABLE:
        raise ImportError("Z3 is not available")

    z3_ctx = make_z3_context(srk_ctx)
    opt = z3.Optimize(ctx=z3_ctx.z3_ctx)

    # Set optimization parameters for box optimization
    params = z3.Params(ctx=z3_ctx.z3_ctx)
    z3.Symbol.mk_string(z3_ctx.z3_ctx, ":opt.priority")
    z3.Symbol.mk_string(z3_ctx.z3_ctx, "box")
    params.add(":opt.priority", "box")
    opt.set_parameters(params)

    # Add the formula
    z3_formula = z3_ctx.z3_of_formula(formula)
    opt.add(z3_formula)

    # Create handles for objectives (minimize and maximize each objective)
    handles = []
    for obj in objectives:
        z3_obj = z3_ctx._z3_of_expr(obj)
        min_handle = opt.minimize(z3_obj)
        max_handle = opt.maximize(z3_obj)
        handles.append((min_handle, max_handle))

    # Check satisfiability
    result = opt.check()
    if result == z3.sat:
        # Extract intervals from handles
        intervals = []
        for min_handle, max_handle in handles:
            try:
                lower = opt.lower(min_handle)
                upper = opt.upper(max_handle)
                intervals.append((lower, upper))
            except:
                # If we can't get bounds, return default
                intervals.append((None, None))
        return Z3Result.SAT, intervals
    elif result == z3.unsat:
        return Z3Result.UNSAT, []
    else:
        return Z3Result.UNKNOWN, []


def quantifier_elimination(srk_ctx: Context, formula: Expression) -> Expression:
    """Eliminate quantifiers from a formula using Z3 tactics."""
    z3_ctx = make_z3_context(srk_ctx)
    return z3_ctx.quantifier_elimination(formula)




def get_unsat_core(srk_ctx: Context, formulas: List[Expression]) -> List[Expression]:
    """Get unsat core for a list of formulas."""
    z3_ctx = make_z3_context(srk_ctx)

    for formula in formulas:
        z3_ctx.add_formula(formula)

    return z3_ctx.get_unsat_core()


def incremental_solve(srk_ctx: Context, formulas: List[Expression]) -> Tuple[Z3Result, Optional[Z3Model]]:
    """Solve formulas incrementally with push/pop support."""
    z3_ctx = make_z3_context(srk_ctx)

    # Add formulas one by one
    for i, formula in enumerate(formulas):
        z3_ctx.add_formula(formula)

        # Check satisfiability after each addition
        result = z3_ctx.check_sat()
        if result == Z3Result.UNSAT:
            # Get unsat core for debugging
            core = z3_ctx.get_unsat_core()
            logf(f"incremental_solve: UNSAT after adding formula {i}, core: {len(core)} expressions", level="info")
            return result, None
        elif result == Z3Result.UNKNOWN:
            logf(f"incremental_solve: UNKNOWN after adding formula {i}", level="warn")
            return result, None

    # All formulas added successfully
    model = z3_ctx.get_model()
    return Z3Result.SAT, model


def prove_implication(srk_ctx: Context, hypothesis: Expression, conclusion: Expression) -> bool:
    """Prove that hypothesis implies conclusion."""
    z3_ctx = make_z3_context(srk_ctx)

    # Add hypothesis and negation of conclusion
    z3_ctx.add_formula(hypothesis)
    z3_ctx.add_formula(mk_not(conclusion))

    # If this is UNSAT, then hypothesis implies conclusion
    result = z3_ctx.check_sat()
    return result == Z3Result.UNSAT


def find_counterexample(srk_ctx: Context, hypothesis: Expression, conclusion: Expression) -> Optional[Z3Model]:
    """Find a counterexample where hypothesis is true but conclusion is false."""
    z3_ctx = make_z3_context(srk_ctx)

    # Add hypothesis and negation of conclusion
    z3_ctx.add_formula(hypothesis)
    z3_ctx.add_formula(mk_not(conclusion))

    # If this is SAT, we have a counterexample
    result = z3_ctx.check_sat()
    if result == Z3Result.SAT:
        return z3_ctx.get_model()
    return None


# CHC (Constrained Horn Clauses) solver functionality
class CHCSolver:
    """Constrained Horn Clauses solver using Z3."""

    def __init__(self, srk_ctx: Context):
        """Initialize CHC solver."""
        if not Z3_AVAILABLE:
            raise ImportError("Z3 is not available")

        self.srk_ctx = srk_ctx
        self.z3_ctx = z3.Context()
        self.fp = z3.Fixedpoint(ctx=self.z3_ctx)

        # Set up fixedpoint parameters
        params = z3.Params(ctx=self.z3_ctx)
        params.add("xform.slice", False)
        params.add("xform.inline_linear", False)
        params.add("xform.inline_eager", False)
        self.fp.set_parameters(params)

        self.head_relations = set()
        self.error_symbol = None

    def register_relation(self, symbol: Symbol) -> None:
        """Register a relation symbol."""
        symbol_type = typ_symbol(self.srk_ctx, symbol)
        if isinstance(symbol_type, tuple) and symbol_type[0] == 'TyFun':
            decl = self._decl_of_symbol(symbol)
            self.fp.register_relation(decl)
            self.head_relations.add(symbol)
        else:
            raise ValueError(f"register_relation: not a relation symbol {symbol}")

    def add_rule(self, hypothesis: Expression, conclusion: Expression) -> None:
        """Add a CHC rule."""
        # Convert to Z3 and add to fixedpoint
        z3_hypothesis = self._z3_of_formula(hypothesis)
        z3_conclusion = self._z3_of_formula(conclusion)

        # Create implication rule
        implication = z3.Boolean.mk_implies(self.z3_ctx, z3_hypothesis, z3_conclusion)
        self.fp.add_rule(implication)

    def check(self) -> Z3Result:
        """Check satisfiability of CHC system."""
        # For now, assume we have an error relation to check
        if self.error_symbol:
            error_decl = self._decl_of_symbol(self.error_symbol)
            try:
                result = self.fp.query(error_decl)
                if result == z3.unsat:
                    return Z3Result.SAT  # No error reachable
                elif result == z3.sat:
                    return Z3Result.UNSAT  # Error reachable
                else:
                    return Z3Result.UNKNOWN
            except Exception as e:
                logf(f"CHCSolver.check: caught exception: {e}", level="warn")
                return Z3Result.UNKNOWN
        else:
            return Z3Result.UNKNOWN

    def _decl_of_symbol(self, symbol: Symbol) -> z3.FuncDecl:
        """Create Z3 function declaration for SRK symbol."""
        z3_ctx = SrkZ3(self.srk_ctx)
        return z3_ctx.decl_of_symbol(symbol)

    def _z3_of_formula(self, formula: Expression) -> z3.ExprRef:
        """Convert SRK formula to Z3."""
        z3_ctx = SrkZ3(self.srk_ctx)
        return z3_ctx.z3_of_formula(formula)


# Alias for backward compatibility
Z3Context = SrkZ3


class Solver:
    """Z3 solver wrapper."""

    def __init__(self, srk_context: Context, theory: str = ""):
        """Initialize solver."""
        if not Z3_AVAILABLE:
            raise ImportError("Z3 is not available")

        self.srk_context = srk_context
        self.z3_context = SrkZ3(srk_context)

        # Create Z3 solver
        if theory:
            self.z3_solver = z3.SolverFor(theory)
        else:
            self.z3_solver = z3.Solver()

    def add(self, formulas: List[Expression]) -> None:
        """Add formulas to the solver."""
        for formula in formulas:
            z3_formula = self.z3_context.z3_of_formula(formula)
            self.z3_solver.add(z3_formula)

    def check(self) -> Z3Result:
        """Check satisfiability."""
        result = self.z3_solver.check()
        if result == z3.sat:
            return Z3Result.SAT
        elif result == z3.unsat:
            return Z3Result.UNSAT
        else:
            return Z3Result.UNKNOWN

    def get_model(self) -> Optional[Z3Model]:
        """Get model if satisfiable."""
        if self.check() == Z3Result.SAT:
            return Z3Model(self.z3_solver.model(), self.srk_context)
        return None


def mk_solver(srk_context: Context, theory: str = "") -> Solver:
    """Create a new Z3 solver."""
    return Solver(srk_context, theory)
