"""
SMT solver interface for SRK.

This module provides a common interface for interacting with SMT solvers,
currently supporting Z3 integration with a framework for adding other solvers.
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple, Optional, Union, Any, Protocol
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
from fractions import Fraction

from aria.srk.syntax import (
    Context,
    Symbol,
    Type,
    Expression,
    FormulaExpression,
    ArithExpression,
    TermExpression,
    Var,
    Const,
    Eq,
    Lt,
    Leq,
    And,
    Or,
    Not,
    TrueExpr,
    FalseExpr,
    Ite,
    Forall,
    Exists,
    Add,
    Mul,
    App,
    Select,
    Store,
)


class SMTResult(Enum):
    """Result of SMT query."""

    SAT = "sat"
    UNSAT = "unsat"
    UNKNOWN = "unknown"


class SMTModel:
    """Represents a model from an SMT solver."""

    def __init__(self, interpretations: Dict[Symbol, Union[Fraction, bool, Dict, Any]]):
        self.interpretations = interpretations

    def get_value(self, symbol: Symbol) -> Union[Fraction, bool, Dict, Any, None]:
        """Get the value of a symbol in the model."""
        return self.interpretations.get(symbol)

    def get_array_value(
        self, symbol: Symbol, index: int
    ) -> Union[Fraction, bool, None]:
        """Get the value of an array element at a specific index."""
        val = self.interpretations.get(symbol)
        if isinstance(val, dict):
            return val.get(index)
        return None

    def evaluate_expression(self, expr: Expression) -> Union[Fraction, bool]:
        """Evaluate an expression in this model."""
        # This is a simplified implementation
        # A full implementation would recursively evaluate expressions

        if isinstance(expr, Const):
            return self.get_value(expr.symbol)
        elif isinstance(expr, Var):
            # Variables don't have direct interpretations
            raise ValueError(f"Cannot evaluate variable {expr} in model")
        elif isinstance(expr, Eq):
            left_val = self.evaluate_expression(expr.left)
            right_val = self.evaluate_expression(expr.right)
            return left_val == right_val
        elif isinstance(expr, Lt):
            left_val = self.evaluate_expression(expr.left)
            right_val = self.evaluate_expression(expr.right)
            return left_val < right_val
        elif isinstance(expr, Leq):
            left_val = self.evaluate_expression(expr.left)
            right_val = self.evaluate_expression(expr.right)
            return left_val <= right_val
        elif isinstance(expr, And):
            for arg in expr.args:
                if not self.evaluate_expression(arg):
                    return False
            return True
        elif isinstance(expr, Or):
            for arg in expr.args:
                if self.evaluate_expression(arg):
                    return True
            return False
        elif isinstance(expr, Not):
            return not self.evaluate_expression(expr.arg)
        elif isinstance(expr, TrueExpr):
            return True
        elif isinstance(expr, FalseExpr):
            return False
        elif isinstance(expr, Var):
            # Variables need to be looked up in the model
            value = self.get_value(expr.symbol)
            if value is None:
                raise ValueError(f"No value for variable {expr.symbol}")
            return value
        elif isinstance(expr, And):
            return all(self.evaluate_expression(arg) for arg in expr.args)
        elif isinstance(expr, Or):
            return any(self.evaluate_expression(arg) for arg in expr.args)
        elif isinstance(expr, Ite):
            cond_val = self.evaluate_expression(expr.condition)
            if cond_val:
                return self.evaluate_expression(expr.then_branch)
            else:
                return self.evaluate_expression(expr.else_branch)
        elif isinstance(expr, Forall):
            # Forall evaluation: check if formula holds for all values of bound variables
            # This is a simplified implementation - in practice, this would need
            # more sophisticated model-based evaluation
            if expr.body is not None:
                # For now, evaluate the body and assume universal quantification
                # A complete implementation would need to handle variable binding properly
                body_result = self.evaluate_expression(expr.body)
                if isinstance(body_result, bool):
                    return body_result  # If body is true, forall is true (simplified)
                else:
                    return False  # If body contains non-boolean, can't evaluate
            return False
        elif isinstance(expr, Exists):
            # Exists evaluation: check if formula holds for some values of bound variables
            # This is a simplified implementation
            if expr.body is not None:
                # For now, evaluate the body and assume existential quantification
                # A complete implementation would need proper existential evaluation
                body_result = self.evaluate_expression(expr.body)
                if isinstance(body_result, bool):
                    return body_result  # If body is true, exists is true (simplified)
                else:
                    return (
                        True  # If body contains non-boolean, conservatively return True
                    )
            return True
        elif isinstance(expr, Add):
            # Arithmetic addition
            if expr.args:
                result = 0
                for arg in expr.args:
                    arg_val = self.evaluate_expression(arg)
                    if not isinstance(arg_val, (int, float, Fraction)):
                        return True  # Conservative approximation for non-numeric
                    result += arg_val
                return result
            return 0
        elif isinstance(expr, Mul):
            # Arithmetic multiplication
            if expr.args:
                result = 1
                for arg in expr.args:
                    arg_val = self.evaluate_expression(arg)
                    if not isinstance(arg_val, (int, float, Fraction)):
                        return True  # Conservative approximation for non-numeric
                    result *= arg_val
                return result
            return 1
        elif isinstance(expr, Select):
            # Array select: evaluate array and index, then look up value
            array_val = self.evaluate_expression(expr.array)
            index_val = self.evaluate_expression(expr.index)
            if isinstance(array_val, dict) and isinstance(index_val, (int, Fraction)):
                idx = int(index_val) if isinstance(index_val, Fraction) else index_val
                return array_val.get(idx, 0)  # Default to 0 for unknown indices
            return 0  # Conservative fallback
        elif isinstance(expr, Store):
            # Array store: evaluate all components
            array_val = self.evaluate_expression(expr.array)
            index_val = self.evaluate_expression(expr.index)
            value_val = self.evaluate_expression(expr.value)
            if isinstance(array_val, dict) and isinstance(index_val, (int, Fraction)):
                idx = int(index_val) if isinstance(index_val, Fraction) else index_val
                new_array = array_val.copy()
                new_array[idx] = value_val
                return new_array
            return array_val  # Conservative fallback
        elif isinstance(expr, App):
            # Function application - try to evaluate
            # If it's a nullary function (constant), look it up
            if not expr.args:
                value = self.get_value(expr.symbol)
                if value is not None:
                    return value
                # If no value, conservative approximation
                return True
            else:
                # Function with arguments - this requires function interpretation
                # For now, return conservative approximation
                # A full implementation would apply the function to evaluated arguments
                return True
        else:
            raise NotImplementedError(f"Cannot evaluate expression type: {type(expr)}")


class SMTSolver(ABC):
    """Abstract base class for SMT solvers."""

    @abstractmethod
    def add(self, formulas: List[FormulaExpression]) -> None:
        """Add formulas to the solver context."""
        pass

    @abstractmethod
    def push(self) -> None:
        """Push a new scope."""
        pass

    @abstractmethod
    def pop(self, levels: int = 1) -> None:
        """Pop scopes."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset the solver."""
        pass

    @abstractmethod
    def check(self, formulas: List[FormulaExpression] = None) -> SMTResult:
        """Check satisfiability of formulas."""
        pass

    @abstractmethod
    def get_model(self) -> Optional[SMTModel]:
        """Get a model if the last check returned SAT."""
        pass

    @abstractmethod
    def get_unsat_core(self) -> List[FormulaExpression]:
        """Get unsatisfiable core if last check returned UNSAT."""
        pass


class Z3Solver(SMTSolver):
    """Z3-based SMT solver implementation."""

    def __init__(self, context: Context):
        self.context = context
        self._formulas: List[FormulaExpression] = []
        self._model: Optional[SMTModel] = None
        self._last_result: SMTResult = SMTResult.UNKNOWN

        # Try to import z3
        try:
            import z3

            self.z3 = z3
            self._z3_solver = z3.Solver()
            self._symbol_map: Dict[Symbol, Any] = {}  # Map SRK symbols to Z3 symbols
            self._z3_symbol_map: Dict[Any, Symbol] = {}  # Map Z3 symbols to SRK symbols
        except ImportError:
            raise ImportError(
                "Z3 is not installed. Please install with: pip install z3-solver"
            )

    def _z3_sort_from_type(self, typ: Type) -> Any:
        """Map SRK Type to a Z3 Sort (best-effort defaults)."""
        if typ == Type.INT:
            return self.z3.IntSort()
        if typ == Type.REAL:
            return self.z3.RealSort()
        if typ == Type.BOOL:
            return self.z3.BoolSort()
        if typ == Type.ARRAY:
            # Default to Int -> Int arrays (no richer type info available here)
            return self.z3.ArraySort(self.z3.IntSort(), self.z3.IntSort())
        # FUN and other cases are handled at application sites
        return self.z3.IntSort()

    def _srk_to_z3_symbol(self, symbol: Symbol) -> Any:
        """Convert SRK symbol to Z3 symbol/const of appropriate sort."""
        if symbol in self._symbol_map:
            return self._symbol_map[symbol]

        sym_name = str(symbol)
        if symbol.typ == Type.BOOL:
            z3_sym = self.z3.Bool(sym_name)
        elif symbol.typ == Type.INT:
            z3_sym = self.z3.Int(sym_name)
        elif symbol.typ == Type.REAL:
            z3_sym = self.z3.Real(sym_name)
        elif symbol.typ == Type.ARRAY:
            z3_sym = self.z3.Const(sym_name, self._z3_sort_from_type(Type.ARRAY))
        else:
            # Default to uninterpreted const of Int sort
            z3_sym = self.z3.Int(sym_name)

        self._symbol_map[symbol] = z3_sym
        self._z3_symbol_map[z3_sym] = symbol
        return z3_sym

    def _srk_to_z3_expr(self, expr: Expression) -> Any:
        return self._srk_to_z3_expr_with_env(expr, {})

    def _srk_to_z3_expr_with_env(self, expr: Expression, env: Dict[str, Any]) -> Any:
        """Convert SRK expression to Z3 expression with a substitution env for bound names."""
        if isinstance(expr, Const):
            # If this constant's name is bound by a quantifier, use the bound var
            name = expr.symbol.name or str(expr.symbol)
            if name in env:
                return env[name]
            return self._srk_to_z3_symbol(expr.symbol)
        elif isinstance(expr, Var):
            # Use bound variable from the environment when available, otherwise
            # create (and record) a symbolic constant tied to the SRK symbol table.
            if expr.name and expr.name in env:
                return env[expr.name]

            symbol = (
                self.context._symbols.get(expr.var_id)
                if hasattr(self.context, "_symbols")
                else None
            )
            if symbol is None:
                symbol = Symbol(expr.var_id, expr.name, expr.var_type)

            return self._srk_to_z3_symbol(symbol)
        elif isinstance(expr, Eq):
            return self._srk_to_z3_expr_with_env(
                expr.left, env
            ) == self._srk_to_z3_expr_with_env(expr.right, env)
        elif isinstance(expr, Lt):
            return self._srk_to_z3_expr_with_env(
                expr.left, env
            ) < self._srk_to_z3_expr_with_env(expr.right, env)
        elif isinstance(expr, Leq):
            return self._srk_to_z3_expr_with_env(
                expr.left, env
            ) <= self._srk_to_z3_expr_with_env(expr.right, env)
        elif isinstance(expr, And):
            if not expr.args:
                return self.z3.BoolVal(True)
            result = self._srk_to_z3_expr_with_env(expr.args[0], env)
            for arg in expr.args[1:]:
                result = self.z3.And(result, self._srk_to_z3_expr_with_env(arg, env))
            return result
        elif isinstance(expr, Or):
            if not expr.args:
                return self.z3.BoolVal(False)
            result = self._srk_to_z3_expr_with_env(expr.args[0], env)
            for arg in expr.args[1:]:
                result = self.z3.Or(result, self._srk_to_z3_expr_with_env(arg, env))
            return result
        elif isinstance(expr, Not):
            return self.z3.Not(self._srk_to_z3_expr_with_env(expr.arg, env))
        elif isinstance(expr, TrueExpr):
            return self.z3.BoolVal(True)
        elif isinstance(expr, FalseExpr):
            return self.z3.BoolVal(False)
        elif isinstance(expr, Ite):
            return self.z3.If(
                self._srk_to_z3_expr_with_env(expr.condition, env),
                self._srk_to_z3_expr_with_env(expr.then_branch, env),
                self._srk_to_z3_expr_with_env(expr.else_branch, env),
            )
        elif isinstance(expr, Forall):
            # Proper binding: introduce a bound variable with the given name and type
            sort = self._z3_sort_from_type(expr.var_type)
            bound = self.z3.Const(expr.var_name, sort)
            body_z3 = self._srk_to_z3_expr_with_env(
                expr.body, {**env, expr.var_name: bound}
            )
            return self.z3.ForAll([bound], body_z3)

        elif isinstance(expr, Exists):
            sort = self._z3_sort_from_type(expr.var_type)
            bound = self.z3.Const(expr.var_name, sort)
            body_z3 = self._srk_to_z3_expr_with_env(
                expr.body, {**env, expr.var_name: bound}
            )
            return self.z3.Exists([bound], body_z3)
        elif isinstance(expr, Add):
            # Arithmetic addition
            if expr.args:
                z3_args = [self._srk_to_z3_expr_with_env(arg, env) for arg in expr.args]
                return self.z3.Sum(z3_args) if len(z3_args) > 1 else z3_args[0]
            return self.z3.IntVal(0)
        elif isinstance(expr, Mul):
            # Arithmetic multiplication
            if expr.args:
                z3_args = [self._srk_to_z3_expr_with_env(arg, env) for arg in expr.args]
                if len(z3_args) == 0:
                    return self.z3.IntVal(1)
                elif len(z3_args) == 1:
                    return z3_args[0]
                else:
                    result = z3_args[0]
                    for arg in z3_args[1:]:
                        result = result * arg
                    return result
            return self.z3.IntVal(1)
        elif isinstance(expr, Select):
            arr = self._srk_to_z3_expr_with_env(expr.array, env)
            idx = self._srk_to_z3_expr_with_env(expr.index, env)
            return self.z3.Select(arr, idx)
        elif isinstance(expr, Store):
            arr = self._srk_to_z3_expr_with_env(expr.array, env)
            idx = self._srk_to_z3_expr_with_env(expr.index, env)
            val = self._srk_to_z3_expr_with_env(expr.value, env)
            return self.z3.Store(arr, idx, val)
        elif isinstance(expr, App):
            # Function application
            if not expr.args:
                # Nullary function - treat as a constant/variable
                return self._srk_to_z3_symbol(expr.symbol)
            else:
                # Function with arguments
                # Create an uninterpreted function in Z3
                func_name = str(expr.symbol)
                arg_z3 = [self._srk_to_z3_expr_with_env(arg, env) for arg in expr.args]

                # Determine result type based on expression type
                # For simplicity, assume boolean result for predicates
                arg_sorts = [arg.sort() for arg in arg_z3]
                if hasattr(expr, "typ") and expr.typ == Type.BOOL:
                    rng_sort = self.z3.BoolSort()
                elif hasattr(expr, "typ") and expr.typ == Type.REAL:
                    rng_sort = self.z3.RealSort()
                else:
                    rng_sort = self.z3.IntSort()
                func = self.z3.Function(func_name, *arg_sorts, rng_sort)
                return func(*arg_z3)
        elif isinstance(expr, Store):
            # Array store operation
            arr = self._srk_to_z3_expr_with_env(expr.array, env)
            idx = self._srk_to_z3_expr_with_env(expr.index, env)
            val = self._srk_to_z3_expr_with_env(expr.value, env)
            return self.z3.Store(arr, idx, val)
        elif hasattr(expr, "left") and hasattr(expr, "right"):
            # Handle binary operations generically
            left = self._srk_to_z3_expr_with_env(expr.left, env)
            right = self._srk_to_z3_expr_with_env(expr.right, env)
            if isinstance(expr, (Eq,)):
                return left == right
            elif isinstance(expr, (Lt,)):
                return left < right
            elif isinstance(expr, (Leq,)):
                return left <= right
            else:
                # Try to determine operation from class name or attributes
                if hasattr(expr, "op"):
                    if expr.op == "Div":
                        return left / right
                    elif expr.op == "Mod":
                        return left % right
                    elif expr.op == "Floor":
                        return self.z3.ToInt(left)
                    elif expr.op == "Neg":
                        return -left
                    else:
                        # Create uninterpreted function for unknown operations
                        op_name = f"op_{expr.op.lower()}"
                        op_sym = self.z3.Function(
                            op_name, left.sort(), right.sort(), left.sort()
                        )
                        return op_sym(left, right)
                else:
                    # Default to equality for unknown binary operations
                    return left == right
        elif hasattr(expr, "args") and hasattr(expr, "symbol"):
            # Handle n-ary operations generically
            if not expr.args:
                return self._srk_to_z3_symbol(expr.symbol)
            else:
                # Try to determine operation type from symbol name or type
                symbol_name = (
                    str(expr.symbol) if hasattr(expr.symbol, "__str__") else ""
                )
                if "add" in symbol_name.lower() or "+" in symbol_name:
                    args_z3 = [
                        self._srk_to_z3_expr_with_env(arg, env) for arg in expr.args
                    ]
                    return self.z3.Sum(args_z3) if len(args_z3) > 1 else args_z3[0]
                elif "mul" in symbol_name.lower() or "*" in symbol_name:
                    args_z3 = [
                        self._srk_to_z3_expr_with_env(arg, env) for arg in expr.args
                    ]
                    if len(args_z3) == 0:
                        return self.z3.IntVal(1)
                    elif len(args_z3) == 1:
                        return args_z3[0]
                    else:
                        result = args_z3[0]
                        for arg in args_z3[1:]:
                            result = result * arg
                        return result
                elif "and" in symbol_name.lower():
                    args_z3 = [
                        self._srk_to_z3_expr_with_env(arg, env) for arg in expr.args
                    ]
                    return self.z3.And(args_z3)
                elif "or" in symbol_name.lower():
                    args_z3 = [
                        self._srk_to_z3_expr_with_env(arg, env) for arg in expr.args
                    ]
                    return self.z3.Or(args_z3)
                else:
                    # Default to function application
                    func_name = str(expr.symbol)
                    arg_z3 = [
                        self._srk_to_z3_expr_with_env(arg, env) for arg in expr.args
                    ]

                    # Determine result type based on expression type
                    arg_sorts = [arg.sort() for arg in arg_z3]
                    if hasattr(expr, "typ") and expr.typ == Type.BOOL:
                        rng_sort = self.z3.BoolSort()
                    elif hasattr(expr, "typ") and expr.typ == Type.REAL:
                        rng_sort = self.z3.RealSort()
                    else:
                        rng_sort = self.z3.IntSort()
                    func = self.z3.Function(func_name, *arg_sorts, rng_sort)
                    return func(*arg_z3)
        else:
            # Try to provide more helpful error messages
            expr_type = type(expr).__name__
            if hasattr(expr, "__str__"):
                expr_str = str(expr)[:100]  # Truncate long expressions
                raise NotImplementedError(
                    f"Cannot convert expression type {expr_type}: {expr_str}"
                )
            else:
                raise NotImplementedError(f"Cannot convert expression type {expr_type}")

    def add(self, formulas: List[FormulaExpression]) -> None:
        """Add formulas to the solver context."""
        for formula in formulas:
            z3_expr = self._srk_to_z3_expr(formula)
            self._z3_solver.add(z3_expr)
        self._formulas.extend(formulas)

    def push(self) -> None:
        """Push a new scope."""
        self._z3_solver.push()

    def pop(self, levels: int = 1) -> None:
        """Pop scopes."""
        self._z3_solver.pop(levels)

    def reset(self) -> None:
        """Reset the solver."""
        self._z3_solver.reset()
        self._formulas.clear()
        self._model = None
        self._last_result = SMTResult.UNKNOWN

    def check(self, formulas: List[FormulaExpression] = None) -> SMTResult:
        """Check satisfiability of formulas."""
        if formulas:
            # Push a new scope and add the formulas for checking
            self._z3_solver.push()
            for formula in formulas:
                z3_expr = self._srk_to_z3_expr(formula)
                self._z3_solver.add(z3_expr)

        result = self._z3_solver.check()

        if formulas:
            # Pop the scope to remove the temporary formulas
            self._z3_solver.pop()

        if result == self.z3.sat:
            self._last_result = SMTResult.SAT
            # Extract model
            self._model = self._extract_model()
        elif result == self.z3.unsat:
            self._last_result = SMTResult.UNSAT
            self._model = None
        else:
            self._last_result = SMTResult.UNKNOWN
            self._model = None

        return self._last_result

    def _extract_model(self) -> Optional[SMTModel]:
        """Extract model from Z3 including arrays and functions."""
        if self._last_result != SMTResult.SAT:
            return None

        model = self._z3_solver.model()
        interpretations: Dict[Symbol, Union[Fraction, bool, Dict, Any]] = {}

        for decl in model.decls():
            z3_sym = decl()
            srk_symbol = self._z3_symbol_map.get(z3_sym)
            if not srk_symbol:
                continue

            val = model.eval(z3_sym, model_completion=True)

            if srk_symbol.typ == Type.BOOL:
                interpretations[srk_symbol] = self.z3.is_true(val)
            elif srk_symbol.typ in (Type.INT, Type.REAL):
                try:
                    if val.is_int_value():
                        interpretations[srk_symbol] = Fraction(int(val.as_long()), 1)
                    elif hasattr(val, "as_decimal"):
                        dec = val.as_decimal(50)
                        if dec.endswith("?"):
                            dec = dec[:-1]
                        interpretations[srk_symbol] = Fraction(dec)
                    else:
                        interpretations[srk_symbol] = Fraction(str(val))
                except Exception:
                    # Fallback: try Python float
                    interpretations[srk_symbol] = Fraction(float(str(val)))
            elif srk_symbol.typ == Type.ARRAY:
                # Extract array model as a dictionary of index->value mappings
                try:
                    array_model = {}
                    # Try to extract some concrete values from the array
                    # This is a simplified approach - a full implementation would
                    # need to handle symbolic array models more sophisticatedly
                    for i in range(10):  # Sample first 10 indices
                        try:
                            idx_val = self.z3.IntVal(i)
                            elem_val = model.eval(
                                self.z3.Select(val, idx_val), model_completion=True
                            )
                            if elem_val.is_int_value():
                                array_model[i] = Fraction(int(elem_val.as_long()), 1)
                            elif hasattr(elem_val, "as_decimal"):
                                dec = elem_val.as_decimal(50)
                                if dec.endswith("?"):
                                    dec = dec[:-1]
                                array_model[i] = Fraction(dec)
                        except:
                            break  # Stop if we can't evaluate more indices
                    interpretations[srk_symbol] = array_model
                except Exception:
                    # Fallback: store the raw Z3 array value
                    interpretations[srk_symbol] = str(val)
            else:
                # For functions and other types, store as string representation
                interpretations[srk_symbol] = str(val)

        return SMTModel(interpretations)

    def get_model(self) -> Optional[SMTModel]:
        """Get a model if the last check returned SAT."""
        return self._model

    def get_unsat_core(self) -> List[FormulaExpression]:
        """Get unsatisfiable core if last check returned UNSAT."""
        # Z3 doesn't provide unsat cores by default in this interface
        # This would require more sophisticated tracking
        return []


class SMTInterface:
    """Main interface for SMT operations."""

    def __init__(self, context: Context, solver: str = "z3"):
        self.context = context
        if solver == "z3":
            self.solver = Z3Solver(context)
        else:
            raise ValueError(f"Unsupported solver: {solver}")

    def is_sat(self, formula: FormulaExpression) -> SMTResult:
        """Check if a formula is satisfiable."""
        return self.solver.check([formula])

    def get_model(self, formula: FormulaExpression) -> Optional[SMTModel]:
        """Get a model for a satisfiable formula."""
        result = self.solver.check([formula])
        if result == SMTResult.SAT:
            return self.solver.get_model()
        return None

    def entails(
        self, premise: FormulaExpression, conclusion: FormulaExpression
    ) -> SMTResult:
        """Check if premise entails conclusion."""
        # Check if (premise ∧ ¬conclusion) is unsatisfiable
        not_conclusion = Not(conclusion)
        combined = And([premise, not_conclusion])
        return self.solver.check([combined])

    def equiv(
        self, formula1: FormulaExpression, formula2: FormulaExpression
    ) -> SMTResult:
        """Check if two formulas are equivalent."""
        # Check if (formula1 ∧ ¬formula2) and (¬formula1 ∧ formula2) are both unsatisfiable
        not_formula2 = Not(formula2)
        not_formula1 = Not(formula1)

        # Check formula1 => formula2 (i.e., formula1 ∧ ¬formula2 is unsat)
        impl1 = And([formula1, not_formula2])
        result1 = self.solver.check([impl1])

        # Check formula2 => formula1 (i.e., formula2 ∧ ¬formula1 is unsat)
        impl2 = And([formula2, not_formula1])
        result2 = self.solver.check([impl2])

        if result1 == SMTResult.UNSAT and result2 == SMTResult.UNSAT:
            return SMTResult.SAT  # Formulas are equivalent
        elif result1 == SMTResult.SAT:
            return SMTResult.UNSAT  # formula1 does not imply formula2
        elif result2 == SMTResult.SAT:
            return SMTResult.UNSAT  # formula2 does not imply formula1
        else:
            return SMTResult.UNKNOWN


# Convenience functions
def make_solver(context: Context, solver: str = "z3") -> SMTInterface:
    """Create an SMT solver interface."""
    return SMTInterface(context, solver)


def check_sat(context: Context, formulas: List[FormulaExpression]) -> str:
    """Check satisfiability of a list of formulas returning 'sat'|'unsat'|'unknown'.

    This helper matches usage from other modules expecting a simple string.
    """
    solver = SMTInterface(context)
    result = solver.solver.check(formulas)
    if result == SMTResult.SAT:
        return "sat"
    if result == SMTResult.UNSAT:
        return "unsat"
    return "unknown"


def get_model(
    formula: FormulaExpression, context: Optional[Context] = None
) -> Optional[SMTModel]:
    """Get a model for a satisfiable formula."""
    ctx = context or Context()
    solver = SMTInterface(ctx)
    return solver.get_model(formula)


def mk_solver(context: Context, theory: str = "QF_LRA") -> SMTInterface:
    """Create an SMT solver for the given theory.

    Args:
        context: The SRK context
        theory: The SMT-LIB theory (e.g., "QF_LRA", "QF_LIRA", "QF_NRA")

    Returns:
        SMT solver interface
    """
    # For now, we ignore the theory parameter and always use Z3
    # A more complete implementation would configure Z3 based on the theory
    return SMTInterface(context, "z3")


def is_sat(context: Context, formula: FormulaExpression) -> SMTResult:
    """Check if a formula is satisfiable.

    Args:
        context: The SRK context
        formula: The formula to check

    Returns:
        SMTResult indicating sat, unsat, or unknown
    """
    solver = SMTInterface(context)
    return solver.is_sat(formula)


# Add constants for backwards compatibility
Sat = SMTResult.SAT
Unsat = SMTResult.UNSAT
Unknown = SMTResult.UNKNOWN


class Solver:
    """Wrapper class for SMT solver operations.

    This provides a simpler interface that matches the usage in other modules.
    """

    def __init__(self, solver: SMTInterface):
        self.solver = solver

    @staticmethod
    def add(solver: SMTInterface, formulas: List[FormulaExpression]) -> None:
        """Add formulas to the solver."""
        solver.solver.add(formulas)

    @staticmethod
    def check(solver: SMTInterface, assumptions: List[FormulaExpression]) -> SMTResult:
        """Check satisfiability with assumptions."""
        if assumptions:
            return solver.solver.check(assumptions)
        return solver.solver.check()

    @staticmethod
    def get_model(solver: SMTInterface) -> Optional[SMTModel]:
        """Get the current model."""
        return solver.solver.get_model()

    @staticmethod
    def push(solver: SMTInterface) -> None:
        """Push a new scope."""
        solver.solver.push()

    @staticmethod
    def pop(solver: SMTInterface, levels: int = 1) -> None:
        """Pop scopes."""
        solver.solver.pop(levels)

    @staticmethod
    def reset(solver: SMTInterface) -> None:
        """Reset the solver."""
        solver.solver.reset()


# Compatibility alias for SRK interface
Smt = SMTInterface
