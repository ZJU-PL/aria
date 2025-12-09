"""Expression to SMT conversion utilities.

This module provides functions to convert expressions from Version Space Algebra
to SMT format for use with ARIA's SMT solvers.
"""

import z3
from typing import Dict, Any, List, Set, Tuple
from .expressions import (
    Expression, Theory, Variable, Constant, BinaryExpr, UnaryExpr,
    BinaryOp, UnaryOp, IfExpr, LoopExpr, FunctionCallExpr
)


class SMTConverter:
    """Converts VSA expressions to SMT format."""

    def __init__(self):
        self.variable_map: Dict[str, z3.ExprRef] = {}
        self.context = z3.Context()

    def convert_expression(self, expr: Expression, var_types: Dict[str, str] = None) -> z3.ExprRef:
        """Convert a VSA expression to SMT format."""
        var_types = var_types or {}

        if isinstance(expr, Variable):
            return self._convert_variable(expr, var_types)
        elif isinstance(expr, Constant):
            return self._convert_constant(expr)
        elif isinstance(expr, BinaryExpr):
            return self._convert_binary_expr(expr, var_types)
        elif isinstance(expr, UnaryExpr):
            return self._convert_unary_expr(expr, var_types)
        elif isinstance(expr, IfExpr):
            return self._convert_if_expr(expr, var_types)
        elif isinstance(expr, FunctionCallExpr):
            return self._convert_function_call(expr, var_types)
        else:
            raise ValueError(f"Unsupported expression type: {type(expr)}")

    def _convert_variable(self, var: Variable, var_types: Dict[str, str]) -> z3.ExprRef:
        """Convert a variable to SMT format."""
        if var.name not in self.variable_map:
            if var.theory == Theory.LIA:
                self.variable_map[var.name] = z3.Int(var.name, self.context)
            elif var.theory == Theory.BV:
                # Default to 32-bit bitvector if not specified
                bitwidth = var_types.get(var.name, 32)
                self.variable_map[var.name] = z3.BitVec(var.name, bitwidth, self.context)
            elif var.theory == Theory.STRING:
                # Z3 doesn't have native string support, use integer codes or sequences
                self.variable_map[var.name] = z3.String(var.name, self.context)
            else:
                raise ValueError(f"Unsupported theory for variable: {var.theory}")

        return self.variable_map[var.name]

    def _convert_constant(self, const: Constant) -> z3.ExprRef:
        """Convert a constant to SMT format."""
        if const.theory == Theory.LIA:
            return z3.IntVal(const.value, self.context)
        elif const.theory == Theory.BV:
            return z3.BitVecVal(const.value, 32, self.context)  # Default 32-bit
        elif const.theory == Theory.STRING:
            return z3.StringVal(str(const.value), self.context)
        else:
            raise ValueError(f"Unsupported theory for constant: {const.theory}")

    def _convert_binary_expr(self, expr: BinaryExpr, var_types: Dict[str, str]) -> z3.ExprRef:
        """Convert a binary expression to SMT format."""
        left = self.convert_expression(expr.left, var_types)
        right = self.convert_expression(expr.right, var_types)

        if expr.theory == Theory.LIA:
            if expr.op == BinaryOp.ADD:
                return left + right
            elif expr.op == BinaryOp.SUB:
                return left - right
            elif expr.op == BinaryOp.MUL:
                return left * right
            elif expr.op == BinaryOp.DIV:
                return left / right
            elif expr.op == BinaryOp.MOD:
                return left % right
            elif expr.op == BinaryOp.EQ:
                return left == right
            elif expr.op == BinaryOp.NEQ:
                return left != right
            elif expr.op == BinaryOp.LT:
                return left < right
            elif expr.op == BinaryOp.LE:
                return left <= right
            elif expr.op == BinaryOp.GT:
                return left > right
            elif expr.op == BinaryOp.GE:
                return left >= right

        elif expr.theory == Theory.BV:
            if expr.op == BinaryOp.BVAND:
                return left & right
            elif expr.op == BinaryOp.BVOR:
                return left | right
            elif expr.op == BinaryOp.BVXOR:
                return left ^ right
            elif expr.op == BinaryOp.BVSLL:
                return left << right
            elif expr.op == BinaryOp.BVSLR:
                return z3.LShR(left, right)  # Logical shift right
            elif expr.op == BinaryOp.BVSRA:
                return left >> right  # Arithmetic shift right

        elif expr.theory == Theory.STRING:
            if expr.op == BinaryOp.CONCAT:
                return z3.Concat(left, right)
            elif expr.op == BinaryOp.EQ:
                return left == right

        raise ValueError(f"Unsupported binary operation: {expr.op} for theory {expr.theory}")

    def _convert_unary_expr(self, expr: UnaryExpr, var_types: Dict[str, str]) -> z3.ExprRef:
        """Convert a unary expression to SMT format."""
        operand = self.convert_expression(expr.operand, var_types)

        if expr.theory == Theory.LIA:
            if expr.op == UnaryOp.NEG:
                return -operand

        elif expr.theory == Theory.BV:
            if expr.op == UnaryOp.BVNOT:
                return ~operand

        elif expr.theory == Theory.STRING:
            if expr.op == UnaryOp.LENGTH:
                return z3.Length(operand)

        raise ValueError(f"Unsupported unary operation: {expr.op} for theory {expr.theory}")

    def _convert_if_expr(self, expr: IfExpr, var_types: Dict[str, str]) -> z3.ExprRef:
        """Convert a conditional expression to SMT format."""
        condition = self.convert_expression(expr.condition, var_types)
        then_expr = self.convert_expression(expr.then_expr, var_types)
        else_expr = self.convert_expression(expr.else_expr, var_types)

        return z3.If(condition, then_expr, else_expr)

    def _convert_function_call(self, expr: FunctionCallExpr, var_types: Dict[str, str]) -> z3.ExprRef:
        """Convert a function call to SMT format."""
        args = [self.convert_expression(arg, var_types) for arg in expr.args]

        if expr.function_name == "abs":
            return z3.Abs(args[0])
        elif expr.function_name == "min":
            return z3.If(args[0] <= args[1], args[0], args[1])
        elif expr.function_name == "max":
            return z3.If(args[0] >= args[1], args[0], args[1])
        elif expr.function_name == "str_substring":
            # str_substring(str, start, length)
            return z3.SubString(args[0], args[1], args[2])
        elif expr.function_name == "str_indexof":
            # str_indexof(str, substr) - simplified implementation
            return z3.IndexOf(args[0], args[1])

        raise ValueError(f"Unsupported function: {expr.function_name}")

    def create_smt_formula(self, expr: Expression, var_types: Dict[str, str] = None) -> z3.ExprRef:
        """Create SMT formula from expression."""
        return self.convert_expression(expr, var_types)

    def get_variables(self) -> Dict[str, z3.ExprRef]:
        """Get all variables in the SMT context."""
        return self.variable_map.copy()


def expression_to_smt(expr: Expression, var_types: Dict[str, str] = None) -> z3.ExprRef:
    """Convert a VSA expression to SMT format."""
    converter = SMTConverter()
    return converter.convert_expression(expr, var_types)


def smt_to_expression(smt_expr: z3.ExprRef, theory: Theory) -> Expression:
    """Convert SMT expression back to VSA format (simplified implementation)."""
    # This is a simplified reverse conversion - a full implementation would be complex
    if smt_expr.is_int():
        return Constant(int(smt_expr.as_long()), theory)
    elif smt_expr.is_bv():
        return Constant(smt_expr.as_long(), theory)
    elif smt_expr.is_string():
        return Constant(smt_expr.as_string(), theory)
    else:
        # For complex expressions, return a placeholder
        return Constant(0, theory)
