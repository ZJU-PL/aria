"""Expression to SMT conversion utilities."""

from typing import Dict, Optional

import z3

from .expressions import (
    BinaryExpr,
    BinaryOp,
    Constant,
    Expression,
    FunctionCallExpr,
    IfExpr,
    Theory,
    UnaryExpr,
    UnaryOp,
    ValueType,
    Variable,
)


class SMTConverter:
    """Converts typed expressions to Z3 expressions."""

    def __init__(self, context: Optional[z3.Context] = None):
        self.context = context or z3.Context()
        self.variable_map: Dict[str, z3.ExprRef] = {}

    def convert_expression(
        self, expr: Expression, var_types: Optional[Dict[str, int]] = None
    ) -> z3.ExprRef:
        """Convert a VSA expression to an SMT term."""
        var_types = var_types or {}

        if isinstance(expr, Variable):
            return self._convert_variable(expr, var_types)
        if isinstance(expr, Constant):
            return self._convert_constant(expr)
        if isinstance(expr, BinaryExpr):
            return self._convert_binary_expr(expr, var_types)
        if isinstance(expr, UnaryExpr):
            return self._convert_unary_expr(expr, var_types)
        if isinstance(expr, IfExpr):
            return self._convert_if_expr(expr, var_types)
        if isinstance(expr, FunctionCallExpr):
            return self._convert_function_call(expr, var_types)
        raise ValueError(f"Unsupported expression type: {type(expr)}")

    def _convert_variable(
        self, variable: Variable, var_types: Dict[str, int]
    ) -> z3.ExprRef:
        if variable.name not in self.variable_map:
            if variable.value_type == ValueType.INT:
                self.variable_map[variable.name] = z3.Int(variable.name, self.context)
            elif variable.value_type == ValueType.BOOL:
                self.variable_map[variable.name] = z3.Bool(variable.name, self.context)
            elif variable.value_type == ValueType.STRING:
                self.variable_map[variable.name] = z3.String(variable.name, self.context)
            elif variable.value_type == ValueType.BV:
                bitwidth = var_types.get(variable.name, variable.bitwidth or 32)
                self.variable_map[variable.name] = z3.BitVec(
                    variable.name, bitwidth, self.context
                )
            else:
                raise ValueError(f"Unsupported variable type: {variable.value_type}")
        return self.variable_map[variable.name]

    def _convert_constant(self, constant: Constant) -> z3.ExprRef:
        if constant.value_type == ValueType.INT:
            return z3.IntVal(constant.value, self.context)
        if constant.value_type == ValueType.BOOL:
            return z3.BoolVal(constant.value, self.context)
        if constant.value_type == ValueType.STRING:
            return z3.StringVal(str(constant.value), self.context)
        if constant.value_type == ValueType.BV:
            bitwidth = constant.bitwidth or 32
            return z3.BitVecVal(int(constant.value), bitwidth, self.context)
        raise ValueError(f"Unsupported constant type: {constant.value_type}")

    def _convert_binary_expr(
        self, expr: BinaryExpr, var_types: Dict[str, int]
    ) -> z3.ExprRef:
        left = self.convert_expression(expr.left, var_types)
        right = self.convert_expression(expr.right, var_types)

        if expr.op == BinaryOp.ADD:
            return left + right
        if expr.op == BinaryOp.SUB:
            return left - right
        if expr.op == BinaryOp.MUL:
            return left * right
        if expr.op == BinaryOp.DIV:
            return left / right
        if expr.op == BinaryOp.MOD:
            return z3.Mod(left, right)
        if expr.op == BinaryOp.EQ:
            return left == right
        if expr.op == BinaryOp.NEQ:
            return left != right
        if expr.op == BinaryOp.LT:
            return left < right
        if expr.op == BinaryOp.LE:
            return left <= right
        if expr.op == BinaryOp.GT:
            return left > right
        if expr.op == BinaryOp.GE:
            return left >= right
        if expr.op == BinaryOp.AND:
            return z3.And(self._coerce_to_bool(left), self._coerce_to_bool(right))
        if expr.op == BinaryOp.OR:
            return z3.Or(self._coerce_to_bool(left), self._coerce_to_bool(right))
        if expr.op == BinaryOp.CONCAT:
            return z3.Concat(left, right)
        if expr.op == BinaryOp.BVAND:
            return left & right
        if expr.op == BinaryOp.BVOR:
            return left | right
        if expr.op == BinaryOp.BVXOR:
            return left ^ right
        if expr.op == BinaryOp.BVSLL:
            return left << right
        if expr.op == BinaryOp.BVSLR:
            return z3.LShR(left, right)
        if expr.op == BinaryOp.BVSRA:
            return left >> right

        raise ValueError(f"Unsupported binary operation: {expr.op}")

    def _convert_unary_expr(
        self, expr: UnaryExpr, var_types: Dict[str, int]
    ) -> z3.ExprRef:
        operand = self.convert_expression(expr.operand, var_types)

        if expr.op == UnaryOp.NEG:
            return -operand
        if expr.op == UnaryOp.NOT:
            return z3.Not(self._coerce_to_bool(operand))
        if expr.op == UnaryOp.BVNOT:
            return ~operand
        if expr.op == UnaryOp.LENGTH:
            return z3.Length(operand)

        raise ValueError(f"Unsupported unary operation: {expr.op}")

    def _convert_if_expr(
        self, expr: IfExpr, var_types: Dict[str, int]
    ) -> z3.ExprRef:
        condition = self.convert_expression(expr.condition, var_types)
        then_expr = self.convert_expression(expr.then_expr, var_types)
        else_expr = self.convert_expression(expr.else_expr, var_types)
        return z3.If(self._coerce_to_bool(condition), then_expr, else_expr)

    def _convert_function_call(
        self, expr: FunctionCallExpr, var_types: Dict[str, int]
    ) -> z3.ExprRef:
        args = [self.convert_expression(arg, var_types) for arg in expr.args]

        if expr.function_name == "abs":
            return z3.Abs(args[0])
        if expr.function_name == "min":
            return z3.If(args[0] <= args[1], args[0], args[1])
        if expr.function_name == "max":
            return z3.If(args[0] >= args[1], args[0], args[1])
        if expr.function_name == "str_substring":
            return z3.SubString(args[0], args[1], args[2])
        if expr.function_name == "str_indexof":
            return z3.IndexOf(args[0], args[1], z3.IntVal(0, self.context))

        raise ValueError(f"Unsupported function: {expr.function_name}")

    def _coerce_to_bool(self, expr: z3.ExprRef) -> z3.BoolRef:
        if z3.is_bool(expr):
            return expr
        if z3.is_int(expr):
            return expr != z3.IntVal(0, self.context)
        if z3.is_bv(expr):
            return expr != z3.BitVecVal(0, expr.size(), self.context)
        if z3.is_string(expr):
            return expr != z3.StringVal("", self.context)
        raise ValueError(f"Cannot coerce SMT term to bool: {expr}")

    def create_smt_formula(
        self, expr: Expression, var_types: Optional[Dict[str, int]] = None
    ) -> z3.ExprRef:
        """Create an SMT formula from an expression."""
        return self.convert_expression(expr, var_types)

    def get_variables(self) -> Dict[str, z3.ExprRef]:
        """Return declared SMT variables."""
        return self.variable_map.copy()


def expression_to_smt(
    expr: Expression, var_types: Optional[Dict[str, int]] = None
) -> z3.ExprRef:
    """Convert a VSA expression to SMT."""
    return SMTConverter().convert_expression(expr, var_types)


def smt_to_expression(smt_expr: z3.ExprRef, theory: Theory) -> Expression:
    """Convert a simple SMT value back to a VSA expression."""
    from .expressions import Constant, ValueType

    if z3.is_true(smt_expr) or z3.is_false(smt_expr):
        return Constant(z3.is_true(smt_expr), theory, value_type=ValueType.BOOL)
    if z3.is_int_value(smt_expr):
        return Constant(smt_expr.as_long(), theory, value_type=ValueType.INT)
    if z3.is_bv_value(smt_expr):
        return Constant(
            smt_expr.as_long(),
            theory,
            value_type=ValueType.BV,
            bitwidth=smt_expr.size(),
        )
    if z3.is_string_value(smt_expr):
        return Constant(smt_expr.as_string(), theory, value_type=ValueType.STRING)
    raise ValueError("Reverse conversion only supports concrete SMT values")
