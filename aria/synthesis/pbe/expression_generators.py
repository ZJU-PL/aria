"""Expression generators for different theories.

This module provides functions to generate possible expressions
for LIA, BV, and String theories.
"""

from typing import List, Set, Dict, Any
from .expressions import (
    Expression, Theory, Variable, Constant, BinaryExpr, UnaryExpr,
    BinaryOp, UnaryOp, IfExpr, FunctionCallExpr, var, const, add, sub, mul, eq, lt, concat, bv_and, length
)


def generate_lia_expressions(variables: List[str], max_depth: int = 3) -> List[Expression]:
    """Generate LIA expressions up to max_depth."""
    expressions = []

    # Variables and constants
    for var_name in variables:
        expressions.append(var(var_name, Theory.LIA))

    # Constants
    expressions.extend([
        const(0, Theory.LIA),
        const(1, Theory.LIA),
        const(-1, Theory.LIA),
        const(2, Theory.LIA),
        const(10, Theory.LIA),
    ])

    # Unary operations (at depth 1)
    if max_depth >= 1:
        base_exprs = [var(v, Theory.LIA) for v in variables] + [
            const(0, Theory.LIA), const(1, Theory.LIA), const(2, Theory.LIA)
        ]

        for expr in base_exprs:
            expressions.append(UnaryExpr(UnaryOp.NEG, expr))

    # Binary operations (at depth 2+)
    if max_depth >= 2:
        base_exprs = [var(v, Theory.LIA) for v in variables] + [
            const(0, Theory.LIA), const(1, Theory.LIA), const(2, Theory.LIA), const(10, Theory.LIA)
        ]

        for left in base_exprs:
            for right in base_exprs:
                expressions.append(add(left, right))
                expressions.append(sub(left, right))
                expressions.append(mul(left, right))
                if right != const(0, Theory.LIA):  # Avoid division by zero in generator
                    expressions.append(BinaryExpr(left, BinaryOp.DIV, right))
                    expressions.append(BinaryExpr(left, BinaryOp.MOD, right))
                expressions.append(eq(left, right))
                expressions.append(lt(left, right))

    # Advanced expressions (conditionals, functions) - depth 3+
    if max_depth >= 3:
        base_exprs = [var(v, Theory.LIA) for v in variables] + [
            const(0, Theory.LIA), const(1, Theory.LIA), const(2, Theory.LIA), const(10, Theory.LIA)
        ]

        # Simple conditionals
        for cond in base_exprs:
            for then_expr in base_exprs:
                for else_expr in base_exprs:
                    # Simple conditions like (x > 0 ? y : z)
                    if isinstance(cond, (Variable, Constant)):
                        continue  # Skip simple variables/constants as conditions
                    expressions.append(IfExpr(cond, then_expr, else_expr))

        # Function calls
        for expr in base_exprs:
            expressions.append(FunctionCallExpr("abs", [expr], Theory.LIA))
            if len(base_exprs) >= 2:
                for expr2 in base_exprs:
                    expressions.append(FunctionCallExpr("min", [expr, expr2], Theory.LIA))
                    expressions.append(FunctionCallExpr("max", [expr, expr2], Theory.LIA))

    # Remove duplicates
    seen = set()
    unique_expressions = []
    for expr in expressions:
        expr_str = str(expr)
        if expr_str not in seen:
            seen.add(expr_str)
            unique_expressions.append(expr)

    return unique_expressions


def generate_bv_expressions(variables: List[str], bitwidth: int = 8, max_depth: int = 3) -> List[Expression]:
    """Generate bitvector expressions up to max_depth with advanced constructs."""
    expressions = []

    # Variables and constants
    for var_name in variables:
        expressions.append(var(var_name, Theory.BV))

    # Constants
    expressions.extend([
        const(0, Theory.BV),
        const(1, Theory.BV),
        const(0xFF, Theory.BV),
        const(0xF0, Theory.BV),
        const(0x0F, Theory.BV),
    ])

    # Unary operations (at depth 1)
    if max_depth >= 1:
        base_exprs = [var(v, Theory.BV) for v in variables] + [
            const(0, Theory.BV), const(1, Theory.BV), const(0xFF, Theory.BV)
        ]

        for expr in base_exprs:
            expressions.append(UnaryExpr(UnaryOp.BVNOT, expr))

    # Binary operations (at depth 2+)
    if max_depth >= 2:
        base_exprs = [var(v, Theory.BV) for v in variables] + [
            const(0, Theory.BV), const(1, Theory.BV), const(0xFF, Theory.BV), const(0xF0, Theory.BV)
        ]

        for left in base_exprs:
            for right in base_exprs:
                expressions.append(bv_and(left, right))
                expressions.append(BinaryExpr(left, BinaryOp.BVOR, right))
                expressions.append(BinaryExpr(left, BinaryOp.BVXOR, right))
                expressions.append(BinaryExpr(left, BinaryOp.BVSLL, right))
                expressions.append(BinaryExpr(left, BinaryOp.BVSLR, right))
                expressions.append(eq(left, right))

    # Advanced expressions (conditionals) - depth 3+
    if max_depth >= 3:
        base_exprs = [var(v, Theory.BV) for v in variables] + [
            const(0, Theory.BV), const(1, Theory.BV), const(0xFF, Theory.BV), const(0xF0, Theory.BV)
        ]

        # Simple conditionals for BV
        for cond in base_exprs:
            for then_expr in base_exprs:
                for else_expr in base_exprs:
                    # Simple conditions like (x != 0 ? y : z)
                    if isinstance(cond, (Variable, Constant)):
                        continue  # Skip simple variables/constants as conditions
                    expressions.append(IfExpr(cond, then_expr, else_expr))

    # Remove duplicates
    seen = set()
    unique_expressions = []
    for expr in expressions:
        expr_str = str(expr)
        if expr_str not in seen:
            seen.add(expr_str)
            unique_expressions.append(expr)

    return unique_expressions


def generate_string_expressions(variables: List[str], max_depth: int = 3) -> List[Expression]:
    """Generate string expressions up to max_depth."""
    expressions = []

    # Variables and constants
    for var_name in variables:
        expressions.append(var(var_name, Theory.STRING))

    # Constants
    expressions.extend([
        const("", Theory.STRING),
        const("a", Theory.STRING),
        const("b", Theory.STRING),
        const("abc", Theory.STRING),
        const("0", Theory.STRING),
        const("1", Theory.STRING),
    ])

    # Unary operations (at depth 1)
    if max_depth >= 1:
        base_exprs = [var(v, Theory.STRING) for v in variables] + [
            const("", Theory.STRING), const("a", Theory.STRING), const("abc", Theory.STRING)
        ]

        for expr in base_exprs:
            expressions.append(length(expr))

    # Binary operations (at depth 2+)
    if max_depth >= 2:
        base_exprs = [var(v, Theory.STRING) for v in variables] + [
            const("", Theory.STRING), const("a", Theory.STRING), const("b", Theory.STRING),
            const("abc", Theory.STRING), const("0", Theory.STRING), const("1", Theory.STRING)
        ]

        for left in base_exprs:
            for right in base_exprs:
                expressions.append(concat(left, right))
                expressions.append(eq(left, right))

    # Advanced expressions (conditionals, functions) - depth 3+
    if max_depth >= 3:
        base_exprs = [var(v, Theory.STRING) for v in variables] + [
            const("", Theory.STRING), const("a", Theory.STRING), const("b", Theory.STRING),
            const("abc", Theory.STRING), const("0", Theory.STRING), const("1", Theory.STRING)
        ]

        # Simple conditionals for String
        for cond in base_exprs:
            for then_expr in base_exprs:
                for else_expr in base_exprs:
                    # Simple conditions like (len(s) > 0 ? s : "")
                    if isinstance(cond, (Variable, Constant)):
                        continue  # Skip simple variables/constants as conditions
                    expressions.append(IfExpr(cond, then_expr, else_expr))

        # Function calls for strings
        for expr in base_exprs:
            expressions.append(FunctionCallExpr("str_substring", [expr, const(0, Theory.STRING), const(1, Theory.STRING)], Theory.STRING))
            expressions.append(FunctionCallExpr("str_indexof", [expr, const("a", Theory.STRING)], Theory.STRING))

    # Remove duplicates
    seen = set()
    unique_expressions = []
    for expr in expressions:
        expr_str = str(expr)
        if expr_str not in seen:
            seen.add(expr_str)
            unique_expressions.append(expr)

    return unique_expressions


def generate_expressions_for_theory(theory: Theory, variables: List[str], **kwargs) -> List[Expression]:
    """Generate expressions for a given theory."""
    if theory == Theory.LIA:
        return generate_lia_expressions(variables, **kwargs)
    elif theory == Theory.BV:
        return generate_bv_expressions(variables, **kwargs)
    elif theory == Theory.STRING:
        return generate_string_expressions(variables, **kwargs)
    else:
        raise ValueError(f"Unsupported theory: {theory}")


def get_theory_from_variables(variables: List[Dict[str, Any]]) -> Theory:
    """Infer the theory from example variable types."""
    if not variables:
        raise ValueError("Cannot infer theory from empty examples")

    # Check the first example to determine theory
    first_example = variables[0]
    sample_values = {}

    for var_name, value in first_example.items():
        if var_name == 'output':
            continue
        sample_values[var_name] = value

    # Check if values are integers (LIA)
    if all(isinstance(v, int) for v in sample_values.values()):
        return Theory.LIA

    # Check if values are strings (String)
    if all(isinstance(v, str) for v in sample_values.values()):
        return Theory.STRING

    # Default to LIA for mixed types or unknown types
    return Theory.LIA
