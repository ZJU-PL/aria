"""
Tests for the syntax module.
"""

import unittest
from fractions import Fraction
from aria.utils.srk.core.syntax import (
    Context,
    Symbol,
    Type,
    ExpressionBuilder,
    Var,
    Const,
    Add,
    Mul,
    Eq,
    Lt,
    Leq,
    And,
    Or,
    Not,
    TrueExpr,
    FalseExpr,
    make_context,
    make_expression_builder,
    mk_var,
    mk_const,
    mk_eq,
    mk_exists,
    mk_iff,
    mk_int,
    mk_ite,
    mk_lt,
    mk_pow,
    mk_true,
    free_vars,
    size,
    substitute,
    substitute_map,
)


class TestContext(unittest.TestCase):
    """Test context functionality."""

    def setUp(self):
        self.context = Context()

    def test_mk_symbol(self):
        """Test symbol creation."""
        sym1 = self.context.mk_symbol("x", Type.INT)
        sym2 = self.context.mk_symbol("y", Type.REAL)

        self.assertEqual(sym1.name, "x")
        self.assertEqual(sym1.typ, Type.INT)
        self.assertEqual(sym2.name, "y")
        self.assertEqual(sym2.typ, Type.REAL)

        # Test unique IDs
        sym3 = self.context.mk_symbol("x", Type.INT)
        self.assertNotEqual(sym1.id, sym3.id)

    def test_register_named_symbol(self):
        """Test named symbol registration."""
        self.context.register_named_symbol("z", Type.BOOL)
        sym = self.context.get_named_symbol("z")

        self.assertEqual(sym.name, "z")
        self.assertEqual(sym.typ, Type.BOOL)
        self.assertTrue(self.context.is_registered_name("z"))

    def test_symbol_operations(self):
        """Test symbol operations."""
        sym = self.context.mk_symbol("test", Type.REAL)
        self.assertEqual(self.context.symbol_name(sym), "test")
        self.assertEqual(self.context.typ_symbol(sym), Type.REAL)


class TestExpressions(unittest.TestCase):
    """Test expression creation and manipulation."""

    def setUp(self):
        self.context = Context()
        self.builder = ExpressionBuilder(self.context)

    def test_variables_and_constants(self):
        """Test variable and constant expressions."""
        var = self.builder.mk_var(0, Type.INT)
        sym = self.context.mk_symbol("x", Type.REAL)
        const = self.builder.mk_const(sym)

        self.assertEqual(var.var_id, 0)
        self.assertEqual(var.var_type, Type.INT)
        self.assertEqual(const.symbol, sym)
        self.assertEqual(const.typ, Type.REAL)

    def test_arithmetic_expressions(self):
        """Test arithmetic expressions."""
        x = self.builder.mk_var(0, Type.INT)
        y = self.builder.mk_var(1, Type.INT)

        add_expr = self.builder.mk_add([x, y])
        mul_expr = self.builder.mk_mul([x, y])

        self.assertIsInstance(add_expr, Add)
        self.assertIsInstance(mul_expr, Mul)
        self.assertEqual(len(add_expr.args), 2)
        self.assertEqual(len(mul_expr.args), 2)

    def test_boolean_expressions(self):
        """Test boolean expressions."""
        x = self.builder.mk_var(0, Type.INT)
        y = self.builder.mk_var(1, Type.INT)

        eq_expr = self.builder.mk_eq(x, y)
        lt_expr = self.builder.mk_lt(x, y)
        leq_expr = self.builder.mk_leq(x, y)

        self.assertIsInstance(eq_expr, Eq)
        self.assertIsInstance(lt_expr, Lt)
        self.assertIsInstance(leq_expr, Leq)

        true_expr = self.builder.mk_true()
        false_expr = self.builder.mk_false()

        self.assertIsInstance(true_expr, TrueExpr)
        self.assertIsInstance(false_expr, FalseExpr)

    def test_compound_formulas(self):
        """Test compound boolean formulas."""
        x = self.builder.mk_var(0, Type.INT)
        y = self.builder.mk_var(1, Type.INT)

        eq1 = self.builder.mk_eq(x, y)
        eq2 = self.builder.mk_eq(x, self.builder.mk_var(2, Type.INT))

        and_expr = self.builder.mk_and([eq1, eq2])
        or_expr = self.builder.mk_or([eq1, eq2])
        not_expr = self.builder.mk_not(eq1)

        self.assertIsInstance(and_expr, And)
        self.assertIsInstance(or_expr, Or)
        self.assertIsInstance(not_expr, Not)

        self.assertEqual(len(and_expr.args), 2)
        self.assertEqual(len(or_expr.args), 2)

    def test_expression_equality(self):
        """Test expression equality."""
        x = self.builder.mk_var(0, Type.INT)
        y = self.builder.mk_var(1, Type.INT)

        eq1 = self.builder.mk_eq(x, y)
        eq2 = self.builder.mk_eq(x, y)
        eq3 = self.builder.mk_eq(y, x)  # Should be equal due to symmetry

        self.assertEqual(eq1, eq2)
        self.assertEqual(eq1, eq3)

        # Different expressions should not be equal
        lt_expr = self.builder.mk_lt(x, y)
        self.assertNotEqual(eq1, lt_expr)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""

    def test_default_context_functions(self):
        """Test functions using default context."""
        var = mk_var(0, Type.INT)
        self.assertIsInstance(var, Var)

        # Test with custom symbol
        sym = Symbol(100, "test", Type.REAL)
        const = mk_const(sym)
        self.assertIsInstance(const, Const)

        eq = mk_eq(var, const)
        self.assertIsInstance(eq, Eq)

    def test_substitute_replaces_vars_and_consts(self):
        ctx = Context()
        x = ctx.mk_symbol("x", Type.INT)
        y = ctx.mk_symbol("y", Type.INT)
        expr = mk_eq(ctx, mk_var(ctx, x), mk_const(ctx, y))
        replacement = mk_int(ctx, 1)

        result = substitute(expr, {x: replacement, y: replacement})

        self.assertEqual(result, mk_eq(ctx, replacement, replacement))

    def test_substitute_does_not_enter_same_named_quantifier(self):
        ctx = Context()
        x = ctx.mk_symbol("x", Type.INT)
        body = mk_lt(ctx, mk_var(ctx, x), mk_int(ctx, 1))
        quantified = mk_exists(ctx, "x", Type.INT, body)

        result = substitute_map(quantified, {x: mk_int(ctx, 0)})

        self.assertEqual(result, quantified)

    def test_free_vars_and_size_helpers(self):
        ctx = Context()
        x = ctx.mk_symbol("x", Type.INT)
        y = ctx.mk_symbol("y", Type.INT)
        expr = mk_eq(ctx, mk_var(ctx, x), mk_const(ctx, y))

        self.assertEqual({symbol.name for symbol in free_vars(expr)}, {"x", "y"})
        self.assertEqual(size(expr), 3)

    def test_mk_iff_and_mk_pow_helpers(self):
        ctx = Context()
        x = ctx.mk_symbol("x", Type.INT)
        left = mk_lt(ctx, mk_var(ctx, x), mk_int(ctx, 1))
        right = mk_true(ctx)

        self.assertIsInstance(mk_iff(ctx, left, right), And)
        self.assertIsNotNone(mk_pow(ctx, mk_var(ctx, x), mk_int(ctx, 2)))


if __name__ == "__main__":
    unittest.main()
