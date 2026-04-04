"""Unit tests for uf_utils module."""

import pytest
import z3
from aria.utils.uf_utils import (
    visitor,
    modify,
    replace_func_with_template,
    instiatiate_func_with_axioms,
    purify,
)


class TestUFUtils:
    """Test cases for uf_utils functionality."""

    def test_visitor(self):
        """Test visitor function."""
        x, y = z3.Ints("x y")
        formula = z3.And(x > 0, y < 10, x + y == 5)

        seen = {}
        expressions = list(visitor(formula, seen))

        assert formula in expressions
        assert x in expressions
        assert y in expressions

    def test_modify(self):
        """Test modify function with transformations."""
        x, y = z3.Ints("x y")
        formula = z3.And(x > 0, y < 10)

        # Test identity transformation
        modified = modify(formula, lambda expr: None)
        assert z3.eq(modified, formula)

        # Test constant doubling
        def double_constants(expr):
            if z3.is_const(expr) and z3.is_int(expr):
                try:
                    return z3.IntVal(expr.as_long() * 2)
                except:
                    pass
            return None

        modified = modify(formula, double_constants)
        assert modified is not None

    def test_replace_func_with_template(self):
        """Test function replacement with template."""
        x, y = z3.Ints("x y")
        f = z3.Function("f", z3.IntSort(), z3.IntSort(), z3.BoolSort())
        formula = z3.And(f(x, y), x > 0)

        a0, a1, a2 = z3.Ints("a0 a1 a2")
        template = a0 + a1 * x + a2 * y >= 0

        result = replace_func_with_template(formula, f, template)

        assert result is not None
        # Check that f is no longer in the result
        seen = {}
        func_apps = [
            expr
            for expr in visitor(result, seen)
            if z3.is_app(expr) and z3.eq(expr.decl(), f)
        ]
        assert len(func_apps) == 0

    def test_instiatiate_func_with_axioms(self):
        """Test function instantiation with axioms."""
        x, y = z3.Ints("x y")
        f = z3.Function("f", z3.IntSort(), z3.IntSort(), z3.BoolSort())

        # Test with function applications
        formula = z3.And(f(x, y), x > 0)
        axiom = z3.ForAll([x, y], z3.Implies(f(x, y), x + y > 0))
        result = instiatiate_func_with_axioms(formula, f, axiom)
        assert result is not None
        assert z3.is_and(result)

        # Test without function applications
        formula_no_f = z3.And(x > 0, y < 10)
        result = instiatiate_func_with_axioms(formula_no_f, f, axiom)
        assert z3.eq(result, formula_no_f)

    def test_purify(self):
        """Test purification of mixed formulas."""
        x, y, z = z3.Ints("x y z")
        f = z3.Function("f", z3.IntSort(), z3.IntSort())
        g = z3.Function("g", z3.IntSort(), z3.IntSort())

        # Test simple mixed formula
        mixed = f(x + y) > 0
        purified = purify(mixed)
        assert purified is not None

        # Test complex mixed formula
        complex_mixed = z3.And(f(x + y) > z, g(x) + g(y) == 10, z > 0)
        purified = purify(complex_mixed)
        assert purified is not None
        assert z3.is_and(purified)

        # Test pure arithmetic formula
        pure = z3.And(x > 0, y < 10, x + y == 5)
        purified = purify(pure)
        assert purified is not None

        # Test UF-only formula
        uf_only = z3.And(f(x) > 0, f(y) < 10)
        purified = purify(uf_only)
        assert purified is not None

    def test_purify_quantified_error(self):
        """Test that quantified formulas raise error."""
        x, y = z3.Ints("x y")
        f = z3.Function("f", z3.IntSort(), z3.IntSort())
        quantified = z3.ForAll([x, y], f(x + y) > 0)

        with pytest.raises(ValueError, match="Quantified formulas not supported"):
            purify(quantified)


if __name__ == "__main__":
    pytest.main([__file__])
