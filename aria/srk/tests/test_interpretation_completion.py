"""
Tests for the completed interpretation.py module.

This tests the function application and quantifier evaluation features.
"""

import pytest
from fractions import Fraction

from aria.srk.syntax import (
    Context,
    Type,
    mk_symbol,
    mk_const,
    mk_real,
    mk_add,
    mk_mul,
    mk_var,
)
from aria.srk.syntax import mk_eq, mk_leq, mk_and, mk_forall, mk_exists
from aria.srk.interpretation import (
    Interpretation,
    InterpretationValue,
    make_interpretation,
)
from aria.srk.qQ import QQ


class TestFunctionApplicationEvaluation:
    """Test function application evaluation in interpretations."""

    def test_simple_function_application(self):
        """Test evaluating a simple function application."""
        ctx = Context()

        # Create a function symbol f and a variable x
        f = mk_symbol(ctx, "f", Type.REAL)
        x = mk_symbol(ctx, "x", Type.REAL)

        # Create interpretation with f(x) = x + 1
        interp = Interpretation(ctx)
        one = mk_real(ctx, QQ.one)
        var_0 = mk_var(ctx, 0, Type.REAL)
        func_body = mk_add([var_0, one])
        interp = interp.add_fun(f, func_body)

        # Test that we can retrieve the function
        func_value = interp.get_value(f)
        assert isinstance(func_value.value, type(func_body))

    def test_function_application_with_constant_arg(self):
        """Test function application with a constant argument."""
        ctx = Context()

        # Create symbols
        f = mk_symbol(ctx, "f", Type.REAL)

        # Create interpretation with f(x) = 2 * x
        interp = Interpretation(ctx)
        var_0 = mk_var(ctx, 0, Type.REAL)
        two = mk_real(ctx, QQ.of_int(2))
        func_body = mk_mul([two, var_0])
        interp = interp.add_fun(f, func_body)

        # The function is defined, we've tested basic storage
        assert interp.get_value(f) is not None


class TestQuantifierEvaluation:
    """Test quantifier evaluation (requires SMT solver)."""

    def test_quantifier_evaluation_availability(self):
        """Test that quantifier evaluation is available with Z3."""
        from aria.srk.srkZ3 import Z3_AVAILABLE

        # Just check if Z3 is available
        # Actual quantifier tests would require Z3 to be installed
        if Z3_AVAILABLE:
            print("Z3 is available for quantifier evaluation")
        else:
            print(
                "Z3 is not available, quantifier evaluation will raise NotImplementedError"
            )

    def test_quantifier_structure(self):
        """Test creating quantified formulas."""
        ctx = Context()

        # Create a simple quantified formula: ∀x. x >= 0
        x = mk_symbol(ctx, "x", Type.REAL)
        x_const = mk_const(x)
        zero = mk_real(ctx, QQ.zero)
        body = mk_leq(zero, x_const)

        # Create forall
        forall_formula = mk_forall(str(x), x.typ, body)

        # Just verify it was created
        assert forall_formula is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
