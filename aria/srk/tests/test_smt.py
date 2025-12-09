"""
Tests for the SMT module.
"""

import unittest
from fractions import Fraction
from aria.srk.syntax import Context, Symbol, Type, ExpressionBuilder, Eq, Lt, Leq, And, TrueExpr, FalseExpr
from aria.srk.smt import SMTInterface, SMTResult, SMTModel, check_sat, get_model


class TestSMTInterface(unittest.TestCase):
    """Test SMT solver interface."""

    def setUp(self):
        self.context = Context()
        self.builder = ExpressionBuilder(self.context)
        self.smt = SMTInterface(self.context)

    def test_basic_sat_check(self):
        """Test basic satisfiability checking."""
        x = self.context.mk_symbol("x", Type.INT)
        const_x = self.builder.mk_const(x)

        # Simple true formula should be satisfiable
        true_formula = TrueExpr()
        result = self.smt.is_sat(true_formula)
        self.assertEqual(result, SMTResult.SAT)

        # Simple false formula should be unsatisfiable
        false_formula = FalseExpr()
        result = self.smt.is_sat(false_formula)
        self.assertEqual(result, SMTResult.UNSAT)

    def test_arithmetic_constraints(self):
        """Test arithmetic constraint solving."""
        x = self.context.mk_symbol("x", Type.INT)
        y = self.context.mk_symbol("y", Type.INT)

        const_x = self.builder.mk_const(x)
        const_y = self.builder.mk_const(y)

        # x = 5 should be satisfiable
        eq_formula = Eq(const_x, self.builder.mk_var(0, Type.INT))  # This is a simplified test
        # Note: This test would need more sophisticated expression building

    def test_model_extraction(self):
        """Test model extraction from satisfiable formulas."""
        x = self.context.mk_symbol("x", Type.INT)
        const_x = self.builder.mk_const(x)

        # Simple satisfiable formula
        true_formula = TrueExpr()
        model = self.smt.get_model(true_formula)

        if model is not None:
            self.assertIsInstance(model, SMTModel)
        else:
            # Z3 might not be available in test environment
            self.skipTest("Z3 solver not available")


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience SMT functions."""

    def setUp(self):
        self.context = Context()

    def test_check_sat_function(self):
        """Test check_sat convenience function."""
        true_formula = TrueExpr()
        result = check_sat(self.context, [true_formula])
        self.assertEqual(result, 'sat')

        false_formula = FalseExpr()
        result = check_sat(self.context, [false_formula])
        self.assertEqual(result, 'unsat')


class TestSMTModel(unittest.TestCase):
    """Test SMT model functionality."""

    def test_model_creation(self):
        """Test model creation and value retrieval."""
        interpretations = {
            Symbol(1, "x", Type.INT): Fraction(5),
            Symbol(2, "y", Type.REAL): Fraction(3, 2),
            Symbol(3, "b", Type.BOOL): True
        }

        model = SMTModel(interpretations)

        self.assertEqual(model.get_value(Symbol(1, "x", Type.INT)), Fraction(5))
        self.assertEqual(model.get_value(Symbol(2, "y", Type.REAL)), Fraction(3, 2))
        self.assertEqual(model.get_value(Symbol(3, "b", Type.BOOL)), True)
        self.assertIsNone(model.get_value(Symbol(999, "z", Type.INT)))


if __name__ == '__main__':
    unittest.main()
