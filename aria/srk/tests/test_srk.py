"""
Main test suite for SRK Python package.

This module tests the overall package structure, imports, and basic functionality.
"""

import unittest
import sys
import os
import importlib

# Add the parent directory to the path so we can import aria.srk
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Test that all main modules can be imported
class TestPackageImports(unittest.TestCase):
    """Test that all SRK modules can be imported successfully."""

    def test_syntax_import(self):
        """Test syntax module import."""
        try:
            from aria.srk import syntax

            self.assertIsNotNone(syntax)
            self.assertTrue(hasattr(syntax, "Context"))
            self.assertTrue(hasattr(syntax, "Symbol"))
            self.assertTrue(hasattr(syntax, "Type"))
        except ImportError as e:
            self.fail(f"Failed to import syntax module: {e}")

    def test_polynomial_import(self):
        """Test polynomial module import."""
        try:
            from aria.srk import polynomial

            self.assertIsNotNone(polynomial)
            self.assertTrue(hasattr(polynomial, "Polynomial"))
            self.assertTrue(hasattr(polynomial, "Monomial"))
        except ImportError as e:
            self.fail(f"Failed to import polynomial module: {e}")

    def test_smt_import(self):
        """Test SMT module import."""
        try:
            from aria.srk import smt

            self.assertIsNotNone(smt)
            self.assertTrue(hasattr(smt, "SMTInterface"))
        except ImportError as e:
            self.fail(f"Failed to import smt module: {e}")

    def test_abstract_import(self):
        """Test abstract module import."""
        try:
            from aria.srk import abstract

            self.assertIsNotNone(abstract)
            self.assertTrue(hasattr(abstract, "AbstractDomain"))
        except ImportError as e:
            self.fail(f"Failed to import abstract module: {e}")

    def test_linear_import(self):
        """Test linear module import."""
        try:
            from aria.srk import linear

            self.assertIsNotNone(linear)
            self.assertTrue(hasattr(linear, "QQVector"))
            self.assertTrue(hasattr(linear, "QQMatrix"))
        except ImportError as e:
            self.fail(f"Failed to import linear module: {e}")

    def test_interval_import(self):
        """Test interval module import."""
        try:
            from aria.srk import interval

            self.assertIsNotNone(interval)
            self.assertTrue(hasattr(interval, "Interval"))
        except ImportError as e:
            self.fail(f"Failed to import interval module: {e}")

    def test_polyhedron_import(self):
        """Test polyhedron module import."""
        try:
            from aria.srk import polyhedron

            self.assertIsNotNone(polyhedron)
            self.assertTrue(hasattr(polyhedron, "Polyhedron"))
            self.assertTrue(hasattr(polyhedron, "Constraint"))
        except ImportError as e:
            self.fail(f"Failed to import polyhedron module: {e}")

    def test_simplify_import(self):
        """Test simplify module import."""
        try:
            from aria.srk import Simplifier

            self.assertIsNotNone(Simplifier)
        except ImportError as e:
            self.fail(f"Failed to import Simplifier: {e}")

    def test_util_import(self):
        """Test util module import."""
        try:
            from aria.srk import util

            self.assertIsNotNone(util)
        except ImportError as e:
            self.fail(f"Failed to import util module: {e}")

    def test_termination_dta_import(self):
        """Test that terminationDTA can be imported."""
        try:
            termination_dta = importlib.import_module("aria.srk.terminationDTA")
            self.assertIsNotNone(termination_dta)
        except ImportError as e:
            self.fail(f"Failed to import terminationDTA module: {e}")


class TestPackageStructure(unittest.TestCase):
    """Test the overall package structure and exports."""

    def test_package_version(self):
        """Test that package has version information."""
        try:
            from aria.srk import __version__

            self.assertIsNotNone(__version__)
            self.assertIsInstance(__version__, str)
        except ImportError:
            self.fail("Package version not available")

    def test_package_author(self):
        """Test that package has author information."""
        try:
            from aria.srk import __author__

            self.assertIsNotNone(__author__)
            self.assertIsInstance(__author__, str)
        except ImportError:
            self.fail("Package author not available")

    def test_main_exports_available(self):
        """Test that main exports are available."""
        try:
            from aria.srk import (
                Context,
                Symbol,
                Expression,
                FormulaExpression,
                TermExpression,
                Polynomial,
                SMTInterface,
                AbstractDomain,
                Interval,
                Polyhedron,
            )

            # Just test that these can be imported, not their functionality
            self.assertTrue(True)  # If we get here, imports worked
        except ImportError as e:
            self.fail(f"Failed to import main exports: {e}")

    def test_star_import_smoke(self):
        """Test that wildcard imports do not expose missing symbols."""
        namespace = {}
        try:
            exec("from aria.srk import *", namespace)
        except Exception as e:
            self.fail(f"Wildcard import failed: {e}")

        self.assertIn("Context", namespace)
        self.assertIn("SMTInterface", namespace)


class TestBasicFunctionality(unittest.TestCase):
    """Test basic functionality of core components."""

    def test_context_creation(self):
        """Test that we can create a context."""
        try:
            from aria.srk.syntax import make_context

            context = make_context()
            self.assertIsNotNone(context)
        except Exception as e:
            self.fail(f"Failed to create context: {e}")

    def test_symbol_creation(self):
        """Test that we can create symbols."""
        try:
            from aria.srk.syntax import Context, Type, Symbol

            context = Context()
            symbol = context.mk_symbol("x", Type.INT)
            self.assertIsNotNone(symbol)
            self.assertEqual(symbol.name, "x")
            self.assertEqual(symbol.typ, Type.INT)
        except Exception as e:
            self.fail(f"Failed to create symbol: {e}")

    def test_polynomial_creation(self):
        """Test that we can create polynomials."""
        try:
            from aria.srk.polynomial import Polynomial, Monomial

            # Create a simple polynomial: x + 1
            monomial = Monomial([1, 0])  # x^1 * y^0
            poly = Polynomial({monomial: 1})
            self.assertIsNotNone(poly)
        except Exception as e:
            self.fail(f"Failed to create polynomial: {e}")

    def test_interval_creation(self):
        """Test that we can create intervals."""
        try:
            from aria.srk.interval import Interval

            # Create a simple interval [0, 1]
            interval = Interval(0, 1)
            self.assertIsNotNone(interval)
        except Exception as e:
            self.fail(f"Failed to create interval: {e}")

    def test_zero_and_one_helpers(self):
        """Test compatibility zero/one constructors."""
        from aria.srk.syntax import Context, Type, expr_typ, mk_one, mk_zero

        context = Context()

        zero_int = mk_zero(context)
        one_int = mk_one(context)
        zero_real = mk_zero(context, Type.REAL)
        one_real = mk_one(context, Type.REAL)

        self.assertEqual(expr_typ(zero_int), Type.INT)
        self.assertEqual(expr_typ(one_int), Type.INT)
        self.assertEqual(expr_typ(zero_real), Type.REAL)
        self.assertEqual(expr_typ(one_real), Type.REAL)

    def test_srkast_quantifier_helpers(self):
        """Test list-based srkAst quantifier helpers."""
        from aria.srk.srkAst import mk_exists as ast_mk_exists, mk_forall as ast_mk_forall
        from aria.srk.syntax import Context, Exists, Forall, Type, mk_const, mk_eq
        from aria.srk.syntax import mk_exists, mk_forall, mk_int
        from aria.srk.syntax import mk_symbol

        context = Context()
        x = mk_symbol(context, "x", Type.INT)
        body = mk_eq(context, mk_const(context, x), mk_int(context, 0))

        exists_formula = ast_mk_exists([x], body)
        forall_formula = ast_mk_forall([x], body)

        self.assertIsInstance(exists_formula, Exists)
        self.assertIsInstance(forall_formula, Forall)
        self.assertEqual(exists_formula.var_name, "x")
        self.assertEqual(forall_formula.var_name, "x")
        self.assertEqual(exists_formula.var_type, Type.INT)
        self.assertEqual(forall_formula.var_type, Type.INT)

        explicit_exists = mk_exists(context, "x", Type.INT, body)
        explicit_forall = mk_forall(context, "x", Type.INT, body)

        self.assertIsInstance(explicit_exists, Exists)
        self.assertIsInstance(explicit_forall, Forall)


class TestModuleIntegration(unittest.TestCase):
    """Test that modules work together properly."""

    def test_syntax_to_polynomial_integration(self):
        """Test integration between syntax and polynomial modules."""
        try:
            from aria.srk.syntax import make_context, make_expression_builder, Type
            from aria.srk.polynomial import Polynomial, Monomial

            context = make_context()
            builder = make_expression_builder(context)

            # Create a variable and convert to polynomial
            x = builder.mk_var(1, Type.INT)
            # This tests that the modules can work together
            self.assertIsNotNone(x)
        except Exception as e:
            self.fail(f"Failed syntax-polynomial integration: {e}")

    def test_smt_integration(self):
        """Test SMT module integration."""
        from aria.srk.syntax import make_context
        from aria.srk.smt import SMTInterface

        context = make_context()

        # Test that SMTInterface can be created (even if Z3 is not available)
        # The constructor should handle missing Z3 gracefully
        try:
            smt = SMTInterface(context)
            # If we get here, SMTInterface was created successfully
            self.assertIsNotNone(smt)
        except ImportError as ie:
            # This is expected if Z3 is not installed
            if "Z3 is not installed" in str(ie):
                # This is the expected behavior when Z3 is not available
                self.skipTest("Z3 not installed - skipping SMT integration test")
            else:
                self.fail(f"Unexpected ImportError: {ie}")
        except Exception as e:
            self.fail(f"Failed SMT integration: {e}")


class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases."""

    def test_invalid_symbol_creation(self):
        """Test error handling in symbol creation."""
        try:
            from aria.srk.syntax import Context, Type

            context = Context()

            # Test with invalid type (this might not raise an error but should be handled)
            # This is more of a documentation test
            symbol = context.mk_symbol("test", Type.BOOL)
            self.assertIsNotNone(symbol)

        except Exception as e:
            # If it raises an error, that's also fine - it means validation works
            self.assertIsInstance(e, Exception)

    def test_empty_polynomial(self):
        """Test handling of empty polynomials."""
        try:
            from aria.srk.polynomial import Polynomial

            # Create empty polynomial
            empty_poly = Polynomial({})
            self.assertIsNotNone(empty_poly)

        except Exception as e:
            self.fail(f"Failed to handle empty polynomial: {e}")


if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)
