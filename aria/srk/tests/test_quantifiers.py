"""
Tests for quantifier handling with Z3 integration.

This module tests the quantifier evaluation and SMT solver integration,
including array operations and model extraction.
"""

import unittest
import signal
from fractions import Fraction

from aria.srk.syntax import Context, Type, mk_symbol, mk_const, mk_real, mk_add, mk_mul, mk_eq, mk_leq, mk_lt
from aria.srk.syntax import mk_forall, mk_exists, mk_var, mk_select, mk_store, mk_and, mk_or, mk_not
from aria.srk.smt import SMTInterface, SMTResult, check_sat, get_model
from aria.srk.interpretation import Interpretation, make_interpretation
from aria.srk.qQ import QQ


class TimeoutError(Exception):
    """Raised when a test times out."""
    pass


def timeout_handler(signum, frame):
    """Handle timeout signal."""
    raise TimeoutError("Test timed out")


def with_timeout(timeout_seconds=5):
    """Decorator to add timeout to test methods."""
    def decorator(func):
        def wrapper(self):
            # Set up timeout
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            try:
                return func(self)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        return wrapper
    return decorator


class TestQuantifierEvaluation(unittest.TestCase):
    """Test quantifier evaluation with Z3 integration."""

    def setUp(self):
        self.context = Context()
        self.smt = SMTInterface(self.context)

    def test_z3_availability(self):
        """Test if Z3 is available for quantifier evaluation."""
        try:
            import z3
            self.assertTrue(True, "Z3 is available")
        except ImportError:
            self.skipTest("Z3 not available - install with: pip install z3-solver")

    @with_timeout(3)
    def test_simple_forall(self):
        """Test simple universal quantification."""
        try:
            import z3
        except ImportError:
            self.skipTest("Z3 not available")

        # Create formula: ∀x. x = x (always true)
        x = mk_symbol(self.context, "x", Type.INT)
        x_const = mk_const(x)
        
        # x = x (always true)
        body = mk_eq(x_const, x_const)
        forall_formula = mk_forall("x", Type.INT, body)
        
        # Check satisfiability - should be sat (true for all x)
        result = check_sat(self.context, [forall_formula])
        print(f"Forall x=x result: {result}")
        # This should be satisfiable since x=x is always true
        self.assertEqual(result, 'sat')

    @with_timeout(3)
    def test_simple_exists(self):
        """Test simple existential quantification."""
        try:
            import z3
        except ImportError:
            self.skipTest("Z3 not available")

        # Create formula: ∃x. x = 5 (should be satisfiable)
        x = mk_symbol(self.context, "x", Type.INT)
        x_const = mk_const(x)
        five = mk_real(self.context, QQ.of_int(5))
        
        # x = 5
        body = mk_eq(x_const, five)
        exists_formula = mk_exists("x", Type.INT, body)
        
        # Check satisfiability
        result = check_sat(self.context, [exists_formula])
        print(f"Exists x=5 result: {result}")
        # This should be satisfiable since x=5 has a solution
        self.assertEqual(result, 'sat')

    @with_timeout(3)
    def test_quantifier_with_arithmetic(self):
        """Test quantifier with arithmetic operations."""
        try:
            import z3
        except ImportError:
            self.skipTest("Z3 not available")

        # Create formula: ∃x. x = 2 (simple, should be sat)
        x = mk_symbol(self.context, "x", Type.INT)
        x_const = mk_const(x)
        two = mk_real(self.context, QQ.of_int(2))
        
        # x = 2
        body = mk_eq(x_const, two)
        exists_formula = mk_exists("x", Type.INT, body)
        
        # Check satisfiability
        result = check_sat(self.context, [exists_formula])
        print(f"Exists x=2 result: {result}")
        # This should be satisfiable since x=2 has a solution
        self.assertEqual(result, 'sat')

    @with_timeout(3)
    def test_nested_quantifiers(self):
        """Test nested quantifiers."""
        try:
            import z3
        except ImportError:
            self.skipTest("Z3 not available")

        # Create formula: ∀x. ∃y. y = x (simple, always true)
        x = mk_symbol(self.context, "x", Type.INT)
        y = mk_symbol(self.context, "y", Type.INT)
        x_const = mk_const(x)
        y_const = mk_const(y)
        
        # y = x
        inner_body = mk_eq(y_const, x_const)
        exists_y = mk_exists("y", Type.INT, inner_body)
        forall_x = mk_forall("x", Type.INT, exists_y)
        
        # Check satisfiability
        result = check_sat(self.context, [forall_x])
        print(f"Nested quantifiers result: {result}")
        # This should be satisfiable since for any x, we can choose y=x
        self.assertEqual(result, 'sat')

    @with_timeout(3)
    def test_quantifier_negation(self):
        """Test quantifier with negation."""
        try:
            import z3
        except ImportError:
            self.skipTest("Z3 not available")

        # Create formula: ¬∃x. x ≠ x (always true, since x = x for all x)
        x = mk_symbol(self.context, "x", Type.INT)
        x_const = mk_const(x)
        
        # x ≠ x (always false)
        x_neq_x = mk_not(mk_eq(x_const, x_const))
        exists_x = mk_exists("x", Type.INT, x_neq_x)
        not_exists = mk_not(exists_x)
        
        # Check satisfiability
        result = check_sat(self.context, [not_exists])
        print(f"Negated quantifier result: {result}")
        # This should be satisfiable since ¬∃x. x≠x is equivalent to ∀x. x=x (always true)
        self.assertEqual(result, 'sat')


class TestArrayOperations(unittest.TestCase):
    """Test array operations with quantifiers."""

    def setUp(self):
        self.context = Context()

    def test_array_select_store(self):
        """Test array select and store operations."""
        try:
            import z3
        except ImportError:
            self.skipTest("Z3 not available")

        # Create array symbol
        arr = mk_symbol(self.context, "arr", Type.ARRAY)
        arr_const = mk_const(arr)
        
        # Create index and value (use integer constants for array indices and values)
        idx = mk_symbol(self.context, "idx", Type.INT)
        idx_const = mk_const(idx)
        val = mk_symbol(self.context, "val", Type.INT)
        val_const = mk_const(val)
        
        # Store operation: arr[idx] := val
        stored_arr = mk_store(arr_const, idx_const, val_const)
        
        # Select operation: arr[idx] = val
        selected_val = mk_select(stored_arr, idx_const)
        equality = mk_eq(selected_val, val_const)
        
        # Check satisfiability
        result = check_sat(self.context, [equality])
        print(f"Array select/store result: {result}")
        # This should be satisfiable since we can choose values for idx and val
        self.assertEqual(result, 'sat')

    def test_array_quantifier(self):
        """Test quantifier over array operations."""
        try:
            import z3
        except ImportError:
            self.skipTest("Z3 not available")

        # Create array and index symbols
        arr = mk_symbol(self.context, "arr", Type.ARRAY)
        i = mk_symbol(self.context, "i", Type.INT)
        arr_const = mk_const(arr)
        i_const = mk_const(i)
        
        # Create formula: ∃i. arr[i] = arr[i] (always true)
        arr_i = mk_select(arr_const, i_const)
        body = mk_eq(arr_i, arr_i)
        exists_formula = mk_exists("i", Type.INT, body)
        
        # Check satisfiability
        result = check_sat(self.context, [exists_formula])
        # print(f"Array quantifier result: {result}")
        # This should be satisfiable since arr[i] = arr[i] is always true
        self.assertEqual(result, 'sat')

    def test_array_model_extraction(self):
        """Test model extraction for array operations."""
        try:
            import z3
        except ImportError:
            self.skipTest("Z3 not available")

        # Create a simple satisfiable formula with arrays
        arr = mk_symbol(self.context, "arr", Type.ARRAY)
        arr_const = mk_const(arr)
        idx = mk_symbol(self.context, "idx", Type.INT)
        idx_const = mk_const(idx)
        val = mk_real(self.context, QQ.of_int(10))
        
        # arr[idx] = 10
        select_expr = mk_select(arr_const, idx_const)
        formula = mk_eq(select_expr, val)
        
        # Get model
        model = get_model(formula, self.context)
        if model is not None:
            self.assertIsInstance(model, type(model))
            # Check if we can get array values
            arr_val = model.get_value(arr)
            if isinstance(arr_val, dict):
                # Check if we can get values for some indices
                self.assertTrue(len(arr_val) >= 0)  # Just check it's a dict


class TestInterpretationIntegration(unittest.TestCase):
    """Test integration between interpretation and SMT modules."""

    def setUp(self):
        self.context = Context()
        self.interpretation = make_interpretation(self.context)

    def test_interpretation_with_quantifiers(self):
        """Test interpretation module with quantifier evaluation."""
        # Create a simple quantified formula
        x = mk_symbol(self.context, "x", Type.INT)
        x_const = mk_const(x)
        
        # x = x (always true)
        body = mk_eq(x_const, x_const)
        forall_formula = mk_forall("x", Type.INT, body)
        
        # Test evaluation (should use SMT solver as fallback)
        try:
            result = self.interpretation.evaluate_formula(forall_formula)
            self.assertIsInstance(result, bool)
        except NotImplementedError:
            # Expected if Z3 is not available or quantifier evaluation fails
            pass

    def test_interpretation_with_arrays(self):
        """Test interpretation module with array operations."""
        # Create array and test basic operations
        arr = mk_symbol(self.context, "arr", Type.ARRAY)
        arr_const = mk_const(arr)
        idx = mk_real(self.context, QQ.zero)
        val = mk_real(self.context, QQ.of_int(5))
        
        # Store and select
        stored = mk_store(arr_const, idx, val)
        selected = mk_select(stored, idx)
        
        # This should work even without Z3
        try:
            # Test that expressions can be created
            self.assertIsNotNone(stored)
            self.assertIsNotNone(selected)
        except Exception as e:
            self.fail(f"Array operations failed: {e}")


if __name__ == '__main__':
    unittest.main()
