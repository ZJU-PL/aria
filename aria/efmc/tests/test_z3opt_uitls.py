import unittest
from unittest.mock import patch, MagicMock, Mock
import z3

from aria.utils.z3.opt import box_optimize, pareto_optimize, maxsmt


class TestBoxOptimize(unittest.TestCase):

    def setUp(self):
        # Common setup for tests
        self.x = z3.Int("x")
        self.y = z3.Int("y")

    def test_box_optimize_basic(self):
        """Test basic functionality with simple constraints."""
        # Create a formula: 0 <= x <= 10, 0 <= y <= 10
        formula = z3.And(self.x >= 0, self.x <= 10, self.y >= 0, self.y <= 10)

        # Minimize x, maximize y
        min_list = [self.x]
        max_list = [self.y]

        # Call the function
        min_res, max_res = box_optimize(formula, min_list, max_list)

        # Check results
        self.assertEqual(min_res[0], 0)  # x should be minimized to 0
        self.assertEqual(max_res[0], 10)  # y should be maximized to 10

    def test_box_optimize_with_timeout(self):
        """Test with timeout parameter."""
        formula = z3.And(self.x >= 0, self.x <= 10, self.y >= 0, self.y <= 10)
        min_list = [self.x]
        max_list = [self.y]

        # Call with timeout
        with patch("z3.Optimize") as mock_optimize:
            mock_instance = MagicMock()
            mock_optimize.return_value = mock_instance

            # Setup mock for check() to return sat
            mock_instance.check.return_value = z3.sat

            # Setup mock for minimize and maximize
            min_obj = MagicMock()
            min_obj.value.return_value = 0
            max_obj = MagicMock()
            max_obj.value.return_value = 10

            mock_instance.minimize.return_value = min_obj
            mock_instance.maximize.return_value = max_obj

            box_optimize(formula, min_list, max_list, timeout=1000)

            # Verify timeout was set
            mock_instance.set.assert_any_call("timeout", 1000)

    def test_box_optimize_multiple_objectives(self):
        """Test with multiple minimize and maximize objectives."""
        # Create a formula with multiple variables
        z = z3.Int("z")
        w = z3.Int("w")
        formula = z3.And(
            self.x >= 0,
            self.x <= 10,
            self.y >= 0,
            self.y <= 10,
            z >= 0,
            z <= 10,
            w >= 0,
            w <= 10,
        )

        # Multiple objectives
        min_list = [self.x, z]
        max_list = [self.y, w]

        # Call the function
        min_res, max_res = box_optimize(formula, min_list, max_list)

        # Check results
        self.assertEqual(len(min_res), 2)
        self.assertEqual(len(max_res), 2)
        self.assertEqual(min_res[0], 0)  # x should be minimized to 0
        self.assertEqual(min_res[1], 0)  # z should be minimized to 0
        self.assertEqual(max_res[0], 10)  # y should be maximized to 10
        self.assertEqual(max_res[1], 10)  # w should be maximized to 10

    def test_box_optimize_unsatisfiable(self):
        """Test with unsatisfiable formula."""
        # Create an unsatisfiable formula: x > 5 and x < 3
        formula = z3.And(self.x > 5, self.x < 3)

        min_list = [self.x]
        max_list = [self.y]

        # Call the function - should return (None, None) for unsatisfiable formula
        min_res, max_res = box_optimize(formula, min_list, max_list)

        # Check results
        self.assertIsNone(min_res)
        self.assertIsNone(max_res)

    def test_box_optimize_empty_objectives(self):
        """Test with empty minimize and maximize lists."""
        formula = z3.And(self.x >= 0, self.x <= 10)

        # Empty objective lists
        min_list = []
        max_list = []

        # Call the function
        min_res, max_res = box_optimize(formula, min_list, max_list)

        # Check results
        self.assertEqual(min_res, [])
        self.assertEqual(max_res, [])

    @patch("z3.Optimize")
    def test_box_optimize_priority_setting(self, mock_optimize):
        """Test that box priority is correctly set."""
        mock_instance = MagicMock()
        mock_optimize.return_value = mock_instance

        # Setup mock for check() to return sat
        mock_instance.check.return_value = z3.sat

        # Setup mock for minimize and maximize
        min_obj = MagicMock()
        min_obj.value.return_value = 0
        mock_instance.minimize.return_value = min_obj

        max_obj = MagicMock()
        max_obj.value.return_value = 10
        mock_instance.maximize.return_value = max_obj

        formula = z3.And(self.x >= 0, self.x <= 10)
        min_list = [self.x]
        max_list = [self.y]

        box_optimize(formula, min_list, max_list)

        # Verify box priority was set
        mock_instance.set.assert_any_call("opt.priority", "box")


class TestParetoOptimize(unittest.TestCase):

    def setUp(self):
        # Create some common variables for tests
        self.x = z3.Int("x")
        self.y = z3.Int("y")

    def test_pareto_optimize_sat_case(self):
        """Test pareto_optimize when the formula is satisfiable"""
        # Create a simple formula: x > 0 and y > 0
        formula = z3.And(self.x > 0, self.y > 0)

        # We want to minimize x and maximize y
        minimize_list = [self.x]
        maximize_list = [self.y]

        # Expected results
        expected_min = [1]  # Minimum value for x is 1
        expected_max = [1]  # Some value for y (will be mocked)

        # Create a mock for z3.Optimize
        with patch("z3.Optimize") as mock_optimize:
            # Configure the mock
            mock_instance = mock_optimize.return_value
            mock_instance.check.return_value = z3.sat

            # Mock the minimize and maximize methods
            mock_min_handle = Mock()
            mock_max_handle = Mock()
            mock_instance.minimize.return_value = mock_min_handle
            mock_instance.maximize.return_value = mock_max_handle

            # Mock the value method for the handles
            mock_min_handle.value.return_value = expected_min[0]
            mock_max_handle.value.return_value = expected_max[0]

            # Call the function
            min_res, max_res = pareto_optimize(formula, minimize_list, maximize_list)

            # Verify the results
            self.assertEqual(min_res, expected_min)
            self.assertEqual(max_res, expected_max)

            # Verify the mock calls
            mock_instance.set.assert_called_with("opt.priority", "pareto")
            mock_instance.add.assert_called_with(formula)
            mock_instance.minimize.assert_called_with(self.x)
            mock_instance.maximize.assert_called_with(self.y)
            mock_instance.check.assert_called_once()

    def test_pareto_optimize_unsat_case(self):
        """Test pareto_optimize when the formula is unsatisfiable"""
        # Create an unsatisfiable formula: x > 0 and x < 0
        formula = z3.And(self.x > 0, self.x < 0)

        # We want to minimize x and maximize y
        minimize_list = [self.x]
        maximize_list = [self.y]

        # Create a mock for z3.Optimize
        with patch("z3.Optimize") as mock_optimize:
            # Configure the mock
            mock_instance = mock_optimize.return_value
            mock_instance.check.return_value = z3.unsat

            # Call the function
            min_res, max_res = pareto_optimize(formula, minimize_list, maximize_list)

            # Verify the results
            self.assertIsNone(min_res)
            self.assertIsNone(max_res)

            # Verify the mock calls
            mock_instance.set.assert_called_with("opt.priority", "pareto")
            mock_instance.add.assert_called_with(formula)
            mock_instance.check.assert_called_once()

    def test_pareto_optimize_with_timeout(self):
        """Test pareto_optimize with a timeout value"""
        # Create a simple formula
        formula = z3.And(self.x > 0, self.y > 0)

        # We want to minimize x and maximize y
        minimize_list = [self.x]
        maximize_list = [self.y]

        # Set a timeout
        timeout = 1000  # 1 second

        # Create a mock for z3.Optimize
        with patch("z3.Optimize") as mock_optimize:
            # Configure the mock
            mock_instance = mock_optimize.return_value
            mock_instance.check.return_value = z3.sat

            # Mock the minimize and maximize methods
            mock_min_handle = Mock()
            mock_max_handle = Mock()
            mock_instance.minimize.return_value = mock_min_handle
            mock_instance.maximize.return_value = mock_max_handle

            # Mock the value method for the handles
            mock_min_handle.value.return_value = 1
            mock_max_handle.value.return_value = 1

            # Call the function with timeout
            pareto_optimize(formula, minimize_list, maximize_list, timeout)

            # Verify the timeout was set
            mock_instance.set.assert_any_call("timeout", timeout)

    def test_pareto_optimize_multiple_objectives(self):
        """Test pareto_optimize with multiple minimize and maximize objectives"""
        # Create a simple formula
        formula = z3.And(self.x > 0, self.y > 0)

        # Create a third variable
        z_var = z3.Int("z")

        # We want to minimize x and z, and maximize y
        minimize_list = [self.x, z_var]
        maximize_list = [self.y]

        # Expected results
        expected_min = [1, 2]  # Minimum values for x and z
        expected_max = [3]  # Maximum value for y

        # Create a mock for z3.Optimize
        with patch("z3.Optimize") as mock_optimize:
            # Configure the mock
            mock_instance = mock_optimize.return_value
            mock_instance.check.return_value = z3.sat

            # Mock the minimize and maximize methods
            mock_min_handle1 = Mock()
            mock_min_handle2 = Mock()
            mock_max_handle = Mock()

            # Configure the minimize method to return different handles for different calls
            mock_instance.minimize.side_effect = [mock_min_handle1, mock_min_handle2]
            mock_instance.maximize.return_value = mock_max_handle

            # Mock the value method for the handles
            mock_min_handle1.value.return_value = expected_min[0]
            mock_min_handle2.value.return_value = expected_min[1]
            mock_max_handle.value.return_value = expected_max[0]

            # Call the function
            min_res, max_res = pareto_optimize(formula, minimize_list, maximize_list)

            # Verify the results
            self.assertEqual(min_res, expected_min)
            self.assertEqual(max_res, expected_max)

            # Verify the mock calls
            self.assertEqual(mock_instance.minimize.call_count, 2)
            self.assertEqual(mock_instance.maximize.call_count, 1)

    def test_pareto_optimize_integration(self):
        """Integration test with actual Z3 solver (no mocks)"""
        # Create a simple formula: x > 0 and y > 0 and x + y < 10
        formula = z3.And(self.x > 0, self.y > 0, self.x + self.y < 10)

        # We want to minimize x and maximize y
        minimize_list = [self.x]
        maximize_list = [self.y]

        # Call the function
        min_res, max_res = pareto_optimize(formula, minimize_list, maximize_list)

        # Verify we got results (not None)
        self.assertIsNotNone(min_res)
        self.assertIsNotNone(max_res)

        # Verify the results make sense
        self.assertEqual(len(min_res), 1)
        self.assertEqual(len(max_res), 1)

        # The minimum value for x should be 1
        self.assertEqual(min_res[0], 1)

        # The maximum value for y should be 8 (since x=1 and x+y<10)
        self.assertEqual(max_res[0], 8)


class TestMaxSMT(unittest.TestCase):

    def setUp(self):
        # Create some common variables for tests
        self.a = z3.Bool("a")
        self.b = z3.Bool("b")
        self.c = z3.Bool("c")

    def test_maxsmt_all_satisfied(self):
        """Test maxsmt when all soft constraints can be satisfied"""
        # Hard constraint: True (always satisfiable)
        hard = z3.BoolVal(True)

        # Soft constraints that can all be satisfied
        soft = [self.a, self.b, self.c]
        weights = [10, 20, 30]

        # Create a mock for z3.Optimize
        with patch("z3.Optimize") as mock_optimize:
            # Configure the mock
            mock_instance = mock_optimize.return_value
            mock_instance.check.return_value = z3.sat

            # Mock the model
            mock_model = Mock()
            mock_instance.model.return_value = mock_model

            # Configure eval to return True for all soft constraints
            mock_model.eval.return_value = z3.BoolVal(True)

            # Call the function
            cost = maxsmt(hard, soft, weights)

            # Verify the results
            self.assertEqual(cost, 0)  # All constraints satisfied, so cost is 0

            # Verify the mock calls
            mock_instance.add.assert_called_with(hard)
            self.assertEqual(mock_instance.add_soft.call_count, 3)
            mock_instance.check.assert_called_once()
            self.assertEqual(mock_model.eval.call_count, 3)

    def test_maxsmt_some_unsatisfied(self):
        """Test maxsmt when some soft constraints cannot be satisfied"""
        # Hard constraint: True (always satisfiable)
        hard = z3.BoolVal(True)

        # Soft constraints
        soft = [self.a, self.b, self.c]
        weights = [10, 20, 30]

        # Create a mock for z3.Optimize
        with patch("z3.Optimize") as mock_optimize:
            # Configure the mock
            mock_instance = mock_optimize.return_value
            mock_instance.check.return_value = z3.sat

            # Mock the model
            mock_model = Mock()
            mock_instance.model.return_value = mock_model

            # Configure eval to return False for the first and third constraints
            def side_effect(expr, model_completion=True):
                if expr == self.a or expr == self.c:
                    return z3.BoolVal(False)
                return z3.BoolVal(True)

            mock_model.eval.side_effect = side_effect

            # Call the function
            cost = maxsmt(hard, soft, weights)

            # Verify the results
            self.assertEqual(
                cost, 40
            )  # a (10) and c (30) are unsatisfied, so cost is 40

            # Verify the mock calls
            mock_instance.add.assert_called_with(hard)
            self.assertEqual(mock_instance.add_soft.call_count, 3)
            mock_instance.check.assert_called_once()
            self.assertEqual(mock_model.eval.call_count, 3)

    def test_maxsmt_unsat(self):
        """Test maxsmt when the hard constraints are unsatisfiable"""
        # Hard constraint: False (unsatisfiable)
        hard = z3.BoolVal(False)

        # Soft constraints
        soft = [self.a, self.b]
        weights = [10, 20]

        # Create a mock for z3.Optimize
        with patch("z3.Optimize") as mock_optimize:
            # Configure the mock
            mock_instance = mock_optimize.return_value
            mock_instance.check.return_value = z3.unsat

            # Call the function
            cost = maxsmt(hard, soft, weights)

            # Verify the results
            self.assertEqual(cost, 0)  # Function returns 0 when unsat

            # Verify the mock calls
            mock_instance.add.assert_called_with(hard)
            self.assertEqual(mock_instance.add_soft.call_count, 2)
            mock_instance.check.assert_called_once()
            # model() should not be called when unsat
            mock_instance.model.assert_not_called()

    def test_maxsmt_with_timeout(self):
        """Test maxsmt with a timeout value"""
        # Hard constraint
        hard = z3.BoolVal(True)

        # Soft constraints
        soft = [self.a, self.b]
        weights = [10, 20]

        # Set a timeout
        timeout = 1000  # 1 second

        # Create a mock for z3.Optimize
        with patch("z3.Optimize") as mock_optimize:
            # Configure the mock
            mock_instance = mock_optimize.return_value
            mock_instance.check.return_value = z3.sat

            # Mock the model
            mock_model = Mock()
            mock_instance.model.return_value = mock_model
            mock_model.eval.return_value = z3.BoolVal(True)

            # Call the function with timeout
            maxsmt(hard, soft, weights, timeout)

            # Verify the timeout was set
            mock_instance.set.assert_called_with("timeout", timeout)

    def test_maxsmt_empty_soft(self):
        """Test maxsmt with empty soft constraints list"""
        # Hard constraint
        hard = z3.BoolVal(True)

        # Empty soft constraints
        soft = []
        weights = []

        # Create a mock for z3.Optimize
        with patch("z3.Optimize") as mock_optimize:
            # Configure the mock
            mock_instance = mock_optimize.return_value
            mock_instance.check.return_value = z3.sat

            # Mock the model
            mock_model = Mock()
            mock_instance.model.return_value = mock_model

            # Call the function
            cost = maxsmt(hard, soft, weights)

            # Verify the results
            self.assertEqual(cost, 0)  # No soft constraints, so cost is 0

            # Verify the mock calls
            mock_instance.add.assert_called_with(hard)
            mock_instance.add_soft.assert_not_called()
            mock_instance.check.assert_called_once()

    def test_maxsmt_integration(self):
        """Integration test with actual Z3 solver (no mocks)"""
        # Hard constraint: a OR b must be true
        hard = z3.Or(self.a, self.b)

        # Soft constraints: prefer a AND b to be false
        soft = [z3.Not(self.a), z3.Not(self.b)]
        weights = [1, 1]

        # Call the function
        cost = maxsmt(hard, soft, weights)

        # At least one of a or b must be true to satisfy the hard constraint
        # So at least one of the soft constraints must be violated
        # The minimum cost should be 1
        self.assertEqual(cost, 1)

        # Another test with conflicting constraints
        # Hard constraint: a AND b must be true
        hard = z3.And(self.a, self.b)

        # Soft constraints: prefer a AND b to be false
        soft = [z3.Not(self.a), z3.Not(self.b)]
        weights = [1, 2]

        # Call the function
        cost = maxsmt(hard, soft, weights)

        # Both a and b must be true to satisfy the hard constraint
        # So both soft constraints must be violated
        # The cost should be 1 + 2 = 3
        self.assertEqual(cost, 3)


if __name__ == "__main__":
    unittest.main()
