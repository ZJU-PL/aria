"""
Tests for the termination analysis module.
"""

import unittest
from aria.srk.syntax import Context, Symbol, Type
from aria.srk.termination import TerminationAnalyzer, RankingFunction, TerminationResult


class TestTerminationAnalyzer(unittest.TestCase):
    """Test termination analyzer functionality."""

    def setUp(self):
        self.context = Context()

    def test_analyzer_creation(self):
        """Test creating a termination analyzer."""
        analyzer = TerminationAnalyzer(self.context)
        self.assertIsNotNone(analyzer)

    def test_analyze_simple_loop(self):
        """Test analyzing a simple loop for termination."""
        analyzer = TerminationAnalyzer(self.context)

        # Create a simple loop transition
        x = self.context.mk_symbol("x", Type.INT)
        from aria.srk.transition import Transition

        # This is a placeholder - real implementation would create actual transitions
        transitions = []  # Empty for now

        result = analyzer.analyze_transitions(transitions)
        self.assertIsInstance(result, TerminationResult)


class TestRankingFunction(unittest.TestCase):
    """Test ranking function functionality."""

    def setUp(self):
        self.context = Context()

    def test_ranking_function_creation(self):
        """Test creating a ranking function."""
        x = self.context.mk_symbol("x", Type.INT)

        # Create a simple linear ranking function: x + 1
        from aria.srk.syntax import mk_add, mk_var, mk_const, Symbol

        one_symbol = Symbol(1, "1", Type.INT)
        expr = mk_add([mk_var(x, Type.INT), mk_const(one_symbol)])

        ranking_func = RankingFunction(expr, True)  # Decreases = True
        self.assertEqual(ranking_func.expression, expr)
        self.assertTrue(ranking_func.decreases)


class TestTerminationResult(unittest.TestCase):
    """Test termination result functionality."""

    def test_terminating_result(self):
        """Test terminating result."""
        result = TerminationResult(True)
        self.assertTrue(result.terminates)
        self.assertIsNone(result.ranking_function)

    def test_non_terminating_result(self):
        """Test non-terminating result."""
        result = TerminationResult(False)
        self.assertFalse(result.terminates)

    def test_result_with_ranking_function(self):
        """Test result with ranking function."""
        x = Context().mk_symbol("x", Type.INT)
        from aria.srk.syntax import mk_add, mk_var, mk_const, Symbol

        one_symbol = Symbol(1, "1", Type.INT)
        expr = mk_add([mk_var(x, Type.INT), mk_const(one_symbol)])
        ranking_func = RankingFunction(expr, True)

        result = TerminationResult(True, ranking_func)
        self.assertTrue(result.terminates)
        self.assertEqual(result.ranking_function, ranking_func)


if __name__ == "__main__":
    unittest.main()
