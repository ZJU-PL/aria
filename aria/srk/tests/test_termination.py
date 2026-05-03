"""
Tests for the termination analysis module.
"""

import unittest
from aria.srk.syntax import Context, Symbol, Type
from aria.srk.termination import TerminationAnalyzer, RankingFunction, TerminationResult
from aria.srk.transition import Transition
from aria.srk.qQ import QQ
from aria.srk.syntax import mk_add, mk_and, mk_const, mk_eq, mk_leq, mk_lt, mk_real


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

        # This is a placeholder - real implementation would create actual transitions
        transitions = []  # Empty for now

        result = analyzer.analyze_transitions(transitions)
        self.assertIsInstance(result, TerminationResult)

    def test_analyze_decrementing_counter_with_guard(self):
        """x > 0; x := x - 1 is proved by ranking function x."""
        analyzer = TerminationAnalyzer(self.context)
        x = self.context.mk_symbol("x", Type.INT)
        zero = mk_real(self.context, QQ.zero())
        minus_one = mk_real(self.context, QQ.of_int(-1))
        tr = Transition(
            transform={x: mk_add([mk_const(x), minus_one])},
            guard=mk_lt(zero, mk_const(x)),
            context=self.context,
        )

        result = analyzer.analyze_transitions([tr])

        self.assertTrue(result.terminates)
        self.assertIsNotNone(result.ranking_function)

    def test_analyze_increasing_counter_to_bound(self):
        """x < n; x := x + 1 is proved by ranking function n - x."""
        analyzer = TerminationAnalyzer(self.context)
        x = self.context.mk_symbol("x", Type.INT)
        n = self.context.mk_symbol("n", Type.INT)
        one = mk_real(self.context, QQ.one())
        tr = Transition(
            transform={x: mk_add([mk_const(x), one])},
            guard=mk_lt(mk_const(x), mk_const(n)),
            context=self.context,
        )

        result = analyzer.analyze_transitions([tr])

        self.assertTrue(result.terminates)
        self.assertIsNotNone(result.ranking_function)

    def test_analyze_decrement_without_lower_bound_is_unknown(self):
        """A decreasing transform without a guard lower bound stays unknown."""
        analyzer = TerminationAnalyzer(self.context)
        x = self.context.mk_symbol("x", Type.INT)
        minus_one = mk_real(self.context, QQ.of_int(-1))
        tr = Transition.assign(self.context, x, mk_add([mk_const(x), minus_one]))

        result = analyzer.analyze_transitions([tr])

        self.assertFalse(result.terminates)

    def test_analyze_havoc_like_transform_is_unknown(self):
        """Identity/havoc-like transforms do not prove a positive decrease."""
        analyzer = TerminationAnalyzer(self.context)
        x = self.context.mk_symbol("x", Type.INT)
        zero = mk_real(self.context, QQ.zero())
        tr = Transition(
            transform={x: mk_const(x)},
            guard=mk_lt(zero, mk_const(x)),
            context=self.context,
        )

        result = analyzer.analyze_transitions([tr])

        self.assertFalse(result.terminates)

    def test_analyze_two_counter_lexicographic_loop(self):
        """A loop with two phases is proved by the lexicographic tuple (x, y)."""
        analyzer = TerminationAnalyzer(self.context)
        x = self.context.mk_symbol("x", Type.INT)
        y = self.context.mk_symbol("y", Type.INT)
        zero = mk_real(self.context, QQ.zero())
        one = mk_real(self.context, QQ.one())
        minus_one = mk_real(self.context, QQ.of_int(-1))

        first_phase = Transition(
            transform={
                x: mk_add([mk_const(x), minus_one]),
                y: mk_add([mk_const(y), one]),
            },
            guard=mk_and(
                self.context,
                [
                    mk_lt(zero, mk_const(x)),
                    mk_leq(zero, mk_const(y)),
                ],
            ),
            context=self.context,
        )
        second_phase = Transition(
            transform={
                x: mk_const(x),
                y: mk_add([mk_const(y), minus_one]),
            },
            guard=mk_and(
                self.context,
                [
                    mk_eq(mk_const(x), zero),
                    mk_lt(zero, mk_const(y)),
                ],
            ),
            context=self.context,
        )

        result = analyzer.analyze_transitions([first_phase, second_phase])

        self.assertTrue(result.terminates)
        self.assertIsNotNone(result.ranking_function)


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
