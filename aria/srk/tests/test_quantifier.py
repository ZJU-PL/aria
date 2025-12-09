"""
Tests for the quantifier elimination module.
"""

import unittest
from aria.srk.syntax import Context, Symbol, Type
from aria.srk.quantifier import QuantifierEngine, StrategyImprovementSolver


class TestQuantifierEngine(unittest.TestCase):
    """Test quantifier engine functionality."""

    def setUp(self):
        self.context = Context()

    def test_engine_creation(self):
        """Test creating a quantifier engine."""
        engine = QuantifierEngine(self.context)
        self.assertIsNotNone(engine)

    def test_simple_elimination(self):
        """Test simple quantifier elimination."""
        engine = QuantifierEngine(self.context)

        x = self.context.mk_symbol("x", Type.INT)
        from aria.srk.syntax import Lt, Var, Const, Exists

        # ∃x. x > 0
        from aria.srk.syntax import Const, Symbol
        zero_symbol = Symbol(0, "0", Type.INT)
        formula = Exists(str(x), x.typ, Lt(Var(x, Type.INT), Const(zero_symbol)))

        # This should eliminate the quantifier
        result = engine.eliminate_quantifiers(formula)
        self.assertIsNotNone(result)


class TestStrategyImprovementSolver(unittest.TestCase):
    """Test strategy improvement solver functionality."""

    def setUp(self):
        self.context = Context()

    def test_solver_creation(self):
        """Test creating a strategy improvement solver."""
        solver = StrategyImprovementSolver(self.context)
        self.assertIsNotNone(solver)

    def test_solve_simple_game(self):
        """Test solving a simple game."""
        solver = StrategyImprovementSolver(self.context)

        # This is a placeholder for a simple game
        # A real implementation would set up a proper game structure

        # For now, just test that the solver can be created and called
        # result = solver.solve(game)
        # self.assertIsNotNone(result)
        pass


if __name__ == '__main__':
    unittest.main()
