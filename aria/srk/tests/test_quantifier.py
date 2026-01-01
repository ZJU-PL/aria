"""
Tests for the quantifier elimination module.
"""

import unittest
from aria.srk.syntax import (
    Context,
    Symbol,
    Type,
    Lt,
    Leq,
    Eq,
    Var,
    Const,
    Exists,
    Forall,
    mk_symbol,
    mk_const,
    mk_real,
    mk_add,
    mk_mul,
    mk_eq,
    mk_lt,
    mk_leq,
    mk_geq,
    mk_and,
    mk_or,
    mk_not,
)
from aria.srk.quantifier import QuantifierEngine, StrategyImprovementSolver


class TestQuantifierEngine(unittest.TestCase):
    """Test quantifier engine functionality."""

    def setUp(self):
        self.context = Context()

    def test_engine_creation(self):
        """Test creating a quantifier engine."""
        engine = QuantifierEngine(self.context)
        self.assertIsNotNone(engine)

    def test_simple_exists_elimination(self):
        """Test simple existential quantifier elimination: ∃x. x > 0."""
        engine = QuantifierEngine(self.context)

        x = mk_symbol(self.context, "x", Type.INT)
        zero = mk_const(mk_symbol(self.context, "0", Type.INT))
        x_var = self.context.mk_var(x)

        # ∃x. x > 0  (Note: Lt(x_var, zero) means x < 0, we want x > 0)
        # So we use mk_lt with swapped arguments: zero < x_var
        formula = Exists(str(x), x.typ, mk_lt(zero, x_var))

        result = engine.eliminate_quantifiers(formula)
        self.assertIsNotNone(result)
        # Result should be quantifier-free
        self.assertFalse(isinstance(result, (Exists, Forall)))

    def test_simple_forall_elimination(self):
        """Test simple universal quantifier elimination: ∀x. x >= 0."""
        engine = QuantifierEngine(self.context)

        x = mk_symbol(self.context, "x", Type.INT)
        zero = mk_const(mk_symbol(self.context, "0", Type.INT))
        x_var = self.context.mk_var(x)

        # ∀x. x >= 0
        formula = Forall(str(x), x.typ, mk_geq(x_var, zero))

        result = engine.eliminate_quantifiers(formula)
        self.assertIsNotNone(result)
        self.assertFalse(isinstance(result, (Exists, Forall)))

    def test_exists_with_constant(self):
        """Test existential quantifier with constant constraint: ∃x. x = 5."""
        engine = QuantifierEngine(self.context)

        x = mk_symbol(self.context, "x", Type.INT)
        five = mk_const(mk_symbol(self.context, "5", Type.INT))
        x_var = self.context.mk_var(x)

        # ∃x. x = 5
        formula = Exists(str(x), x.typ, mk_eq(x_var, five))

        result = engine.eliminate_quantifiers(formula)
        self.assertIsNotNone(result)
        self.assertFalse(isinstance(result, (Exists, Forall)))

    def test_exists_with_inequality(self):
        """Test existential quantifier with inequality: ∃x. x < 10."""
        engine = QuantifierEngine(self.context)

        x = mk_symbol(self.context, "x", Type.INT)
        ten = mk_const(mk_symbol(self.context, "10", Type.INT))
        x_var = self.context.mk_var(x)

        # ∃x. x < 10
        formula = Exists(str(x), x.typ, mk_lt(x_var, ten))

        result = engine.eliminate_quantifiers(formula)
        self.assertIsNotNone(result)
        self.assertFalse(isinstance(result, (Exists, Forall)))

    def test_exists_with_disjunction(self):
        """Test existential quantifier with disjunction: ∃x. x < 0 ∨ x > 10."""
        engine = QuantifierEngine(self.context)

        x = mk_symbol(self.context, "x", Type.INT)
        zero = mk_const(mk_symbol(self.context, "0", Type.INT))
        ten = mk_const(mk_symbol(self.context, "10", Type.INT))
        x_var = self.context.mk_var(x)

        # ∃x. x < 0 ∨ x > 10
        formula = Exists(str(x), x.typ, mk_or([mk_lt(x_var, zero), mk_lt(ten, x_var)]))

        result = engine.eliminate_quantifiers(formula)
        print(f"Formula: {formula}")
        print(f"Result: {result}")
        self.assertIsNotNone(result)
        self.assertFalse(isinstance(result, (Exists, Forall)))

    def test_forall_with_implication(self):
        """Test universal quantifier: ∀x. x > 0 → x > -1."""
        engine = QuantifierEngine(self.context)

        x = mk_symbol(self.context, "x", Type.INT)
        zero = mk_const(mk_symbol(self.context, "0", Type.INT))
        neg_one = mk_const(mk_symbol(self.context, "-1", Type.INT))
        x_var = self.context.mk_var(x)

        # ∀x. x > 0 → x > -1
        # This is equivalent to: ∀x. ¬(x > 0) ∨ (x > -1)
        formula = Forall(
            str(x), x.typ, mk_or([mk_not(mk_lt(zero, x_var)), mk_lt(neg_one, x_var)])
        )

        result = engine.eliminate_quantifiers(formula)
        print(f"Formula: {formula}")
        print(f"Result: {result}")
        self.assertIsNotNone(result)
        self.assertFalse(isinstance(result, (Exists, Forall)))

    def test_multiple_exists_quantifiers(self):
        """Test multiple existential quantifiers: ∃x. ∃y. x + y > 0."""
        engine = QuantifierEngine(self.context)

        x = mk_symbol(self.context, "x", Type.INT)
        y = mk_symbol(self.context, "y", Type.INT)
        zero = mk_const(mk_symbol(self.context, "0", Type.INT))
        x_var = self.context.mk_var(x)
        y_var = self.context.mk_var(y)

        # ∃x. ∃y. x + y > 0
        x_plus_y = mk_add([x_var, y_var])
        inner = Exists(str(y), y.typ, mk_lt(zero, x_plus_y))
        formula = Exists(str(x), x.typ, inner)

        result = engine.eliminate_quantifiers(formula)
        print(f"Formula: {formula}")
        print(f"Result: {result}")
        self.assertIsNotNone(result)
        self.assertFalse(isinstance(result, (Exists, Forall)))

    def test_multiple_forall_quantifiers(self):
        """Test multiple universal quantifiers: ∀x. ∀y. x + y >= x."""
        engine = QuantifierEngine(self.context)

        x = mk_symbol(self.context, "x", Type.INT)
        y = mk_symbol(self.context, "y", Type.INT)
        x_var = self.context.mk_var(x)
        y_var = self.context.mk_var(y)

        # ∀x. ∀y. x + y >= x
        x_plus_y = mk_add([x_var, y_var])
        inner = Forall(str(y), y.typ, mk_geq(x_plus_y, x_var))
        formula = Forall(str(x), x.typ, inner)

        result = engine.eliminate_quantifiers(formula)
        print(f"Formula: {formula}")
        print(f"Result: {result}")
        self.assertIsNotNone(result)
        self.assertFalse(isinstance(result, (Exists, Forall)))

    def test_mixed_quantifiers(self):
        """Test mixed quantifiers: ∃x. ∀y. x + y > y."""
        engine = QuantifierEngine(self.context)

        x = mk_symbol(self.context, "x", Type.INT)
        y = mk_symbol(self.context, "y", Type.INT)
        x_var = self.context.mk_var(x)
        y_var = self.context.mk_var(y)

        # ∃x. ∀y. x + y > y
        x_plus_y = mk_add([x_var, y_var])
        inner = Forall(str(y), y.typ, mk_lt(y_var, x_plus_y))
        formula = Exists(str(x), x.typ, inner)

        result = engine.eliminate_quantifiers(formula)
        print(f"Formula: {formula}")
        print(f"Result: {result}")
        self.assertIsNotNone(result)
        self.assertFalse(isinstance(result, (Exists, Forall)))

    def test_quantifier_free_formula(self):
        """Test that quantifier-free formulas are returned unchanged."""
        engine = QuantifierEngine(self.context)

        x = mk_symbol(self.context, "x", Type.INT)
        zero = mk_const(mk_symbol(self.context, "0", Type.INT))
        x_var = self.context.mk_var(x)

        # x > 0 (no quantifiers)
        formula = mk_lt(zero, x_var)

        result = engine.eliminate_quantifiers(formula)
        self.assertEqual(result, formula)

    def test_nested_quantifiers(self):
        """Test deeply nested quantifiers: ∃x. (∀y. y > x) ∧ (x > 0)."""
        engine = QuantifierEngine(self.context)

        x = mk_symbol(self.context, "x", Type.INT)
        y = mk_symbol(self.context, "y", Type.INT)
        zero = mk_const(mk_symbol(self.context, "0", Type.INT))
        x_var = self.context.mk_var(x)
        y_var = self.context.mk_var(y)

        # ∃x. (∀y. y > x) ∧ (x > 0)
        forall_part = Forall(str(y), y.typ, mk_lt(x_var, y_var))
        exists_part = mk_lt(zero, x_var)
        formula = Exists(str(x), x.typ, mk_and([forall_part, exists_part]))

        result = engine.eliminate_quantifiers(formula)
        print(f"Formula: {formula}")
        print(f"Result: {result}")
        self.assertIsNotNone(result)
        self.assertFalse(isinstance(result, (Exists, Forall)))

    def test_forall_negation(self):
        """Test universal quantifier with negation: ∀x. ¬(x < 0)."""
        engine = QuantifierEngine(self.context)

        x = mk_symbol(self.context, "x", Type.INT)
        zero = mk_const(mk_symbol(self.context, "0", Type.INT))
        x_var = self.context.mk_var(x)

        # ∀x. ¬(x < 0) which is equivalent to ∀x. x >= 0
        formula = Forall(str(x), x.typ, mk_not(mk_lt(x_var, zero)))

        result = engine.eliminate_quantifiers(formula)
        print(f"Formula: {formula}")
        print(f"Result: {result}")
        self.assertIsNotNone(result)
        self.assertFalse(isinstance(result, (Exists, Forall)))


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


if __name__ == "__main__":
    unittest.main()
