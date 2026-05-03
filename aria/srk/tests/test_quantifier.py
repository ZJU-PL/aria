"""
Tests for the quantifier elimination module.
"""

import unittest
from fractions import Fraction
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
from aria.srk.quantifier import (
    QuantifierEngine,
    StrategyImprovementSolver,
    check_strategy,
    maximize,
    simsat,
)


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

    def test_simsat_uses_quantifier_prefix_for_linear_arithmetic(self):
        """Check a simple quantified linear arithmetic game with Z3."""
        x = mk_symbol(self.context, "qx", Type.REAL)
        x_const = mk_const(x)
        zero = mk_real(self.context, 0)

        formula = Exists(str(x), x.typ, mk_lt(zero, x_const))

        self.assertEqual(simsat(self.context, formula), "Sat")

    def test_maximize_returns_bounded_linear_optimum(self):
        """Maximize a bounded linear objective instead of returning a fallback."""
        x = mk_symbol(self.context, "mx", Type.REAL)
        x_const = mk_const(x)
        zero = mk_real(self.context, 0)
        five = mk_real(self.context, 5)
        constraints = mk_and([mk_leq(zero, x_const), mk_leq(x_const, five)])

        result = maximize(self.context, constraints, x_const)

        self.assertEqual(result, ("Bounded", 5))

    def test_maximize_reports_unbounded_linear_objective(self):
        """Recognize an unbounded supported linear optimization problem."""
        x = mk_symbol(self.context, "ux", Type.REAL)
        x_const = mk_const(x)
        zero = mk_real(self.context, 0)
        constraints = mk_leq(zero, x_const)

        self.assertEqual(maximize(self.context, constraints, x_const), "Infinity")

    def test_maximize_returns_unknown_for_nonlinear_objective(self):
        """Unsupported nonlinear objectives should not get optimistic fallbacks."""
        x = mk_symbol(self.context, "nx", Type.REAL)
        x_const = mk_const(x)
        zero = mk_real(self.context, 0)
        constraints = mk_leq(zero, x_const)
        nonlinear_objective = mk_mul([x_const, x_const])

        self.assertEqual(
            maximize(self.context, constraints, nonlinear_objective), "Unknown"
        )

    def test_check_strategy_returns_false_for_unsat_linear_game(self):
        """Strategy validation should inspect the actual game formula."""
        x = mk_symbol(self.context, "sx", Type.REAL)
        x_const = mk_const(x)
        zero = mk_real(self.context, 0)
        formula = mk_and([mk_lt(x_const, zero), mk_leq(zero, x_const)])

        self.assertFalse(check_strategy(self.context, [], formula, {"strategy": []}))

    def test_check_strategy_validates_concrete_existential_move(self):
        """Concrete existential strategies are checked against the matrix."""
        x = mk_symbol(self.context, "cx", Type.REAL)
        x_const = mk_const(x)
        zero = mk_real(self.context, 0)
        formula = mk_lt(zero, x_const)
        prefix = [("Exists", x)]

        self.assertTrue(
            check_strategy(
                self.context,
                prefix,
                formula,
                {"assignments": {x: Fraction(1)}},
            )
        )
        self.assertFalse(
            check_strategy(
                self.context,
                prefix,
                formula,
                {"assignments": {x: Fraction(-1)}},
            )
        )

    def test_check_strategy_rejects_constant_move_under_universal_counterplay(self):
        """A fixed existential move must satisfy every remaining universal branch."""
        y = mk_symbol(self.context, "uy", Type.REAL)
        x = mk_symbol(self.context, "ex", Type.REAL)
        y_const = mk_const(y)
        x_const = mk_const(x)
        formula = mk_lt(y_const, x_const)
        prefix = [("Forall", y), ("Exists", x)]

        self.assertFalse(
            check_strategy(
                self.context,
                prefix,
                formula,
                {"assignments": {x: Fraction(0)}},
            )
        )

    def test_check_strategy_unknown_for_nonlinear_strategy_formula(self):
        """Unsupported nonlinear strategy validation stays explicitly unknown."""
        x = mk_symbol(self.context, "nsx", Type.REAL)
        x_const = mk_const(x)
        formula = mk_lt(mk_real(self.context, 0), mk_mul([x_const, x_const]))

        self.assertIsNone(
            check_strategy(
                self.context,
                [("Exists", x)],
                formula,
                {"assignments": {x: Fraction(2)}},
            )
        )


if __name__ == "__main__":
    unittest.main()
