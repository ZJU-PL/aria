"""
Tests for the CHC (Constrained Horn Clauses) module.
"""

import unittest
from aria.srk.syntax import Context, Symbol, Type
from aria.srk.chc import CHCClause, CHCSystem, CHCSolver


class TestCHCClause(unittest.TestCase):
    """Test CHC clause functionality."""

    def setUp(self):
        self.context = Context()

    def test_simple_clause(self):
        """Test creating a simple CHC clause."""
        x = self.context.mk_symbol("x", Type.INT)
        y = self.context.mk_symbol("y", Type.INT)

        # Create a simple clause: x > 0 ∧ y > x ⇒ y > 0
        from aria.srk.syntax import Lt, Var, Const, Symbol
        zero_symbol = Symbol(0, "0", Type.INT)  # Create a symbol for constant 0
        premise1 = Lt(Var(x, Type.INT), Const(zero_symbol))
        premise2 = Lt(Var(y, Type.INT), Var(x, Type.INT))
        conclusion = Lt(Var(y, Type.INT), Const(zero_symbol))

        clause = CHCClause([premise1, premise2], conclusion)

        self.assertEqual(len(clause.premises), 2)
        self.assertEqual(clause.conclusion, conclusion)

    def test_clause_without_premises(self):
        """Test clause with no premises (just a conclusion)."""
        x = self.context.mk_symbol("x", Type.INT)

        from aria.srk.syntax import Lt, Var, Const, Symbol
        zero_symbol = Symbol(0, "0", Type.INT)
        conclusion = Lt(Var(x, Type.INT), Const(zero_symbol))
        clause = CHCClause([], conclusion)

        self.assertEqual(len(clause.premises), 0)
        self.assertEqual(clause.conclusion, conclusion)


class TestCHCSystem(unittest.TestCase):
    """Test CHC system functionality."""

    def setUp(self):
        self.context = Context()

    def test_empty_system(self):
        """Test creating an empty CHC system."""
        system = CHCSystem([])
        self.assertEqual(len(system.clauses), 0)

    def test_system_with_clauses(self):
        """Test CHC system with clauses."""
        x = self.context.mk_symbol("x", Type.INT)

        from aria.srk.syntax import Lt, Var, Const, Symbol
        zero_symbol = Symbol(0, "0", Type.INT)
        one_symbol = Symbol(1, "1", Type.INT)
        clause1 = CHCClause([], Lt(Var(x, Type.INT), Const(zero_symbol)))
        clause2 = CHCClause([Lt(Var(x, Type.INT), Const(zero_symbol))], Lt(Var(x, Type.INT), Const(one_symbol)))

        system = CHCSystem([clause1, clause2])
        self.assertEqual(len(system.clauses), 2)


class TestCHCSolver(unittest.TestCase):
    """Test CHC solver functionality."""

    def setUp(self):
        self.context = Context()

    def test_solver_creation(self):
        """Test creating a CHC solver."""
        solver = CHCSolver(self.context)
        self.assertIsNotNone(solver)

    def test_solve_simple_system(self):
        """Test solving a simple CHC system."""
        solver = CHCSolver(self.context)

        # Create a simple system
        x = self.context.mk_symbol("x", Type.INT)
        from aria.srk.syntax import Lt, Var, Const, Symbol
        zero_symbol = Symbol(0, "0", Type.INT)
        clause = CHCClause([], Lt(Var(x, Type.INT), Const(zero_symbol)))
        system = CHCSystem([clause])

        # This should be solvable (conclusion is not necessarily true)
        result = solver.solve(system)
        self.assertIsNotNone(result)


if __name__ == '__main__':
    unittest.main()
