# coding: utf-8
"""
Tests for the QF_DT solver.
"""

import z3

from aria.tests import TestCase, main
from aria.smt.adt import QFDTSolver
from aria.utils import SolverResult


class TestQFDT(TestCase):
    """
    Test the algebraic datatype solver wrapper.
    """

    def test_enum_smt_string_sat(self):
        solver = QFDTSolver()
        formula = """
        (set-logic QF_DT)
        (declare-datatypes () ((Color red green blue)))
        (declare-const c Color)
        (assert (not (= c red)))
        """

        result = solver.solve_smt_string(formula)
        self.assertEqual(result, SolverResult.SAT)
        self.assertIsNotNone(solver.model)

    def test_enum_smt_string_unsat(self):
        solver = QFDTSolver()
        formula = """
        (set-logic QF_DT)
        (declare-datatypes () ((Color red green)))
        (declare-const c Color)
        (assert (= c red))
        (assert (= c green))
        """

        result = solver.solve_smt_string(formula)
        self.assertEqual(result, SolverResult.UNSAT)

    def test_recursive_datatype_unsat(self):
        list_dt = z3.Datatype("IntList")
        list_dt.declare("nil")
        list_dt.declare("cons", ("head", z3.IntSort()), ("tail", list_dt))
        int_list = list_dt.create()

        xs = z3.Const("xs", int_list)
        formula = z3.And(int_list.is_cons(xs), int_list.tail(xs) == xs)

        solver = QFDTSolver()
        result = solver.solve_smt_formula(formula)
        self.assertEqual(result, SolverResult.UNSAT)

    def test_constructor_and_selector_sat(self):
        tree_dt = z3.Datatype("Tree")
        tree_dt.declare("leaf", ("value", z3.IntSort()))
        tree_dt.declare("node", ("left", tree_dt), ("right", tree_dt))
        tree = tree_dt.create()

        leaf = tree.leaf(z3.IntVal(7))
        t = z3.Const("t", tree)
        formula = z3.And(t == leaf, tree.is_leaf(t), tree.value(t) == 7)

        solver = QFDTSolver()
        result = solver.solve_smt_formula(formula)
        self.assertEqual(result, SolverResult.SAT)
        self.assertIsNotNone(solver.model)

    def test_unrecognized_logic_falls_back_to_generic_solver(self):
        color_dt = z3.Datatype("ColorFallback")
        color_dt.declare("red")
        color_dt.declare("green")
        color = color_dt.create()

        chooser = z3.Function("chooser", color, color)
        c = z3.Const("c", color)
        formula = z3.And(chooser(color.red) == color.green, c == chooser(color.red))

        solver = QFDTSolver(logic="QF_UFDT")
        result = solver.solve_smt_formula(formula)
        self.assertEqual(result, SolverResult.SAT)
        self.assertIsNotNone(solver.model)


if __name__ == "__main__":
    main()
