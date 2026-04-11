"""Regression tests for Shannon-expansion Boolean QE."""

from pysat.formula import CNF

from aria.quant.qe.qe_expansion import QuantifierElimination
from aria.tests import TestCase, main


class TestExpansionQuantifierElimination(TestCase):
    def test_unsatisfiable_formula_remains_false(self):
        qe = QuantifierElimination()
        x = qe.get_var_id("x")

        formula = CNF(from_clauses=[[x], [-x]])
        result = qe.eliminate_quantifiers(formula, ["x"])

        self.assertEqual(result.clauses, [[]])
        self.assertFalse(qe.is_satisfiable(result))

    def test_existential_projection_keeps_remaining_clause(self):
        qe = QuantifierElimination()
        x = qe.get_var_id("x")
        y = qe.get_var_id("y")

        formula = CNF(from_clauses=[[x], [y]])
        result = qe.eliminate_quantifiers(formula, ["x"])

        self.assertEqual(result.clauses, [[y]])

    def test_tautological_resolvent_is_dropped(self):
        qe = QuantifierElimination()
        x = qe.get_var_id("x")
        y = qe.get_var_id("y")

        formula = CNF(from_clauses=[[x, y], [-x, -y]])
        result = qe.eliminate_quantifiers(formula, ["x"])

        self.assertEqual(result.clauses, [])
        self.assertTrue(qe.is_satisfiable(result))


if __name__ == "__main__":
    main()
