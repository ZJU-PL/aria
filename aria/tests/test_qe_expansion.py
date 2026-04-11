"""Regression tests for Shannon-expansion Boolean QE."""

from itertools import product

from pysat.formula import CNF

from aria.quant.qe.qe_expansion import QuantifierElimination
from aria.tests import TestCase, main


class TestExpansionQuantifierElimination(TestCase):
    @staticmethod
    def _evaluate_clause(clause, assignment):
        return any(
            assignment[abs(lit)] if lit > 0 else not assignment[abs(lit)]
            for lit in clause
        )

    @classmethod
    def _evaluate_cnf(cls, formula, assignment):
        return all(
            cls._evaluate_clause(clause, assignment) for clause in formula.clauses
        )

    @classmethod
    def _expected_projection(cls, formula, free_vars, quantified_blocks):
        expected = {}

        for free_values in product([False, True], repeat=len(free_vars)):
            free_assignment = dict(zip(free_vars, free_values))
            expected[free_values] = cls._eval_quantified_formula(
                formula, free_assignment, quantified_blocks, 0
            )

        return expected

    @classmethod
    def _eval_quantified_formula(cls, formula, free_assignment, quantified_blocks, block_idx):
        if block_idx == len(quantified_blocks):
            return cls._evaluate_cnf(formula, free_assignment)

        quantifier, block = quantified_blocks[block_idx]
        choices = product([False, True], repeat=len(block))

        if quantifier == "exists":
            return any(
                cls._eval_quantified_formula(
                    formula,
                    {
                        **free_assignment,
                        **dict(zip(block, block_values)),
                    },
                    quantified_blocks,
                    block_idx + 1,
                )
                for block_values in choices
            )

        return all(
            cls._eval_quantified_formula(
                formula,
                {
                    **free_assignment,
                    **dict(zip(block, block_values)),
                },
                quantified_blocks,
                block_idx + 1,
            )
            for block_values in choices
        )

    @classmethod
    def _assert_semantically_equivalent(cls, formula, projected, free_vars, blocks):
        expected = cls._expected_projection(formula, free_vars, blocks)

        for free_values in product([False, True], repeat=len(free_vars)):
            assignment = dict(zip(free_vars, free_values))
            actual = cls._evaluate_cnf(projected, assignment)
            if actual != expected[free_values]:
                raise AssertionError(
                    f"projection mismatch for assignment {assignment}: "
                    f"expected {expected[free_values]}, got {actual}"
                )

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

    def test_universal_projection_collects_both_cofactors(self):
        qe = QuantifierElimination()
        x = qe.get_var_id("x")
        y = qe.get_var_id("y")
        z = qe.get_var_id("z")

        formula = CNF(from_clauses=[[x, y], [-x, z]])
        result = qe.eliminate_quantifiers(formula, ["x"], quantifier="forall")

        self.assertEqual(result.clauses, [[y], [z]])

    def test_mixed_quantifier_blocks_are_eliminated_in_order(self):
        qe = QuantifierElimination()
        x = qe.get_var_id("x")
        y = qe.get_var_id("y")
        z = qe.get_var_id("z")

        formula = CNF(from_clauses=[[x, y], [-x, z], [-y, z]])
        result = qe.eliminate_quantifier_blocks(
            formula, [("exists", "x"), ("forall", "y")]
        )

        self.assertEqual(result.clauses, [[z]])

    def test_grouped_quantifier_blocks_are_supported(self):
        qe = QuantifierElimination()
        x = qe.get_var_id("x")
        y = qe.get_var_id("y")
        z = qe.get_var_id("z")
        w = qe.get_var_id("w")

        formula = CNF(from_clauses=[[x, z], [y, z], [-x, -y, w], [-z, w]])
        result = qe.eliminate_quantifier_blocks(
            formula, [("exists", ["x", "y"]), ("forall", ["z"])]
        )

        self._assert_semantically_equivalent(
            formula,
            result,
            free_vars=[w],
            blocks=[("exists", [x, y]), ("forall", [z])],
        )

    def test_existential_projection_matches_truth_table(self):
        qe = QuantifierElimination()
        x = qe.get_var_id("x")
        y = qe.get_var_id("y")
        z = qe.get_var_id("z")

        formula = CNF(from_clauses=[[x, y], [-x, z], [-y, z]])
        result = qe.eliminate_quantifiers(formula, ["x"], quantifier="exists")

        self._assert_semantically_equivalent(
            formula,
            result,
            free_vars=[y, z],
            blocks=[("exists", [x])],
        )

    def test_alternating_grouped_projection_matches_truth_table(self):
        qe = QuantifierElimination()
        x = qe.get_var_id("x")
        y = qe.get_var_id("y")
        z = qe.get_var_id("z")
        w = qe.get_var_id("w")

        formula = CNF(
            from_clauses=[[x, z], [-x, y], [-y, w], [y, -z], [-z, w]]
        )
        blocks = [("exists", [x, y]), ("forall", [z])]
        result = qe.eliminate_quantifier_blocks(formula, blocks)

        self._assert_semantically_equivalent(
            formula,
            result,
            free_vars=[w],
            blocks=[("exists", [x, y]), ("forall", [z])],
        )

    def test_subsumed_clauses_are_removed_after_universal_elimination(self):
        qe = QuantifierElimination()
        x = qe.get_var_id("x")
        y = qe.get_var_id("y")

        formula = CNF(from_clauses=[[x, y], [y], [-x, y]])
        result = qe.eliminate_quantifiers(formula, [x], quantifier="forall")

        self.assertEqual(result.clauses, [[y]])


if __name__ == "__main__":
    main()
