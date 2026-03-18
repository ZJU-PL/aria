"""Tests for the typed PBE solver stack."""

import time

from aria.synthesis.pbe import PBESolver, PBETask, SMTPBESolver, Theory
from aria.tests import TestCase, main


class TestPBESolver(TestCase):
    """Regression tests for programming-by-example synthesis."""

    def test_synthesizes_lia_addition(self):
        """The solver should recover a simple arithmetic relation."""
        examples = [
            {"x": 1, "y": 2, "output": 3},
            {"x": 3, "y": 4, "output": 7},
            {"x": 5, "y": 1, "output": 6},
        ]

        result = PBESolver(max_expression_depth=2, max_candidates=200).synthesize(
            examples
        )

        self.assertTrue(result.success)
        self.assertIsNotNone(result.expression)
        self.assertEqual(str(result.expression), "(x + y)")
        self.assertEqual(result.statistics["consistent_expressions"], 1)

    def test_synthesizes_string_length_with_integer_output(self):
        """String-domain tasks can now target integer outputs."""
        examples = [
            {"s": "hello", "output": 5},
            {"s": "hi", "output": 2},
            {"s": "", "output": 0},
        ]

        result = PBESolver(max_expression_depth=2, max_candidates=200).synthesize(
            examples
        )

        self.assertTrue(result.success)
        self.assertIsNotNone(result.expression)
        self.assertEqual(str(result.expression), "len(s)")
        self.assertEqual(result.statistics["output_type"], "int")

    def test_uses_example_literals_for_constants(self):
        """Observed literals from examples should be available to the DSL."""
        examples = [
            {"input": 10, "output": 15},
            {"input": 20, "output": 25},
            {"input": 5, "output": 10},
            {"input": 0, "output": 5},
        ]

        result = PBESolver(max_expression_depth=2, max_candidates=200).synthesize(
            examples
        )

        self.assertTrue(result.success)
        self.assertIsNotNone(result.expression)
        self.assertEqual(result.expression.evaluate({"input": 7}), 12)

    def test_rejects_malformed_examples(self):
        """Malformed example sets should fail cleanly."""
        result = PBESolver().synthesize([{"x": 1}, {"x": 2, "output": 2}])

        self.assertFalse(result.success)
        self.assertIn("missing an 'output' field", result.message)

    def test_supports_bitvectors_with_theory_hint(self):
        """Bitvector tasks require an explicit theory hint for integer inputs."""
        examples = [
            {"x": 0b1010, "y": 0b1100, "output": 0b1000},
            {"x": 0b1111, "y": 0b0011, "output": 0b0011},
            {"x": 0b0000, "y": 0b1111, "output": 0b0000},
        ]

        result = PBESolver(
            max_expression_depth=2,
            theory_hint=Theory.BV,
            max_candidates=200,
        ).synthesize(examples)

        self.assertTrue(result.success)
        self.assertEqual(str(result.expression), "(x & y)")

    def test_infers_typed_tasks_for_mixed_inputs(self):
        """Task inference should preserve per-variable sorts."""
        examples = [
            {"flag": True, "x": 10, "output": 10},
            {"flag": False, "x": 10, "output": 0},
        ]

        task = PBETask.from_examples(examples)

        self.assertEqual(task.theory, Theory.LIA)
        self.assertEqual(task.statistics()["inputs"], {"flag": "bool", "x": "int"})
        self.assertEqual(task.candidate_values("flag"), [True, False])

    def test_supports_boolean_guarded_lia_synthesis(self):
        """Boolean inputs should participate directly in conditional synthesis."""
        examples = [
            {"flag": True, "x": 10, "output": 10},
            {"flag": False, "x": 10, "output": 0},
            {"flag": True, "x": 3, "output": 3},
        ]

        result = PBESolver(max_expression_depth=2, max_candidates=400).synthesize(
            examples
        )

        self.assertTrue(result.success)
        self.assertEqual(str(result.expression), "(if flag then x else 0)")
        self.assertEqual(
            result.statistics["inputs"], {"flag": "bool", "x": "int"}
        )

    def test_supports_string_synthesis_with_integer_indices(self):
        """String-domain synthesis should allow auxiliary integer inputs."""
        examples = [
            {"s": "abcd", "i": 1, "output": "b"},
            {"s": "wxyz", "i": 2, "output": "y"},
            {"s": "hello", "i": 4, "output": "o"},
        ]

        result = PBESolver(max_expression_depth=2, max_candidates=400).synthesize(
            examples
        )

        self.assertTrue(result.success)
        self.assertEqual(str(result.expression), "str_substring(s, i, 1)")
        self.assertEqual(result.expression.evaluate({"s": "logic", "i": 0}), "l")

    def test_one_shot_synthesis_reports_ambiguity(self):
        """One-shot synthesis should surface semantically distinct alternatives."""
        examples = [
            {"x": 1, "output": 1},
            {"x": 2, "output": 2},
        ]

        result = PBESolver(max_expression_depth=2, max_candidates=200).synthesize(
            examples
        )

        self.assertTrue(result.success)
        self.assertTrue(result.is_ambiguous)
        self.assertEqual(str(result.expression), "x")
        self.assertGreater(len(result.version_space), 1)
        self.assertTrue(any(item["x"] < 0 for item in result.distinguishing_inputs))

    def test_oracle_guided_refinement_resolves_ambiguity(self):
        """Heuristic counterexamples should drive the solver to a unique program."""
        examples = [
            {"x": 1, "output": 1},
            {"x": 2, "output": 2},
        ]

        result = PBESolver(
            max_expression_depth=2,
            max_candidates=200,
        ).synthesize_with_oracle(examples, lambda assignment: abs(assignment["x"]))

        self.assertTrue(result.success)
        self.assertFalse(result.is_ambiguous)
        self.assertEqual(str(result.expression), "abs(x)")
        self.assertGreater(result.statistics["initial_consistent_expressions"], 1)
        self.assertEqual(result.statistics["final_consistent_expressions"], 1)
        self.assertEqual(result.statistics["refinement_rounds"], 1)
        self.assertEqual(result.statistics["oracle_calls"], 1)
        self.assertTrue(result.statistics["ambiguity_resolved"])

    def test_smt_oracle_guided_refinement_prefers_smt_counterexamples(self):
        """SMTPBESolver should consult SMT for refinement counterexamples first."""
        examples = [
            {"x": 1, "output": 1},
            {"x": 2, "output": 2},
        ]

        solver = SMTPBESolver(max_expression_depth=2, max_candidates=200)
        calls = {"count": 0}
        original = solver.smt_verifier.find_counterexample

        def wrapped(*args, **kwargs):
            calls["count"] += 1
            return original(*args, **kwargs)

        solver.smt_verifier.find_counterexample = wrapped

        result = solver.synthesize_with_oracle(
            examples,
            lambda assignment: abs(assignment["x"]),
        )

        self.assertTrue(result.success)
        self.assertFalse(result.is_ambiguous)
        self.assertEqual(str(result.expression), "abs(x)")
        self.assertGreater(calls["count"], 0)
        self.assertEqual(result.statistics["oracle_calls"], 1)

    def test_oracle_refinement_rejects_incompatible_outputs(self):
        """Oracle outputs must respect the declared task output type."""
        examples = [
            {"x": 1, "output": 1},
            {"x": 2, "output": 2},
        ]

        result = PBESolver(
            max_expression_depth=2,
            max_candidates=200,
        ).synthesize_with_oracle(examples, lambda assignment: f"value:{assignment['x']}")

        self.assertFalse(result.success)
        self.assertIn("Oracle returned", result.message)
        self.assertEqual(result.statistics["oracle_calls"], 1)
        self.assertEqual(result.statistics["refinement_rounds"], 0)

    def test_oracle_refinement_reports_exceptions(self):
        """Oracle exceptions should fail the refinement run cleanly."""
        examples = [
            {"x": 1, "output": 1},
            {"x": 2, "output": 2},
        ]

        def failing_oracle(assignment):
            raise RuntimeError(f"bad input {assignment['x']}")

        result = PBESolver(
            max_expression_depth=2,
            max_candidates=200,
        ).synthesize_with_oracle(examples, failing_oracle)

        self.assertFalse(result.success)
        self.assertIn("Oracle failed", result.message)
        self.assertEqual(result.statistics["oracle_calls"], 1)
        self.assertEqual(result.statistics["refinement_rounds"], 0)

    def test_oracle_refinement_respects_round_limit(self):
        """Refinement should stop cleanly when the round budget is zero."""
        examples = [
            {"x": 1, "output": 1},
            {"x": 2, "output": 2},
        ]

        result = PBESolver(
            max_expression_depth=2,
            max_candidates=200,
        ).synthesize_with_oracle(
            examples,
            lambda assignment: abs(assignment["x"]),
            max_refinement_rounds=0,
        )

        self.assertTrue(result.success)
        self.assertTrue(result.is_ambiguous)
        self.assertEqual(result.statistics["refinement_rounds"], 0)
        self.assertEqual(result.statistics["oracle_calls"], 0)
        self.assertIn("round limit", result.message)

    def test_oracle_refinement_respects_timeout_budget(self):
        """The refinement loop should share the solver's overall timeout budget."""
        examples = [
            {"x": 1, "output": 1},
            {"x": 2, "output": 2},
        ]

        def slow_oracle(assignment):
            time.sleep(2.5)
            return abs(assignment["x"])

        result = PBESolver(
            max_expression_depth=2,
            max_candidates=200,
            timeout=2.0,
        ).synthesize_with_oracle(examples, slow_oracle)

        self.assertTrue(result.success)
        self.assertTrue(result.is_ambiguous)
        self.assertEqual(result.statistics["oracle_calls"], 1)
        self.assertEqual(result.statistics["refinement_rounds"], 1)
        self.assertIn("timeout budget expired", result.message)

    def test_smt_verification_matches_example_consistency(self):
        """SMT verification should agree with the concrete evaluator."""
        examples = [
            {"x": 1, "y": 2, "output": 3},
            {"x": 3, "y": 4, "output": 7},
            {"x": 5, "y": 1, "output": 6},
        ]

        solver = SMTPBESolver(max_expression_depth=2, max_candidates=200)
        result = solver.synthesize(examples)

        self.assertTrue(result.success)
        self.assertIsNotNone(result.expression)
        self.assertTrue(solver.verify_with_smt(result.expression, examples))


if __name__ == "__main__":
    main()
