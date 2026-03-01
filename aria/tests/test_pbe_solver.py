"""Tests for the typed PBE solver stack."""

from aria.synthesis.pbe import PBESolver, SMTPBESolver, Theory
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
