"""Unit tests for NL abduction pipeline (no network calls)."""

from __future__ import annotations

from typing import Tuple

from aria.tests import TestCase

from aria.ml.llm.abduction import NLAbductionCompiler, NLAbductor


class _FakeLLM:
    def __init__(self, responses):
        self._responses = list(responses)
        self._i = 0

    def infer(self, message: str, is_measure_cost: bool = False) -> Tuple[str, int, int]:
        if self._i >= len(self._responses):
            raise RuntimeError("No more fake responses")
        r = self._responses[self._i]
        self._i += 1
        return r, 0, 0


class TestNLAbduction(TestCase):
    def test_compile_success(self):
        text = """Premise: x and y are positive integers.
Conclusion: x+y>10 and x>5."""
        llm = _FakeLLM(
            [
                """```json
{
  \"premise_text\": \"x and y are positive integers.\",
  \"conclusion_text\": \"x+y>10 and x>5.\",
  \"variables\": [
    {\"name\": \"x\", \"sort\": \"Int\", \"description\": \"Alice apples\"},
    {\"name\": \"y\", \"sort\": \"Int\", \"description\": \"Bob apples\"}
  ],
  \"glossary\": {\"x\": \"Alice apples\", \"y\": \"Bob apples\"},
  \"domain\": \"(and (>= x 0) (>= y 0))\",
  \"premise\": \"(and (> x 0) (> y 0))\",
  \"conclusion\": \"(and (> (+ x y) 10) (> x 5))\"
}
```"""
            ]
        )
        c = NLAbductionCompiler(llm=llm, max_attempts=1)
        res = c.compile(text)
        self.assertIsNotNone(res.problem)
        self.assertIsNone(res.error)
        self.assertEqual(len(res.problem.variables), 2)

    def test_abduce_with_feedback_finds_solution(self):
        text = """Premise: x and y are positive integers.
Conclusion: x+y>10 and x>5."""
        # 1) compiler json
        # 2) hypothesis attempt 1: psi_smt = [] (insufficient)
        # 3) hypothesis attempt 2: psi_smt = [(> x 5)] (still insufficient)
        # 4) hypothesis attempt 3: psi_smt includes both constraints (valid)
        llm = _FakeLLM(
            [
                """{
  \"premise_text\": \"x and y are positive integers.\",
  \"conclusion_text\": \"x+y>10 and x>5.\",
  \"variables\": [
    {\"name\": \"x\", \"sort\": \"Int\", \"description\": \"Alice\"},
    {\"name\": \"y\", \"sort\": \"Int\", \"description\": \"Bob\"}
  ],
  \"glossary\": {},
  \"domain\": \"(and (>= x 0) (>= y 0))\",
  \"premise\": \"(and (> x 0) (> y 0))\",
  \"conclusion\": \"(and (> (+ x y) 10) (> x 5))\"
}""",
                """{\"psi_smt\": [], \"psi_nl\": []}""",
                """{\"psi_smt\": [\"(> x 5)\"], \"psi_nl\": []}""",
                """{\"psi_smt\": [\"(> x 5)\", \"(> (+ x y) 10)\"], \"psi_nl\": []}""",
            ]
        )
        abd = NLAbductor(llm=llm, max_iterations=5)
        res = abd.abduce(text)
        self.assertTrue(res.is_valid)
        self.assertIsNotNone(res.hypothesis)
        self.assertGreaterEqual(len(res.iterations), 1)

    def test_exchange_adds_lemmas_for_nl_constraints(self):
        text = """Premise: x is a positive integer.
Conclusion: x is greater than 10."""

        # Hypothesis has an NL-only constraint that rules out the SMT counterexample.
        # The verifier exchange returns an SMT lemma that encodes it.
        llm = _FakeLLM(
            [
                """{
  \"premise_text\": \"x is a positive integer.\",
  \"conclusion_text\": \"x is greater than 10.\",
  \"variables\": [
    {\"name\": \"x\", \"sort\": \"Int\", \"description\": \"value\"}
  ],
  \"glossary\": {},
  \"domain\": \"(>= x 0)\",
  \"premise\": \"(> x 0)\",
  \"conclusion\": \"(> x 10)\"
}""",
                """{
  \"psi_smt\": [],
  \"psi_nl\": [\"x is at least 11\"]
}""",
                """{
  \"verdict\": \"reject\",
  \"lemmas_smt\": [\"(>= x 11)\"],
  \"note\": \"NL constraint rules out x=1\"
}""",
            ]
        )

        abd = NLAbductor(llm=llm, max_iterations=2, max_exchange_rounds=2)
        res = abd.abduce(text)
        self.assertTrue(res.is_valid)
        self.assertIsNotNone(res.hypothesis)
