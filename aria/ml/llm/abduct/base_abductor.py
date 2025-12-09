"""Basic LLM-based abductor implementation."""

import time
from aria.ml.llm.llmtool.LLM_utils import LLM
from .data_structures import AbductionProblem, AbductionResult
from .validation import validate_hypothesis
from .prompts import create_basic_prompt
from .utils import extract_smt_from_llm_response, parse_smt2_string


class LLMAbductor:
    """Generates abductive hypotheses and validates them with Z3."""

    def __init__(self, llm: LLM, max_attempts: int = 3, temperature: float = 0.7):
        self.llm = llm
        self.max_attempts = max_attempts
        self.temperature = temperature

    def _invoke_llm(self, prompt: str) -> str:
        response, _, _ = self.llm.infer(prompt, True)
        return response or ""

    def abduce(self, problem: AbductionProblem) -> AbductionResult:
        """Generate an abductive hypothesis for the given problem."""
        start_time = time.time()
        prompt = create_basic_prompt(problem)
        llm_response = ""

        for attempt in range(self.max_attempts):
            try:
                llm_response = self._invoke_llm(prompt)
                smt_string = extract_smt_from_llm_response(llm_response)
                if not smt_string:
                    continue

                hypothesis = parse_smt2_string(smt_string, problem)
                if hypothesis is None:
                    continue

                is_consistent, is_sufficient = validate_hypothesis(problem, hypothesis)
                result = AbductionResult(
                    problem=problem,
                    hypothesis=hypothesis,
                    is_consistent=is_consistent,
                    is_sufficient=is_sufficient,
                    llm_response=llm_response,
                    prompt=prompt,
                    execution_time=time.time() - start_time
                )

                if is_consistent and is_sufficient:
                    return result

            except Exception as e:
                return AbductionResult(
                    problem=problem,
                    error=str(e),
                    llm_response=llm_response,
                    prompt=prompt,
                    execution_time=time.time() - start_time
                )

        return (result if 'result' in locals() else
                AbductionResult(
                    problem=problem,
                    error="Failed to generate a valid hypothesis after multiple attempts",
                    llm_response=llm_response,
                    prompt=prompt,
                    execution_time=time.time() - start_time
                ))
