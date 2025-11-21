"""Feedback-based LLM abductor implementation."""

import time
from arlib.ml.llm.llmtool.LLM_utils import LLM
from .data_structures import AbductionProblem, AbductionIterationResult, FeedbackAbductionResult
from .base_abductor import LLMAbductor
from .validation import validate_hypothesis, generate_counterexample
from .prompts import create_basic_prompt, create_feedback_prompt
from .utils import extract_smt_from_llm_response, parse_smt2_string


class FeedbackLLMAbductor(LLMAbductor):
    """Uses LLMs to generate abductive hypotheses with SMT-based feedback."""

    def __init__(self, llm: LLM, max_iterations: int = 5, temperature: float = 0.7):
        super().__init__(llm=llm, max_attempts=1, temperature=temperature)
        self.max_iterations = max_iterations

    def abduce_with_feedback(self, problem: AbductionProblem) -> FeedbackAbductionResult:
        """Iterative abduction with feedback from counterexamples."""
        start_time = time.time()
        iteration_results = []
        result = FeedbackAbductionResult(problem=problem, prompt=create_basic_prompt(problem))
        current_prompt = result.prompt

        for iteration in range(self.max_iterations):
            try:
                llm_response = self._invoke_llm(current_prompt)
                smt_string = extract_smt_from_llm_response(llm_response)
                if not smt_string:
                    continue

                hypothesis = parse_smt2_string(smt_string, problem)
                if hypothesis is None:
                    continue

                is_consistent, is_sufficient = validate_hypothesis(problem, hypothesis)
                iter_result = AbductionIterationResult(
                    hypothesis=hypothesis,
                    is_consistent=is_consistent,
                    is_sufficient=is_sufficient,
                    is_valid=is_consistent and is_sufficient,
                    llm_response=llm_response,
                    prompt=current_prompt,
                    iteration=iteration + 1
                )

                if is_consistent and is_sufficient:
                    iteration_results.append(iter_result)
                    break

                counterexample = generate_counterexample(problem, hypothesis)
                iter_result.counterexample = counterexample
                iteration_results.append(iter_result)

                if counterexample is None:
                    break

                current_prompt = create_feedback_prompt(problem, iteration_results, counterexample)

            except Exception as e:
                result.error = str(e)
                break

        for iter_result in iteration_results:
            result.add_iteration(iter_result)

        if iteration_results:
            last_iter = iteration_results[-1]
            result.hypothesis = last_iter.hypothesis
            result.is_consistent = last_iter.is_consistent
            result.is_sufficient = last_iter.is_sufficient
            result.is_valid = last_iter.is_valid
            result.llm_response = last_iter.llm_response

        result.execution_time = time.time() - start_time
        return result

    def abduce(self, problem: AbductionProblem) -> FeedbackAbductionResult:
        """Override the parent abduce method to use feedback-based abduction."""
        return self.abduce_with_feedback(problem)
