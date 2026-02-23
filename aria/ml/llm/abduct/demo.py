"""Demo script for LLM-based abduction.

Run with: python -m aria.ml.llm.abduct.demo
"""

import asyncio
import z3

from aria.llmtools.providers.cli.opencode import chat_opencode
from aria.ml.llm.abduct import (
    AbductionProblem,
    validate_hypothesis,
    create_basic_prompt,
)
from aria.ml.llm.abduct.parsing import extract_smt_from_llm_response, parse_smt2_string
from aria.ml.llm.abduct.validation import generate_counterexample
from aria.ml.llm.abduct.prompts.feedback import create_feedback_prompt


async def llm_infer(model: str, message: str) -> str:
    """Call LLM and return response content."""
    messages = [
        {
            "role": "system",
            "content": "You are an expert in formal logic and SMT solving. Respond only with SMT-LIB2 formulas.",
        },
        {"role": "user", "content": message},
    ]
    response = await chat_opencode(model=model, messages=messages, max_tokens=1024)
    return response.content or ""


async def demo_abduction(model: str = "glm-5-free"):
    """Demonstrate abduction with LLM and iterative feedback."""
    x, y = z3.Ints("x y")
    problem = AbductionProblem(
        premise=z3.And(x > 0, y > 0),
        conclusion=z3.And(x + y > 10, x > 5),
    )

    print("=" * 60)
    print(f"LLM-Based Abduction Demo (model: {model})")
    print("=" * 60 + "\n")

    print(f"Premise: {problem.premise}")
    print(f"Conclusion: {problem.conclusion}\n")

    prompt = create_basic_prompt(problem)
    iteration_results = []

    for iteration in range(3):
        print(f"--- Iteration {iteration + 1} ---")
        try:
            response = await llm_infer(model, prompt)
            print(f"LLM Response: {response[:200]}...")

            smt_string = extract_smt_from_llm_response(response)
            hypothesis = parse_smt2_string(smt_string, problem)

            is_consistent, is_sufficient = validate_hypothesis(problem, hypothesis)
            print(f"Hypothesis: {hypothesis}")
            print(f"Consistent: {is_consistent}, Sufficient: {is_sufficient}")

            result = type(
                "Result",
                (),
                {"hypothesis": hypothesis, "is_consistent": is_consistent, "is_sufficient": is_sufficient, "counterexample": None},
            )()
            iteration_results.append(result)

            if is_consistent and is_sufficient:
                print("\nFound valid hypothesis!")
                break

            counterexample = generate_counterexample(problem, hypothesis)
            result.counterexample = counterexample
            if counterexample:
                print(f"Counterexample: {counterexample}")
                prompt = create_feedback_prompt(problem, iteration_results, counterexample)

        except Exception as e:
            print(f"Error: {e}")
            break
        print()

    print(f"\nTotal iterations: {len(iteration_results)}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(demo_abduction())
