"""Feedback-based prompt template for abduction."""

from typing import List, Dict, Any
from aria.ml.llm.abduct.data_structures import (
    AbductionProblem,
    AbductionIterationResult,
)


def create_feedback_prompt(
    problem: AbductionProblem,
    previous_iterations: List[AbductionIterationResult],
    last_counterexample: Dict[str, Any],
) -> str:
    """Build a feedback-augmented prompt for iterative abduction."""
    last_iteration = previous_iterations[-1]
    ce_formatted = "\n".join(
        [f"{var} = {value}" for var, value in last_counterexample.items()]
    )

    if not last_iteration.is_consistent:
        issue = "inconsistent with the premise"
    else:
        issue = "doesn't imply the conclusion"

    history = ""
    for i, result in enumerate(previous_iterations[:-1]):
        ce_str = ""
        if result.counterexample:
            ce_str = ", ".join(
                [f"{var}={val}" for var, val in result.counterexample.items()]
            )
        history += (
            f"Attempt {i+1}: {result.hypothesis} "
            f"(Consistent: {result.is_consistent}, "
            f"Sufficient: {result.is_sufficient})\n"
        )
        if ce_str:
            history += f"Counterexample: {ce_str}\n"

    return f"""Problem:
```
{problem.to_smt2_string()}
```

Goal: Find ψ such that (premise ∧ ψ) is satisfiable and implies: \
{problem.conclusion}

Your previous attempt: {last_iteration.hypothesis}
Issue: Your hypothesis is {issue}.

Counterexample:
{ce_formatted}

Previous attempts:
{history}

Provide ONLY the revised SMT-LIB2 formula for ψ."""
