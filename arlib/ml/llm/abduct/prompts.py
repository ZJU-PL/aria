"""Prompt generation for LLM-based abduction.
1. basic
2. feedback-guided
3. chain-of-thought (CoT)
4. few-shot
"""

from typing import List, Dict, Any
from arlib.ml.llm.abduct.data_structures import AbductionProblem, AbductionIterationResult


def create_basic_prompt(problem: AbductionProblem) -> str:
    """Build a basic instruction prompt for generating abductive hypotheses."""
    return f"""You are an expert in logical abduction and SMT.

Problem in SMT-LIB2 format:
```
{problem.to_smt2_string()}
```

Variables: {', '.join([str(var) for var in problem.variables])}

Find hypothesis ψ such that:
1. (premise ∧ ψ) is satisfiable
2. (premise ∧ ψ) implies: {problem.conclusion}

Provide ONLY the SMT-LIB2 formula for ψ. Examples:
(assert (formula))
or
(formula)

NO explanations, NO declare-const statements."""


def create_feedback_prompt(
    problem: AbductionProblem,
    previous_iterations: List[AbductionIterationResult],
    last_counterexample: Dict[str, Any]
) -> str:
    """Build a feedback-augmented prompt for iterative abduction."""
    last_iteration = previous_iterations[-1]
    ce_formatted = "\n".join([f"{var} = {value}" for var, value in last_counterexample.items()])

    issue = ("inconsistent with the premise" if not last_iteration.is_consistent
             else "doesn't imply the conclusion")

    history = ""
    for i, result in enumerate(previous_iterations[:-1]):
        ce_str = ", ".join([f"{var}={val}" for var, val in result.counterexample.items()]) if result.counterexample else ""
        history += f"Attempt {i+1}: {result.hypothesis} (Consistent: {result.is_consistent}, Sufficient: {result.is_sufficient})\n"
        if ce_str:
            history += f"Counterexample: {ce_str}\n"

    return f"""Problem:
```
{problem.to_smt2_string()}
```

Goal: Find ψ such that (premise ∧ ψ) is satisfiable and implies: {problem.conclusion}

Your previous attempt: {last_iteration.hypothesis}
Issue: Your hypothesis is {issue}.

Counterexample:
{ce_formatted}

Previous attempts:
{history}

Provide ONLY the revised SMT-LIB2 formula for ψ."""


def create_cot_prompt(problem: AbductionProblem) -> str:
    """Build a Chain of Thought prompt that guides step-by-step reasoning."""
    return f"""You are an expert in logical abduction and SMT. Solve this step by step.

Problem in SMT-LIB2 format:
```
{problem.to_smt2_string()}
```

Variables: {', '.join([str(var) for var in problem.variables])}

Step-by-step reasoning:
1. First, analyze the premise: What constraints does it impose on the variables?
2. Next, examine the conclusion: What must be true for the conclusion to hold?
3. Then, identify the gap: What additional constraints are needed to bridge premise and conclusion?
4. Finally, formulate hypothesis ψ: What formula would make (premise ∧ ψ) imply the conclusion?

Think through each step carefully:

Step 1 - Premise analysis:
[Analyze the premise constraints here]

Step 2 - Conclusion analysis:
[Analyze what the conclusion requires here]

Step 3 - Gap identification:
[Identify what's missing to connect premise to conclusion]

Step 4 - Hypothesis formulation:
[Formulate the hypothesis ψ]

Provide ONLY the final SMT-LIB2 formula for ψ. Examples:
(assert (formula))
or
(formula)

NO explanations, NO declare-const statements."""


def create_few_shot_prompt(problem: AbductionProblem) -> str:
    """Build a few-shot prompt with concrete examples."""
    return f"""You are an expert in logical abduction and SMT. Here are some examples:

Example 1:
Problem: Find ψ such that (x > 0 ∧ ψ) implies (x > 1)
Variables: [x]
Solution: ψ = (x > 1)

Example 2:
Problem: Find ψ such that (x + y = 5 ∧ ψ) implies (x = 3)
Variables: [x, y]
Solution: ψ = (y = 2)

Example 3:
Problem: Find ψ such that (a ∧ b ∧ ψ) implies (a ∨ c)
Variables: [a, b, c]
Solution: ψ = (c)

Now solve this problem:

Problem in SMT-LIB2 format:
```
{problem.to_smt2_string()}
```

Variables: {', '.join([str(var) for var in problem.variables])}

Find hypothesis ψ such that:
1. (premise ∧ ψ) is satisfiable
2. (premise ∧ ψ) implies: {problem.conclusion}

Provide ONLY the SMT-LIB2 formula for ψ. Examples:
(assert (formula))
or
(formula)

NO explanations, NO declare-const statements."""
