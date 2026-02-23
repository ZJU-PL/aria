"""Chain-of-Thought prompt template for abduction."""

from aria.ml.llm.abduct.data_structures import AbductionProblem


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
