"""Few-shot prompt template for abduction."""

from aria.ml.llm.abduct.data_structures import AbductionProblem


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
