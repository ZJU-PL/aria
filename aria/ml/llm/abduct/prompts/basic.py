"""Basic prompt template for abduction."""

from aria.ml.llm.abduct.data_structures import AbductionProblem


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
