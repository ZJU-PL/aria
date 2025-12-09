"""
Prompts for LLM-based Craig interpolant generation.

1. Basics
2. Few-Shot
3. CoT
"""
from typing import List
import z3
from z3.z3util import get_vars


def _common_symbols(asA: List[z3.ExprRef], asB: List[z3.ExprRef]) -> List[str]:
    get_names = lambda forms: {v.decl().name() for e in forms for v in get_vars(e)}
    return sorted(get_names(asA) & get_names(asB))


def mk_interpolant_prompt(A: List[z3.ExprRef], B: List[z3.ExprRef]) -> str:
    """Generate a basic prompt for LLM to create a Craig interpolant between sets A and B."""
    A_text = "\n".join(f"(assert {e.sexpr()})" for e in A)
    B_text = "\n".join(f"(assert {e.sexpr()})" for e in B)
    commons = ", ".join(_common_symbols(A, B))
    return f"""Generate a Craig interpolant I between sets A and B.
Requirements: A⟹I valid, I∧B unsat, use only shared symbols: {commons}
Return ONLY the S-expression, no explanations.

A: (set-logic ALL)\n{A_text}
B: (set-logic ALL)\n{B_text}"""


def mk_interpolant_cot_prompt(A: List[z3.ExprRef], B: List[z3.ExprRef]) -> str:
    """Generate a Chain-of-Thought prompt for LLM to create a Craig interpolant.
    TODO: for CoT, shoud we just add an additional sentence 'think step-by-step' or guide the steps such as the ones in the current prompt?
    """
    A_text = "\n".join(f"(assert {e.sexpr()})" for e in A)
    B_text = "\n".join(f"(assert {e.sexpr()})" for e in B)
    commons = ", ".join(_common_symbols(A, B))
    return f"""Generate a Craig interpolant I between sets A and B using step-by-step reasoning.

Requirements:
- A⟹I must be valid (A implies I)
- I∧B must be unsat (I and B are inconsistent)
- I can only use symbols that appear in both A and B: {commons}

Step-by-step approach:
1. Analyze what A asserts about the shared variables
2. Analyze what B asserts about the shared variables
3. Find a formula I that captures the "gap" between A and B
4. Verify that A⟹I and I∧B is unsat
5. Express I using only shared symbols

A: (set-logic ALL)\n{A_text}
B: (set-logic ALL)\n{B_text}

Reasoning:
Step 1: A asserts: [analyze A's constraints on shared variables]
Step 2: B asserts: [analyze B's constraints on shared variables]
Step 3: The interpolant should: [explain the logical relationship]
Step 4: Verification: [check the requirements]
Step 5: Interpolant: [provide the S-expression]"""


def mk_interpolant_fewshot_prompt(A: List[z3.ExprRef], B: List[z3.ExprRef]) -> str:
    """Generate a few-shot prompt with examples for Craig interpolant generation."""
    A_text = "\n".join(f"(assert {e.sexpr()})" for e in A)
    B_text = "\n".join(f"(assert {e.sexpr()})" for e in B)
    commons = ", ".join(_common_symbols(A, B))

    return f"""Generate a Craig interpolant I between sets A and B.

Requirements: A⟹I valid, I∧B unsat, use only shared symbols: {commons}

Examples:

Example 1:
A: (assert (> x 5))
B: (assert (< x 3))
Shared symbols: x
Interpolant: (<= x 5)
Reasoning: A says x > 5, B says x < 3. The interpolant x ≤ 5 is implied by A and contradicts B.

Example 2:
A: (assert (and (> x 0) (< x 10)))
B: (assert (or (< x 0) (> x 5)))
Shared symbols: x
Interpolant: (<= x 5)
Reasoning: A says 0 < x < 10, B says x < 0 or x > 5. The interpolant x ≤ 5 captures the overlap.

Now solve:
A: (set-logic ALL)\n{A_text}
B: (set-logic ALL)\n{B_text}
Interpolant:"""


def mk_interpolant_structured_prompt(A: List[z3.ExprRef], B: List[z3.ExprRef]) -> str:
    """Generate a structured prompt with detailed analysis sections.
    TODO: What is the point of this prompt?... add a few related papers...
    """
    A_text = "\n".join(f"(assert {e.sexpr()})" for e in A)
    B_text = "\n".join(f"(assert {e.sexpr()})" for e in B)
    commons = ", ".join(_common_symbols(A, B))

    return f"""# Craig Interpolant Generation

## Problem Statement
Find an interpolant I between formula sets A and B such that:
- A ⟹ I (A implies I)
- I ∧ B is unsatisfiable (I and B are inconsistent)
- I uses only symbols common to both A and B

## Shared Symbols
{commons}

## Formula Set A
```smt2
(set-logic ALL)
{A_text}
```

## Formula Set B
```smt2
(set-logic ALL)
{B_text}
```

## Analysis Framework

### Step 1: Variable Analysis
- Variables in A: [list variables in A]
- Variables in B: [list variables in B]
- Shared variables: {commons}

### Step 2: Constraint Analysis
- What does A constrain about shared variables?
- What does B constrain about shared variables?
- What is the logical relationship between these constraints?

### Step 3: Interpolant Design
- What formula I would bridge the gap between A and B?
- Does I follow from A?
- Does I contradict B?
- Does I use only shared symbols?

### Step 4: Verification
- Check: A ⟹ I
- Check: I ∧ B is unsat
- Confirm: I uses only shared symbols

## Solution
Provide the interpolant as an SMT-LIB expression:"""


def mk_interpolant_prompt_with_type(A: List[z3.ExprRef], B: List[z3.ExprRef], prompt_type: str = "basic") -> str:
    """Generate interpolant prompt with specified type.

    Args:
        A: List of Z3 expressions for formula set A
        B: List of Z3 expressions for formula set B
        prompt_type: Type of prompt ("basic", "cot", "fewshot", "structured")

    Returns:
        Formatted prompt string
    """
    if prompt_type == "cot":
        return mk_interpolant_cot_prompt(A, B)
    elif prompt_type == "fewshot":
        return mk_interpolant_fewshot_prompt(A, B)
    elif prompt_type == "structured":
        return mk_interpolant_structured_prompt(A, B)
    else:  # default to basic
        return mk_interpolant_prompt(A, B)


def get_available_prompt_types() -> List[str]:
    """Get list of available prompt types."""
    return ["basic", "cot", "fewshot", "structured"]


def get_prompt_description(prompt_type: str) -> str:
    """Get description of a specific prompt type."""
    descriptions = {
        "basic": "Simple, direct prompt asking for interpolant with minimal instructions",
        "cot": "Chain-of-Thought prompt with step-by-step reasoning framework",
        "fewshot": "Few-shot learning prompt with concrete examples and solutions",
        "structured": "Highly structured prompt with detailed analysis sections and markdown formatting"
    }
    return descriptions.get(prompt_type, "Unknown prompt type")
