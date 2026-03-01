"""Prompt for compiling natural language into SMT-backed abduction problems."""

from __future__ import annotations

from typing import Optional


def create_compile_prompt(text: str, previous_error: Optional[str] = None) -> str:
    """Ask the LLM to compile NL into a structured JSON + SMT terms."""

    err = ""
    if previous_error:
        err = (
            "\nPrevious attempt failed with this error. Fix the JSON/SMT and try again:\n"
            "{0}\n".format(previous_error)
        )

    # Use an f-string so we can include literal braces in the JSON schema.
    return f"""You are a formal methods expert. Convert the natural-language problem into a precise SMT-backed abduction instance.

The input uses this required structure:
Premise: <text>
Conclusion: <text>

Output MUST be a single JSON object. No extra keys beyond those documented. No surrounding prose.

JSON schema (all fields required):
{{
  \"premise_text\": \"<copy the premise from the input>\", 
  \"conclusion_text\": \"<copy the conclusion from the input>\",
  \"variables\": [ {{\"name\": \"x\", \"sort\": \"Int\", \"description\": \"...\"}}, ... ],
  \"glossary\": {{ \"x\": \"...\", \"some_predicate\": \"...\" }},
  \"domain\": \"<SMT Bool term>\",
  \"premise\": \"<SMT Bool term>\",
  \"conclusion\": \"<SMT Bool term>\"
}}

Rules for SMT terms:
- Provide a Bool term only, e.g. '(and (> x 0) (< y 10))'.
- Do NOT include (assert ...) wrappers. Do NOT include declare-const.
- Use only the declared variables.
- Keep it in quantifier-free logic when possible.

Natural-language problem:
---
{text.strip()}
---
{err}"""
