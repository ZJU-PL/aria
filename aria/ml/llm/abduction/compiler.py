"""Compile natural language into an SMT-backed abduction instance.

This module intentionally depends only on a tiny LLM duck-typed interface
(`infer(message, is_measure_cost=False) -> (content, in_tokens, out_tokens)`).
That keeps the compiler usable even when optional LLM provider dependencies
are not installed.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Protocol, Tuple

from .data_structures import (
    CompiledAbductionProblem,
    CompilationResult,
    SmtVarDecl,
)
from .json_extract import extract_json_object, JsonExtractError
from .prompts import create_compile_prompt
from .smt import build_env, parse_bool_term, SmtParseError


class InputFormatError(ValueError):
    pass


_PREMISE_RE = re.compile(
    r"(?is)\bpremise\s*:\s*(.*?)\bconclusion\s*:\s*(.*)$"
)


def split_premise_conclusion(text: str) -> Tuple[str, str]:
    """Extract (premise_text, conclusion_text) from the raw input.

    Required format:
      Premise: ...\nConclusion: ...
    """

    m = _PREMISE_RE.search(text or "")
    if not m:
        raise InputFormatError(
            "Input must include 'Premise:' and 'Conclusion:' sections"
        )
    premise_text = m.group(1).strip()
    conclusion_text = m.group(2).strip()
    if not premise_text:
        raise InputFormatError("Premise section is empty")
    if not conclusion_text:
        raise InputFormatError("Conclusion section is empty")
    return premise_text, conclusion_text


class NLAbductionCompiler:
    """LLM compiler from text to (domain, premise, conclusion)."""

    def __init__(self, llm: "LLMClient", max_attempts: int = 3) -> None:
        self.llm = llm
        self.max_attempts = max_attempts

    def compile(self, text: str) -> CompilationResult:
        last_error: Optional[str] = None
        result = CompilationResult(attempts=0)

        try:
            premise_text, conclusion_text = split_premise_conclusion(text)
        except InputFormatError as e:
            result.error = str(e)
            result.attempts = 0
            return result

        for attempt in range(1, self.max_attempts + 1):
            prompt = create_compile_prompt(text, previous_error=last_error)
            result.prompt = prompt
            result.attempts = attempt

            llm_response, _, _ = self.llm.infer(prompt, True)
            result.llm_response = llm_response

            try:
                obj = extract_json_object(llm_response)
                compiled = self._compile_from_obj(
                    text=text,
                    premise_text=premise_text,
                    conclusion_text=conclusion_text,
                    obj=obj,
                )
                result.problem = compiled
                result.error = None
                return result
            except (JsonExtractError, SmtParseError, KeyError, TypeError, ValueError) as e:
                last_error = str(e)
                result.error = last_error
                continue

        return result

    def _compile_from_obj(
        self,
        text: str,
        premise_text: str,
        conclusion_text: str,
        obj: Dict[str, Any],
    ) -> CompiledAbductionProblem:
        obj_premise_text = obj.get("premise_text", "")
        obj_conclusion_text = obj.get("conclusion_text", "")
        if not isinstance(obj_premise_text, str) or not isinstance(
            obj_conclusion_text, str
        ):
            raise ValueError("premise_text and conclusion_text must be strings")
        if not obj_premise_text.strip() or not obj_conclusion_text.strip():
            raise ValueError("premise_text and conclusion_text are required")
        variables_raw = obj["variables"]
        if not isinstance(variables_raw, list):
            raise ValueError("'variables' must be a list")

        variables: List[SmtVarDecl] = []
        for item in variables_raw:
            if not isinstance(item, dict):
                raise ValueError("Each variables[] item must be an object")
            name = item.get("name", "")
            sort = item.get("sort", "")
            desc = item.get("description", "")
            if not isinstance(name, str) or not isinstance(sort, str):
                raise ValueError("Variable name/sort must be strings")
            if desc is None:
                desc = ""
            if not isinstance(desc, str):
                raise ValueError("Variable description must be a string")
            variables.append(SmtVarDecl(name=name.strip(), sort=sort.strip(), description=desc))

        glossary = obj.get("glossary", {})
        if glossary is None:
            glossary = {}
        if not isinstance(glossary, dict):
            raise ValueError("'glossary' must be an object")
        glossary_str: Dict[str, str] = {}
        for k, v in glossary.items():
            if isinstance(k, str) and isinstance(v, str):
                glossary_str[k] = v

        env = build_env(variables)
        domain = parse_bool_term(obj["domain"], env)
        premise = parse_bool_term(obj["premise"], env)
        conclusion = parse_bool_term(obj["conclusion"], env)

        return CompiledAbductionProblem(
            text=text,
            premise_text=premise_text,
            conclusion_text=conclusion_text,
            variables=variables,
            domain_constraints=domain,
            premise=premise,
            conclusion=conclusion,
            glossary=glossary_str,
        )


class LLMClient(Protocol):
    def infer(self, message: str, is_measure_cost: bool = False) -> Tuple[str, int, int]:
        ...
