"""
LLM-backed trigger suggestion for E-matching.

The LLM never synthesizes raw SMT terms. Instead, it chooses trigger
combinations from a pre-computed list of safe candidates. The model is
asked to return JSON like:
{"triggers": [[0, 2], [1]]}
where numbers index into the candidate list provided in the prompt.
"""

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import z3

try:
    from aria.ml.llm.llmtool.LLM_utils import LLM  # type: ignore
    from aria.ml.llm.llmtool.logger import Logger  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    LLM = None  # type: ignore
    Logger = None  # type: ignore


@dataclass
class TriggerCandidate:
    """
    A potential trigger term extracted from a quantifier.

    Attributes:
        expr: The Z3 expression for the trigger term, using canonical bound vars.
        text: A human-readable rendering of the term.
        variables: Names of bound variables that appear in the term.
    """

    expr: z3.ExprRef
    text: str
    variables: Sequence[str]


class LLMTriggerGenerator:
    """
    Use an LLM to pick trigger combinations from candidate terms.

    The generator is intentionally conservative: it only returns triggers built
    from the provided candidates, and it enforces that all bound variables are
    covered across the selected trigger groups.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        llm: Optional[object] = None,
        logger: Optional[object] = None,
        verbose: bool = False,
        max_groups: int = 3,
    ) -> None:
        self.verbose = verbose
        self.max_groups = max_groups
        self.logger = logger or (Logger(self._default_log_path()) if Logger else None)

        # Allow dependency injection for testing
        if llm is not None:
            self.llm = llm
        elif LLM and Logger:
            # Lazily initialize the shared LLM wrapper
            self.llm = LLM(
                online_model_name=model,
                logger=self.logger,
                temperature=temperature,
                system_role="You are an expert at E-matching trigger selection.",
            )
        else:  # pragma: no cover - exercised when LLM deps are missing
            self.llm = None

    @staticmethod
    def _default_log_path() -> str:
        return str(Path(tempfile.gettempdir()) / "aria_llm_trigger.log")

    def _debug(self, message: str) -> None:
        if self.verbose:
            print(f"[llm-trigger] {message}")  # noqa: T201

    def build_prompt(
        self,
        quantifier: z3.QuantifierRef,
        candidates: Sequence[TriggerCandidate],
        bound_var_names: Sequence[str],
    ) -> str:
        """
        Build a structured prompt that asks the LLM to pick trigger groups.
        """
        candidate_lines = "\n".join(
            f"{idx}: {cand.text}    vars={','.join(cand.variables)}"
            for idx, cand in enumerate(candidates)
        )
        body_text = quantifier.body().sexpr()
        return (
            "You choose E-matching triggers for a quantified SMT formula.\n"
            f"Pick at most {self.max_groups} trigger groups. Each group is a list "
            "of candidate ids. Use JSON: {{\"triggers\": [[id1, id2], [id3]]}}.\n"
            "Rules:\n"
            "1) Cover every bound variable across the chosen groups "
            f"({', '.join(bound_var_names)}).\n"
            "2) Prefer terms that mention more bound variables and deeper "
            "function structure. Avoid arithmetic or boolean connective nodes.\n"
            "3) Do not invent new terms; only use the ids shown below.\n"
            "4) Keep groups small (1-2 terms) unless a larger multi-pattern is "
            "needed to cover all variables.\n"
            f"Quantifier body:\n{body_text}\n"
            f"Candidates:\n{candidate_lines}\n"
            "Return only JSON. If no good trigger exists, return "
            '{{"triggers": []}}.'
        )

    def suggest_trigger_groups(
        self,
        quantifier: z3.QuantifierRef,
        candidates: Sequence[TriggerCandidate],
        bound_var_names: Sequence[str],
    ) -> List[List[z3.ExprRef]]:
        """
        Ask the LLM to pick trigger groups. Returns empty on failure.
        """
        if not candidates:
            return []
        if self.llm is None:
            self._debug("LLM backend not configured; skipping LLM trigger selection.")
            return []

        prompt = self.build_prompt(quantifier, candidates, bound_var_names)
        response, _, _ = self.llm.infer(prompt)  # type: ignore[attr-defined]
        self._debug(f"LLM raw response: {response}")

        index_groups = self._parse_response(response, len(candidates))
        index_groups = index_groups[: self.max_groups]
        trigger_groups = self._indexes_to_triggers(index_groups, candidates)

        # Ensure coverage; otherwise drop the suggestion.
        if not self._covers_all_bound_vars(trigger_groups, bound_var_names):
            self._debug("LLM suggestion rejected: missing bound variable coverage.")
            return []
        return trigger_groups

    def _indexes_to_triggers(
        self, index_groups: Sequence[Sequence[int]], candidates: Sequence[TriggerCandidate]
    ) -> List[List[z3.ExprRef]]:
        seen: set[str] = set()
        result: List[List[z3.ExprRef]] = []
        for group in index_groups:
            exprs: List[z3.ExprRef] = []
            for idx in group:
                if 0 <= idx < len(candidates):
                    exprs.append(candidates[idx].expr)
            if not exprs:
                continue
            key = "|".join(sorted(candidates[idx].text for idx in group if 0 <= idx < len(candidates)))
            if key in seen:
                continue
            seen.add(key)
            result.append(exprs)
        return result

    def _parse_response(
        self, response: str, num_candidates: int
    ) -> List[List[int]]:
        """
        Parse the LLM output and return index groups.
        """
        if not response:
            return []
        json_blob = self._extract_json_blob(response)
        if json_blob is None:
            return []
        try:
            data = json.loads(json_blob)
        except json.JSONDecodeError:
            return []

        payload = data.get("triggers") or data.get("patterns")
        if not isinstance(payload, list):
            return []

        index_groups: List[List[int]] = []
        for entry in payload:
            if isinstance(entry, int):
                entry = [entry]
            if not isinstance(entry, list):
                continue
            cleaned: List[int] = []
            for idx in entry:
                if isinstance(idx, int) and 0 <= idx < num_candidates:
                    cleaned.append(idx)
            if cleaned:
                index_groups.append(cleaned)
        return index_groups

    def _extract_json_blob(self, text: str) -> Optional[str]:
        """
        Extract the first JSON object or array from the text.
        """
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            return brace_match.group(0)
        list_match = re.search(r"\[.*\]", text, re.DOTALL)
        if list_match:
            return f'{{"triggers": {list_match.group(0)}}}'
        return None

    @staticmethod
    def _covers_all_bound_vars(
        trigger_groups: Iterable[Sequence[z3.ExprRef]], bound_var_names: Sequence[str]
    ) -> bool:
        needed = set(bound_var_names)
        if not needed:
            return True
        present: set[str] = set()
        for group in trigger_groups:
            for expr in group:
                present.update(_collect_var_names(expr, needed))
        return needed.issubset(present)


def _collect_var_names(expr: z3.ExprRef, whitelist: Optional[set[str]] = None) -> set[str]:
    """
    Collect variable names that appear as uninterpreted constants inside expr.
    """
    names: set[str] = set()
    worklist = [expr]
    while worklist:
        current = worklist.pop()
        if z3.is_const(current) and current.decl().kind() == z3.Z3_OP_UNINTERPRETED:
            name = str(current)
            if whitelist is None or name in whitelist:
                names.add(name)
        worklist.extend(list(current.children()))
    return names
