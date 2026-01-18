"""
LLM-backed trigger suggestion for E-matching.

The default mode chooses trigger combinations from a pre-computed list of
safe candidates. The model is asked to return JSON like:
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
        direct_terms: bool = False,
    ) -> None:
        self.verbose = verbose
        self.max_groups = max_groups
        self.direct_terms = direct_terms
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

    def build_direct_prompt(
        self,
        quantifier: z3.QuantifierRef,
        bound_var_names: Sequence[str],
        allowed_symbols: Sequence[str],
    ) -> str:
        """
        Build a prompt that asks for SMT-LIB terms directly.
        """
        body_text = quantifier.body().sexpr()
        symbol_lines = ", ".join(sorted(set(allowed_symbols)))
        return (
            "You choose E-matching triggers for a quantified SMT formula.\n"
            f"Pick at most {self.max_groups} trigger groups. Each group is a list "
            "of SMT-LIB terms. Use JSON: "
            '{{"triggers": [["(f x)", "(g x y)"], ["(h y)"]]}}.\n'
            "Rules:\n"
            "1) Cover every bound variable across the chosen groups "
            f"({', '.join(bound_var_names)}).\n"
            "2) Use only symbols already present in the formula body or bound "
            "variables; do not invent new symbols.\n"
            "3) Do not include quantifiers in the trigger terms.\n"
            "4) Keep groups small (1-2 terms) unless a larger multi-pattern is "
            "needed to cover all variables.\n"
            f"Quantifier body:\n{body_text}\n"
            f"Allowed symbols:\n{symbol_lines}\n"
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

    def suggest_direct_trigger_groups(
        self,
        quantifier: z3.QuantifierRef,
        bound_vars: Sequence[z3.ExprRef],
    ) -> List[List[z3.ExprRef]]:
        """
        Ask the LLM to synthesize trigger terms directly. Returns empty on failure.
        """
        if self.llm is None:
            self._debug("LLM backend not configured; skipping direct trigger selection.")
            return []

        bound_var_names = [str(var) for var in bound_vars]
        decls = self._collect_decls(quantifier.body(), bound_vars)
        allowed_symbols = self._collect_allowed_symbols(quantifier.body(), bound_var_names)
        prompt = self.build_direct_prompt(quantifier, bound_var_names, allowed_symbols)
        response, _, _ = self.llm.infer(prompt)  # type: ignore[attr-defined]
        self._debug(f"LLM raw response: {response}")

        term_groups = self._parse_direct_response(response)
        if not term_groups:
            return []

        parsed_groups: List[List[z3.ExprRef]] = []
        seen: set[str] = set()
        for group in term_groups[: self.max_groups]:
            exprs: List[z3.ExprRef] = []
            for term in group:
                expr = self._parse_term(term, decls)
                if expr is None:
                    exprs = []
                    break
                if self._contains_quantifier(expr):
                    exprs = []
                    break
                if not self._term_uses_allowed_symbols(expr, allowed_symbols):
                    exprs = []
                    break
                if not _collect_var_names(expr, set(bound_var_names)):
                    exprs = []
                    break
                exprs.append(expr)
            if not exprs:
                continue
            key = "|".join(sorted(expr.sexpr() for expr in exprs))
            if key in seen:
                continue
            seen.add(key)
            parsed_groups.append(exprs)

        if not parsed_groups:
            return []
        if not self._covers_all_bound_vars(parsed_groups, bound_var_names):
            self._debug("LLM direct suggestion rejected: missing bound variable coverage.")
            return []
        return parsed_groups

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

    def _parse_direct_response(self, response: str) -> List[List[str]]:
        """
        Parse the LLM output for direct SMT-LIB terms.
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

        groups: List[List[str]] = []
        for entry in payload:
            if isinstance(entry, str):
                entry = [entry]
            if not isinstance(entry, list):
                continue
            cleaned: List[str] = []
            for term in entry:
                if isinstance(term, str) and term.strip():
                    cleaned.append(term.strip())
            if cleaned:
                groups.append(cleaned)
        return groups

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

    def _collect_decls(
        self, expr: z3.ExprRef, bound_vars: Sequence[z3.ExprRef]
    ) -> dict[str, object]:
        decls: dict[str, object] = {str(var): var for var in bound_vars}

        def visit(node: z3.ExprRef) -> None:
            if z3.is_quantifier(node):
                return
            if z3.is_app(node):
                decl = node.decl()
                if decl.kind() == z3.Z3_OP_UNINTERPRETED:
                    if node.num_args() == 0:
                        decls.setdefault(decl.name(), node)
                    else:
                        decls.setdefault(decl.name(), decl)
            for child in node.children():
                visit(child)

        visit(expr)
        return decls

    def _collect_allowed_symbols(
        self, expr: z3.ExprRef, bound_var_names: Sequence[str]
    ) -> List[str]:
        symbols: set[str] = set(bound_var_names)

        def visit(node: z3.ExprRef) -> None:
            if z3.is_quantifier(node):
                return
            if z3.is_app(node):
                symbols.add(node.decl().name())
            for child in node.children():
                visit(child)

        visit(expr)
        return sorted(symbols)

    def _parse_term(self, term: str, decls: dict[str, object]) -> Optional[z3.ExprRef]:
        term_text = term.strip()
        if not term_text:
            return None
        payload = f"(assert (= {term_text} {term_text}))"
        try:
            parsed = z3.parse_smt2_string(payload, decls=decls)
        except z3.Z3Exception:
            return None
        if not parsed:
            return None
        expr = parsed[0]
        if not z3.is_app(expr):
            return None
        if expr.decl().kind() != z3.Z3_OP_EQ or expr.num_args() != 2:
            return None
        return expr.arg(0)

    def _contains_quantifier(self, expr: z3.ExprRef) -> bool:
        worklist = [expr]
        while worklist:
            current = worklist.pop()
            if z3.is_quantifier(current):
                return True
            worklist.extend(current.children())
        return False

    def _term_uses_allowed_symbols(
        self, expr: z3.ExprRef, allowed_symbols: Sequence[str]
    ) -> bool:
        allowed = set(allowed_symbols)
        seen: set[str] = set()
        worklist = [expr]
        while worklist:
            current = worklist.pop()
            if z3.is_app(current):
                seen.add(current.decl().name())
            if z3.is_const(current) and current.decl().kind() == z3.Z3_OP_UNINTERPRETED:
                seen.add(str(current))
            worklist.extend(current.children())
        return seen.issubset(allowed)


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
