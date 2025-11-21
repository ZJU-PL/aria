"""Rule-based auto-formalisation for translating natural language into SMT-LIB.

The goal is not to be exhaustive, but to provide a compact and deterministic
baseline that can cover common sentence patterns without calling an LLM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Pattern


@dataclass
class NL2SMTArtifacts:
    """Artifacts produced during the translation pipeline."""

    declarations: Dict[str, str] = field(default_factory=dict)
    assertions: List[str] = field(default_factory=list)
    unparsed: List[str] = field(default_factory=list)


class NL2SMTConverter:
    """Lightweight rule-based converter from natural language to SMT-LIB."""

    _DECLARATION_PATTERNS: Iterable[tuple[Pattern[str], str]] = (
        (re.compile(r"^(?:let\s+)?(?P<var>\w+)\s+is\s+an?\s+integer$", re.IGNORECASE), "Int"),
        (re.compile(r"^(?:let\s+)?(?P<var>\w+)\s+is\s+an?\s+natural\s+number$", re.IGNORECASE), "Int"),
        (re.compile(r"^(?:let\s+)?(?P<var>\w+)\s+is\s+an?\s+real(?:\s+number)?$", re.IGNORECASE), "Real"),
        (re.compile(r"^(?:let\s+)?(?P<var>\w+)\s+is\s+(?:a\s+)?boolean$", re.IGNORECASE), "Bool"),
    )

    def __init__(self, logic: str = "QF_LIA") -> None:
        self.logic = logic

    def convert(self, description: str) -> str:
        """Translate a short English description into an SMT-LIB script."""
        artifacts = self._analyse(description)
        return self._render(artifacts)

    # --------------------------------------------------------------------- #
    # Internal helpers

    def _analyse(self, description: str) -> NL2SMTArtifacts:
        artifacts = NL2SMTArtifacts()
        for clause in self._tokenise(description):
            if self._try_declaration(clause, artifacts):
                continue

            if self._try_assertion(clause, artifacts):
                continue

            artifacts.unparsed.append(clause)
        return artifacts

    def _tokenise(self, description: str) -> List[str]:
        raw_parts = re.split(r"[.;\n]+", description.strip())
        clauses: List[str] = []
        for part in raw_parts:
            part = part.strip()
            if not part:
                continue
            # Opportunistically break on "and" when it likely separates clauses.
            lower_part = part.lower()
            if " and " in lower_part:
                subparts = re.split(r"\band\b", part, flags=re.IGNORECASE)
                clauses.extend(sub.strip(" ,") for sub in subparts if sub.strip(" ,"))
            else:
                clauses.append(part)
        return clauses

    def _try_declaration(self, clause: str, artifacts: NL2SMTArtifacts) -> bool:
        for pattern, smt_type in self._DECLARATION_PATTERNS:
            match = pattern.match(clause)
            if match:
                var = match.group("var")
                artifacts.declarations[var] = smt_type
                return True
        return False

    def _try_assertion(self, clause: str, artifacts: NL2SMTArtifacts) -> bool:
        builders = (
            self._match_equality_literal,
            self._match_equality_binary_op,
            self._match_comparison,
        )
        for builder in builders:
            assertion = builder(clause)
            if assertion:
                artifacts.assertions.append(assertion)
                return True
        return False

    def _match_equality_literal(self, clause: str) -> Optional[str]:
        patterns = (
            re.compile(r"^(?P<lhs>\w+)\s+(?:equals|is\s+equal\s+to)\s+(?P<rhs>-?\d+)$", re.IGNORECASE),
            re.compile(r"^(?P<lhs>\w+)\s+(?:equals|is\s+equal\s+to)\s+(?P<rhs>\w+)$", re.IGNORECASE),
        )
        for pattern in patterns:
            match = pattern.match(clause)
            if match:
                lhs, rhs = match.group("lhs"), match.group("rhs")
                return f"(= {lhs} {rhs})"
        return None

    def _match_equality_binary_op(self, clause: str) -> Optional[str]:
        add_pattern = re.compile(
            r"^(?P<lhs>\w+)\s+(?:equals|is\s+equal\s+to)\s+(?P<rhs>\w+)\s+plus\s+(?P<term>-?\d+)$",
            re.IGNORECASE,
        )
        sub_pattern = re.compile(
            r"^(?P<lhs>\w+)\s+(?:equals|is\s+equal\s+to)\s+(?P<rhs>\w+)\s+minus\s+(?P<term>-?\d+)$",
            re.IGNORECASE,
        )
        match = add_pattern.match(clause)
        if match:
            lhs, rhs, term = match.group("lhs", "rhs", "term")
            return f"(= {lhs} (+ {rhs} {term}))"
        match = sub_pattern.match(clause)
        if match:
            lhs, rhs, term = match.group("lhs", "rhs", "term")
            return f"(= {lhs} (- {rhs} {term}))"
        return None

    def _match_comparison(self, clause: str) -> Optional[str]:
        comparisons = (
            (re.compile(r"^(?P<lhs>\w+)\s+is\s+greater\s+than\s+(?P<rhs>\w+)$", re.IGNORECASE), ">"),
            (
                re.compile(
                    r"^(?P<lhs>\w+)\s+is\s+greater\s+than\s+or\s+equal\s+to\s+(?P<rhs>\w+)$",
                    re.IGNORECASE,
                ),
                ">=",
            ),
            (re.compile(r"^(?P<lhs>\w+)\s+is\s+less\s+than\s+(?P<rhs>\w+)$", re.IGNORECASE), "<"),
            (
                re.compile(
                    r"^(?P<lhs>\w+)\s+is\s+less\s+than\s+or\s+equal\s+to\s+(?P<rhs>\w+)$",
                    re.IGNORECASE,
                ),
                "<=",
            ),
        )
        for pattern, operator in comparisons:
            match = pattern.match(clause)
            if match:
                lhs, rhs = match.group("lhs", "rhs")
                return f"({operator} {lhs} {rhs})"
        return None

    def _render(self, artifacts: NL2SMTArtifacts) -> str:
        lines = [
            "; Auto-generated SMT-LIB script (rule-based NL → SMT)",
            f"(set-logic {self.logic})",
        ]
        for var, smt_type in sorted(artifacts.declarations.items()):
            lines.append(f"(declare-const {var} {smt_type})")
        for assertion in artifacts.assertions:
            lines.append(f"(assert {assertion})")
        if artifacts.unparsed:
            lines.append("; Unparsed clauses retained as comments:")
            lines.extend(f";   {clause}" for clause in artifacts.unparsed)
        lines.append("(check-sat)")
        lines.append("(get-model)")
        return "\n".join(lines)


def nl_to_smt(description: str, logic: str = "QF_LIA") -> str:
    """Convenience function mirroring :meth:`NL2SMTConverter.convert`."""

    converter = NL2SMTConverter(logic=logic)
    return converter.convert(description)
