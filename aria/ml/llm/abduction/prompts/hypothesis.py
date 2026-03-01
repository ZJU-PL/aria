"""Prompt for generating abductive hypotheses (psi)."""

from __future__ import annotations

from typing import List

from ..data_structures import CompiledAbductionProblem


def _vars_block(problem: CompiledAbductionProblem) -> str:
    parts: List[str] = []
    for v in problem.variables:
        desc = v.description or problem.glossary.get(v.name, "")
        if desc:
            parts.append("- {0}: {1}  ({2})".format(v.name, v.sort, desc))
        else:
            parts.append("- {0}: {1}".format(v.name, v.sort))
    return "\n".join(parts)


def create_hypothesis_prompt(problem: CompiledAbductionProblem) -> str:
    """Ask the LLM for a hypothesis psi to complete the explanation."""

    return (
        "You are doing abductive reasoning with an SMT solver as verifier.\n\n"
        "Natural-language input is REQUIRED to include an explicit Premise and Conclusion.\n\n"
        "Premise (text):\n"
        "---\n"
        "{0}\n"
        "---\n\n"
        "Conclusion (text):\n"
        "---\n"
        "{1}\n"
        "---\n\n"
        "Variables:\n"
        "{2}\n\n"
        "SMT meaning:\n"
        "- domain: {3}\n"
        "- premise: {4}\n"
        "- conclusion: {5}\n\n"
        "Task: propose a hypothesis psi such that:\n"
        "1) (domain AND premise AND psi_smt) is satisfiable\n"
        "2) (domain AND premise AND psi_smt) implies conclusion\n"
        "You MAY also include NL-only constraints that are not SMT-encodable.\n\n"
        "Output MUST be a single JSON object with exactly these keys:\n"
        "{{\n"
        '  \"psi_smt\": [\"<SMT Bool term>\", ...],\n'
        '  \"psi_nl\": [\"<natural language constraint>\", ...]\n'
        "}}\n\n"
        "Rules:\n"
        "- Put everything you can encode into psi_smt.\n"
        "- Use psi_nl only for constraints you cannot encode.\n"
        "- SMT terms are Bool terms only (no declare-const, no assert).\n"
        "- No extra prose outside JSON.\n"
    ).format(
        problem.premise_text,
        problem.conclusion_text,
        _vars_block(problem),
        problem.domain_constraints.sexpr(),
        problem.premise.sexpr(),
        problem.conclusion.sexpr(),
    )
