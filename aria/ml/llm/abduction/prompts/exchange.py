"""Prompts for SMT/LLM information exchange during verification."""

from __future__ import annotations

from typing import Any, Dict, List

from ..data_structures import CompiledAbductionProblem


def create_counterexample_exchange_prompt(
    problem: CompiledAbductionProblem,
    psi_nl: List[str],
    counterexample: Dict[str, Any],
) -> str:
    """Ask LLM whether the SMT counterexample is compatible with NL constraints.

    If the counterexample violates NL-only constraints, the LLM should propose SMT
    lemmas that rule it out (a Nelson-Oppen-like exchange).
    """

    ce_lines: List[str] = []
    for k, v in counterexample.items():
        ce_lines.append("{0} = {1}".format(k, v))
    ce_block = "\n".join(ce_lines) if ce_lines else "(no assignment)"

    nl_block = "\n".join(["- {0}".format(x) for x in psi_nl]) if psi_nl else "(none)"

    return (
        "You are coordinating a hybrid verifier: SMT + natural language.\n\n"
        "We have an SMT counterexample assignment that satisfies domain, premise, and the SMT part of the hypothesis, but violates the SMT conclusion.\n"
        "Your job: decide if this counterexample is a REAL counterexample when also considering the NL-only constraints.\n\n"
        "Premise (text):\n---\n{0}\n---\n\n"
        "Conclusion (text):\n---\n{1}\n---\n\n"
        "NL-only hypothesis constraints:\n{2}\n\n"
        "Counterexample assignment:\n{3}\n\n"
        "Output MUST be a single JSON object with exactly these keys:\n"
        "{{\n"
        '  \"verdict\": \"accept\" | \"reject\",\n'
        '  \"lemmas_smt\": [\"<SMT Bool term>\", ...],\n'
        '  \"note\": \"short reason\"\n'
        "}}\n\n"
        "Rules:\n"
        "- If verdict=accept: counterexample is compatible with NL constraints; lemmas_smt MUST be [].\n"
        "- If verdict=reject: counterexample violates NL constraints; provide 1-3 SMT Bool terms over declared variables that would rule it out.\n"
        "- SMT terms must be quantifier-free if possible; no declare-const; no assert.\n"
        "- No extra prose outside JSON.\n"
    ).format(problem.premise_text, problem.conclusion_text, nl_block, ce_block)
