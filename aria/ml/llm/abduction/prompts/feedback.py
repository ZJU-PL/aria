"""Prompt for counterexample-driven hypothesis refinement."""

from __future__ import annotations

from typing import Any, Dict

from ..data_structures import AbductionIteration, CompiledAbductionProblem


def create_hypothesis_feedback_prompt(
    problem: CompiledAbductionProblem,
    last: AbductionIteration,
    counterexample: Dict[str, Any],
) -> str:
    issue = "inconsistent with the premise" if not last.is_consistent else "insufficient"

    ce_lines = []
    for k, v in counterexample.items():
        ce_lines.append("{0} = {1}".format(k, v))
    ce_block = "\n".join(ce_lines) if ce_lines else "(no model)"

    prev_smt = []
    prev_nl = []
    if last.hypothesis is not None:
        prev_smt = [t.sexpr() for t in last.hypothesis.smt_terms]
        prev_nl = list(last.hypothesis.nl_terms)

    return (
        "Refine your abductive hypothesis using solver feedback.\n\n"
        "Premise (text):\n"
        "---\n"
        "{0}\n"
        "---\n\n"
        "Conclusion (text):\n"
        "---\n"
        "{1}\n"
        "---\n\n"
        "SMT meaning:\n"
        "- domain: {2}\n"
        "- premise: {3}\n"
        "- conclusion: {4}\n\n"
        "Your previous hypothesis:\n"
        "- psi_smt: {5}\n"
        "- psi_nl: {6}\n\n"
        "Issue: {7}\n\n"
        "Counterexample assignment:\n"
        "{8}\n\n"
        "Output MUST be a single JSON object with exactly these keys:\n"
        "{{\n"
        '  \"psi_smt\": [\"<SMT Bool term>\", ...],\n'
        '  \"psi_nl\": [\"<natural language constraint>\", ...]\n'
        "}}\n\n"
        "No extra prose outside JSON.\n"
    ).format(
        problem.premise_text,
        problem.conclusion_text,
        problem.domain_constraints.sexpr(),
        problem.premise.sexpr(),
        problem.conclusion.sexpr(),
        prev_smt,
        prev_nl,
        issue,
        ce_block,
    )
