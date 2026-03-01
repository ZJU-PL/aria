"""Demo: NL abduction pipeline.

Run with:
  python -m aria.ml.llm.abduction.demo

This demo requires your LLM provider credentials to be configured.
"""

from __future__ import annotations

import os

from aria.llmtools.client import LLM
from aria.llmtools.logger import Logger

from .abductor import NLAbductor


def main() -> None:
    model = os.environ.get("ARIA_LLM_MODEL", "gpt-4.1-mini")
    logger = Logger(log_file_path=".logs/nl_abduction_demo.log")
    llm = LLM(online_model_name=model, logger=logger, temperature=0.2)
    abd = NLAbductor(llm=llm)

    text = """Premise: Alice and Bob each have a positive integer number of apples.
Conclusion: Alice has more than 5 apples, and together they have more than 10 apples."""

    res = abd.abduce(text)
    if res.error:
        print("ERROR:", res.error)
        if res.compilation and res.compilation.error:
            print("COMPILATION ERROR:", res.compilation.error)
        return

    assert res.compiled is not None
    print("Compiled premise:", res.compiled.premise)
    print("Compiled conclusion:", res.compiled.conclusion)
    assert res.hypothesis is not None
    print("Hypothesis psi_smt:", [t.sexpr() for t in res.hypothesis.smt_terms])
    print("Hypothesis psi_nl:", list(res.hypothesis.nl_terms))


if __name__ == "__main__":
    main()
