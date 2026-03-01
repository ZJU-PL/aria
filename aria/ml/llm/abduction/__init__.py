"""Natural-language abduction with LLM + SMT.

This package treats the *input problem* as natural language. An LLM first
compiles the text into an SMT-backed representation (variables, domain,
premise, conclusion). Another LLM stage generates abductive hypotheses (psi)
which are validated by an SMT solver (Z3), optionally with counterexample-
driven feedback.
"""

from .data_structures import (
    SmtVarDecl,
    CompiledAbductionProblem,
    CompilationResult,
    HybridHypothesis,
    AbductionIteration,
    AbductionResult,
)
from .compiler import NLAbductionCompiler
from .abductor import NLAbductor

__all__ = [
    "SmtVarDecl",
    "CompiledAbductionProblem",
    "CompilationResult",
    "HybridHypothesis",
    "AbductionIteration",
    "AbductionResult",
    "NLAbductionCompiler",
    "NLAbductor",
]
