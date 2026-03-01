"""Data structures for natural-language abduction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import z3


@dataclass
class SmtVarDecl:
    name: str
    sort: str
    description: str = ""


@dataclass
class CompiledAbductionProblem:
    """SMT-backed abduction problem compiled from natural language."""

    text: str
    premise_text: str
    conclusion_text: str
    variables: List[SmtVarDecl]
    domain_constraints: z3.BoolRef
    premise: z3.BoolRef
    conclusion: z3.BoolRef
    glossary: Dict[str, str] = field(default_factory=dict)


@dataclass
class HybridHypothesis:
    """A hypothesis with an SMT part and an NL-only part."""

    smt_terms: List[z3.BoolRef] = field(default_factory=list)
    nl_terms: List[str] = field(default_factory=list)

    def smt_conjunction(self) -> z3.BoolRef:
        if not self.smt_terms:
            return z3.BoolVal(True)
        return z3.And(*self.smt_terms)


@dataclass
class CompilationResult:
    problem: Optional[CompiledAbductionProblem] = None
    prompt: str = ""
    llm_response: str = ""
    error: Optional[str] = None
    attempts: int = 0


@dataclass
class AbductionIteration:
    iteration: int
    hypothesis: Optional[HybridHypothesis] = None
    is_consistent: bool = False
    is_sufficient: bool = False
    is_valid: bool = False
    counterexample: Optional[Dict[str, Any]] = None
    bridge_lemmas_smt: List[str] = field(default_factory=list)
    verifier_response: str = ""
    prompt: str = ""
    llm_response: str = ""
    error: Optional[str] = None


@dataclass
class AbductionResult:
    compiled: Optional[CompiledAbductionProblem] = None
    hypothesis: Optional[HybridHypothesis] = None
    is_consistent: bool = False
    is_sufficient: bool = False
    is_valid: bool = False
    iterations: List[AbductionIteration] = field(default_factory=list)
    compilation: Optional[CompilationResult] = None
    error: Optional[str] = None
    execution_time: float = 0.0
