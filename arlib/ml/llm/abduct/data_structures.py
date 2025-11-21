"""Data structures for LLM-based abduction."""

import z3
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from arlib.utils.z3_expr_utils import get_variables


@dataclass
class AbductionProblem:
    premise: z3.BoolRef
    conclusion: z3.BoolRef
    description: str = ""
    variables: List[z3.ExprRef] = field(default_factory=list)
    domain_constraints: z3.BoolRef = None

    def __post_init__(self):
        if not self.variables:
            self.variables = list(set(get_variables(self.premise) + get_variables(self.conclusion)))
        if self.domain_constraints is None:
            self.domain_constraints = z3.BoolVal(True)

    def to_smt2_string(self) -> str:
        smt2 = []
        for var in self.variables:
            name = str(var)
            if z3.is_int(var):
                smt2.append(f"(declare-const {name} Int)")
            elif z3.is_real(var):
                smt2.append(f"(declare-const {name} Real)")
            elif z3.is_bool(var):
                smt2.append(f"(declare-const {name} Bool)")
            elif z3.is_bv(var):
                smt2.append(f"(declare-const {name} (_ BitVec {var.size()}))")
            else:
                smt2.append(f"(declare-const {name} {var.sort()})")

        if str(self.domain_constraints) != "True":
            smt2.append(f"(assert {self.domain_constraints.sexpr()})")
        smt2.append(f"(assert {self.premise.sexpr()})")
        smt2.extend([
            ";; Goal: find ψ such that:",
            ";; 1. (premise ∧ ψ) is satisfiable",
            ";; 2. (premise ∧ ψ) |= conclusion",
            f";; where conclusion is: {self.conclusion.sexpr()}"
        ])
        return "\n".join(smt2)


@dataclass
class AbductionResult:
    problem: AbductionProblem
    hypothesis: Optional[z3.BoolRef] = None
    is_consistent: bool = False
    is_sufficient: bool = False
    is_valid: bool = False
    llm_response: str = ""
    prompt: str = ""
    execution_time: float = 0.0
    error: Optional[str] = None

    def __post_init__(self):
        self.is_valid = self.is_consistent and self.is_sufficient


@dataclass
class AbductionIterationResult:
    hypothesis: Optional[z3.BoolRef] = None
    is_consistent: bool = False
    is_sufficient: bool = False
    is_valid: bool = False
    counterexample: Optional[Dict[str, Any]] = None
    llm_response: str = ""
    prompt: str = ""
    iteration: int = 0


@dataclass
class FeedbackAbductionResult(AbductionResult):
    iterations: List[AbductionIterationResult] = field(default_factory=list)
    total_iterations: int = 0

    def add_iteration(self, iteration_result: AbductionIterationResult) -> None:
        self.iterations.append(iteration_result)
        self.total_iterations = len(self.iterations)
