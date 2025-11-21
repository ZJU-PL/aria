"""Validation logic for abduction hypotheses."""

import z3
from typing import Dict, Any, Optional, Tuple
from arlib.utils.z3_solver_utils import is_sat, is_entail
from .data_structures import AbductionProblem


def validate_hypothesis(problem: AbductionProblem, hypothesis: z3.BoolRef) -> Tuple[bool, bool]:
    """Return (is_consistent, is_sufficient) for a hypothesis."""
    if z3.is_true(hypothesis):
        return True, False

    full_premise = (z3.And(problem.domain_constraints, problem.premise)
                   if not z3.is_true(problem.domain_constraints)
                   else problem.premise)

    is_consistent = is_sat(z3.And(full_premise, hypothesis))
    if not is_consistent:
        return False, False

    is_sufficient = is_entail(z3.And(full_premise, hypothesis), problem.conclusion)
    return is_consistent, is_sufficient


def generate_counterexample(problem: AbductionProblem, hypothesis: z3.BoolRef) -> Optional[Dict[str, Any]]:
    """Generate a counterexample showing why the hypothesis is invalid."""
    full_premise = (z3.And(problem.domain_constraints, problem.premise)
                   if not z3.is_true(problem.domain_constraints)
                   else problem.premise)

    # Check for inconsistency
    if not is_sat(z3.And(full_premise, hypothesis)):
        s = z3.Solver()
        s.add(full_premise, z3.Not(hypothesis))
        if s.check() == z3.sat:
            model = s.model()
            return {str(v): model.eval(v, model_completion=True) for v in problem.variables}
        return None

    # Check for insufficiency
    if not is_entail(z3.And(full_premise, hypothesis), problem.conclusion):
        s = z3.Solver()
        s.add(full_premise, hypothesis, z3.Not(problem.conclusion))
        if s.check() == z3.sat:
            model = s.model()
            return {str(v): model.eval(v, model_completion=True) for v in problem.variables}

    return None
