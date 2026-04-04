from typing import Any, List

from aria.utils.z3.expr import (
    get_variables,
    get_atoms,
    to_smtlib2,
    big_and,
    big_or,
    negate,
    ctx_simplify,
    eval_predicates,
)
from aria.utils.z3.solver import is_sat, is_equiv, is_valid, is_entail

from .verification_utils import (
    VerificationResult,
    SolverTimeout,
    check_entailment,
    are_expressions_equivalent,
    check_invariant,
    get_counterexample,
)


def extract_all(lst: List[Any]) -> List[Any]:
    """Extract all elements from nested lists."""
    results = []
    for elem in lst:
        if isinstance(elem, list):
            results.extend(extract_all(elem))
        else:
            results.append(elem)
    return results
