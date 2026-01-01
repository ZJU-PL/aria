"""
PySMT counterparts of the Z3-based unary satisfiability helpers.

Result encoding (shared across helpers):
- 1: constraint satisfiable under the given precondition
- 0: unsatisfiable
- 2: unknown (e.g., solver returned unknown)
"""

from typing import List, Optional

from pysmt.exceptions import SolverReturnedUnknownResultError
from pysmt.shortcuts import Solver


def _check(solver: Solver) -> str:
    """
    Run the solver and normalize the outcome.

    Returns:
        "sat" | "unsat" | "unknown"
    """
    try:
        res = solver.solve()
    except SolverReturnedUnknownResultError:
        return "unknown"

    if res is True:
        return "sat"
    if res is False:
        return "unsat"
    return "unknown"


def unary_check(precond, cnt_list: List) -> List[int]:
    """Check each constraint independently under the precondition (fresh solver per check)."""
    results: List[Optional[int]] = [None] * len(cnt_list)

    for i, cnt in enumerate(cnt_list):
        solver = Solver()
        solver.add_assertion(precond)
        solver.add_assertion(cnt)

        res = _check(solver)
        if res == "sat":
            results[i] = 1
        elif res == "unsat":
            results[i] = 0
        else:
            results[i] = 2

    return results  # type: ignore[return-value]


def unary_check_incremental(precond, cnt_list: List) -> List[int]:
    """Check each constraint with a shared solver using push/pop for efficiency."""
    results: List[Optional[int]] = [None] * len(cnt_list)
    solver = Solver()
    solver.add_assertion(precond)

    for i, cnt in enumerate(cnt_list):
        solver.push()
        solver.add_assertion(cnt)
        res = _check(solver)
        if res == "sat":
            results[i] = 1
        elif res == "unsat":
            results[i] = 0
        else:
            results[i] = 2
        solver.pop()

    return results  # type: ignore[return-value]


def unary_check_cached(precond, cnt_list: List) -> List[int]:
    """Reuse satisfying models to mark other constraints true when implied by the model."""
    results: List[Optional[int]] = [None] * len(cnt_list)

    for i, cnt in enumerate(cnt_list):
        if results[i] is not None:
            continue

        solver = Solver()
        solver.add_assertion(precond)
        solver.add_assertion(cnt)
        res = _check(solver)
        if res == "sat":
            model = solver.get_model()
            results[i] = 1
            for j, other_cnt in enumerate(cnt_list):
                if results[j] is None and model.get_value(other_cnt).is_true():
                    results[j] = 1
        elif res == "unsat":
            results[i] = 0
        else:
            results[i] = 2

    return results  # type: ignore[return-value]


def unary_check_incremental_cached(precond, cnt_list: List) -> List[int]:
    """Incremental + caching: share solver state and propagate model truths across constraints."""
    results: List[Optional[int]] = [None] * len(cnt_list)
    solver = Solver()
    solver.add_assertion(precond)

    for i, cnt in enumerate(cnt_list):
        if results[i] is not None:
            continue

        solver.push()
        solver.add_assertion(cnt)
        res = _check(solver)
        if res == "sat":
            model = solver.get_model()
            results[i] = 1
            for j, other_cnt in enumerate(cnt_list):
                if results[j] is None and model.get_value(other_cnt).is_true():
                    results[j] = 1
        elif res == "unsat":
            results[i] = 0
        else:
            results[i] = 2
        solver.pop()

    return results  # type: ignore[return-value]
