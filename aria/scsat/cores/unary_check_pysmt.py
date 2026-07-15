"""
PySMT counterparts of the Z3-based unary satisfiability helpers.

Result encoding:
- 1: satisfiable under the precondition
- 0: unsatisfiable under the precondition
- 2: unknown (solver returned unknown or not yet decided)
"""

from typing import Tuple, List, Optional

from pysmt.exceptions import SolverReturnedUnknownResultError
from pysmt.shortcuts import Solver


# ── Internal solver helper ───────────────────────────────────────────────────

def _check(solver: Solver, solver_calls: list[int]) -> str:
    """
    Normalize solver status to sat/unsat/unknown/timeout.
    """
    solver_calls[0] += 1
    try:
        res = solver.solve()
    except SolverReturnedUnknownResultError:
        return "timeout"

    if res is True:
        return "sat"
    if res is False:
        return "unsat"
    
    return "unknown"


# ── Algorithm 1: LS ───────────────────────────────────────────────────────────

def unary_check(precond, cnt_list: List, timeout_ms: int = 0, solver_calls: Optional[list[int]] = None) -> Tuple[List[int], int]:
    """
    Check each constraint independently under the precondition (fresh solver per check).
    """
    if solver_calls is None:
        solver_calls = [0]

    results: List[int] = []

    for i, cnt in enumerate(cnt_list):
        with Solver(name="z3", solver_options={"timeout": timeout_ms}) as solver:
            solver.add_assertion(precond)
            solver.add_assertion(cnt)

            status = _check(solver, solver_calls)
            if status == "sat":
                results.append(1)
            elif status == "unsat":
                results.append(0)
            else:   # timeout or unknown
                results.append(2)

    return results, solver_calls[0]


# ── Algorithm 2: LS-Inc ───────────────────────────────────────────────────────

def unary_check_incremental(precond, cnt_list: List, timeout_ms: int = 0, solver_calls: Optional[list[int]] = None) -> Tuple[List[int], int]:
    """
    Check each constraint with a shared solver using push/pop for efficiency.
    """
    if solver_calls is None:
        solver_calls = [0]

    results: List[int] = []

    with Solver(name="z3", solver_options={"timeout": timeout_ms}) as solver:
        solver.add_assertion(precond)

        for i, cnt in enumerate(cnt_list):
            solver.push()
            solver.add_assertion(cnt)
            status = _check(solver, solver_calls)
            if status == "sat":
                results.append(1)
            elif status == "unsat":
                results.append(0)
            else:   # timeout or unknown
                results.append(2)
            solver.pop()

    return results, solver_calls[0]


# ── Algorithm 3: LS-Reuse ───────────────────────────────────────────────────────

def unary_check_cached(precond, cnt_list: List, timeout_ms: int = 0, solver_calls: Optional[list[int]] = None) -> Tuple[List[int], int]:
    """
    Reuse satisfying models to mark other constraints true when implied by the model.
    """
    if solver_calls is None:
        solver_calls = [0]

    results: List[Optional[int]] = [None] * len(cnt_list)

    for i, cnt in enumerate(cnt_list):
        if results[i] is not None:
            continue

        with Solver(name="z3", solver_options={"timeout": timeout_ms}) as solver:
            solver.add_assertion(precond)
            solver.add_assertion(cnt)
            status = _check(solver, solver_calls)

            if status == "sat":
                model = solver.get_model()
                results[i] = 1
                for j, other_cnt in enumerate(cnt_list):
                    if results[j] is None and model.get_value(other_cnt).is_true():
                        results[j] = 1
            elif status == "unsat":
                results[i] = 0
            else:   # timeout or unknown
                results[i] = 2

    return results, solver_calls[0]


# ── Algorithm 4: LS-IncReuse ───────────────────────────────────────────────────────

def unary_check_incremental_cached(precond, cnt_list: List, timeout_ms: int = 0, solver_calls: Optional[list[int]] = None) -> Tuple[List[int], int]:
    """
    Incremental + caching: share solver state and propagate model truths across constraints.
    """
    if solver_calls is None:
        solver_calls = [0]

    results: List[Optional[int]] = [None] * len(cnt_list)

    with Solver(name="z3", solver_options={"timeout": timeout_ms}) as solver:
        solver.add_assertion(precond)

        for i, cnt in enumerate(cnt_list):
            if results[i] is not None:
                continue

            solver.push()
            solver.add_assertion(cnt)
            status = _check(solver, solver_calls)

            if status == "sat":
                model = solver.get_model()
                results[i] = 1
                for j, other_cnt in enumerate(cnt_list):
                    if results[j] is None and model.get_value(other_cnt).is_true():
                        results[j] = 1
            elif status == "unsat":
                results[i] = 0
            else:   # timeout or unknown
                results[i] = 2
            solver.pop()

    return results, solver_calls[0]
