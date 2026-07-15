"""
PySMT disjunctive over-approximation helpers.

Result encoding:
- 1: satisfiable under the precondition
- 0: unsatisfiable under the precondition
- 2: unknown (solver returned unknown or not yet decided)
"""

from typing import Tuple, List

from pysmt.exceptions import SolverReturnedUnknownResultError
from pysmt.shortcuts import And, Or, Solver


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


# ── Algorithm 1: OA ───────────────────────────────────────────────────────────

def _compact_check_cached(precond, cnt_list: List, res_label: List[int], solver_calls: list[int], timeout_ms: int = 0):
    """
    Recursive disjunctive check (fresh solver each recursion), with model caching.
    """
    conditions = [cnt_list[i] for i, lbl in enumerate(res_label) if lbl == 2]

    if len(conditions) == 0:
        return

    f = Or(conditions)

    if f.is_false():
        return

    with Solver(name="z3", solver_options={"timeout": timeout_ms}) as solver:
        solver.add_assertion(And(precond, f))
        status = _check(solver, solver_calls)
        if status == "unsat":
            for i, lbl in enumerate(res_label):
                if lbl == 2:
                    res_label[i] = 0
        elif status == "sat":
            m = solver.get_model()
            for i, lbl in enumerate(res_label):
                if lbl == 2 and m.get_value(cnt_list[i]).is_true():
                    res_label[i] = 1
        else:   # timeout or unknown
            return
    _compact_check_cached(precond, cnt_list, res_label, solver_calls, timeout_ms)


def disjunctive_check_cached(precond, cnt_list: List, timeout_ms: int = 0) -> Tuple[List[int], int]:
    """
    Entry point for cached disjunctive checking (non-incremental solver usage).
    """
    res = [2] * len(cnt_list)  # 0 means unsat, 1 means sat, 2 means "unknown"
    solver_calls = [0]
    _compact_check_cached(precond, cnt_list, res, solver_calls, timeout_ms)
    return res, solver_calls[0]


# ── Algorithm 2: OA-Inc ───────────────────────────────────────────────────────

def _compact_check_incremental_cached(solver: Solver, precond, cnt_list: List, res_label: List[int], solver_calls: list[int]):
    """
    Recursive disjunctive check using a shared solver with push/pop for efficiency.
    """
    conditions = [cnt_list[i] for i, lbl in enumerate(res_label) if lbl == 2]

    if len(conditions) == 0:
        return

    f = Or(conditions)

    if f.is_false():
        return

    solver.push()
    solver.add_assertion(f)
    status = _check(solver, solver_calls)
    if status == "unsat":
        for i, lbl in enumerate(res_label):
            if lbl == 2:
                res_label[i] = 0
    elif status == "sat":
        m = solver.get_model()
        for i, lbl in enumerate(res_label):
            if lbl == 2 and m.get_value(cnt_list[i]).is_true():
                res_label[i] = 1
    else:   # timeout or unknown
        solver.pop()
        return
    solver.pop()
    _compact_check_incremental_cached(solver, precond, cnt_list, res_label, solver_calls)


def disjunctive_check_incremental_cached(precond, cnt_list: List, timeout_ms: int = 0) -> Tuple[List[int], int]:
    """
    Entry point for cached disjunctive checking with a shared incremental solver.
    """
    results = [2] * len(cnt_list)
    solver_calls = [0]
    with Solver(name="z3", solver_options={"timeout": timeout_ms}) as solver:
        solver.add_assertion(precond)
        _compact_check_incremental_cached(solver, precond, cnt_list, results, solver_calls)
    return results, solver_calls[0]
