"""
PySMT disjunctive over-approximation helpers.

Result encoding:
- 1: satisfiable under the precondition
- 0: unsatisfiable under the precondition
- 2: unknown (solver returned unknown or not yet decided)
"""

from typing import List

from pysmt.exceptions import SolverReturnedUnknownResultError
from pysmt.shortcuts import And, Or, Solver


def _check(solver: Solver) -> str:
    """Normalize solver status to sat/unsat/unknown."""
    try:
        res = solver.solve()
    except SolverReturnedUnknownResultError:
        return "unknown"

    if res is True:
        return "sat"
    if res is False:
        return "unsat"
    return "unknown"


def compact_check_cached(precond, cnt_list: List, res_label: List[int]):
    """Recursive disjunctive check (fresh solver each recursion), with model caching."""
    conditions = [cnt_list[i] for i, lbl in enumerate(res_label) if lbl == 2]

    if len(conditions) == 0:
        return

    f = Or(conditions)

    if f.is_false():
        return

    solver = Solver()
    solver.add_assertion(And(precond, f))
    res = _check(solver)
    if res == "unsat":
        for i, lbl in enumerate(res_label):
            if lbl == 2:
                res_label[i] = 0
    elif res == "sat":
        m = solver.get_model()
        for i, lbl in enumerate(res_label):
            if lbl == 2 and m.get_value(cnt_list[i]).is_true():
                res_label[i] = 1
    else:
        return
    compact_check_cached(precond, cnt_list, res_label)


def disjunctive_check_cached(precond, cnt_list: List) -> List[int]:
    """Entry point for cached disjunctive checking (non-incremental solver usage)."""
    res = [2] * len(cnt_list)  # 0 means unsat, 1 means sat, 2 means "unknown"
    compact_check_cached(precond, cnt_list, res)
    return res


def compact_check_incremental_cached(
    solver: Solver, precond, cnt_list: List, res_label: List[int]
):
    """Recursive disjunctive check using a shared solver with push/pop for efficiency."""
    conditions = [cnt_list[i] for i, lbl in enumerate(res_label) if lbl == 2]

    if len(conditions) == 0:
        return

    f = Or(conditions)

    if f.is_false():
        return

    solver.push()
    solver.add_assertion(f)
    res = _check(solver)
    if res == "unsat":
        for i, lbl in enumerate(res_label):
            if lbl == 2:
                res_label[i] = 0
    elif res == "sat":
        m = solver.get_model()
        for i, lbl in enumerate(res_label):
            if lbl == 2 and m.get_value(cnt_list[i]).is_true():
                res_label[i] = 1
    else:
        solver.pop()
        return
    solver.pop()
    compact_check_incremental_cached(solver, precond, cnt_list, res_label)


def disjunctive_check_incremental_cached(precond, cnt_list: List) -> List[int]:
    """Entry point for cached disjunctive checking with a shared incremental solver."""
    results = [2] * len(cnt_list)
    solver = Solver()
    solver.add_assertion(precond)
    compact_check_incremental_cached(solver, precond, cnt_list, results)
    return results
