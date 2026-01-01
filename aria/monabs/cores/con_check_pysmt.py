"""
PySMT versions of conjunctive/disjunctive satisfiability helpers with optional caching.

Result encoding (shared across helpers):
- 1: constraint satisfiable under the given precondition
- 0: unsatisfiable
- 2: unknown (e.g., solver returned unknown)
"""

from typing import List, Optional

from pysmt.exceptions import SolverReturnedUnknownResultError
from pysmt.shortcuts import Or, Solver


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


def _core_indices(core) -> List[int]:
    """Extract integer indices from a PySMT named unsat core."""
    indices: List[int] = []
    # core may be a dict(name -> formula) or an iterable of names
    items = core.keys() if hasattr(core, "keys") else core
    for item in items:
        name = str(item)
        if name.startswith("idx_"):
            try:
                indices.append(int(name[4:]))
                continue
            except ValueError:
                pass
        try:
            indices.append(int(name))
        except ValueError:
            # Skip entries we cannot map back to indices
            continue
    return indices


def unary_check_cached(
    precond, cnt_list: List, results: List[Optional[int]], check_list: List[int]
):
    """Non-incremental check with model-based caching over a subset of constraints."""
    for i in check_list:
        if results[i] is not None:
            continue
        solver = Solver()
        solver.add_assertion(precond)
        solver.add_assertion(cnt_list[i])
        res = _check(solver)
        if res == "sat":
            model = solver.get_model()
            results[i] = 1
            for j in check_list:
                if results[j] is None and model.get_value(cnt_list[j]).is_true():
                    results[j] = 1
        elif res == "unsat":
            results[i] = 0
        else:
            results[i] = 2


def unary_check_incremental_cached(
    solver: Solver, cnt_list: List, results: List[Optional[int]], check_list: List[int]
):
    """Incremental version of unary_check_cached using a shared solver with push/pop."""
    for i in check_list:
        if results[i] is not None:
            continue
        solver.push()
        solver.add_assertion(cnt_list[i])
        res = _check(solver)
        if res == "sat":
            model = solver.get_model()
            results[i] = 1
            for j in check_list:
                if results[j] is None and model.get_value(cnt_list[j]).is_true():
                    results[j] = 1
            solver.pop()
        elif res == "unsat":
            core = solver.get_named_unsat_core()
            solver.pop()
            results[i] = 0
            # core content not used here, but keep symmetry
        else:
            results[i] = 2
            solver.pop()


def disjunctive_check_incremental_cached(
    solver: Solver, cnt_list: List, results: List[Optional[int]], check_list: List[int]
):
    """Recursively solve a disjunction of pending constraints, caching satisfiable members."""
    conditions = [cnt_list[i] for i in check_list if results[i] is None]
    if len(conditions) == 0:
        return

    f = Or(conditions)
    if f.is_false():
        return

    solver.push()
    solver.add_assertion(f)
    res = _check(solver)
    if res == "unsat":
        for i in check_list:
            results[i] = 0
        solver.pop()
    elif res == "sat":
        m = solver.get_model()
        solver.pop()
        new_check_list = []
        for i in check_list:
            if results[i] is None and m.get_value(cnt_list[i]).is_true():
                results[i] = 1
            elif results[i] is None:
                new_check_list.append(i)
        disjunctive_check_incremental_cached(solver, cnt_list, results, new_check_list)
    else:
        solver.pop()
        for i in check_list:
            if results[i] is None:
                results[i] = 2


def conjunctive_check_incremental(
    precond, cnt_list: List, alogorithm: int = 0
) -> List[int]:
    """
    Perform a conjunctive satisfiability check on a list of constraints under a given precondition.
    Mirrors the Z3-based version but uses PySMT solvers and APIs.
    """
    results: List[Optional[int]] = [None] * len(cnt_list)
    solver = Solver(name="z3", unsat_cores_mode="named")  # need unsat cores
    waiting_list_idx: List[int] = []
    queue: List[List[int]] = [list(range(len(cnt_list)))]

    while queue:
        current_subset = queue.pop(0)
        solver.push()
        for idx in current_subset:
            solver.add_assertion(cnt_list[idx], named=f"idx_{idx}")
        solver_result = _check(solver)
        if (
            solver_result == "sat"
        ):  # no conflicts within the predicates, no need to split
            while True:
                solver.add_assertion(precond)
                solver_result = _check(solver)
                if solver_result == "sat":
                    solver.pop()
                    for idx in current_subset:
                        results[idx] = 1
                    break
                elif solver_result == "unsat":
                    unsat_core = _core_indices(solver.get_named_unsat_core())
                    solver.pop()
                    for idx in unsat_core:
                        if idx in current_subset:
                            current_subset.remove(idx)
                        if idx not in waiting_list_idx:
                            waiting_list_idx.append(idx)
                    if len(current_subset) == 0:
                        break
                    solver.push()
                    for idx in current_subset:
                        solver.add_assertion(cnt_list[idx], named=f"idx_{idx}")
                else:  # unknown
                    solver.pop()
                    for idx in current_subset:
                        if results[idx] is None:
                            results[idx] = 2
                    break

        elif solver_result == "unsat":  # conflicts within the predicates, need to split
            core_indices = set(_core_indices(solver.get_named_unsat_core()))
            solver.pop()
            unsat_set_indices = list(core_indices)
            sat_set_indices = [i for i in current_subset if i not in core_indices]
            if len(unsat_set_indices) == 1:
                waiting_list_idx.append(unsat_set_indices[0])
                if sat_set_indices:
                    queue.append(sat_set_indices)
            else:
                subsets = [
                    [unsat_set_indices[i]] for i in range(len(unsat_set_indices))
                ]
                for i, sat_item in enumerate(sat_set_indices):
                    subsets[i % len(unsat_set_indices)].append(sat_item)
                queue.extend(subsets)
        else:
            solver.pop()
            for idx in current_subset:
                if results[idx] is None:
                    results[idx] = 2

    solver.add_assertion(precond)
    if alogorithm == 0:
        unary_check_cached(precond, cnt_list, results, waiting_list_idx)
    elif alogorithm == 1:
        unary_check_incremental_cached(solver, cnt_list, results, waiting_list_idx)
    elif alogorithm == 2:
        disjunctive_check_incremental_cached(
            solver, cnt_list, results, waiting_list_idx
        )
    else:
        raise ValueError("Invalid algorithm choice. Choose 0, 1, or 2.")

    return results  # type: ignore[return-value]


def conjunctive_check(precond, cnt_list: List, alogorithm: int = 0) -> List[int]:
    """
    Non-incremental version of conjunctive satisfiability check using PySMT.
    """
    results: List[Optional[int]] = [None] * len(cnt_list)
    waiting_list_idx: List[int] = []
    queue: List[List[int]] = [list(range(len(cnt_list)))]
    i = 0
    while queue:
        i += 1
        current_subset = queue.pop(0)
        solver_split = Solver(name="z3", unsat_cores_mode="named")
        for idx in current_subset:
            solver_split.add_assertion(cnt_list[idx], named=f"idx_{idx}")
        solver_result = _check(solver_split)
        if (
            solver_result == "sat"
        ):  # no conflicts within the predicates, no need to split
            while True:
                solver_check = Solver(name="z3", unsat_cores_mode="named")
                solver_check.add_assertion(precond)
                for idx in current_subset:
                    solver_check.add_assertion(cnt_list[idx], named=f"idx_{idx}")
                solver_result = _check(solver_check)

                if solver_result == "sat":
                    for idx in current_subset:
                        results[idx] = 1
                    break
                elif solver_result == "unsat":
                    unsat_core = _core_indices(solver_check.get_named_unsat_core())
                    for idx in unsat_core:
                        if idx in current_subset:
                            current_subset.remove(idx)
                        if idx not in waiting_list_idx:
                            waiting_list_idx.append(idx)
                    if len(current_subset) == 0:
                        break
                else:
                    for idx in current_subset:
                        if results[idx] is None:
                            results[idx] = 2
                    break

        elif solver_result == "unsat":  # conflicts within the predicates, need to split
            unsat_core_indices = set(_core_indices(solver_split.get_named_unsat_core()))
            unsat_set_indices = list(unsat_core_indices)
            sat_set_indices = [i for i in current_subset if i not in unsat_core_indices]
            if len(unsat_set_indices) == 1:
                waiting_list_idx.append(unsat_set_indices[0])
                if sat_set_indices:
                    queue.append(sat_set_indices)
            else:
                subsets = [
                    [unsat_set_indices[i]] for i in range(len(unsat_set_indices))
                ]
                for i, sat_item in enumerate(sat_set_indices):
                    subsets[i % len(unsat_set_indices)].append(sat_item)
                queue.extend(subsets)
        else:
            for idx in current_subset:
                if results[idx] is None:
                    results[idx] = 2

    solver_fallback = Solver(name="z3", unsat_cores_mode="named")
    solver_fallback.add_assertion(precond)
    if alogorithm == 0:
        unary_check_cached(precond, cnt_list, results, waiting_list_idx)
    elif alogorithm == 1:
        unary_check_incremental_cached(
            solver_fallback, cnt_list, results, waiting_list_idx
        )
    elif alogorithm == 2:
        disjunctive_check_incremental_cached(
            solver_fallback, cnt_list, results, waiting_list_idx
        )
    else:
        raise ValueError("Invalid algorithm choice. Choose 0, 1, or 2.")

    return results  # type: ignore[return-value]
