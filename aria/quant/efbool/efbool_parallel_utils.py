"""
Some parallel utilities for efbool.
Is this file used for now?
"""
from multiprocessing import Pool
from typing import List, Tuple

from pysat.solvers import Solver


def check_sat_assuming(clauses: List[List[int]], assumptions: List[int]) -> Tuple[bool, List[int]]:
    """Check satisfiability of a formula under some assumptions.

    Args:
        clauses: A Boolean formula represented as a list of lists of integers.
        assumptions: A set of literals forming the assumptions.

    Returns:
        A tuple containing the satisfiability result (True or False) and either the model or the unsatisfiable core.
    """
    with Solver(name="m22", bootstrap_with=clauses) as solver:
        ans: bool = solver.solve(assumptions=assumptions)
        if ans:
            return ans, solver.get_model()
        return ans, solver.get_core()


def parallel_check_assumptions(
    clauses: List[List[int]], assumptions_lists: List[List[int]], num_workers: int
) -> List[List[int]]:
    """Solve clauses under a set of assumptions (deal with each one in parallel).

    Args:
        clauses: A Boolean formula represented as a list of lists of integers.
        assumptions_lists: A list of assumption lists to be checked in parallel.
        num_workers: The number of worker processes to use for parallel processing.

    Returns:
        A list of results, where each result is either the model or the unsatisfiable core.
    """
    assert num_workers >= 1

    with Pool(num_workers) as pool:
        results = pool.starmap(check_sat_assuming, [(clauses, assumptions) for assumptions in assumptions_lists])

    return [result for _, result in results]
