"""Checker utilities for EFBV parallel module.

FIXME: the file is very likely buggy
"""

import multiprocessing
import concurrent.futures
from typing import List

import z3

from aria.quant.efbv.efbv_parallel.exceptions import (
    ForAllSolverSuccess,
    ForAllSolverUnknown,
)


def check_candidate(fml: z3.BoolRef):
    """
    Check candidate provided by the ExistsSolver.

    Args:
        fml: the formula to be checked, which is based on
             a new z3 context different from the main thread

    Returns:
        A model (to be translated to the main context by ForAllSolver)

    TODO: we may pass a set of formulas to this function (any ways, the code in
      this function is thread-local?
    """
    # print("Checking one ...", fml)
    solver = z3.SolverFor("QF_BV", ctx=fml.ctx)
    solver.add(fml)
    res = solver.check()
    if res == z3.sat:
        m = solver.model()
        return m  # to the original context?
    if res == z3.unsat:
        raise ForAllSolverSuccess()
    raise ForAllSolverUnknown()


def parallel_check_candidates(fmls: List[z3.BoolRef], num_workers: int):
    """Check candidates in parallel.

    Create new context for the computation. Note that we need to do this
    sequentially, as parallel access to the current context or its objects
    will result in a segfault.
    """
    # origin_ctx = fmls[0].ctx
    tasks = []
    for fml in fmls:
        # tasks.append((fml, main_ctx()))
        i_context = z3.Context()
        i_fml = fml.translate(i_context)
        tasks.append(i_fml)

    # TODO: try processes?
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        #  with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(check_candidate, task) for task in tasks]
        results = [f.result() for f in futures]
        return results


def parallel_check_candidates_multiprocessing(fmls: List[z3.ExprRef], num_workers):
    """Solve clauses under a set of assumptions (deal with each one in parallel).

    Note: still experimental; prefer the thread-based version.
    """
    assert num_workers >= 1
    tasks = []
    for fml in fmls:
        i_context = z3.Context()
        i_fml = fml.translate(i_context)
        tasks.append(i_fml)

    with multiprocessing.Pool(num_workers) as p:
        try:
            answers = p.map(check_candidate, tasks)
        finally:
            p.close()
            p.join()

    return answers
