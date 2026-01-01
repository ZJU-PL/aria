"""
Reduce OMT(BV) to Weighted MaxSAT.

This module provides functions to reduce bit-vector optimization problems to
weighted MaxSAT problems using:
1. OBV-BS and its variants
2. Existing weighted MaxSAT solvers
"""

import logging
from typing import Optional, Union

import z3

from aria.optimization.omtbv.bit_blast_omt_solver import BitBlastOMTBVSolver

logger = logging.getLogger(__name__)


def bv_opt_with_maxsat(
    z3_fml: z3.ExprRef,
    z3_obj: z3.ExprRef,
    minimize: bool,
    solver_name: str,
) -> Optional[Union[int, float]]:
    """Reduce OMT(BV) to Weighted MaxSAT.

    Args:
        z3_fml: Z3 formula to optimize
        z3_obj: Objective variable to optimize
        minimize: If True, minimize the objective; if False, maximize
        solver_name: Name of the MaxSAT solver to use

    Returns:
        Optimal value found by the solver, or None if optimization failed

    Note:
        Currently all objectives are converted to "maximize" internally.
        For minimization, we maximize the negation and convert the result.
        TODO: Consider adding a dedicated minimize_with_maxsat API.
    """
    omt = BitBlastOMTBVSolver()
    omt.from_smt_formula(z3_fml)
    omt.set_engine(solver_name)

    return omt.maximize_with_maxsat(z3_obj, is_signed=False, minimize=minimize)


def demo_maxsat() -> None:
    """Demo function for MaxSAT-based bit-vector optimization."""
    import time  # pylint: disable=import-outside-toplevel

    y = z3.BitVec("y", 4)
    fml = z3.And(z3.UGT(y, 3), z3.ULT(y, 10))
    logger.info("Starting MaxSAT-based optimization")
    start = time.time()
    res = bv_opt_with_maxsat(fml, y, minimize=True, solver_name="FM")
    elapsed_time = time.time() - start
    logger.info("Result: %s", res)
    logger.info("Solving time: %.3f seconds", elapsed_time)


if __name__ == "__main__":
    demo_maxsat()
