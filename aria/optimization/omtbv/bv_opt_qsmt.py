"""
Reduce OMT(BV) to QSMT and call SMT solvers.

This module reduces bit-vector optimization problems to quantified SMT problems
and calls SMT solvers that support quantified bit-vector formulas:
- Z3
- CVC5
- Q3B
- ...?
"""

import logging

import z3

from aria.optimization.bin_solver import solve_with_bin_smt
from aria.utils.z3_expr_utils import get_expr_vars

logger = logging.getLogger(__name__)


def bv_opt_with_pysmt() -> None:
    """PySMT-based bit-vector optimization (not implemented)."""
    raise NotImplementedError


def bv_opt_with_qsmt(
    fml: z3.ExprRef,
    obj: z3.ExprRef,
    minimize: bool,
    solver_name: str,
) -> str:
    """Solve OMT(BV) using quantified SMT reduction.

    Args:
        fml: Z3 formula to optimize
        obj: Objective variable to optimize
        minimize: If True, minimize the objective; if False, maximize
        solver_name: Name of the solver to use

    Returns:
        String result from the solver

    Note:
        Currently all objectives are converted to "maximize" internally.
        TODO: Consider distinguishing between unsigned and signed
        comparisons (bvule vs <).
    """
    objname = obj
    all_vars = get_expr_vars(fml)
    if obj not in all_vars:
        # Create a new variable to represent obj (a term, e.g., x + y)
        objname = z3.BitVec(str(obj), obj.sort().size())
        fml = z3.And(fml, objname == obj)

    obj_misc = z3.BitVec("m_" + str(objname), obj.size())
    new_fml = z3.substitute(fml, (obj, obj_misc))

    if minimize:
        # Minimize: for all other values, if they satisfy the formula,
        # then obj <= that value
        qfml = z3.And(
            fml, z3.ForAll([obj_misc], z3.Implies(new_fml, z3.ULE(obj, obj_misc)))
        )
    else:
        # Maximize: for all other values, if they satisfy the formula,
        # then that value <= obj
        qfml = z3.And(
            fml, z3.ForAll([obj_misc], z3.Implies(new_fml, z3.ULE(obj_misc, obj)))
        )

    logger.debug("Quantified formula: %s", qfml)

    if z3.is_bv(obj):
        return solve_with_bin_smt(
            "BV", qfml=qfml, obj_name=obj.sexpr(), solver_name=solver_name
        )
    return solve_with_bin_smt(
        "ALL", qfml=qfml, obj_name=obj.sexpr(), solver_name=solver_name
    )


def demo_qsmt() -> None:
    """Demo function for QSMT-based bit-vector optimization."""
    import time  # pylint: disable=import-outside-toplevel

    y = z3.BitVec("y", 16)
    fml = z3.And(z3.UGT(y, 0), z3.ULT(y, 10))
    logger.info("Starting QSMT-based optimization")
    start = time.time()
    res = bv_opt_with_qsmt(fml, y, minimize=True, solver_name="z3")
    elapsed_time = time.time() - start
    logger.info("Result: %s", res)
    logger.info("Solving time: %.3f seconds", elapsed_time)


if __name__ == "__main__":
    demo_qsmt()
