"""
Iterative search-based optimization for bit-vector OMT problems.
"""
import logging
from typing import Any, Union

import z3
from pysmt.shortcuts import And, BV, BVUGT, BVULE, BVULT, BVUGE, Solver

from aria.optimization.pysmt_utils import z3_to_pysmt
from aria.utils.z3_expr_utils import get_expr_vars

logger = logging.getLogger(__name__)


def _preprocess_objective(z3_fml: z3.ExprRef, z3_obj: z3.ExprRef) -> tuple:
    """Preprocess objective variable for optimization."""
    objname = z3_obj
    all_vars = get_expr_vars(z3_fml)
    if z3_obj not in all_vars:
        objname = z3.BitVec(str(z3_obj), z3_obj.sort().size())
        z3_fml = z3.And(z3_fml, objname == z3_obj)
    return z3_to_pysmt(z3_fml, objname)


def bv_opt_with_linear_search(
    z3_fml: z3.ExprRef,
    z3_obj: z3.ExprRef,
    minimize: bool,
    solver_name: str,
) -> Union[int, str]:
    """Linear search-based OMT for bit-vectors."""
    obj, fml = _preprocess_objective(z3_fml, z3_obj)
    logger.info("Linear search %simization", "min" if minimize else "max")  # noqa: W1203

    with Solver(name=solver_name) as solver:
        solver.add_assertion(fml)
        return _minimize_linear_search(solver, obj) if minimize else _maximize_linear_search(solver, obj)


def bv_opt_with_binary_search(
    z3_fml: z3.ExprRef,
    z3_obj: z3.ExprRef,
    minimize: bool,
    solver_name: str,
) -> int:
    """Binary search-based OMT for bit-vectors."""
    obj, fml = _preprocess_objective(z3_fml, z3_obj)
    bv_width = obj.bv_width()
    max_bv = (1 << bv_width) - 1
    logger.info("Binary search %simization", "min" if minimize else "max")

    with Solver(name=solver_name) as solver:
        solver.add_assertion(fml)
        if minimize:
            return _minimize_binary_search(solver, obj, bv_width, max_bv)
        return _maximize_binary_search(solver, obj, bv_width, max_bv)


def _minimize_linear_search(solver: Solver, obj: Any) -> int:
    """Linear search minimization."""
    lower = BV(0, obj.bv_width())
    iteration = 0
    while solver.solve():
        iteration += 1
        lower = solver.get_model().get_value(obj)
        solver.add_assertion(BVULT(obj, lower))
    logger.info("Minimized in %d iterations: %d", iteration, int(lower.constant_value()))
    return int(lower.constant_value())


def _maximize_linear_search(solver: Solver, obj: Any) -> Union[int, str]:
    """Linear search maximization."""
    cur_upper = None
    iteration = 0
    while solver.solve():
        iteration += 1
        cur_upper = solver.get_model().get_value(obj)
        solver.add_assertion(BVUGT(obj, cur_upper))

    if cur_upper is not None:
        result = int(cur_upper.constant_value())
        logger.info("Maximized in %d iterations: %d", iteration, result)
        return result
    logger.info("Unsatisfiable")
    return "unsatisfiable"


def _minimize_binary_search(solver: Solver, obj: Any, bv_width: int, max_bv: int) -> int:
    """Binary search minimization."""
    cur_min, cur_max = 0, max_bv
    lower = BV(max_bv, bv_width)
    iteration = 0

    while cur_min <= cur_max:
        iteration += 1
        solver.push()
        cur_mid = cur_min + ((cur_max - cur_min) >> 1)

        cond = And(
            BVUGE(obj, BV(cur_min, bv_width)),
            BVULE(obj, BV(cur_mid, bv_width))
        )
        solver.add_assertion(cond)

        if not solver.solve():
            cur_min = cur_mid + 1
        else:
            lower = solver.get_model().get_value(obj)
            cur_max = int(lower.constant_value()) - 1
        solver.pop()

    logger.info("Minimized in %d iterations: %d", iteration, int(lower.constant_value()))
    return int(lower.constant_value())


def _maximize_binary_search(solver: Solver, obj: Any, bv_width: int, max_bv: int) -> int:
    """Binary search maximization."""
    cur_min, cur_max = 0, max_bv
    upper = BV(0, bv_width)
    iteration = 0

    while cur_min <= cur_max:
        iteration += 1
        solver.push()
        cur_mid = cur_min + ((cur_max - cur_min) >> 1)

        cond = And(
            BVUGE(obj, BV(cur_mid, bv_width)),
            BVULE(obj, BV(cur_max, bv_width))
        )
        solver.add_assertion(cond)

        if not solver.solve():
            cur_max = cur_mid - 1
        else:
            upper = solver.get_model().get_value(obj)
            cur_min = int(upper.constant_value()) + 1
        solver.pop()

    logger.info("Maximized in %d iterations: %d", iteration, int(upper.constant_value()))
    return int(upper.constant_value())


def demo_iterative() -> None:
    """Demonstrate iterative search optimization."""
    import time  # pylint: disable=import-outside-toplevel

    y = z3.BitVec("y", 16)
    fml = z3.And(z3.UGT(y, 3), z3.ULT(y, 10))
    start = time.time()

    try:
        logger.info("Linear search maximization:")
        lin_res = bv_opt_with_linear_search(fml, y, minimize=False, solver_name="z3")
        logger.info("Result: %s", lin_res)

        logger.info("Binary search minimization:")
        bin_res = bv_opt_with_binary_search(fml, y, minimize=True, solver_name="z3")
        logger.info("Result: %s", bin_res)

        logger.info("Total time: %.3fs", time.time() - start)
    except (ValueError, RuntimeError) as e:
        logger.error("Demo failed: %s", e)


def init_logger(log_level: str = 'INFO') -> None:
    """Initialize logger."""
    logger.handlers.clear()
    level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(level)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)


if __name__ == '__main__':
    init_logger('DEBUG')
    demo_iterative()
