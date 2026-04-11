"""Iterative OMT(QF_FP) search using IEEE-754 total ordering."""

import logging
from typing import Optional, cast

import z3

from aria.optimization.omtfp.fp_opt_qsmt import (
    fp_value_from_bits,
    prepare_fp_objective_with_key,
)

logger = logging.getLogger(__name__)


def fp_opt_with_linear_search(
    z3_fml: z3.ExprRef, z3_obj: z3.ExprRef, minimize: bool, solver_name: str = "z3"
) -> Optional[z3.ExprRef]:
    """Optimize a floating-point objective using IEEE-total-order linear search."""
    if solver_name != "z3":
        raise ValueError("Floating-point OMT currently supports only the z3 backend")

    fml, _obj_var, obj_bits, obj_key = prepare_fp_objective_with_key(
        z3_fml, z3_obj, prefix="iter_fp_obj"
    )
    obj_bits = cast(z3.BitVecRef, obj_bits)
    obj_key = cast(z3.BitVecRef, obj_key)
    key_width = cast(int, obj_key.size())
    solver = z3.Solver()
    solver.add(fml)

    best_key = None
    best_value = None
    while solver.check() == z3.sat:
        model = solver.model()
        key_value = model.eval(obj_key, model_completion=True)
        best_key = int(str(key_value))
        best_value = fp_value_from_bits(int(str(model.eval(obj_bits, model_completion=True))), z3_obj.sort())
        if minimize:
            solver.add(z3.ULT(obj_key, z3.BitVecVal(best_key, key_width)))
        else:
            solver.add(z3.UGT(obj_key, z3.BitVecVal(best_key, key_width)))

    logger.info(
        "FP %simization result: %s",
        "min" if minimize else "max",
        best_value,
    )
    return best_value


def fp_opt_with_binary_search(
    z3_fml: z3.ExprRef, z3_obj: z3.ExprRef, minimize: bool, solver_name: str = "z3"
) -> Optional[z3.ExprRef]:
    """Optimize a floating-point objective using binary search over total-order keys."""
    if solver_name != "z3":
        raise ValueError("Floating-point OMT currently supports only the z3 backend")

    fml, _obj_var, obj_bits, obj_key = prepare_fp_objective_with_key(
        z3_fml, z3_obj, prefix="iter_fp_obj"
    )
    obj_bits = cast(z3.BitVecRef, obj_bits)
    obj_key = cast(z3.BitVecRef, obj_key)
    base_solver = z3.Solver()
    base_solver.add(fml)
    if base_solver.check() != z3.sat:
        return None

    width = cast(int, obj_key.size())
    lower = 0
    upper = (1 << width) - 1
    best_key = None

    while lower <= upper:
        mid = lower + ((upper - lower) >> 1)
        base_solver.push()
        bound = z3.BitVecVal(mid, width)
        if minimize:
            base_solver.add(z3.ULE(obj_key, bound))
        else:
            base_solver.add(z3.UGE(obj_key, bound))

        if base_solver.check() == z3.sat:
            key_value = base_solver.model().eval(obj_key, model_completion=True)
            best_key = int(str(key_value))
            if minimize:
                upper = best_key - 1
            else:
                lower = best_key + 1
        else:
            if minimize:
                lower = mid + 1
            else:
                upper = mid - 1
        base_solver.pop()

    if best_key is None:
        return None

    final_solver = z3.Solver()
    final_solver.add(fml)
    final_solver.add(obj_key == z3.BitVecVal(best_key, width))
    if final_solver.check() != z3.sat:
        return None

    result = fp_value_from_bits(
        int(str(final_solver.model().eval(obj_bits, model_completion=True))),
        z3_obj.sort(),
    )
    logger.info(
        "FP %simization result: %s",
        "min" if minimize else "max",
        result,
    )
    return result
