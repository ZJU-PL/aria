"""Exact OMT(QF_FP) via quantified SMT and IEEE-754 total ordering."""

import itertools
import logging
from typing import List, Optional, Sequence, Tuple, cast

import z3

logger = logging.getLogger(__name__)

_FRESH_ID = itertools.count()


def fresh_fp_const(sort: z3.FPSortRef, prefix: str = "fp_obj") -> z3.ExprRef:
    """Create a fresh FP constant of the given sort."""
    return cast(z3.FPRef, z3.Const(f"__aria_{prefix}_{next(_FRESH_ID)}", sort))


def fp_total_key(fp_expr: z3.ExprRef) -> z3.BitVecRef:
    """Encode IEEE-754 totalOrder as an unsigned bit-vector key.

    For interchange formats, flipping the sign bit for non-negative encodings and
    complementing all bits for negative encodings yields a key whose unsigned order
    matches IEEE-754 `totalOrder`, including signed zeros, infinities, and NaNs.
    """
    bits = cast(z3.BitVecRef, z3.fpToIEEEBV(fp_expr))
    return fp_total_key_from_bits(bits)


def fp_total_key_from_bits(bits: z3.BitVecRef) -> z3.BitVecRef:
    """Encode totalOrder directly from IEEE-754 bit-vector representations."""
    bits_bv = cast(z3.BitVecRef, bits)
    width = bits_bv.size()
    sign = z3.Extract(width - 1, width - 1, bits_bv)
    sign_mask = z3.BitVecVal(1 << (width - 1), width)
    return cast(
        z3.BitVecRef,
        z3.If(sign == z3.BitVecVal(1, 1), ~bits_bv, bits_bv ^ sign_mask),
    )


def fp_total_lt(left: z3.ExprRef, right: z3.ExprRef) -> z3.ExprRef:
    """IEEE-754 totalOrder strict comparison."""
    return z3.ULT(fp_total_key(left), fp_total_key(right))


def fp_total_le(left: z3.ExprRef, right: z3.ExprRef) -> z3.ExprRef:
    """IEEE-754 totalOrder non-strict comparison."""
    return z3.ULE(fp_total_key(left), fp_total_key(right))


def fp_value_from_bits(bits: int, sort: z3.FPSortRef) -> z3.ExprRef:
    """Construct an exact FP constant from its IEEE-754 bit pattern."""
    width = sort.ebits() + sort.sbits()
    return z3.fpBVToFP(z3.BitVecVal(bits, width), sort)


def fp_value_from_bits_expr(bits: z3.BitVecRef, sort: z3.FPSortRef) -> z3.ExprRef:
    """Construct an FP term from a bit-vector expression."""
    return z3.fpBVToFP(bits, sort)


def fp_model_value(model: z3.ModelRef, fp_expr: z3.ExprRef) -> z3.ExprRef:
    """Extract an exact FP value from a model, preserving NaN payload/sign."""
    bits = model.eval(z3.fpToIEEEBV(fp_expr), model_completion=True)
    return fp_value_from_bits(int(str(bits)), cast(z3.FPSortRef, fp_expr.sort()))


def fp_value_bits(value: z3.ExprRef) -> int:
    """Extract the exact IEEE-754 bit pattern from an exact FP value term."""
    if value.num_args() == 1:
        return int(str(value.arg(0)))

    probe = z3.FP("__aria_fp_bits_probe", cast(z3.FPSortRef, value.sort()))
    solver = z3.Solver()
    solver.add(z3.fpToIEEEBV(probe) == z3.fpToIEEEBV(value))
    if solver.check() != z3.sat:
        raise ValueError("Unable to extract floating-point bit pattern")
    return int(str(solver.model().eval(z3.fpToIEEEBV(probe), model_completion=True)))


def format_fp_value(value: z3.ExprRef) -> str:
    """Render an exact FP value with both readable and exact bit forms."""
    sort = cast(z3.FPSortRef, value.sort())
    width = sort.ebits() + sort.sbits()
    hex_width = (width + 3) // 4
    return f"{z3.simplify(value)} [bits=0x{fp_value_bits(value):0{hex_width}x}]"


def format_fp_values(values: Sequence[Optional[z3.ExprRef]]) -> str:
    """Render a list of exact FP values."""
    rendered = ["None" if value is None else format_fp_value(value) for value in values]
    return "[" + ", ".join(rendered) + "]"


def prepare_fp_objective(
    z3_fml: z3.ExprRef, z3_obj: z3.ExprRef, prefix: str = "fp_obj"
) -> Tuple[z3.ExprRef, z3.ExprRef]:
    """Ensure the objective is represented by a named FP variable."""
    if z3_obj.sort_kind() != z3.Z3_FLOATING_POINT_SORT:
        raise ValueError("Expected a floating-point objective")

    if z3.is_const(z3_obj) and z3_obj.decl().arity() == 0:
        return z3_fml, z3_obj

    obj_var = fresh_fp_const(cast(z3.FPSortRef, z3_obj.sort()), prefix=prefix)
    return cast(z3.ExprRef, z3.And(z3_fml, obj_var == z3_obj)), obj_var


def prepare_fp_objective_with_key(
    z3_fml: z3.ExprRef, z3_obj: z3.ExprRef, prefix: str = "fp_obj"
) -> Tuple[z3.ExprRef, z3.ExprRef, z3.BitVecRef, z3.BitVecRef]:
    """Ensure the objective has both named FP and total-order-key variables."""
    fml, obj_var = prepare_fp_objective(z3_fml, z3_obj, prefix=prefix)
    obj_sort = cast(z3.FPSortRef, obj_var.sort())
    key_width = obj_sort.ebits() + obj_sort.sbits()
    bits_var = z3.BitVec(f"__aria_{prefix}_bits_{next(_FRESH_ID)}", key_width)
    key_var = z3.BitVec(f"__aria_{prefix}_key_{next(_FRESH_ID)}", key_width)
    return (
        cast(
            z3.ExprRef,
            z3.And(
                fml,
                obj_var == fp_value_from_bits_expr(cast(z3.BitVecRef, bits_var), obj_sort),
                key_var == fp_total_key_from_bits(bits_var),
            ),
        ),
        obj_var,
        cast(z3.BitVecRef, bits_var),
        cast(z3.BitVecRef, key_var),
    )


def pin_fp_value(fp_var: z3.ExprRef, value: z3.ExprRef) -> z3.ExprRef:
    """Constrain an FP term to an exact IEEE-754 value, preserving NaNs and zeros."""
    sort = cast(z3.FPSortRef, fp_var.sort())
    width = sort.ebits() + sort.sbits()
    return cast(z3.BoolRef, z3.fpToIEEEBV(fp_var) == z3.BitVecVal(fp_value_bits(value), width))


def _build_optimality_formula(
    z3_fml: z3.ExprRef, obj_var: z3.ExprRef, minimize: bool
) -> z3.ExprRef:
    obj_misc = fresh_fp_const(cast(z3.FPSortRef, obj_var.sort()), prefix="fp_misc")
    new_fml = cast(z3.ExprRef, z3.substitute(z3_fml, (obj_var, obj_misc)))
    if minimize:
        return cast(
            z3.ExprRef,
            z3.And(
                z3_fml,
                z3.ForAll([obj_misc], z3.Implies(new_fml, fp_total_le(obj_var, obj_misc))),
            ),
        )
    return cast(
        z3.ExprRef,
        z3.And(
            z3_fml,
            z3.ForAll([obj_misc], z3.Implies(new_fml, fp_total_le(obj_misc, obj_var))),
        ),
    )


def _has_better_model(
    z3_fml: z3.ExprRef, obj_var: z3.ExprRef, candidate: z3.ExprRef, minimize: bool
) -> bool:
    """Check whether a candidate can be improved under IEEE totalOrder."""
    solver = z3.Solver()
    solver.add(z3_fml)
    if minimize:
        solver.add(fp_total_lt(obj_var, candidate))
    else:
        solver.add(fp_total_lt(candidate, obj_var))
    return solver.check() == z3.sat


def fp_opt_with_qsmt(
    z3_fml: z3.ExprRef, z3_obj: z3.ExprRef, minimize: bool, solver_name: str = "z3"
) -> Optional[z3.ExprRef]:
    """Solve single-objective OMT(QF_FP) exactly via quantified SMT.

    The objective order is IEEE-754 `totalOrder`, not the partial numeric order
    induced by `fp.lt`/`fp.leq`.
    """
    if solver_name != "z3":
        raise ValueError("Exact OMT(QF_FP) currently supports only the z3 backend")

    fml, obj_var = prepare_fp_objective(z3_fml, z3_obj, prefix="qsmt_fp_obj")
    base_solver = z3.Solver()
    base_solver.add(fml)
    if base_solver.check() != z3.sat:
        return None

    solver = z3.Solver()
    solver.add(_build_optimality_formula(fml, obj_var, minimize))
    check_result = solver.check()
    if check_result == z3.sat:
        result = fp_model_value(solver.model(), obj_var)
        if not _has_better_model(fml, obj_var, result, minimize):
            logger.info(
                "QSMT FP %simization result: %s",
                "min" if minimize else "max",
                result,
            )
            return result
        logger.warning("QSMT candidate failed optimality check, using exact fallback")
    else:
        logger.warning(
            "QSMT solver returned %s, using exact fallback",
            check_result,
        )

    from aria.optimization.omtfp.fp_opt_iterative_search import fp_opt_with_binary_search

    return fp_opt_with_binary_search(fml, obj_var, minimize=minimize, solver_name="z3")


def fp_optimize_boxed(
    z3_fml: z3.ExprRef,
    objectives: Sequence[z3.ExprRef],
    directions: Sequence[str],
    engine: str,
    solver_name: str,
) -> List[Optional[z3.ExprRef]]:
    """Optimize each objective independently under boxed semantics."""
    results: List[Optional[z3.ExprRef]] = []
    for objective, direction in zip(objectives, directions):
        results.append(
            solve_fp_objective(
                z3_fml,
                objective,
                minimize=direction == "min",
                engine=engine,
                solver_name=solver_name,
            )
        )
    return results


def fp_optimize_lex(
    z3_fml: z3.ExprRef,
    objectives: Sequence[z3.ExprRef],
    directions: Sequence[str],
    engine: str,
    solver_name: str,
) -> List[Optional[z3.ExprRef]]:
    """Optimize objectives lexicographically under IEEE totalOrder."""
    current_fml = z3_fml
    results: List[Optional[z3.ExprRef]] = []

    for index, (objective, direction) in enumerate(zip(objectives, directions)):
        current_fml, obj_var = prepare_fp_objective(
            current_fml, objective, prefix=f"lex_fp_obj_{index}"
        )
        result = solve_fp_objective(
            current_fml,
            obj_var,
            minimize=direction == "min",
            engine=engine,
            solver_name=solver_name,
        )
        results.append(result)
        if result is None:
            break
        current_fml = cast(z3.ExprRef, z3.And(current_fml, pin_fp_value(obj_var, result)))

    return results


def solve_fp_objective(
    z3_fml: z3.ExprRef,
    z3_obj: z3.ExprRef,
    minimize: bool,
    engine: str,
    solver_name: str,
) -> Optional[z3.ExprRef]:
    """Dispatch single-objective OMT(QF_FP) solving."""
    if engine == "iter":
        from aria.optimization.omtfp.fp_opt_iterative_search import (
            fp_opt_with_binary_search,
            fp_opt_with_linear_search,
        )

        search_type = solver_name.split("-")[-1]
        backend = solver_name.split("-")[0]
        if search_type == "ls":
            return fp_opt_with_linear_search(z3_fml, z3_obj, minimize, backend)
        if search_type == "bs":
            return fp_opt_with_binary_search(z3_fml, z3_obj, minimize, backend)
        raise ValueError(f"Unsupported FP iterative solver configuration: {solver_name}")
    if engine == "qsmt":
        return fp_opt_with_qsmt(z3_fml, z3_obj, minimize, solver_name)
    if engine == "maxsat":
        raise ValueError("OMT(QF_FP) does not support MaxSAT reduction")
    if engine == "z3py":
        raise ValueError("z3 Optimize does not support floating-point objectives")
    raise ValueError(f"Unsupported FP optimization engine: {engine}")
