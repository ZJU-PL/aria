"""
Cmd line interface for solving OMT problems with different backends.
"""

import argparse
import logging
import random
import re
from fractions import Fraction
from typing import Any, Callable, List, Optional, Sequence, cast

import z3
from z3.z3consts import Z3_OP_BNEG, Z3_OP_UMINUS

from aria.pyomt.omtarith.arith_opt_ls import arith_opt_with_ls
from aria.pyomt.omtarith.arith_opt_qsmt import arith_opt_with_qsmt
from aria.pyomt.omtbv.boxed.bv_boxed_compact import (
    get_input as get_compact_box_input,
    map_bitvector,
    res_2int,
    solve as solve_compact_boxed,
)
from aria.pyomt.omtbv.boxed.bv_boxed_obj_divide import solve_boxed_parallel
from aria.pyomt.omtbv.bv_opt_iterative_search import (
    bv_opt_with_binary_search,
    bv_opt_with_linear_search,
)
from aria.pyomt.omtbv.bv_opt_maxsat import bv_opt_with_maxsat
from aria.pyomt.omtbv.bv_opt_qsmt import bv_opt_with_qsmt
from aria.pyomt.omtfp.fp_omt_parser import FPOMTParser
from aria.pyomt.omtfp.fp_opt_multiobj import (
    fp_optimize_boxed,
    fp_optimize_lex,
    fp_optimize_pareto,
    solve_fp_objective,
)
from aria.pyomt.omtfp.fp_opt_utils import (
    format_fp_frontier,
    format_fp_value,
    format_fp_values,
)
from aria.pyomt.result import OptimizationResult, OptimizationStatus


class OptimizationResultError(RuntimeError):
    """Raised when an optimization backend returns no usable result."""


def _result_with_value(
    value: Any,
    engine: str,
    solver_name: str,
    detail: Optional[str] = None,
) -> OptimizationResult:
    """Build a normalized optimal result."""
    return OptimizationResult(
        status=OptimizationStatus.OPTIMAL,
        value=value,
        engine=engine,
        solver=solver_name,
        detail=detail,
    )


def _detect_objectives(opt: z3.Optimize) -> tuple[List[z3.ExprRef], List[str]]:
    """Extract original objective expressions and directions from Z3 Optimize."""
    objectives: List[z3.ExprRef] = []
    directions: List[str] = []
    for obj in opt.objectives():
        if obj.decl().kind() in (Z3_OP_UMINUS, Z3_OP_BNEG):
            directions.append("max")
            objectives.append(obj.children()[0])
        else:
            directions.append("min")
            objectives.append(obj)
    return objectives, directions


def _is_bv_expr(expr: z3.ExprRef) -> bool:
    return expr.sort_kind() == z3.Z3_BV_SORT


def _is_fp_expr(expr: z3.ExprRef) -> bool:
    return expr.sort_kind() == z3.Z3_FLOATING_POINT_SORT


def _is_arith_expr(expr: z3.ExprRef) -> bool:
    return z3.is_int(expr) or z3.is_real(expr)


def _normalize_z3_value(value: Any) -> Any:
    """Convert Z3 numerals to plain Python values when possible."""
    if hasattr(value, "as_long"):
        try:
            return value.as_long()
        except z3.Z3Exception:
            pass
    if hasattr(value, "as_fraction"):
        try:
            frac = value.as_fraction()
            if frac.denominator == 1:
                return frac.numerator
            return float(frac)
        except (AttributeError, ValueError, ZeroDivisionError):
            pass
    if hasattr(value, "as_decimal"):
        try:
            return float(value.as_decimal(16).rstrip("?"))
        except ValueError:
            pass
    return value


def _parse_solver_output_value(result: str) -> Any:
    """Parse a scalar value from `(get-value ...)` style solver output."""
    if not result or "unsat" in result.lower():
        return None

    match = re.search(r"\(\([^\s)]+\s+(.+?)\)\)", result, flags=re.DOTALL)
    if not match:
        stripped = result.strip()
        try:
            return int(stripped)
        except ValueError:
            try:
                return float(stripped)
            except ValueError:
                return stripped

    value_str = match.group(1).strip()
    if value_str.startswith("#x"):
        return int(value_str[2:], 16)
    if value_str.startswith("#b"):
        return int(value_str[2:], 2)

    frac_match = re.fullmatch(r"\(/\s*(-?\d+)\s+(\d+)\)", value_str)
    if frac_match:
        frac = Fraction(int(frac_match.group(1)), int(frac_match.group(2)))
        return frac.numerator if frac.denominator == 1 else float(frac)

    neg_match = re.fullmatch(r"\(-\s+(.+)\)", value_str)
    if neg_match:
        inner = _parse_solver_output_value(f"sat\n((tmp {neg_match.group(1)}))")
        if isinstance(inner, (int, float)):
            return -inner
        return value_str

    try:
        return int(value_str)
    except ValueError:
        try:
            return float(value_str)
        except ValueError:
            return value_str


def _maybe_parse_backend_value(value: Any) -> Any:
    """Normalize backend results to Python scalars when possible."""
    if isinstance(value, str):
        parsed = _parse_solver_output_value(value)
        return value if parsed is None else parsed
    return _normalize_z3_value(value)


def _solve_fp_opt_file(
    filename: str, engine: str, solver_name: str, opt_priority: str
) -> OptimizationResult:
    """Solve OMT(QF_FP) instances with the OFPBS floating-point semantics."""
    logger = logging.getLogger(__name__)

    parser = FPOMTParser()
    parser.parse_with_z3(filename, is_file=True)
    fml = cast(z3.ExprRef, z3.And(parser.assertions))

    if len(parser.objectives) == 1:
        result = solve_fp_objective(
            fml,
            parser.objectives[0],
            minimize=parser.original_directions[0] == "min",
            engine=engine,
            solver_name=solver_name,
        )
        logger.info("FP optimization result: %s", result)
        if result is None:
            raise OptimizationResultError(
                f"FP optimization returned no result for engine '{engine}' "
                f"and solver '{solver_name}'"
            )
        return _result_with_value(
            format_fp_value(result),
            engine=engine,
            solver_name=solver_name,
            detail="floating-point objective",
        )

    if opt_priority == "box":
        results = fp_optimize_boxed(
            fml, parser.objectives, parser.original_directions, engine, solver_name
        )
        logger.info("FP boxed optimization results: %s", results)
        return _result_with_value(
            format_fp_values(results),
            engine=engine,
            solver_name=solver_name,
            detail="floating-point boxed objectives",
        )
    if opt_priority == "lex":
        results = fp_optimize_lex(
            fml, parser.objectives, parser.original_directions, engine, solver_name
        )
        logger.info("FP lex optimization results: %s", results)
        return _result_with_value(
            format_fp_values(results),
            engine=engine,
            solver_name=solver_name,
            detail="floating-point lexicographic objectives",
        )
    if opt_priority == "par":
        results = fp_optimize_pareto(
            fml, parser.objectives, parser.original_directions, engine, solver_name
        )
        logger.info("FP pareto optimization results: %s", results)
        return _result_with_value(
            format_fp_frontier(results),
            engine=engine,
            solver_name=solver_name,
            detail="floating-point pareto frontier",
        )

    raise ValueError(f"Unsupported FP optimization priority: {opt_priority}")


def _solve_bv_single(
    fml: z3.ExprRef,
    obj: z3.ExprRef,
    engine: str,
    solver_name: str,
    minimize: bool,
) -> Any:
    """Solve a single bit-vector objective."""
    logger = logging.getLogger(__name__)

    if engine == "iter":
        if "-" in solver_name:
            solver_type = solver_name.split("-")[0]
            search_type = solver_name.split("-")[-1]
        else:
            solver_type = solver_name
            search_type = "bs"
        if search_type == "ls":
            result = bv_opt_with_linear_search(
                fml, obj, minimize=minimize, solver_name=solver_type
            )
            logger.info("BV linear-search result: %s", result)
            return result
        if search_type in ("bs", "ofpbs"):
            result = bv_opt_with_binary_search(
                fml, obj, minimize=minimize, solver_name=solver_type
            )
            logger.info("BV binary-search result: %s", result)
            return result
        raise ValueError(
            f"Unsupported iterative BV search strategy '{search_type}' "
            f"from solver '{solver_name}'"
        )
    if engine == "maxsat":
        result = bv_opt_with_maxsat(fml, obj, minimize=minimize, solver_name=solver_name)
        logger.info("BV MaxSAT result: %s", result)
        return result
    if engine == "qsmt":
        result = bv_opt_with_qsmt(fml, obj, minimize=minimize, solver_name=solver_name)
        logger.info("BV QSMT result: %s", result)
        return result
    raise ValueError(f"Unsupported BV optimization engine: {engine}")


def _solve_arith_single(
    fml: z3.ExprRef,
    obj: z3.ExprRef,
    engine: str,
    solver_name: str,
    minimize: bool,
) -> Any:
    """Solve a single arithmetic objective."""
    logger = logging.getLogger(__name__)

    if engine == "qsmt":
        result = arith_opt_with_qsmt(fml, obj, minimize=minimize, solver_name=solver_name)
        logger.info("Arithmetic QSMT result: %s", result)
        return result
    if engine == "iter":
        result = arith_opt_with_ls(fml, obj, minimize=minimize, solver_name=solver_name)
        logger.info("Arithmetic local-search result: %s", result)
        return result
    raise ValueError(f"Unsupported arithmetic optimization engine: {engine}")


def _solve_boxed_independent(
    objectives: Sequence[z3.ExprRef],
    directions: Sequence[str],
    solve_one: Callable[[z3.ExprRef, bool], Any],
    shuffle: bool,
    seed: int,
) -> List[Any]:
    """Solve boxed objectives independently, with optional randomized order."""
    order = list(range(len(objectives)))
    if shuffle:
        random.Random(seed).shuffle(order)

    results: List[Any] = [None] * len(objectives)
    for index in order:
        results[index] = _maybe_parse_backend_value(
            solve_one(objectives[index], directions[index] == "min")
        )
    return results


def _solve_multi_with_z3(
    filename: str, raw_objectives: Sequence[z3.ExprRef], opt_priority: str
) -> List[Any]:
    """Delegate lexicographic and pareto multi-objective solving to Z3."""
    opt = z3.Optimize()
    opt.from_file(filename)
    opt.set(priority=opt_priority)
    if opt.check() != z3.sat:
        raise OptimizationResultError(
            f"Z3 Optimize returned no model for priority '{opt_priority}'"
        )
    model = opt.model()
    return [_normalize_z3_value(model.eval(obj, model_completion=True)) for obj in raw_objectives]


def solve_opt_file_result(
    filename: str,
    engine: str,
    solver_name: str,
    opt_priority: str = "box",
    bv_engine: Optional[str] = None,
    int_engine: Optional[str] = None,
    opt_box_engine: str = "seq",
    opt_box_shuffle: bool = False,
    seed: int = 1,
) -> OptimizationResult:
    """Solve an OMT instance and return a structured result."""
    logger = logging.getLogger(__name__)

    try:
        opt = z3.Optimize()
        opt.from_file(filename)
        raw_objectives, directions = _detect_objectives(opt)
        if not raw_objectives:
            raise ValueError("No objectives found in the supplied formula/file")
        fml = cast(z3.ExprRef, z3.And(opt.assertions()))
    except z3.Z3Exception as ex:
        if "Objective must be bit-vector, integer or real" not in str(ex):
            raise
        return _solve_fp_opt_file(filename, engine, solver_name, opt_priority)

    first_objective = raw_objectives[0]
    if _is_fp_expr(first_objective):
        return _solve_fp_opt_file(filename, engine, solver_name, opt_priority)

    if engine == "z3py":
        opt = z3.Optimize()
        opt.from_file(filename=filename)
        if opt.check() == z3.sat:
            print("Solution found:")
            model = opt.model()
            for decl in model:
                print(f"{decl} = {model[decl]}")
            return OptimizationResult(
                status=OptimizationStatus.OPTIMAL,
                model=model,
                engine=engine,
                solver=solver_name,
                detail="z3py printed model to stdout",
            )
        print("No solution")
        return OptimizationResult(
            status=OptimizationStatus.UNSAT,
            engine=engine,
            solver=solver_name,
            detail="z3py reported no solution",
        )

    if _is_bv_expr(first_objective):
        selected_engine = bv_engine or engine
        if len(raw_objectives) == 1:
            value = _maybe_parse_backend_value(
                _solve_bv_single(
                    fml,
                    first_objective,
                    engine=selected_engine,
                    solver_name=solver_name,
                    minimize=directions[0] == "min",
                )
            )
            return _result_with_value(
                value,
                engine=selected_engine,
                solver_name=solver_name,
                detail="bit-vector objective",
            )

        if opt_priority == "box":
            if opt_box_engine == "compact":
                formu, objec = get_compact_box_input(filename)
                boxed_bits = solve_compact_boxed(formu, map_bitvector(objec))
                values = res_2int(boxed_bits, objec)
            elif opt_box_engine == "par" and len(set(directions)) == 1:
                values = solve_boxed_parallel(
                    fml,
                    list(raw_objectives),
                    minimize=directions[0] == "min",
                    engine=selected_engine,
                    solver_name=solver_name,
                )
            else:
                values = _solve_boxed_independent(
                    raw_objectives,
                    directions,
                    lambda obj, minimize: _solve_bv_single(
                        fml,
                        obj,
                        engine=selected_engine,
                        solver_name=solver_name,
                        minimize=minimize,
                    ),
                    shuffle=opt_box_shuffle,
                    seed=seed,
                )
            logger.info("BV boxed optimization results: %s", values)
            return _result_with_value(
                values,
                engine=selected_engine,
                solver_name=solver_name,
                detail=f"bit-vector boxed objectives via {opt_box_engine}",
            )

        values = _solve_multi_with_z3(filename, raw_objectives, opt_priority)
        logger.info("BV %s optimization results via Z3: %s", opt_priority, values)
        return _result_with_value(
            values,
            engine="z3py",
            solver_name="z3py",
            detail=f"bit-vector {opt_priority} objectives",
        )

    if _is_arith_expr(first_objective):
        if engine == "maxsat" and int_engine not in ("qsmt", "iter"):
            raise ValueError(
                "The maxsat engine only supports bit-vector objectives; "
                "use engine 'qsmt' or 'iter' for arithmetic objectives"
            )
        selected_engine = int_engine or (engine if engine in ("qsmt", "iter") else "qsmt")
        arith_solver = (
            solver_name
            if selected_engine == "qsmt" and engine != "maxsat"
            else "z3"
        )
        if len(raw_objectives) == 1:
            value = _maybe_parse_backend_value(
                _solve_arith_single(
                    fml,
                    first_objective,
                    engine=selected_engine,
                    solver_name=arith_solver,
                    minimize=directions[0] == "min",
                )
            )
            return _result_with_value(
                value,
                engine=selected_engine,
                solver_name=arith_solver,
                detail="arithmetic objective",
            )

        if opt_priority == "box":
            values = _solve_boxed_independent(
                raw_objectives,
                directions,
                lambda obj, minimize: _solve_arith_single(
                    fml,
                    obj,
                    engine=selected_engine,
                    solver_name=arith_solver,
                    minimize=minimize,
                ),
                shuffle=opt_box_shuffle,
                seed=seed,
            )
            logger.info("Arithmetic boxed optimization results: %s", values)
            return _result_with_value(
                values,
                engine=selected_engine,
                solver_name=arith_solver,
                detail="arithmetic boxed objectives",
            )

        values = _solve_multi_with_z3(filename, raw_objectives, opt_priority)
        logger.info("Arithmetic %s optimization results via Z3: %s", opt_priority, values)
        return _result_with_value(
            values,
            engine="z3py",
            solver_name="z3py",
            detail=f"arithmetic {opt_priority} objectives",
        )

    raise ValueError(f"Unsupported optimization engine: {engine}")


def solve_opt_file(
    filename: str,
    engine: str,
    solver_name: str,
    opt_priority: str = "box",
    bv_engine: Optional[str] = None,
    int_engine: Optional[str] = None,
    opt_box_engine: str = "seq",
    opt_box_shuffle: bool = False,
    seed: int = 1,
) -> Optional[str]:
    """Compatibility wrapper returning the legacy string-or-None result."""
    result = solve_opt_file_result(
        filename,
        engine,
        solver_name,
        opt_priority,
        bv_engine=bv_engine,
        int_engine=int_engine,
        opt_box_engine=opt_box_engine,
        opt_box_shuffle=opt_box_shuffle,
        seed=seed,
    )
    if result.value is None:
        return None
    return str(result.value)


def main() -> None:
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(
        description="Solve OMT(BV) problems with different solvers."
    )
    parser.add_argument(
        "filename", type=str, help="The filename of the problem to solve."
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="qsmt",
        choices=["qsmt", "maxsat", "iter", "z3py"],
        help="Legacy default engine to use when no theory-specific engine is set.",
    )

    qsmt_group = parser.add_argument_group(
        "qsmt", "Arguments for the QSMT-based engine"
    )
    qsmt_group.add_argument(
        "--solver-qsmt",
        type=str,
        default="z3",
        choices=["z3", "cvc5", "yices", "msat", "bitwuzla", "q3b"],
        help="Choose the quantified SMT solver to use.",
    )

    maxsat_group = parser.add_argument_group(
        "maxsat", "Arguments for the MaxSAT-based engine"
    )
    maxsat_group.add_argument(
        "--solver-maxsat",
        type=str,
        default="FM",
        choices=["FM", "RC2", "OBV-BS"],
        help="Choose the weighted MaxSAT solver to use",
    )

    iter_group = parser.add_argument_group(
        "iter", "Arguments for the iterative search-based engine"
    )
    iter_group.add_argument(
        "--solver-iter",
        type=str,
        default="z3-ofpbs",
        choices=[i + "-ls" for i in ["z3", "cvc5", "yices", "msat", "btor"]]
        + [i + "-bs" for i in ["z3", "cvc5", "yices", "msat", "btor"]]
        + [i + "-ofpbs" for i in ["z3", "cvc5", "yices", "msat", "btor"]],
        help="Choose the quantifier-free SMT solver to use. ls - linear search,"
        " bs - binary search, ofpbs - objective-fp bitwise search",
    )

    opt_general_group = parser.add_argument_group("Optimization General Options")
    opt_general_group.add_argument(
        "--opt-priority",
        type=str,
        default="box",
        choices=["box", "lex", "par"],
        help="Multi-objective combination method: "
        "box - boxed/multi-independent optimization (default), "
        "lex - lexicographic optimization, follows input order, "
        "par - pareto optimization",
    )

    opt_box_group = parser.add_argument_group("Optimization Boxed-Search Options")
    opt_box_group.add_argument(
        "--opt-box-engine",
        type=str,
        default="seq",
        choices=["seq", "compact", "par"],
        help="Boxed BV strategy: seq - solve each objective independently, "
        "compact - compact optimization (OOPSLA'21), par - parallel optimization",
    )
    opt_box_group.add_argument(
        "--opt-box-shuffle",
        action="store_true",
        help="Optimize boxed objectives in random order (default: false)",
    )

    opt_theory_group = parser.add_argument_group("Optimization Theory Options")
    opt_theory_group.add_argument(
        "--opt-theory-bv-engine",
        type=str,
        default=None,
        choices=["qsmt", "maxsat", "iter"],
    )
    opt_theory_group.add_argument(
        "--opt-theory-int-engine",
        type=str,
        default=None,
        choices=["qsmt", "iter"],
    )

    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if args.opt_theory_bv_engine == "maxsat":
        solver = args.solver_maxsat
    elif args.opt_theory_bv_engine == "iter":
        solver = args.solver_iter
    elif args.opt_theory_bv_engine == "qsmt":
        solver = args.solver_qsmt
    elif args.opt_theory_int_engine == "qsmt":
        solver = args.solver_qsmt
    elif args.opt_theory_int_engine == "iter":
        solver = "z3"
    elif args.engine == "qsmt":
        solver = args.solver_qsmt
    elif args.engine == "maxsat":
        solver = args.solver_maxsat
    elif args.engine == "iter":
        solver = args.solver_iter
    elif args.engine == "z3py":
        solver = "z3py"
    else:
        raise ValueError("Invalid engine specified")

    solve_opt_file(
        args.filename,
        args.engine,
        solver,
        args.opt_priority,
        bv_engine=args.opt_theory_bv_engine,
        int_engine=args.opt_theory_int_engine,
        opt_box_engine=args.opt_box_engine,
        opt_box_shuffle=args.opt_box_shuffle,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
