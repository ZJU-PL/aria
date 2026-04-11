"""
Cmd line interface for solving OMT(BV) problems with different solvers.
"""

import argparse
import logging
from typing import Any, Optional, cast

import z3

from aria.optimization.result import OptimizationResult, OptimizationStatus
from aria.optimization.omtfp.fp_omt_parser import FPOMTParser
from aria.optimization.omtfp.fp_opt_qsmt import (
    format_fp_value,
    format_fp_frontier,
    format_fp_values,
    fp_optimize_boxed,
    fp_optimize_lex,
    fp_optimize_pareto,
    solve_fp_objective,
)
from aria.optimization.omtbv.bv_opt_iterative_search import (
    bv_opt_with_linear_search,
    bv_opt_with_binary_search,
)
from aria.optimization.omtbv.bv_opt_maxsat import bv_opt_with_maxsat
from aria.optimization.omtbv.bv_opt_qsmt import bv_opt_with_qsmt
from aria.optimization.omt_parser import OMTParser


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


def _solve_fp_opt_file(
    filename: str, engine: str, solver_name: str, opt_priority: str
) -> OptimizationResult:
    """Solve OMT(QF_FP) instances with IEEE-754 faithful semantics."""
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


def solve_opt_file_result(
    filename: str, engine: str, solver_name: str, opt_priority: str = "box"
) -> OptimizationResult:
    """Interface for solving single-objective optimization problems.

    Args:
        filename: Path to the OMT problem file
        engine: Optimization engine to use
        solver_name: Name of the solver to use

    Note:
        The OMTParser converts all objectives to "maximize" internally.
    """
    logger = logging.getLogger(__name__)

    try:
        s = OMTParser()
        s.parse_with_z3(filename, is_file=True)
        fml = cast(z3.ExprRef, z3.And(s.assertions))
        obj = s.objective
    except z3.Z3Exception as ex:
        if "Objective must be bit-vector, integer or real" not in str(ex):
            raise
        return _solve_fp_opt_file(filename, engine, solver_name, opt_priority)

    if obj is None:
        raise ValueError("Expected a single optimization objective")

    if engine == "iter":
        solver_type = solver_name.split("-")[0]
        search_type = solver_name.split("-")[-1]
        if search_type == "ls":
            lin_res = bv_opt_with_linear_search(
                fml, obj, minimize=False, solver_name=solver_type
            )
            logger.info("Linear search result: %s", lin_res)
            return _result_with_value(lin_res, engine=engine, solver_name=solver_name)
        if search_type == "bs":
            bin_res = bv_opt_with_binary_search(
                fml, obj, minimize=False, solver_name=solver_type
            )
            logger.info("Binary search result: %s", bin_res)
            return _result_with_value(bin_res, engine=engine, solver_name=solver_name)
        raise ValueError(
            f"Unsupported iterative search strategy '{search_type}' "
            f"from solver '{solver_name}'"
        )
    if engine == "maxsat":
        maxsat_res = bv_opt_with_maxsat(
            fml, obj, minimize=False, solver_name=solver_name
        )
        logger.info("MaxSAT result: %s", maxsat_res)
        if maxsat_res is None:
            raise OptimizationResultError(
                f"MaxSAT optimization produced no result for solver '{solver_name}'"
            )
        return _result_with_value(maxsat_res, engine=engine, solver_name=solver_name)
    if engine == "qsmt":
        qsmt_res = bv_opt_with_qsmt(fml, obj, minimize=False, solver_name=solver_name)
        logger.info("QSMT result: %s", qsmt_res)
        if not qsmt_res.strip():
            raise OptimizationResultError(
                f"QSMT optimization produced empty output for solver '{solver_name}'"
            )
        return _result_with_value(qsmt_res, engine=engine, solver_name=solver_name)
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
        else:
            print("No solution")
            return OptimizationResult(
                status=OptimizationStatus.UNSAT,
                engine=engine,
                solver=solver_name,
                detail="z3py reported no solution",
            )

    raise ValueError(f"Unsupported optimization engine: {engine}")


def solve_opt_file(
    filename: str, engine: str, solver_name: str, opt_priority: str = "box"
) -> Optional[str]:
    """Compatibility wrapper returning the legacy string-or-None result."""
    result = solve_opt_file_result(filename, engine, solver_name, opt_priority)
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
        help="Choose the engine to use",
    )

    # Create argument groups for each engine

    # for single-objective optimization
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

    # for single-objective optimization
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

    # for single-objective optimization
    iter_group = parser.add_argument_group(
        "iter", "Arguments for the iterative search-based engine"
    )
    iter_group.add_argument(
        "--solver-iter",
        type=str,
        default="z3-ls",
        choices=[i + "-ls" for i in ["z3", "cvc5", "yices", "msat", "btor"]]
        + [i + "-bs" for i in ["z3", "cvc5", "yices", "msat", "btor"]],
        help="Choose the quantifier-free SMT solver to use. ls - linear search,"
        " bs - binary search",
    )

    # Optimization General Options
    opt_general_group = parser.add_argument_group("Optimization General Options")

    # Set the priority of objectives in multi-objective optimization
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

    # Optimization Boxed-Search Options
    opt_box_group = parser.add_argument_group("Optimization Boxed-Search Options")

    opt_box_group.add_argument(
        "--opt-box-engine",
        type=str,
        default="seq",
        choices=["seq", "compact", "par"],
        help="Optimize objectives in sequence (default: seq)."
        "compact - compact optimization (OOPSLA'21), "
        "par - parallel optimization",
    )

    opt_box_group.add_argument(
        "--opt-box-shuffle",
        action="store_false",
        help="Optimize objectives in random order (default: false)",
    )

    # Optimization Theory Options (mainly for QF_BV and QF_LIA)
    opt_theory_group = parser.add_argument_group("Optimization Theory Options")
    opt_theory_group.add_argument(
        "--opt-theory-bv-engine",
        type=str,
        default="qsmt",
        choices=["qsmt", "maxsat", "iter"],
    )

    opt_theory_group.add_argument(
        "--opt-theory-int-engine", type=str, default="qsmt", choices=["qsmt", "iter"]
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

    # Configure logging with format
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Ensure the correct solver is used based on the selected engine
    if args.engine == "qsmt":
        solver = args.solver_qsmt
    elif args.engine == "maxsat":
        solver = args.solver_maxsat
    elif args.engine == "iter":
        solver = args.solver_iter
    elif args.engine == "z3py":
        solver = "z3py"
    else:
        raise ValueError("Invalid engine specified")

    solve_opt_file(args.filename, args.engine, solver, args.opt_priority)


if __name__ == "__main__":
    main()
