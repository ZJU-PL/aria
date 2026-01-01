"""
Cmd line interface for solving OMT(BV) problems with different solvers.
"""

import argparse
import logging

import z3

from aria.optimization.omtbv.bv_opt_iterative_search import (
    bv_opt_with_linear_search,
    bv_opt_with_binary_search,
)
from aria.optimization.omtbv.bv_opt_maxsat import bv_opt_with_maxsat
from aria.optimization.omtbv.bv_opt_qsmt import bv_opt_with_qsmt
from aria.optimization.omt_parser import OMTParser


def solve_opt_file(filename: str, engine: str, solver_name: str) -> None:
    """Interface for solving single-objective optimization problems.

    Args:
        filename: Path to the OMT problem file
        engine: Optimization engine to use
        solver_name: Name of the solver to use

    Currently, the OMTParser will convert all objectives to the "maximal objecives"
    """
    logger = logging.getLogger(__name__)  # Move logger here

    s = OMTParser()
    s.parse_with_z3(filename, is_file=True)
    # print(s.objectives)
    fml = z3.And(s.assertions)
    obj = s.objective
    # print(fml, obj)
    # 1. use z3 OPT
    # z3_res = optimize_as_long(fml, obj)
    # print("z3 res: ", z3_res)
    # print("----------------------------------")

    if engine == "iter":
        solver_type = solver_name.split("-")[0]
        search_type = solver_name.split("-")[-1]
        # 2. use SMT-based linear search
        if search_type == "ls":
            lin_res = bv_opt_with_linear_search(
                fml, obj, minimize=False, solver_name=solver_type
            )
            # print("lin res: ", lin_res)
            logger.info("Linear search result: %s", lin_res)
        elif search_type == "bs":
            # 2. use SMT-based binary search
            bin_res = bv_opt_with_binary_search(
                fml, obj, minimize=False, solver_name=solver_type
            )
            logger.info("Binary search result: %s", bin_res)
    elif engine == "maxsat":
        # 3. use MaxSAT
        maxsat_res = bv_opt_with_maxsat(
            fml, obj, minimize=False, solver_name=solver_name
        )
        logger.info("MaxSAT result: %s", maxsat_res)
    elif engine == "qsmt":
        # 4. use QSMT
        qsmt_res = bv_opt_with_qsmt(fml, obj, minimize=False, solver_name=solver_name)
        logger.info("QSMT result: %s", qsmt_res)
    elif engine == "z3py":
        opt = z3.Optimize()
        opt.from_file(filename=filename)
        # Check and get result
        if opt.check() == z3.sat:
            print("Solution found:")
            model = opt.model()
            # Print all variables in the model
            for decl in model:
                print(f"{decl} = {model[decl]}")
        else:
            print("No solution")

    else:
        logger.warning("No result - invalid engine specified")


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

    solve_opt_file(args.filename, args.engine, solver)


if __name__ == "__main__":
    main()
