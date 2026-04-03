"""CLI tool for Exists-Forall SMT (EFSMT) solving."""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import z3

from aria.quant.efsmt_parser import EFSMTParser, EFSMTZ3Parser
from aria.quant.efsmt_solver import simple_cegar_efsmt
from aria.quant.efbv.efbv_parallel.efbv_cegis_parallel import ParallelEFBVSolver
from aria.quant.efbv.efbv_parallel.efbv_utils import EFBVResult
from aria.quant.efbv.efbv_seq.efbv_solver import EFBVSequentialSolver
from aria.quant.eflira.eflira_parallel import ParallelEFLIRASolver, EFLIRAResult
from aria.quant.eflira.eflira_seq import solve_with_simple_cegar as lira_cegar
from aria.quant.eflira.eflira_seq import solve_with_z3 as lira_z3
from aria.utils.z3_expr_utils import get_variables, get_z3_logic


def _parse_efsmt_file(
    filename: str, parser_name: str
) -> Tuple[List[z3.ExprRef], List[z3.ExprRef], z3.ExprRef]:
    if parser_name == "sexpr":
        parser = EFSMTParser()
        return parser.parse_smt2_file(filename)
    parser = EFSMTZ3Parser()
    return parser.parse_smt2_file(filename)


def _infer_theory(
    exists_vars: List[z3.ExprRef], forall_vars: List[z3.ExprRef], phi: z3.ExprRef
) -> str:
    vars_list = exists_vars + forall_vars
    if not vars_list:
        vars_list = get_variables(phi)

    has_bv = any(v.sort().kind() == z3.Z3_BV_SORT for v in vars_list)
    has_int = any(v.sort().kind() == z3.Z3_INT_SORT for v in vars_list)
    has_real = any(v.sort().kind() == z3.Z3_REAL_SORT for v in vars_list)
    if has_bv:
        return "bv"
    if has_int or has_real:
        return "lira"
    return "bool"


def _z3_check(
    forall_vars: List[z3.ExprRef], phi: z3.ExprRef, timeout: Optional[int]
) -> z3.CheckSatResult:
    solver = z3.Solver()
    if timeout is not None:
        solver.set("timeout", timeout * 1000)
    solver.add(z3.ForAll(forall_vars, phi))
    return solver.check()


def _format_result(result) -> str:
    if isinstance(result, z3.CheckSatResult):
        if result == z3.sat:
            return "sat"
        if result == z3.unsat:
            return "unsat"
        return "unknown"
    if isinstance(result, EFBVResult):
        if result == EFBVResult.SAT:
            return "sat"
        if result == EFBVResult.UNSAT:
            return "unsat"
        return "unknown"
    if isinstance(result, EFLIRAResult):
        if result == EFLIRAResult.SAT:
            return "sat"
        if result == EFLIRAResult.UNSAT:
            return "unsat"
        return "unknown"
    if isinstance(result, str):
        if result in ("sat", "unsat", "unknown"):
            return result
    return "unknown"


def _solve_bool(
    engine: str,
    forall_vars: List[z3.ExprRef],
    phi: z3.ExprRef,
    timeout: Optional[int],
    max_loops: Optional[int],
) -> str:
    if engine in ("auto", "z3"):
        return _format_result(_z3_check(forall_vars, phi, timeout))
    if engine == "cegar":
        logic = "QF_BOOL"
        return _format_result(simple_cegar_efsmt(logic, forall_vars, phi, max_loops))
    raise ValueError(f"Unsupported engine for bool: {engine}")


def _solve_bv(
    engine: str,
    exists_vars: List[z3.ExprRef],
    forall_vars: List[z3.ExprRef],
    phi: z3.ExprRef,
    bv_solver: str,
) -> str:
    if engine in ("auto", "efbv-par"):
        solver = ParallelEFBVSolver(mode="canary")
        return _format_result(solver.solve_efsmt_bv(exists_vars, forall_vars, phi))
    if engine == "efbv-seq":
        solver = EFBVSequentialSolver("BV", solver=bv_solver)
        solver.init(exists_vars, forall_vars, phi)
        return _format_result(solver.solve())
    raise ValueError(f"Unsupported engine for bv: {engine}")


def _solve_lira(
    engine: str,
    exists_vars: List[z3.ExprRef],
    forall_vars: List[z3.ExprRef],
    phi: z3.ExprRef,
    timeout: Optional[int],
    max_loops: Optional[int],
    forall_solver: str,
) -> str:
    if engine in ("auto", "eflira-par"):
        solver = ParallelEFLIRASolver(
            mode="cegis",
            bin_solver_name=forall_solver,
        )
        return _format_result(
            solver.solve_efsmt_lira(exists_vars, forall_vars, phi)
        )
    if engine == "eflira-seq":
        return _format_result(lira_cegar(exists_vars, forall_vars, phi, max_loops))
    if engine == "z3":
        return _format_result(_z3_check(forall_vars, phi, timeout))
    if engine == "cegar":
        return _format_result(lira_cegar(exists_vars, forall_vars, phi, max_loops))
    raise ValueError(f"Unsupported engine for lira: {engine}")


def main() -> int:
    """Main entry point for EFSMT CLI."""
    parser = argparse.ArgumentParser(
        description="Solve Exists-Forall SMT (EFSMT) problems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("file", type=str, help="EFSMT SMT-LIB2 file (.smt2)")
    parser.add_argument(
        "--parser",
        choices=["z3", "sexpr"],
        default="z3",
        help="Parsing backend (default: z3)",
    )
    parser.add_argument(
        "--theory",
        choices=["auto", "bool", "bv", "lira"],
        default="auto",
        help="Theory selection (default: auto)",
    )
    parser.add_argument(
        "--engine",
        choices=[
            "auto",
            "z3",
            "cegar",
            "efbv-par",
            "efbv-seq",
            "eflira-par",
            "eflira-seq",
        ],
        default="auto",
        help="Solver engine (default: auto)",
    )
    parser.add_argument(
        "--bv-solver",
        type=str,
        default="z3",
        help="Backend for efbv-seq (default: z3)",
    )
    parser.add_argument(
        "--forall-solver",
        type=str,
        default="z3",
        help="Binary solver for eflira-par (default: z3)",
    )
    parser.add_argument(
        "--max-loops",
        type=int,
        help="Max iterations for CEGAR-based engines",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Timeout in seconds for Z3-based checks",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()
    if not Path(args.file).exists():
        print(f"Error: File not found: {args.file}", file=sys.stderr)
        return 1

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        exists_vars, forall_vars, phi = _parse_efsmt_file(args.file, args.parser)
        theory = args.theory
        if theory == "auto":
            theory = _infer_theory(exists_vars, forall_vars, phi)
        engine = args.engine
        if engine == "auto":
            if theory == "bv":
                engine = "efbv-par"
            elif theory == "lira":
                engine = "eflira-par"
            else:
                engine = "z3"

        logging.info("Detected theory: %s (logic: %s)", theory, get_z3_logic(phi))
        logging.info("Using engine: %s", engine)

        if theory == "bool":
            result = _solve_bool(
                engine,
                forall_vars,
                phi,
                timeout=args.timeout,
                max_loops=args.max_loops,
            )
        elif theory == "bv":
            result = _solve_bv(
                engine, exists_vars, forall_vars, phi, bv_solver=args.bv_solver
            )
        elif theory == "lira":
            result = _solve_lira(
                engine,
                exists_vars,
                forall_vars,
                phi,
                timeout=args.timeout,
                max_loops=args.max_loops,
                forall_solver=args.forall_solver,
            )
        else:
            raise ValueError(f"Unsupported theory: {theory}")

        print(result)
        return 0
    except (ValueError, IOError, OSError, z3.Z3Exception) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if args.log_level == "DEBUG":
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
