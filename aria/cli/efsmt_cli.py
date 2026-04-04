"""CLI tool for Exists-Forall SMT (EFSMT) solving."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, cast

import z3

from aria.quant.efsmt_parser import EFSMTParser, EFSMTZ3Parser
from aria.utils.z3.expr import get_variables, get_z3_logic

if TYPE_CHECKING:
    from aria.quant.efbv.efbv_parallel.efbv_cegis_parallel import ParallelEFBVSolver
    from aria.quant.efbv.efbv_parallel.efbv_utils import EFBVResult
    from aria.quant.efbv.efbv_seq.efbv_solver import EFBVSequentialSolver
    from aria.quant.eflira.eflira_parallel import EFLIRAResult, ParallelEFLIRASolver
    from aria.quant.eflira.eflira_parallel import FSolverMode as EFLIRAForallMode
    from aria.quant.eflira.eflira_sampling_utils import ESolverSampleStrategy


ParallelEFBVSolver: Any = None
EFBVSequentialSolver: Any = None
ParallelEFLIRASolver: Any = None
EFLIRAForallMode: Any = None
ESolverSampleStrategy: Any = None
simple_cegar_efsmt: Any = None
lira_cegar: Any = None


def _get_simple_cegar_solver() -> Any:
    global simple_cegar_efsmt
    if simple_cegar_efsmt is None:
        from aria.quant.efsmt_solver import simple_cegar_efsmt as solver

        simple_cegar_efsmt = solver
    return simple_cegar_efsmt


def _get_parallel_efbv_solver_cls() -> Any:
    global ParallelEFBVSolver
    if ParallelEFBVSolver is None:
        from aria.quant.efbv.efbv_parallel.efbv_cegis_parallel import (
            ParallelEFBVSolver as parallel_efbv_solver,
        )

        ParallelEFBVSolver = parallel_efbv_solver
    return ParallelEFBVSolver


def _get_efbv_sequential_solver_cls() -> Any:
    global EFBVSequentialSolver
    if EFBVSequentialSolver is None:
        from aria.quant.efbv.efbv_seq.efbv_solver import (
            EFBVSequentialSolver as efbv_sequential_solver,
        )

        EFBVSequentialSolver = efbv_sequential_solver
    return EFBVSequentialSolver


def _get_parallel_eflira_solver_cls() -> Any:
    global ParallelEFLIRASolver
    if ParallelEFLIRASolver is None:
        from aria.quant.eflira.eflira_parallel import (
            ParallelEFLIRASolver as parallel_eflira_solver,
        )

        ParallelEFLIRASolver = parallel_eflira_solver
    return ParallelEFLIRASolver


def _get_eflira_forall_mode_enum() -> Any:
    global EFLIRAForallMode
    if EFLIRAForallMode is None:
        from aria.quant.eflira.eflira_parallel import FSolverMode as eflira_forall_mode

        EFLIRAForallMode = eflira_forall_mode
    return EFLIRAForallMode


def _get_eflira_sample_strategy_enum() -> Any:
    global ESolverSampleStrategy
    if ESolverSampleStrategy is None:
        from aria.quant.eflira.eflira_sampling_utils import (
            ESolverSampleStrategy as eflira_sample_strategy,
        )

        ESolverSampleStrategy = eflira_sample_strategy
    return ESolverSampleStrategy


def _get_lira_cegar_solver() -> Any:
    global lira_cegar
    if lira_cegar is None:
        from aria.quant.eflira.eflira_seq import solve_with_simple_cegar as solver

        lira_cegar = solver
    return lira_cegar


def _load_lira_runtime() -> None:
    _get_parallel_eflira_solver_cls()
    _get_eflira_forall_mode_enum()


def _load_bv_runtime() -> None:
    _get_parallel_efbv_solver_cls()
    _get_efbv_sequential_solver_cls()


def _load_lira_runtime_for_solver() -> None:
    if ParallelEFLIRASolver is None or EFLIRAForallMode is None:
        _load_lira_runtime()
    _get_eflira_sample_strategy_enum()
    _get_lira_cegar_solver()


def _parse_efsmt_file(
    filename: str, parser_name: str
) -> Tuple[List[z3.ExprRef], List[z3.ExprRef], z3.ExprRef]:
    if parser_name == "sexpr":
        parser = EFSMTParser()
        exists_vars, forall_vars, phi = parser.parse_smt2_file(filename)
        return exists_vars, forall_vars, cast(z3.ExprRef, phi)
    parser = EFSMTZ3Parser()
    exists_vars, forall_vars, phi = parser.parse_smt2_file(filename)
    return exists_vars, forall_vars, cast(z3.ExprRef, phi)


def _infer_theory(
    exists_vars: Sequence[z3.ExprRef],
    forall_vars: Sequence[z3.ExprRef],
    phi: z3.ExprRef,
) -> str:
    vars_list = list(exists_vars) + list(forall_vars)
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
    result_type = type(result).__name__
    if result_type == "EFBVResult":
        if getattr(result, "name", None) == "SAT":
            return "sat"
        if getattr(result, "name", None) == "UNSAT":
            return "unsat"
        return "unknown"
    if result_type == "EFLIRAResult":
        if getattr(result, "name", None) == "SAT":
            return "sat"
        if getattr(result, "name", None) == "UNSAT":
            return "unsat"
        return "unknown"
    if isinstance(result, str):
        if result in ("sat", "unsat", "unknown"):
            return result
    return "unknown"


def _parse_eflira_forall_mode(mode: Optional[str]) -> Optional[object]:
    if mode is None:
        return None
    forall_mode_enum = _get_eflira_forall_mode_enum()
    mode_map = {
        "sequential": forall_mode_enum.SEQUENTIAL,
        "parallel-thread": forall_mode_enum.PARALLEL_THREAD,
        "parallel-process-ipc": forall_mode_enum.PARALLEL_PROCESS_IPC,
    }
    return mode_map[mode]


def _parse_eflira_sample_strategy(strategy: str) -> object:
    sample_strategy_enum = _get_eflira_sample_strategy_enum()

    strategy_map = {
        "blocking": sample_strategy_enum.BLOCKING,
        "random-seed": sample_strategy_enum.RANDOM_SEED,
        "optimize": sample_strategy_enum.OPTIMIZE,
        "lexicographic": sample_strategy_enum.LEXICOGRAPHIC,
        "jitter": sample_strategy_enum.JITTER,
        "portfolio": sample_strategy_enum.PORTFOLIO,
    }
    return strategy_map[strategy]


def _build_eflira_sample_config(args: argparse.Namespace) -> Dict[str, object]:
    sample_config: Dict[str, object] = {}
    if args.eflira_lex_order:
        sample_config["lex_order"] = [
            item.strip() for item in args.eflira_lex_order.split(",") if item.strip()
        ]
    if args.eflira_optimize_objectives is not None:
        sample_config["optimize_objectives"] = args.eflira_optimize_objectives
    if args.eflira_optimize_max_tries is not None:
        sample_config["optimize_max_tries"] = args.eflira_optimize_max_tries
    if args.eflira_optimize_coeff_low is not None:
        sample_config["optimize_coeff_low"] = args.eflira_optimize_coeff_low
    if args.eflira_optimize_coeff_high is not None:
        sample_config["optimize_coeff_high"] = args.eflira_optimize_coeff_high
    if args.eflira_jitter_int_delta is not None:
        sample_config["jitter_int_delta"] = args.eflira_jitter_int_delta
    if args.eflira_jitter_real_delta is not None:
        sample_config["jitter_real_delta"] = args.eflira_jitter_real_delta
    if args.eflira_jitter_max_tries is not None:
        sample_config["jitter_max_tries"] = args.eflira_jitter_max_tries
    return sample_config


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
        return _format_result(
            _get_simple_cegar_solver()(logic, forall_vars, phi, max_loops)
        )
    raise ValueError(f"Unsupported engine for bool: {engine}")


def _solve_bv(
    engine: str,
    exists_vars: List[z3.ExprRef],
    forall_vars: List[z3.ExprRef],
    phi: z3.ExprRef,
    bv_solver: str,
    bv_pysmt_solver: str,
    max_loops: Optional[int],
    efbv_num_samples: int,
) -> str:
    if engine in ("auto", "efbv-par"):
        solver = _get_parallel_efbv_solver_cls()(
            mode="canary",
            maxloops=max_loops,
            num_samples=efbv_num_samples,
        )
        return _format_result(solver.solve_efsmt_bv(exists_vars, forall_vars, phi))
    if engine == "efbv-seq":
        solver = _get_efbv_sequential_solver_cls()(
            "BV", solver=bv_solver, pysmt_solver=bv_pysmt_solver
        )
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
    eflira_forall_mode: Optional[str],
    eflira_num_workers: int,
    eflira_num_samples: int,
    eflira_sample_strategy: str,
    eflira_sample_max_tries: int,
    eflira_sample_seed_low: int,
    eflira_sample_seed_high: int,
    eflira_sample_config: Dict[str, object],
) -> str:
    if engine in ("auto", "eflira-par"):
        solver = _get_parallel_eflira_solver_cls()(
            mode="cegis",
            forall_mode=_parse_eflira_forall_mode(eflira_forall_mode),
            bin_solver_name=forall_solver,
            num_workers=eflira_num_workers,
            num_samples=eflira_num_samples,
            sample_strategy=_parse_eflira_sample_strategy(eflira_sample_strategy),
            sample_max_tries=eflira_sample_max_tries,
            sample_seed_low=eflira_sample_seed_low,
            sample_seed_high=eflira_sample_seed_high,
            sample_config=eflira_sample_config or None,
        )
        return _format_result(
            solver.solve_efsmt_lira(exists_vars, forall_vars, phi)
        )
    if engine == "eflira-seq":
        return _format_result(
            _get_lira_cegar_solver()(exists_vars, forall_vars, phi, max_loops)
        )
    if engine == "z3":
        return _format_result(_z3_check(forall_vars, phi, timeout))
    if engine == "cegar":
        return _format_result(
            _get_lira_cegar_solver()(exists_vars, forall_vars, phi, max_loops)
        )
    raise ValueError(f"Unsupported engine for lira: {engine}")


def main() -> int:
    """Main entry point for EFSMT CLI."""
    parser = argparse.ArgumentParser(
        description="Solve Exists-Forall SMT (EFSMT) problems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("file", type=str, help="EFSMT SMT-LIB2 file (.smt2)")
    general_group = parser.add_argument_group("general options")
    bv_group = parser.add_argument_group("bit-vector options")
    lira_group = parser.add_argument_group("LIA/LRA options")

    general_group.add_argument(
        "--parser",
        choices=["z3", "sexpr"],
        default="z3",
        help="Parsing backend (default: z3)",
    )
    general_group.add_argument(
        "--theory",
        choices=["auto", "bool", "bv", "lira"],
        default="auto",
        help="Theory selection (default: auto)",
    )
    general_group.add_argument(
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
    bv_group.add_argument(
        "--bv-solver",
        type=str,
        default="z3",
        help=(
            "Backend for efbv-seq; supports z3, cvc5, btor, yices2, mathsat, "
            "bitwuzla, z3qbf, caqe, q3b, z3sat, cegis, and PySAT backends"
        ),
    )
    bv_group.add_argument(
        "--bv-pysmt-solver",
        type=str,
        default="z3",
        help="PySMT backend used when --bv-solver=cegis (default: z3)",
    )
    bv_group.add_argument(
        "--efbv-num-samples",
        type=int,
        default=5,
        help="Number of existential samples per efbv-par iteration (default: 5)",
    )
    lira_group.add_argument(
        "--forall-solver",
        type=str,
        default="z3",
        help="Binary solver for eflira-par (default: z3)",
    )
    lira_group.add_argument(
        "--eflira-forall-mode",
        choices=["sequential", "parallel-thread", "parallel-process-ipc"],
        help="Forall-check scheduling mode for eflira-par",
    )
    lira_group.add_argument(
        "--eflira-num-workers",
        type=int,
        default=4,
        help="Worker count for eflira-par forall checks (default: 4)",
    )
    lira_group.add_argument(
        "--eflira-num-samples",
        type=int,
        default=5,
        help="Number of existential samples per eflira-par iteration (default: 5)",
    )
    lira_group.add_argument(
        "--eflira-sample-strategy",
        choices=[
            "blocking",
            "random-seed",
            "optimize",
            "lexicographic",
            "jitter",
            "portfolio",
        ],
        default="blocking",
        help="Existential sampling strategy for eflira-par (default: blocking)",
    )
    lira_group.add_argument(
        "--eflira-sample-max-tries",
        type=int,
        default=25,
        help="Max retries for randomized eflira-par sampling (default: 25)",
    )
    lira_group.add_argument(
        "--eflira-sample-seed-low",
        type=int,
        default=1,
        help="Lower bound for randomized eflira-par seeds (default: 1)",
    )
    lira_group.add_argument(
        "--eflira-sample-seed-high",
        type=int,
        default=1000,
        help="Upper bound for randomized eflira-par seeds (default: 1000)",
    )
    lira_group.add_argument(
        "--eflira-lex-order",
        type=str,
        help="Comma-separated variable order for lexicographic eflira-par sampling",
    )
    lira_group.add_argument(
        "--eflira-optimize-objectives",
        type=int,
        help="Number of random objectives for optimize sampling",
    )
    lira_group.add_argument(
        "--eflira-optimize-max-tries",
        type=int,
        help="Retry budget for optimize sampling",
    )
    lira_group.add_argument(
        "--eflira-optimize-coeff-low",
        type=int,
        help="Lower coefficient bound for optimize sampling",
    )
    lira_group.add_argument(
        "--eflira-optimize-coeff-high",
        type=int,
        help="Upper coefficient bound for optimize sampling",
    )
    lira_group.add_argument(
        "--eflira-jitter-int-delta",
        type=int,
        help="Integer delta for jitter sampling",
    )
    lira_group.add_argument(
        "--eflira-jitter-real-delta",
        type=str,
        help="Real delta for jitter sampling",
    )
    lira_group.add_argument(
        "--eflira-jitter-max-tries",
        type=int,
        help="Retry budget for jitter sampling",
    )
    general_group.add_argument(
        "--max-loops",
        type=int,
        help="Max iterations for CEGAR-based engines",
    )
    general_group.add_argument(
        "--timeout",
        type=int,
        help="Timeout in seconds for Z3-based checks",
    )
    general_group.add_argument(
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
        eflira_sample_config = _build_eflira_sample_config(args)
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
                engine,
                exists_vars,
                forall_vars,
                phi,
                bv_solver=args.bv_solver,
                bv_pysmt_solver=args.bv_pysmt_solver,
                max_loops=args.max_loops,
                efbv_num_samples=args.efbv_num_samples,
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
                eflira_forall_mode=args.eflira_forall_mode,
                eflira_num_workers=args.eflira_num_workers,
                eflira_num_samples=args.eflira_num_samples,
                eflira_sample_strategy=args.eflira_sample_strategy,
                eflira_sample_max_tries=args.eflira_sample_max_tries,
                eflira_sample_seed_low=args.eflira_sample_seed_low,
                eflira_sample_seed_high=args.eflira_sample_seed_high,
                eflira_sample_config=eflira_sample_config,
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
