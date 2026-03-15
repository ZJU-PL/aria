#!/usr/bin/env python3
"""QF_BV portfolio solver built from Z3 tactics and PySAT backends."""

import argparse
import json
import logging
import multiprocessing
import os
import queue
import sys
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import z3
from pysat.formula import CNF
from pysat.solvers import Solver

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


class SolverResult(Enum):
    """Possible results from solvers."""

    SAT = "sat"
    UNSAT = "unsat"
    UNKNOWN = "unknown"

    @property
    def return_code(self) -> int:
        """Maps solver results to return codes."""
        return {self.SAT: 10, self.UNSAT: 20, self.UNKNOWN: 0}[self]


@dataclass
class WorkerResult:
    """Single worker outcome recorded by the portfolio."""

    result: str
    backend: str
    stage: str
    elapsed_seconds: float
    error: Optional[str] = None

    def to_solver_result(self) -> SolverResult:
        """Convert the serialized result value back to the enum."""
        return SolverResult(self.result)


@dataclass
class PortfolioResult:
    """Full portfolio outcome with worker metadata."""

    result: SolverResult
    winner: Optional[WorkerResult]
    worker_results: List[WorkerResult]
    error: Optional[str] = None


@dataclass
class SolverConfig:
    """Configuration for portfolio execution."""

    sat_solvers: List[str] = field(
        default_factory=lambda: [
            "cd",
            "cd15",
            "gc3",
            "gc4",
            "g3",
            "g4",
            "lgl",
            "mcb",
            "mpl",
            "mg3",
            "mc",
            "m22",
        ]
    )
    z3_preamble_ids: List[int] = field(default_factory=lambda: [0, 1])
    sat_timeout_seconds: float = 300.0
    overall_timeout_seconds: float = 600.0
    start_method: Optional[str] = None

    def multiprocessing_context(self):
        """Return the multiprocessing context for this run."""
        if self.start_method is None:
            return multiprocessing.get_context()
        return multiprocessing.get_context(self.start_method)


def _build_z3_preamble(preamble_id: int) -> z3.Tactic:
    """Build one of the supported QF_BV preprocessing pipelines."""
    if preamble_id == 0:
        return z3.AndThen(
            z3.With("simplify"),
            z3.With("propagate-values"),
            z3.Tactic("elim-uncnstr"),
            z3.With("solve-eqs", solve_eqs_max_occs=2),
            z3.Tactic("reduce-bv-size"),
            z3.With(
                "simplify",
                som=True,
                pull_cheap_ite=True,
                push_ite_bv=False,
                local_ctx=True,
                local_ctx_limit=10000000,
                flat=True,
                hoist_mul=False,
            ),
            z3.With("simplify", hoist_mul=False, som=False),
            "max-bv-sharing",
            "ackermannize_bv",
            "bit-blast",
            z3.With("simplify", local_ctx=True, flat=False),
            z3.With("solve-eqs", solve_eqs_max_occs=2),
            "aig",
            "tseitin-cnf",
        )
    if preamble_id == 1:
        return z3.AndThen(
            z3.With("simplify"),
            z3.With("propagate-values"),
            z3.With("solve-eqs", solve_eqs_max_occs=2),
            z3.Tactic("elim-uncnstr"),
            z3.With(
                "simplify",
                som=True,
                pull_cheap_ite=True,
                push_ite_bv=False,
                local_ctx=True,
                local_ctx_limit=10000000,
                flat=True,
                hoist_mul=False,
            ),
            z3.Tactic("max-bv-sharing"),
            z3.Tactic("bit-blast"),
            z3.With("simplify", local_ctx=True, flat=False),
            "aig",
            "tseitin-cnf",
        )
    raise ValueError(f"Unknown Z3 preamble id: {preamble_id}")


def _parse_formula(file_name: str) -> z3.ExprRef:
    """Parse a SMT2 file into a single Z3 expression."""
    with open(file_name, encoding="utf-8") as formula_file:
        formula_text = formula_file.read()

    ctx = z3.Context()
    fml_vec = z3.parse_smt2_string(formula_text, ctx=ctx)
    return fml_vec[0] if len(fml_vec) == 1 else z3.And(fml_vec)


def _solver_result_payload(
    result: SolverResult,
    backend: str,
    stage: str,
    started_at: float,
    error: Optional[str] = None,
) -> WorkerResult:
    """Build a worker result payload."""
    return WorkerResult(
        result=result.value,
        backend=backend,
        stage=stage,
        elapsed_seconds=time.monotonic() - started_at,
        error=error,
    )


def _put_worker_result(result_queue, worker_result: WorkerResult) -> None:
    """Put a serializable worker result on the queue."""
    result_queue.put(asdict(worker_result))


def _available_sat_solvers(candidates: List[str]) -> List[str]:
    """Return the PySAT backends available in this environment."""
    available = []
    for solver_name in candidates:
        try:
            with Solver(name=solver_name):
                available.append(solver_name)
        except (ValueError, RuntimeError, OSError):
            logger.info("Skipping unavailable SAT backend: %s", solver_name)
    return available


def _solve_sat_worker(solver_name: str, dimacs: str, result_queue) -> None:
    """Solve a CNF instance with one PySAT backend."""
    started_at = time.monotonic()
    try:
        cnf = CNF(from_string=dimacs)
        with Solver(name=solver_name, bootstrap_with=cnf) as solver:
            logger.info("Solving with %s", solver_name)
            result = SolverResult.SAT if solver.solve() else SolverResult.UNSAT
            _put_worker_result(
                result_queue,
                _solver_result_payload(
                    result=result,
                    backend=solver_name,
                    stage="sat",
                    started_at=started_at,
                ),
            )
    except (ValueError, RuntimeError, OSError) as exc:
        logger.error("Error in %s: %s", solver_name, exc)
        _put_worker_result(
            result_queue,
            _solver_result_payload(
                result=SolverResult.UNKNOWN,
                backend=solver_name,
                stage="sat",
                started_at=started_at,
                error=str(exc),
            ),
        )


def _wait_for_first_result(result_queue, timeout_seconds: float) -> Optional[WorkerResult]:
    """Wait for the first worker result, or return None on timeout."""
    try:
        payload = result_queue.get(timeout=timeout_seconds)
    except queue.Empty:
        return None
    return WorkerResult(**payload)


def _wait_for_decisive_result(
    result_queue,
    processes: List[multiprocessing.Process],
    timeout_seconds: float,
) -> List[WorkerResult]:
    """Collect worker results until a decisive answer is found or time expires."""
    deadline = time.monotonic() + timeout_seconds
    worker_results: List[WorkerResult] = []

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break

        worker_result = _wait_for_first_result(result_queue, remaining)
        if worker_result is None:
            break

        worker_results.append(worker_result)
        if worker_result.to_solver_result() != SolverResult.UNKNOWN:
            break

        if not any(process.is_alive() for process in processes):
            break

    return worker_results


def _terminate_processes(processes: List[multiprocessing.Process]) -> None:
    """Terminate and reap all worker processes."""
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=1.0)


def _run_sat_portfolio(
    dimacs: str, sat_solvers: List[str], timeout_seconds: float, mp_context
) -> WorkerResult:
    """Run the SAT-only subportfolio on a bit-blasted CNF."""
    available_solvers = _available_sat_solvers(sat_solvers)
    if not available_solvers:
        return WorkerResult(
            result=SolverResult.UNKNOWN.value,
            backend="sat-portfolio",
            stage="sat-portfolio",
            elapsed_seconds=0.0,
            error="No SAT backends available",
        )

    result_queue = mp_context.Queue()
    processes = []
    started_at = time.monotonic()

    try:
        for solver_name in available_solvers:
            process = mp_context.Process(
                target=_solve_sat_worker,
                args=(solver_name, dimacs, result_queue),
            )
            processes.append(process)
            process.start()

        worker_results = _wait_for_decisive_result(
            result_queue, processes, timeout_seconds
        )
        if not worker_results:
            return WorkerResult(
                result=SolverResult.UNKNOWN.value,
                backend="sat-portfolio",
                stage="sat-portfolio",
                elapsed_seconds=time.monotonic() - started_at,
                error=f"Timed out after {timeout_seconds:.3f}s",
            )
        for worker_result in worker_results:
            if worker_result.to_solver_result() != SolverResult.UNKNOWN:
                return worker_result
        return worker_results[-1]
    finally:
        _terminate_processes(processes)
        result_queue.close()
        result_queue.join_thread()


def _preprocess_and_solve_sat_worker(
    file_name: str,
    preamble_id: int,
    sat_solvers: List[str],
    sat_timeout_seconds: float,
    start_method: Optional[str],
    result_queue,
) -> None:
    """Preprocess a QF_BV formula and solve the resulting CNF portfolio-style."""
    started_at = time.monotonic()
    backend = f"z3-preamble-{preamble_id}"
    try:
        fml = _parse_formula(file_name)
        qfbv_preamble = _build_z3_preamble(preamble_id)
        qfbv_tactic = z3.With(
            qfbv_preamble, elim_and=True, push_ite_bv=True, blast_distinct=True
        )
        after_simp = qfbv_tactic(fml).as_expr()

        if z3.is_true(after_simp):
            _put_worker_result(
                result_queue,
                _solver_result_payload(
                    result=SolverResult.SAT,
                    backend=backend,
                    stage="preprocess",
                    started_at=started_at,
                ),
            )
            return

        if z3.is_false(after_simp):
            _put_worker_result(
                result_queue,
                _solver_result_payload(
                    result=SolverResult.UNSAT,
                    backend=backend,
                    stage="preprocess",
                    started_at=started_at,
                ),
            )
            return

        goal = z3.Goal(ctx=fml.ctx)
        goal.add(after_simp)
        sat_result = _run_sat_portfolio(
            dimacs=goal.dimacs(),
            sat_solvers=sat_solvers,
            timeout_seconds=sat_timeout_seconds,
            mp_context=multiprocessing.get_context(start_method),
        )
        sat_result.backend = f"{backend}:{sat_result.backend}"
        sat_result.stage = "preprocess+sat"
        sat_result.elapsed_seconds = time.monotonic() - started_at
        _put_worker_result(result_queue, sat_result)
    except (ValueError, RuntimeError, z3.Z3Exception, OSError) as exc:
        logger.error("Error in preprocessing worker %s: %s", backend, exc)
        _put_worker_result(
            result_queue,
            _solver_result_payload(
                result=SolverResult.UNKNOWN,
                backend=backend,
                stage="preprocess+sat",
                started_at=started_at,
                error=str(exc),
            ),
        )


def _solve_with_z3_worker(file_name: str, result_queue) -> None:
    """Solve the original formula directly with Z3."""
    started_at = time.monotonic()
    try:
        fml = _parse_formula(file_name)
        solver = z3.Solver(ctx=fml.ctx)
        solver.add(fml)
        result = SolverResult(str(solver.check()))
        _put_worker_result(
            result_queue,
            _solver_result_payload(
                result=result,
                backend="z3",
                stage="z3",
                started_at=started_at,
            ),
        )
    except (ValueError, RuntimeError, z3.Z3Exception, OSError) as exc:
        logger.error("Error in Z3 worker: %s", exc)
        _put_worker_result(
            result_queue,
            _solver_result_payload(
                result=SolverResult.UNKNOWN,
                backend="z3",
                stage="z3",
                started_at=started_at,
                error=str(exc),
            ),
        )


class FormulaParser:
    """Handles SMT formula parsing and solving."""

    @staticmethod
    def solve(file_name: str, config: Optional[SolverConfig] = None) -> SolverResult:
        """Solve the formula and return only the final SAT result."""
        return FormulaParser.solve_detailed(file_name, config=config).result

    @staticmethod
    def solve_detailed(
        file_name: str, config: Optional[SolverConfig] = None
    ) -> PortfolioResult:
        """Run the portfolio and return the winning worker metadata."""
        run_config = config or SolverConfig()
        worker_results: List[WorkerResult] = []

        try:
            _parse_formula(file_name)
        except (ValueError, RuntimeError, z3.Z3Exception, OSError) as exc:
            logger.error("Error parsing formula: %s", exc)
            return PortfolioResult(
                result=SolverResult.UNKNOWN,
                winner=None,
                worker_results=[],
                error=str(exc),
            )

        mp_context = run_config.multiprocessing_context()
        result_queue = mp_context.Queue()
        processes = []

        try:
            for preamble_id in run_config.z3_preamble_ids:
                process = mp_context.Process(
                    target=_preprocess_and_solve_sat_worker,
                    args=(
                        file_name,
                        preamble_id,
                        run_config.sat_solvers,
                        run_config.sat_timeout_seconds,
                        run_config.start_method,
                        result_queue,
                    ),
                )
                processes.append(process)
                process.start()

            process = mp_context.Process(
                target=_solve_with_z3_worker,
                args=(file_name, result_queue),
            )
            processes.append(process)
            process.start()

            worker_results = _wait_for_decisive_result(
                result_queue, processes, run_config.overall_timeout_seconds
            )
            if not worker_results:
                timeout_error = (
                    f"Timed out after {run_config.overall_timeout_seconds:.3f}s"
                )
                return PortfolioResult(
                    result=SolverResult.UNKNOWN,
                    winner=None,
                    worker_results=[],
                    error=timeout_error,
                )
            winner = worker_results[-1]
            for worker_result in worker_results:
                if worker_result.to_solver_result() != SolverResult.UNKNOWN:
                    winner = worker_result
                    break
            return PortfolioResult(
                result=winner.to_solver_result(),
                winner=winner,
                worker_results=worker_results,
            )
        finally:
            _terminate_processes(processes)
            result_queue.close()
            result_queue.join_thread()


def _write_artifact_logs(
    request_directory: str, result: PortfolioResult, formula_file: str
) -> Dict[str, str]:
    """Write portfolio summaries to the advertised stdout/stderr artifact paths."""
    stdout_path = os.path.join(request_directory, "stdout.log")
    stderr_path = os.path.join(request_directory, "stderr.log")

    stdout_payload = {
        "formula_file": formula_file,
        "result": result.result.value,
        "winner": asdict(result.winner) if result.winner is not None else None,
        "worker_results": [asdict(worker) for worker in result.worker_results],
    }

    with open(stdout_path, "w", encoding="utf-8") as stdout_file:
        json.dump(stdout_payload, stdout_file, indent=2)
        stdout_file.write("\n")

    with open(stderr_path, "w", encoding="utf-8") as stderr_file:
        if result.error is not None:
            stderr_file.write(result.error)
            stderr_file.write("\n")
        for worker in result.worker_results:
            if worker.error is not None:
                stderr_file.write(
                    f"{worker.backend} [{worker.stage}]: {worker.error}\n"
                )

    return {"stdout_path": stdout_path, "stderr_path": stderr_path}


def main() -> None:
    """CLI entry point used by the request-directory wrapper."""
    parser = argparse.ArgumentParser(description="QF_BV SMT portfolio solver")
    parser.add_argument("request_directory", help="Directory containing input files")
    args = parser.parse_args()

    try:
        with open(
            os.path.join(args.request_directory, "input.json"), encoding="utf-8"
        ) as input_file:
            input_json = json.load(input_file)

        formula_file = input_json.get("formula_file")
        if not formula_file:
            raise ValueError("No formula file specified in input.json")

        result = FormulaParser.solve_detailed(formula_file)
        solver_output = {
            "return_code": result.result.return_code,
            "result": result.result.value,
            "winner": asdict(result.winner) if result.winner is not None else None,
            "artifacts": _write_artifact_logs(
                args.request_directory, result, formula_file
            ),
        }

        output_path = os.path.join(args.request_directory, "solver_out.json")
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(solver_output, output_file, indent=2)

        logger.info("Result: %s (code: %s)", result.result.value, result.result.return_code)
    except (ValueError, FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        logger.error("Fatal error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
