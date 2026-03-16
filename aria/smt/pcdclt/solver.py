"""Parallel CDCL(T) SMT Solver"""

import logging
import time
from dataclasses import dataclass
from multiprocessing import Process, Queue, cpu_count
from typing import List

from aria.bool import PySATSolver, simplify_numeric_clauses
from aria.utils import SolverResult
from aria.global_params import SMT_SOLVERS_PATH
from aria.smt.pcdclt.preprocessor import FormulaAbstraction
from aria.smt.pcdclt.theory_solver import TheorySolver
from aria.smt.pcdclt.config import (
    NUM_SAMPLES_PER_ROUND,
    MAX_T_CHECKING_PROCESSES,
    SIMPLIFY_CLAUSES,
    WORKER_SHUTDOWN_TIMEOUT,
    BOOL_MODEL_SAMPLING_STRATEGY,
)

logger = logging.getLogger(__name__)


@dataclass
class BlockingClauseMetrics:
    generated_total: int = 0
    added_total: int = 0
    theory_unsat_checks_total: int = 0
    theory_sat_checks_total: int = 0
    theory_unknown_or_error_total: int = 0
    completed_rounds_total: int = 0
    wall_time_seconds: float = 0.0


def new_blocking_clause_metrics() -> BlockingClauseMetrics:
    return BlockingClauseMetrics()


def record_theory_result(
    metrics: BlockingClauseMetrics, theory_result, bool_model: List[int]
) -> List[int] | None:
    if theory_result == SolverResult.UNSAT or theory_result == SolverResult.UNSAT.name:
        metrics.generated_total += 1
        metrics.theory_unsat_checks_total += 1
        return [-literal for literal in bool_model]
    if theory_result == SolverResult.SAT or theory_result == SolverResult.SAT.name:
        metrics.theory_sat_checks_total += 1
        return None
    metrics.theory_unknown_or_error_total += 1
    return None


def record_added_clauses(metrics: BlockingClauseMetrics, clauses: List[List[int]]) -> None:
    metrics.added_total += len(clauses)


def _sanitize_incremental_smt2(smt2_string: str) -> str:
    """Drop terminal commands so the file can be loaded into an interactive solver."""
    filtered = []
    for line in smt2_string.splitlines():
        stripped = line.strip()
        if stripped.startswith("(check-sat"):
            continue
        if stripped.startswith("(get-model"):
            continue
        if stripped.startswith("(get-value"):
            continue
        if stripped == "(exit)":
            continue
        filtered.append(line)
    return "\n".join(filtered) + "\n"


def _theory_worker(
    worker_id: int, init_theory_formula, task_queue, result_queue, solver_bin: str
):
    """
    Theory checking worker process

    Args:
        worker_id: Worker identifier
        init_theory_formula: Initial theory constraints (shared)
        task_queue: Queue to receive (task_id, assumptions) tuples
        result_queue: Queue to send (task_id, unsat_core) results
        solver_bin: Path to SMT solver binary
    """
    logger.debug("Theory worker %d starting", worker_id)

    # Use context manager to ensure proper cleanup
    theory_solver = None
    try:
        theory_solver = TheorySolver(solver_bin, worker_id=worker_id)
        theory_solver.add_formula(init_theory_formula)

        while True:
            task_id, assumptions = task_queue.get()

            # Shutdown signal
            if task_id == -1:
                logger.debug("Theory worker %d shutting down cleanly", worker_id)
                break

            try:
                # Check theory consistency
                logger.info(
                    f"worker {worker_id} check sat start", extra={"is_timing": True}
                )
                result = theory_solver.check_sat_assuming(assumptions)
                logger.info(
                    f"worker {worker_id} check sat over", extra={"is_timing": True}
                )
                result_queue.put((task_id, result.name))

            except (OSError, RuntimeError, ValueError) as e:
                logger.error("Worker %d error processing task: %s", worker_id, e)
                result_queue.put((task_id, f"ERROR:{e}"))

    except (OSError, RuntimeError, ValueError) as e:
        logger.error("Worker %d fatal error during initialization: %s", worker_id, e)

    finally:
        # Ensure theory solver subprocess is cleaned up
        if theory_solver is not None:
            try:
                theory_solver.close()
            except (OSError, RuntimeError) as e:
                logger.warning("Worker %d cleanup error: %s", worker_id, e)
        logger.debug("Theory worker %d exiting", worker_id)

def _models_to_assumptions(
    bool_models: List[List[int]], abstraction: FormulaAbstraction
) -> List[List[str]]:
    """
    Convert Boolean models to theory solver assumptions

    Args:
        bool_models: List of Boolean models (e.g., [[1, -2, 3], [-1, 2, -3]])
        abstraction: Formula abstraction with var mappings

    Returns:
        List of assumption lists (e.g., [['p@1', '(not p@2)', 'p@3'], ...])
    """
    all_assumptions = []

    for model in bool_models:
        assumptions = []
        for literal in model:
            if abs(literal) not in abstraction.id_to_atom:
                continue
            atom = abstraction.get_atom_sexpr(abs(literal))
            if literal > 0:
                assumptions.append(atom)
            else:
                assumptions.append(f"(not {atom})")
        all_assumptions.append(assumptions)

    return all_assumptions


def solve(smt2_string: str, logic: str = "ALL") -> SolverResult:
    """
    Solve SMT formula using parallel CDCL(T)

    Args:
        smt2_string: SMT-LIB2 formula string
        logic: SMT-LIB2 logic (e.g., 'QF_LRA', 'ALL')

    Returns:
        SolverResult.SAT, SolverResult.UNSAT, or SolverResult.UNKNOWN
    """
    metrics = new_blocking_clause_metrics()
    solve_start = time.monotonic()

    # Step 1: Preprocess and build Boolean abstraction
    abstraction = FormulaAbstraction()
    preprocess_result = abstraction.preprocess(smt2_string)

    if preprocess_result != SolverResult.UNKNOWN:
        # Decided during preprocessing
        logger.debug("Solved during preprocessing: %s", preprocess_result)
        return preprocess_result

    # Step 2: Initialize Boolean solver
    bool_solver = PySATSolver()
    bool_solver.add_clauses(abstraction.numeric_clauses)

    # Step 3: Setup theory solver workers
    if MAX_T_CHECKING_PROCESSES == 0:
        num_workers = cpu_count()
    else:
        num_workers = MAX_T_CHECKING_PROCESSES
    num_workers = min(num_workers, cpu_count())

    # Build theory formula
    theory_formula = _sanitize_incremental_smt2(smt2_string)

    # Get solver binary
    z3_config = SMT_SOLVERS_PATH["z3"]
    solver_bin = z3_config["path"]
    if "-in" not in z3_config.get("args", ""):
        solver_bin = f"{solver_bin} -in"
    else:
        solver_bin = f"{solver_bin} {z3_config['args']}"

    # Create worker queues
    task_queue = Queue()
    result_queue = Queue()

    # Start worker processes
    workers = []
    for worker_id in range(num_workers):
        worker = Process(
            target=_theory_worker,
            args=(
                worker_id,
                theory_formula,
                task_queue,
                result_queue,
                solver_bin,
            ),
        )
        worker.daemon = True
        worker.start()
        workers.append(worker)

    logger.debug("Started %d theory workers", num_workers)

    # Step 4: Main CDCL(T) loop
    result = SolverResult.UNKNOWN

    try:
        while True:
            # Check Boolean satisfiability
            check_start = time.monotonic()
            logger.info("bool check start", extra={"is_timing": True})
            bool_result = bool_solver.check_sat()
            logger.info(
                "bool check over result=%s elapsed=%.3fs",
                bool_result.name,
                time.monotonic() - check_start,
                extra={"is_timing": True},
            )
            if bool_result != SolverResult.SAT:
                result = SolverResult.UNSAT
                break

            logger.debug("Boolean abstraction is SAT")

            # Sample multiple Boolean models
            sample_start = time.monotonic()
            logger.info("bool sample start", extra={"is_timing": True})
            bool_models = bool_solver.sample_models(
                to_enum=NUM_SAMPLES_PER_ROUND,
                strategy=BOOL_MODEL_SAMPLING_STRATEGY,
            )
            logger.info(
                "bool sample over strategy=%s models=%d elapsed=%.3fs",
                BOOL_MODEL_SAMPLING_STRATEGY,
                len(bool_models),
                time.monotonic() - sample_start,
                extra={"is_timing": True},
            )

            if not bool_models:
                result = SolverResult.UNSAT
                break

            logger.debug("Sampled %d Boolean models", len(bool_models))

            # Convert to assumptions and submit to theory workers
            all_assumptions = _models_to_assumptions(bool_models, abstraction)

            submit_start = time.monotonic()
            logger.info("theory submit start", extra={"is_timing": True})
            for task_id, assumptions in enumerate(all_assumptions):
                task_queue.put((task_id, assumptions))
            logger.info(
                "theory submit over tasks=%d elapsed=%.3fs",
                len(all_assumptions),
                time.monotonic() - submit_start,
                extra={"is_timing": True},
            )

            # Collect results from workers
            blocking_clauses = []
            saw_unknown = False
            collect_start = time.monotonic()
            logger.info("theory collect start", extra={"is_timing": True})
            for _ in range(len(all_assumptions)):
                task_id, core_result = result_queue.get()

                if isinstance(core_result, str) and core_result.startswith("ERROR:"):
                    logger.error("Theory solver error: %s", core_result)
                    record_theory_result(metrics, core_result, bool_models[task_id])
                    saw_unknown = True
                    continue

                if core_result == SolverResult.SAT.name:
                    record_theory_result(metrics, core_result, bool_models[task_id])
                    # Found theory-consistent model - SAT!
                    logger.debug("Found theory-consistent model")
                    result = SolverResult.SAT
                    break

                clause = record_theory_result(
                    metrics, core_result, bool_models[task_id]
                )
                if clause is not None:
                    blocking_clauses.append(clause)
                else:
                    saw_unknown = True

            logger.info(
                "theory collect over results=%d elapsed=%.3fs",
                len(all_assumptions),
                time.monotonic() - collect_start,
                extra={"is_timing": True},
            )

            if result == SolverResult.SAT:
                break

            if not blocking_clauses and saw_unknown:
                result = SolverResult.UNKNOWN
                break

            # All models are theory-inconsistent - learn by blocking sampled models
            logger.debug(
                "All models theory-inconsistent, processing %d blocking clauses",
                len(blocking_clauses),
            )

            if SIMPLIFY_CLAUSES:
                blocking_clauses = simplify_numeric_clauses(blocking_clauses)

            logger.debug("Adding %d blocking clauses", len(blocking_clauses))

            record_added_clauses(metrics, blocking_clauses)
            for clause in blocking_clauses:
                bool_solver.add_clause(clause)
            metrics.completed_rounds_total += 1

    except (OSError, RuntimeError, ValueError) as e:
        logger.error("Error in CDCL(T) main loop: %s", e)
        result = SolverResult.UNKNOWN

    finally:
        metrics.wall_time_seconds = time.monotonic() - solve_start
        logger.info(
            "pcdclt_metrics generated_total=%d added_total=%d theory_unsat_checks_total=%d theory_sat_checks_total=%d theory_unknown_or_error_total=%d completed_rounds_total=%d wall_time_seconds=%.3f sampling_strategy=%s simplify_clauses=%s worker_count=%d",
            metrics.generated_total,
            metrics.added_total,
            metrics.theory_unsat_checks_total,
            metrics.theory_sat_checks_total,
            metrics.theory_unknown_or_error_total,
            metrics.completed_rounds_total,
            metrics.wall_time_seconds,
            BOOL_MODEL_SAMPLING_STRATEGY,
            SIMPLIFY_CLAUSES,
            num_workers,
        )
        # Shutdown workers gracefully
        logger.debug("Shutting down %d theory workers", num_workers)

        # Send shutdown signal to all workers
        for _ in range(num_workers):
            try:
                task_queue.put((-1, None))
            except (OSError, RuntimeError) as e:
                logger.warning("Error sending shutdown signal: %s", e)

        # Wait for workers to finish gracefully
        alive_workers = []
        for worker in workers:
            worker.join(timeout=WORKER_SHUTDOWN_TIMEOUT)
            if worker.is_alive():
                alive_workers.append(worker)

        # Force terminate any workers that didn't exit gracefully
        if alive_workers:
            logger.warning(
                "%d workers didn't exit gracefully, terminating", len(alive_workers)
            )
            for worker in alive_workers:
                try:
                    worker.terminate()
                    worker.join(timeout=0.5)
                except (OSError, RuntimeError) as e:
                    logger.error("Error terminating worker: %s", e)

        # Final check for zombie workers
        still_alive = [w for w in workers if w.is_alive()]
        if still_alive:
            logger.error(
                "%d workers are still alive after termination!", len(still_alive)
            )
            # Last resort: kill
            for worker in still_alive:
                try:
                    worker.kill()
                except (OSError, RuntimeError) as e:
                    logger.error("Error killing worker: %s", e)
        else:
            logger.debug("All workers terminated cleanly")

        # Ensure queues are properly cleaned up to avoid hangs
        try:
            task_queue.close()
            task_queue.join_thread()
        except (OSError, RuntimeError) as e:
            logger.debug("task_queue cleanup error: %s", e)
        try:
            result_queue.close()
            result_queue.join_thread()
        except (OSError, RuntimeError) as e:
            logger.debug("result_queue cleanup error: %s", e)

    return result


class CDCLTSolver:
    """CDCL(T) SMT Solver interface"""

    def solve_smt2_string(self, smt2_string: str, logic: str = "ALL") -> SolverResult:
        """Solve SMT-LIB2 string"""
        return solve(smt2_string, logic)

    def solve_smt2_file(self, filename: str, logic: str = "ALL") -> SolverResult:
        """Solve SMT-LIB2 file"""
        with open(filename, "r", encoding="utf-8") as f:
            smt2_string = f.read()
        return solve(smt2_string, logic)
