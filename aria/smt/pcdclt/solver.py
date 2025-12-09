"""Parallel CDCL(T) SMT Solver"""

import logging
from multiprocessing import cpu_count, Queue, Process, Manager
from ctypes import c_char_p
from typing import List

from aria.bool import PySATSolver, simplify_numeric_clauses
from aria.utils import SolverResult, SExprParser
from aria.global_params import SMT_SOLVERS_PATH
from aria.smt.pcdclt.preprocessor import FormulaAbstraction
from aria.smt.pcdclt.theory_solver import TheorySolver
from aria.smt.pcdclt.config import (
    NUM_SAMPLES_PER_ROUND,
    MAX_T_CHECKING_PROCESSES,
    SIMPLIFY_CLAUSES,
    WORKER_SHUTDOWN_TIMEOUT,
)

logger = logging.getLogger(__name__)


def _theory_worker(worker_id: int, init_theory_formula, task_queue, result_queue, solver_bin: str):
    """
    Theory checking worker process

    Args:
        worker_id: Worker identifier
        init_theory_formula: Initial theory constraints (shared)
        task_queue: Queue to receive (task_id, assumptions) tuples
        result_queue: Queue to send (task_id, unsat_core) results
        solver_bin: Path to SMT solver binary
    """
    logger.debug(f"Theory worker {worker_id} starting")

    # Use context manager to ensure proper cleanup
    theory_solver = None
    try:
        theory_solver = TheorySolver(solver_bin, worker_id=worker_id)
        theory_solver.add_formula(init_theory_formula.value)

        while True:
            task_id, assumptions = task_queue.get()

            # Shutdown signal
            if task_id == -1:
                logger.debug(f"Theory worker {worker_id} shutting down cleanly")
                break

            try:
                # Check theory consistency
                result = theory_solver.check_sat_assuming(assumptions)

                if result == SolverResult.UNSAT:
                    # Get unsat core
                    unsat_core = theory_solver.get_unsat_core()
                    result_queue.put((task_id, unsat_core))
                else:
                    # Theory consistent - signal SAT
                    result_queue.put((task_id, ""))

            except Exception as e:
                logger.error(f"Worker {worker_id} error processing task: {e}")
                result_queue.put((task_id, f"ERROR:{e}"))

    except Exception as e:
        logger.error(f"Worker {worker_id} fatal error during initialization: {e}")

    finally:
        # Ensure theory solver subprocess is cleaned up
        if theory_solver is not None:
            try:
                theory_solver.close()
            except Exception as e:
                logger.warning(f"Worker {worker_id} cleanup error: {e}")
        logger.debug(f"Theory worker {worker_id} exiting")


def _parse_unsat_core(core: str, abstraction: FormulaAbstraction) -> List[int]:
    """
    Convert unsat core s-expression to blocking clause

    Args:
        core: Unsat core string like '(p@4 p@7 (not p@6))'
        abstraction: Formula abstraction with var mappings

    Returns:
        Blocking clause as list of integers, e.g., [-4, -7, 6]
    """
    parsed = SExprParser.parse_sexpr_string(core)
    blocking_clause = []

    for element in parsed:
        if isinstance(element, list):
            # Negated literal: (not p@X)
            var_name = element[1]
            blocking_clause.append(abstraction.var_to_id[var_name])
        else:
            # Positive literal: p@X
            blocking_clause.append(-abstraction.var_to_id[element])

    return blocking_clause


def _models_to_assumptions(bool_models: List[List[int]], abstraction: FormulaAbstraction) -> List[List[str]]:
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
            var_name = abstraction.id_to_var[abs(literal)]
            if literal > 0:
                assumptions.append(var_name)
            else:
                assumptions.append(f"(not {var_name})")
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
    # Step 1: Preprocess and build Boolean abstraction
    abstraction = FormulaAbstraction()
    preprocess_result = abstraction.preprocess(smt2_string)

    if preprocess_result != SolverResult.UNKNOWN:
        # Decided during preprocessing
        logger.debug(f"Solved during preprocessing: {preprocess_result}")
        return preprocess_result

    # Step 2: Initialize Boolean solver
    bool_solver = PySATSolver()
    bool_solver.add_clauses(abstraction.numeric_clauses)

    # Step 3: Setup theory solver workers
    num_workers = cpu_count() if MAX_T_CHECKING_PROCESSES == 0 else MAX_T_CHECKING_PROCESSES
    num_workers = min(num_workers, cpu_count())

    # Build theory formula
    theory_formula = (
        f" (set-logic {logic}) "
        f" (set-option :produce-unsat-cores true) "
        + " ".join(abstraction.theory_signature)
        + f"(assert {abstraction.theory_constraints})"
    )

    # Get solver binary
    z3_config = SMT_SOLVERS_PATH["z3"]
    solver_bin = z3_config["path"]
    if "-in" not in z3_config.get("args", ""):
        solver_bin = f"{solver_bin} -in"
    else:
        solver_bin = f"{solver_bin} {z3_config['args']}"

    # Create shared theory formula
    manager = Manager()
    shared_theory_formula = manager.Value(c_char_p, theory_formula)

    # Create worker queues
    task_queue = Queue()
    result_queue = Queue()

    # Start worker processes
    workers = []
    for worker_id in range(num_workers):
        worker = Process(
            target=_theory_worker,
            args=(worker_id, shared_theory_formula, task_queue, result_queue, solver_bin)
        )
        worker.daemon = True
        worker.start()
        workers.append(worker)

    logger.debug(f"Started {num_workers} theory workers")

    # Step 4: Main CDCL(T) loop
    result = SolverResult.UNKNOWN

    try:
        while True:
            # Check Boolean satisfiability
            if not bool_solver.check_sat():
                result = SolverResult.UNSAT
                break

            logger.debug("Boolean abstraction is SAT")

            # Sample multiple Boolean models
            bool_models = bool_solver.sample_models(to_enum=NUM_SAMPLES_PER_ROUND)

            if not bool_models:
                result = SolverResult.UNSAT
                break

            logger.debug(f"Sampled {len(bool_models)} Boolean models")

            # Convert to assumptions and submit to theory workers
            all_assumptions = _models_to_assumptions(bool_models, abstraction)

            for task_id, assumptions in enumerate(all_assumptions):
                task_queue.put((task_id, assumptions))

            # Collect results from workers
            unsat_cores = []
            for _ in range(len(all_assumptions)):
                task_id, core_result = result_queue.get()

                if isinstance(core_result, str) and core_result.startswith("ERROR:"):
                    logger.error(f"Theory solver error: {core_result}")
                    continue

                if core_result == "":
                    # Found theory-consistent model - SAT!
                    logger.debug("Found theory-consistent model")
                    result = SolverResult.SAT
                    break

                unsat_cores.append(core_result)

            if result == SolverResult.SAT:
                break

            # All models are theory-inconsistent - learn from unsat cores
            logger.debug(f"All models theory-inconsistent, processing {len(unsat_cores)} unsat cores")

            blocking_clauses = [_parse_unsat_core(core, abstraction) for core in unsat_cores]

            if SIMPLIFY_CLAUSES:
                blocking_clauses = simplify_numeric_clauses(blocking_clauses)

            logger.debug(f"Adding {len(blocking_clauses)} blocking clauses")

            for clause in blocking_clauses:
                bool_solver.add_clause(clause)

    except Exception as e:
        logger.error(f"Error in CDCL(T) main loop: {e}")
        result = SolverResult.UNKNOWN

    finally:
        # Shutdown workers gracefully
        logger.debug(f"Shutting down {num_workers} theory workers")

        # Send shutdown signal to all workers
        for _ in range(num_workers):
            try:
                task_queue.put((-1, None))
            except Exception as e:
                logger.warning(f"Error sending shutdown signal: {e}")

        # Wait for workers to finish gracefully
        alive_workers = []
        for worker in workers:
            worker.join(timeout=WORKER_SHUTDOWN_TIMEOUT)
            if worker.is_alive():
                alive_workers.append(worker)

        # Force terminate any workers that didn't exit gracefully
        if alive_workers:
            logger.warning(f"{len(alive_workers)} workers didn't exit gracefully, terminating")
            for worker in alive_workers:
                try:
                    worker.terminate()
                    worker.join(timeout=0.5)
                except Exception as e:
                    logger.error(f"Error terminating worker: {e}")

        # Final check for zombie workers
        still_alive = [w for w in workers if w.is_alive()]
        if still_alive:
            logger.error(f"{len(still_alive)} workers are still alive after termination!")
            # Last resort: kill
            for worker in still_alive:
                try:
                    worker.kill()
                except Exception as e:
                    logger.error(f"Error killing worker: {e}")
        else:
            logger.debug("All workers terminated cleanly")

        # Ensure queues and manager are properly cleaned up to avoid hangs
        try:
            task_queue.close()
            task_queue.join_thread()
        except Exception as e:
            logger.debug(f"task_queue cleanup error: {e}")
        try:
            result_queue.close()
            result_queue.join_thread()
        except Exception as e:
            logger.debug(f"result_queue cleanup error: {e}")
        try:
            # Shutdown the manager server process explicitly
            manager.shutdown()
        except Exception as e:
            logger.debug(f"manager shutdown error: {e}")

    return result


class CDCLTSolver:
    """CDCL(T) SMT Solver interface"""

    def solve_smt2_string(self, smt2_string: str, logic: str = "ALL") -> SolverResult:
        """Solve SMT-LIB2 string"""
        return solve(smt2_string, logic)

    def solve_smt2_file(self, filename: str, logic: str = "ALL") -> SolverResult:
        """Solve SMT-LIB2 file"""
        with open(filename, 'r') as f:
            smt2_string = f.read()
        return solve(smt2_string, logic)
