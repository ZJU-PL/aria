"""
Objective-level divide-and-conquer for solving for boxed optimization over bit-vectors
1. Solve each objective in parallel
2. Combine the results

Possible improvements:
- Use different solver configurations for different objectives.
   - Linear search, binary search, QSMT, MaxSAT
   - Each of the above can be run with different configurations

FIXME: by LLM, to check if this is correct
"""

import multiprocessing as mp
from multiprocessing import Process, Queue
from dataclasses import dataclass
from enum import Enum
import signal
from typing import Optional, List
import time
import logging
import os
import re
import z3

from aria.optimization.omtbv.bv_opt_iterative_search import bv_opt_with_binary_search, bv_opt_with_linear_search
from aria.optimization.omtbv.bv_opt_qsmt import bv_opt_with_qsmt
from aria.optimization.omtbv.bv_opt_maxsat import bv_opt_with_maxsat

logger = logging.getLogger(__name__)

class SolverStatus(Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    TIMEOUT = "timeout"

def parse_smt2_value_output(output: str) -> Optional[int]:
    """Parse integer value from SMT-LIB2 solver output like 'sat\n ((var #x40))'."""
    if not output or "sat" not in output.lower():
        return None

    pattern = r'\(\([^\s)]+\s+([^\s)]+)\)\)'
    match = re.search(pattern, output)
    if not match:
        return None

    value_str = match.group(1).strip()
    try:
        if value_str.startswith('#x'):
            return int(value_str[2:], 16)
        elif value_str.startswith('#b'):
            return int(value_str[2:], 2)
        return int(value_str)
    except (ValueError, AttributeError):
        return None

@dataclass
class ObjectiveResult:
    objective_id: int
    value: Optional[int]
    status: SolverStatus
    solve_time: float

def solve_objective(complete_smt2: str, obj_id: int, minimize: bool, engine: str,
                   solver_name: str, result_queue: Queue, error_queue: Queue):
    """Worker function to solve a single objective. SMT2 string must contain (assert (= obj_var <expr>))."""
    try:
        formula_vec = z3.parse_smt2_string(complete_smt2)
        obj = None
        other_assertions = []

        for assertion in formula_vec:
            if z3.is_eq(assertion) and len(assertion.children()) == 2:
                left, right = assertion.children()
                if z3.is_const(left) and left.decl().name() == 'obj_var':
                    obj = right  # Extract actual objective (x, y, etc.)
                else:
                    other_assertions.append(assertion)
            else:
                other_assertions.append(assertion)

        if obj is None:
            raise ValueError("Could not find 'obj_var' in SMT2 string")

        formula = z3.And(other_assertions) if other_assertions else z3.BoolVal(True)
        start_time = time.time()

        if engine == "qsmt":
            result = bv_opt_with_qsmt(formula, obj, minimize, solver_name)
        elif engine == "maxsat":
            result = bv_opt_with_maxsat(formula, obj, minimize, solver_name)
        elif engine == "iter":
            solver_type = solver_name.split('-')[0]
            if solver_name.endswith("-ls"):
                result = bv_opt_with_linear_search(formula, obj, minimize, solver_type)
            elif solver_name.endswith("-bs"):
                result = bv_opt_with_binary_search(formula, obj, minimize, solver_type)
        else:
            result = None

        solve_time = time.time() - start_time

        if result is None or result == "unknown":
            status, value = SolverStatus.ERROR, None
        else:
            status = SolverStatus.COMPLETED
            if isinstance(result, str):
                value = parse_smt2_value_output(result)
                if value is None:
                    try:
                        value = int(result.strip())
                    except (ValueError, TypeError):
                        value = None
            else:
                value = result if isinstance(result, int) else None

        result_queue.put(ObjectiveResult(obj_id, value, status, solve_time))
    except Exception as e:
        error_queue.put((obj_id, str(e)))

def solve_boxed_parallel(formula: z3.BoolRef,
                        objectives: List[z3.ExprRef],
                        minimize: bool = False,
                        engine: str = "qsmt",
                        solver_name: str = "z3",
                        timeout: float = 3600) -> List[Optional[int]]:
    """
    Solve multiple objectives in parallel using divide-and-conquer strategy

    Args:
        formula: The base formula (constraints)
        objectives: List of objectives to optimize
        minimize: Whether to minimize (True) or maximize (False)
        engine: Optimization engine ("qsmt", "maxsat", "iter")
        solver_name: Specific solver to use
        timeout: Maximum time in seconds for each objective

    Returns:
        List of optimal values (None for failed objectives)
    """
    result_queue = mp.Queue()
    error_queue = mp.Queue()
    processes = []
    results = [None] * len(objectives)

    # Serialize formula and objectives to SMT-LIB2 strings
    # Create a solver to get the base formula as SMT-LIB2
    base_solver = z3.Solver()
    base_solver.add(formula)
    base_smt2 = base_solver.to_smt2()

    for i, obj in enumerate(objectives):
        lines = base_smt2.split('\n')
        insert_idx = next((idx for idx, line in enumerate(lines) if line.strip().startswith("(assert")), len(lines))
        sort_str = obj.sort().sexpr()
        lines.insert(insert_idx, f"(declare-const obj_var {sort_str})")
        lines.insert(insert_idx + 1, f"(assert (= obj_var {obj.sexpr()}))")
        complete_smt2 = '\n'.join(lines)

        p = Process(target=solve_objective,
                   args=(complete_smt2, i, minimize, engine, solver_name,
                         result_queue, error_queue))
        processes.append(p)
        p.start()

    completed = 0
    start_time = time.time()

    try:
        while completed < len(objectives):
            if time.time() - start_time > timeout:
                logger.warning("Global timeout reached")
                break

            try:
                result = result_queue.get_nowait()
                results[result.objective_id] = result.value
                completed += 1
                logger.info(f"Objective {result.objective_id} completed in {result.solve_time:.2f}s "
                          f"with value {result.value}")
            except mp.queues.Empty:
                pass

            try:
                obj_id, error_msg = error_queue.get_nowait()
                logger.error(f"Error in objective {obj_id}: {error_msg}")
                completed += 1
            except mp.queues.Empty:
                pass

            time.sleep(0.1)

    finally:
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1)
                if p.is_alive():
                    os.kill(p.pid, signal.SIGKILL)

    return results

def demo():
    """Demo: parallel boxed optimization with 8-bit wrapped arithmetic."""
    x, y = z3.BitVecs('x y', 8)
    formula = z3.And(z3.UGE(x, 0), z3.UGE(y, 0), z3.ULE(x + y, 10))

    for engine, solver in [("qsmt", "z3"), ("maxsat", "FM"), ("iter", "z3-ls")]:
        print(f"\nTrying {engine} engine with {solver} solver:")
        try:
            results = solve_boxed_parallel(formula, [x, y], False, engine, solver)
            print(f"Results: {results}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demo()
