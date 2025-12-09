"""
For calling bin solvers
"""
import os
import subprocess
import logging
import uuid
from typing import List, Dict, Callable
from threading import Timer

import z3

from aria.global_params import global_config

logger = logging.getLogger(__name__)
BIN_SOLVER_TIMEOUT = 100
# Result = Literal["sat", "unsat", "unknown"]

def terminate(process, is_timeout: List):
    """Terminate a process and set timeout flag."""
    if process.poll() is None:
        try:
            process.terminate()
            is_timeout[0] = True
            logger.debug("Process terminated due to timeout.")
        except Exception as ex:
            logger.error("Error interrupting process: %s", ex)


def get_solver_command(solver_type: str, solver_name: str, tmp_filename: str) -> List[str]:
    """Get the command to run the specified solver."""
    # Map solver names to GlobalConfig names
    solver_name_map = {
        "yices": "yices2",
    }
    config_solver_name = solver_name_map.get(solver_name, solver_name)

    # Get solver path using the GlobalConfig API
    def get_path(solver: str) -> str:
        path = global_config.get_solver_path(solver)
        if path is None:
            raise RuntimeError(f"Solver {solver} not found. Please ensure it is installed.")
        return path

    # Define solver commands (lazy evaluation - paths are resolved when needed)
    solver_configs: Dict[str, Dict[str, Callable[[], List[str]]]] = {
        "smt": {
            "z3": lambda: [get_path("z3"), tmp_filename],
            "cvc5": lambda: [get_path("cvc5"), "-q", "--produce-models", tmp_filename],
            "yices": lambda: [get_path("yices2"), tmp_filename],
            "mathsat": lambda: [get_path("mathsat"), tmp_filename],
        },
        "maxsat": {
            "z3": lambda: [get_path("z3"), tmp_filename],
        }
    }

    # Get command factory for the specific solver
    cmd_factory = solver_configs.get(solver_type, {}).get(solver_name)
    if cmd_factory is None:
        # Default to z3
        return [get_path("z3"), tmp_filename]

    return cmd_factory()


def run_solver(cmd: List[str]) -> str:
    """Run solver command and handle timeout."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    is_timeout = [False]
    timer = Timer(BIN_SOLVER_TIMEOUT, terminate, args=[p, is_timeout])

    try:
        timer.start()
        out = p.stdout.readlines()
        out = ' '.join([line.decode('UTF-8') for line in out])

        if is_timeout[0]:
            return "unknown"
        elif "unsat" in out:
            return out
        elif "sat" in out:
            return out
        else:
            return "unknown"
    finally:
        timer.cancel()
        if p.poll() is None:
            p.terminate()
        p.stdout.close()


def solve_with_bin_smt(logic: str, qfml: z3.ExprRef, obj_name: str, solver_name: str) -> str:
    """Call binary SMT solvers to solve quantified SMT problems."""
    logger.debug(f"Solving QSMT via {solver_name}")

    # Prepare SMT2 formula
    fml_str = "(set-option :produce-models true)\n"
    fml_str += f"(set-logic {logic})\n"
    s = z3.Solver()
    s.add(qfml)
    fml_str += s.to_smt2()
    fml_str += f"(get-value ({obj_name}))\n"

    # Create temporary file
    tmp_filename = f"/tmp/{uuid.uuid1()}_temp.smt2"
    try:
        with open(tmp_filename, "w") as tmp:
            tmp.write(fml_str)

        cmd = get_solver_command("smt", solver_name, tmp_filename)
        logger.debug("Command: %s", cmd)
        return run_solver(cmd)
    finally:
        if os.path.isfile(tmp_filename):
            os.remove(tmp_filename)


def solve_with_bin_maxsat(wcnf: str, solver_name: str) -> str:
    """Solve weighted MaxSAT via binary solvers."""
    logger.debug(f"Solving MaxSAT via {solver_name}")

    tmp_filename = f"/tmp/{uuid.uuid1()}_temp.wcnf"
    try:
        with open(tmp_filename, "w") as tmp:
            tmp.write(wcnf)

        cmd = get_solver_command("maxsat", solver_name, tmp_filename)
        logger.debug("Command: %s", cmd)
        return run_solver(cmd)
    finally:
        if os.path.isfile(tmp_filename):
            os.remove(tmp_filename)


def demo_solver():
    """Demo function to test solver functionality."""
    z3_path = global_config.get_solver_path("z3")
    if z3_path is None:
        raise RuntimeError("Z3 solver not found. Please ensure Z3 is installed.")
    cmd = [z3_path, 'tmp.smt2']
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    is_timeout = [False]
    timer = Timer(BIN_SOLVER_TIMEOUT, terminate, args=[p, is_timeout])

    try:
        timer.start()
        out = p.stdout.readlines()
        out = ' '.join([line.decode('UTF-8') for line in out])
        print(out)
    finally:
        timer.cancel()
        if p.poll() is None:
            p.terminate()
        p.stdout.close()


if __name__ == "__main__":
    demo_solver()
