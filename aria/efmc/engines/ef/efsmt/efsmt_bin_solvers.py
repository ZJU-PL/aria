"""
For calling SMT (that support quantified formulas) and QBF solvers

The available APIs:
- solve_with_bin_qbf: solve QBF via bin QBF solvers
- solve_with_bin_smt: solve SMT via bin SMT solvers
"""

import os
import time
import subprocess
import logging
import uuid
from typing import List
from threading import Timer

import z3

from aria.efmc.efmc_config import (
    z3_exec,
    cvc5_exec,
    g_bin_solver_timeout,
    caqe_exec,
    btor_exec,
    bitwuzla_exec,
    yices_exec,
    math_exec,
    q3b_exec,
)

logger = logging.getLogger(__name__)


def terminate(process, is_timeout_flag: List):
    """Terminates a process and sets the timeout flag to True."""
    if process.poll() is None:
        try:
            process.terminate()
            # Wait briefly for graceful termination
            for _ in range(10):
                if process.poll() is not None:
                    break
                time.sleep(0.1)
            # Force kill if still running
            if process.poll() is None:
                process.kill()
            is_timeout_flag[0] = True
        except (OSError, ProcessLookupError) as ex:
            # Process may have already terminated or doesn't exist
            logger.error("Error while interrupting process: %s", str(ex))
            try:
                process.kill()
            except (OSError, ProcessLookupError):
                pass
            is_timeout_flag[0] = True
        except Exception as ex:  # pylint: disable=broad-exception-caught
            # Catch-all for any other unexpected errors during process termination
            logger.error("Unexpected error while interrupting process: %s", str(ex))
            try:
                process.kill()
            except Exception:  # pylint: disable=broad-exception-caught
                pass
            is_timeout_flag[0] = True


def _run_solver_with_timeout(cmd, timeout=g_bin_solver_timeout):
    """Run solver command with timeout and return output."""
    # Using Popen directly due to timer-based termination logic
    p = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )  # pylint: disable=consider-using-with
    is_timeout_flag = [False]
    timer = Timer(timeout, terminate, args=[p, is_timeout_flag])
    timer.start()

    try:
        out_lines = p.stdout.readlines()
        out_str = " ".join([element.decode("UTF-8") for element in out_lines])
    finally:
        p.stdout.close()
        timer.cancel()
        if p.poll() is None:
            p.terminate()

    return out_str, is_timeout_flag[0]


def solve_with_bin_qbf(fml_str: str, solver_name: str):
    """Call bin QBF solvers"""
    print(f"Solving QBF via {solver_name}")
    tmp_filename = f"/tmp/{uuid.uuid1()}_temp.qdimacs"

    try:
        with open(tmp_filename, "w", encoding="utf-8") as tmp:
            tmp.write(fml_str)

        cmd = [caqe_exec, tmp_filename]  # Default to caqe for all QBF solvers
        out, is_timeout_flag = _run_solver_with_timeout(cmd)

        print(out)
        if is_timeout_flag:
            return "unknown"
        if "unsatisfiable" in out:
            return "unsat"
        if "satisfiable" in out:
            return "sat"
        return "unknown"
    finally:
        if os.path.isfile(tmp_filename):
            os.remove(tmp_filename)


def solve_with_bin_smt(  # pylint: disable=too-many-locals
    logic: str,
    x: List[z3.ExprRef],
    y: List[z3.ExprRef],
    phi: z3.ExprRef,
    solver_name: str,
):
    """Call bin SMT solvers to solve exists forall"""
    logger.debug("Solving EFSMT(BV) via %s", solver_name)

    # Build SMT-LIB formula
    fml_str = f"(set-logic {logic})\n"

    # Declare exists variables (remove duplicates)
    exits_vars_names = set()
    for v in x:
        name = str(v)
        if name not in exits_vars_names:
            exits_vars_names.add(name)
            fml_str += f"(declare-const {v.sexpr()} {v.sort().sexpr()})\n"

    # Build quantified formula
    quant_vars = "(" + " ".join(f"({v.sexpr()} {v.sort().sexpr()})" for v in y) + ")\n"
    quant_fml_body = (
        "(and \n" + "\n".join(f"  {fml.sexpr()}" for fml in phi.children()) + ")"
    )
    fml_str += f"(assert (forall {quant_vars} {quant_fml_body}))\n(check-sat)\n"

    # Get solver command
    solver_cmds = {
        "z3": [z3_exec],
        "cvc5": [cvc5_exec, "-q", "--produce-models"],
        "btor": [btor_exec],
        "boolector": [btor_exec],
        "yices2": [yices_exec],
        "mathsat": [math_exec],
        "bitwuzla": [bitwuzla_exec],
        "q3b": [q3b_exec],
    }
    cmd = solver_cmds.get(solver_name, [z3_exec])

    tmp_filename = f"/tmp/{uuid.uuid1()}_temp.smt2"

    try:
        with open(tmp_filename, "w", encoding="utf-8") as tmp:
            tmp.write(fml_str)

        cmd.append(tmp_filename)
        out, is_timeout_flag = _run_solver_with_timeout(cmd)

        if is_timeout_flag:
            return "unknown"
        if "unsat" in out:
            return "unsat"
        if "sat" in out:
            return "sat"
        return "unknown"
    finally:
        if os.path.isfile(tmp_filename):
            os.remove(tmp_filename)


if __name__ == "__main__":
    # Demo functionality
    demo_cmd = [cvc5_exec, "-q", "--produce-models", "tmp.smt2"]
    demo_out, demo_is_timeout = _run_solver_with_timeout(demo_cmd)
    print(demo_out)
