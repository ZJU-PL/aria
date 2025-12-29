"""Some basic functions for efsmt
"""
import time
import z3

from aria.global_params import global_config
from aria.utils.smtlib_solver import SMTLIBSolver


def solve_with_bin_smt(y, phi: z3.ExprRef, logic: str, solver_name: str):
    """Call bin SMT solvers to solve exists forall"""
    smt2string = f"(set-logic {logic})\n"
    sol = z3.Solver()
    sol.add(z3.ForAll(y, phi))
    smt2string += sol.to_smt2()

    # Note: build/download bin solvers in the project
    # bin_cmd = ""
    if solver_name == "z3":
        bin_cmd = global_config.get_solver_path("z3")
    elif solver_name == "cvc5":
        bin_cmd = global_config.get_solver_path("cvc5") + " -q --produce-models"
    else:
        bin_cmd = global_config.get_solver_path("z3")

    bin_solver = SMTLIBSolver(bin_cmd)
    start = time.time()
    res = bin_solver.check_sat_from_scratch(smt2string)
    if res == "sat":
        # print(bin_solver.get_expr_values(["p1", "p0", "p2"]))
        print("External solver success time: ", time.time() - start)
        # Note: get the model to build the invariant
    elif res == "unsat":
        print("External solver fails time: ", time.time() - start)
    else:
        print("Seems timeout or error in the external solver")
        print(res)
    bin_solver.stop()
    return res
