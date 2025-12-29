"""
For calling SMT and QBF solvers
"""
import os
import time
from typing import List
import subprocess
from threading import Timer
import logging
import uuid

import z3

from aria.utils.smtlib_solver import SMTLIBSolver
from aria.global_params.paths import global_config

G_BIN_SOLVER_TIMEOUT = 100

z3_exec = global_config.get_solver_path("z3")
cvc5_exec = global_config.get_solver_path("cvc5")
yices_exec = global_config.get_solver_path("yices2")

# FIXME: the followings do not exist
caqe_exec = global_config.get_solver_path("caqe")
btor_exec = global_config.get_solver_path("btor")
bitwuzla_exec = global_config.get_solver_path("bitwuzla")
math_exec = global_config.get_solver_path("mathsat")
q3b_exec = global_config.get_solver_path("q3b")
# caqe_exec, \   btor_exec, bitwuzla_exec, math_exec, q3b_exec

logger = logging.getLogger(__name__)


def terminate(process, is_timeout: List):
    """
    Terminate a process and set the timeout flag to True.

    Args:
        process: The process to be terminated.
        is_timeout: A list containing a single boolean item. If the process
            exceeds the timeout limit, the boolean item will be set to True.
    """
    if process.poll() is None:
        try:
            process.terminate()
            is_timeout[0] = True
        except Exception:
            print("error for interrupting")


def solve_with_bin_qbf(fml_str: str, solver_name: str):
    """Call binary QBF solvers."""
    print(f"Solving QBF via {solver_name}")
    tmp_filename = f"/tmp/{uuid.uuid1()}_temp.qdimacs"
    try:
        with open(tmp_filename, "w", encoding="utf-8") as tmp:
            tmp.write(fml_str)
        if solver_name == "caqe":
            cmd = [caqe_exec, tmp_filename]
        else:
            cmd = [caqe_exec, tmp_filename]
        # print(cmd)
        p_gene = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        is_timeout_gene = [False]
        timer_gene = Timer(G_BIN_SOLVER_TIMEOUT, terminate,
                          args=[p_gene, is_timeout_gene])
        timer_gene.start()
        out_gene = p_gene.stdout.readlines()
        out_gene = ' '.join([str(element.decode('UTF-8')) for element in out_gene])
        p_gene.stdout.close()  # close?
        timer_gene.cancel()
        if p_gene.poll() is None:
            p_gene.terminate()  # TODO: need this?

        print(out_gene)
        if is_timeout_gene[0]:
            return "unknown"
        if "unsatisfiable" in out_gene:
            return "unsat"
        if "satisfiable" in out_gene:
            return "sat"
        return "unknown"
    finally:
        if os.path.isfile(tmp_filename):
            os.remove(tmp_filename)


def solve_with_bin_smt(logic: str, x: List[z3.ExprRef], y: List[z3.ExprRef],
                       phi: z3.ExprRef, solver_name: str):
    """Call binary SMT solvers to solve exists forall.

    In this version, we create a temp file, and ...
    """
    logger.debug("Solving EFSMT(BV) via %s", solver_name)
    fml_str = f"(set-logic {logic})\n"
    # there are duplicates in self.exists_vars???
    dump_strategy = 1

    if dump_strategy == 1:
        # there are duplicates in self.exists_vars???
        exits_vars_names = set()
        for v in x:
            name = str(v)
            if name not in exits_vars_names:
                exits_vars_names.add(name)
                fml_str += f"(declare-const {v.sexpr()} {v.sort().sexpr()})\n"
        # print(exits_vars_names)

        quant_vars = "("
        for v in y:
            quant_vars += f"({v.sexpr()} {v.sort().sexpr()}) "
        quant_vars += ")\n"

        quant_fml_body = "(and \n"
        s = z3.Solver()
        s.add(phi)
        # self.phi is in the form of
        #  and (Init, Trans, Post)
        assert z3.is_app(phi)
        for fml in phi.children():
            quant_fml_body += "  {}\n".format(fml.sexpr())
        quant_fml_body += ")"

        fml_body = f"(assert (forall {quant_vars} {quant_fml_body}))\n"
        fml_str += fml_body
        fml_str += "(check-sat)\n"
    else:
        # Another more direct strategy
        # But we cannot see the definition of the VC clearly
        sol = z3.Solver()
        sol.add(z3.ForAll(y, phi))
        fml_str += sol.to_smt2()

    tmp_filename = f"/tmp/{uuid.uuid1()}_temp.smt2"
    try:
        with open(tmp_filename, "w", encoding="utf-8") as tmp:
            tmp.write(fml_str)
        if solver_name == "z3":
            cmd = [z3_exec, tmp_filename]
        elif solver_name == "cvc5":
            cmd = [cvc5_exec, "-q", "--produce-models", tmp_filename]
        elif solver_name in ("btor", "boolector"):
            cmd = [btor_exec, tmp_filename]
        elif solver_name == "yices2":
            cmd = [yices_exec, tmp_filename]
        elif solver_name == "mathsat":
            cmd = [math_exec, tmp_filename]
        elif solver_name == "bitwuzla":
            cmd = [bitwuzla_exec, tmp_filename]
        elif solver_name == "q3b":
            cmd = [q3b_exec, tmp_filename]
        else:
            print("Cannot find corresponding solver")
            cmd = [z3_exec, tmp_filename]
        # print(cmd)
        p_gene = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        is_timeout_gene = [False]
        timer_gene = Timer(G_BIN_SOLVER_TIMEOUT, terminate,
                          args=[p_gene, is_timeout_gene])
        timer_gene.start()
        out_gene = p_gene.stdout.readlines()
        out_gene = ' '.join([str(element.decode('UTF-8')) for element in out_gene])
        p_gene.stdout.close()  # close?
        timer_gene.cancel()
        if p_gene.poll() is None:
            p_gene.terminate()  # TODO: need this?

        if is_timeout_gene[0]:
            return "unknown"
        if "unsat" in out_gene:
            return "unsat"
        if "sat" in out_gene:
            return "sat"
        return "unknown"
    finally:
        if os.path.isfile(tmp_filename):
            os.remove(tmp_filename)


def solve_with_bin_smt_v2(logic: str, y, phi: z3.ExprRef, solver_name: str):
    """Call binary SMT solvers to solve exists forall.

    In this version, I use the SMLIBSolver (We can send strings to the bin solver)
    """
    smt2string = f"(set-logic {logic})\n"
    sol = z3.Solver()
    sol.add(z3.ForAll(y, phi))
    smt2string += sol.to_smt2()

    # bin_cmd = ""
    if solver_name == "z3":
        bin_cmd = z3_exec
    elif solver_name == "cvc5":
        bin_cmd = cvc5_exec + " -q --produce-models"
    else:
        bin_cmd = z3_exec

    bin_solver = SMTLIBSolver(bin_cmd)
    start = time.time()
    res = bin_solver.check_sat_from_scratch(smt2string)
    if res == "sat":
        # print(bin_solver.get_expr_values(["p1", "p0", "p2"]))
        print(f"External solver success time: {time.time() - start}")
        # TODO: get the model to build the invariant
    elif res == "unsat":
        print(f"External solver fails time: {time.time() - start}")
    else:
        print("Seems timeout or error in the external solver")
        print(res)
    bin_solver.stop()
    return res


def demo_solver():
    """Demo function for solver."""
    cmd = [cvc5_exec, "-q", "--produce-models", 'tmp.smt2']
    p_gene = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    is_timeout_gene = [False]
    timer_gene = Timer(G_BIN_SOLVER_TIMEOUT, terminate,
                      args=[p_gene, is_timeout_gene])
    timer_gene.start()
    out_gene = p_gene.stdout.readlines()
    out_gene = ' '.join([str(element.decode('UTF-8')) for element in out_gene])
    p_gene.stdout.close()  # close?
    timer_gene.cancel()
    if p_gene.poll() is None:
        p_gene.terminate()  # TODO: need this?

    print(out_gene)


if __name__ == "__main__":
    demo_solver()
