"""
Call other SMT solvers to manipulate Z3's expr (e.g., quantifier elimination,
abduction, etc.)

Use z3 as the default solve for handling "normal queries" (e.g., sat, equiv,
entail, etc, but use cvc5 and other solvers for specific queries, e.g.,
  - binary interpolant (cvc5, SMTInterpol, mathsat), sequence interpolant
    (SMTInterpol)
  - non-linear interpolant (Yices2)
  - abduction (cvc5)
  - quantifier elimination (cvc5, z3,..)
  - sygus (cvc5)
  - OMT
  - MaxSMT
  - etc.
"""

import logging
import subprocess
from enum import Enum, auto
from threading import Timer
from typing import List

import z3

from aria.utils.smtlib_solver import SmtlibProc
from aria.global_params.paths import global_config

logger = logging.getLogger(__name__)


class BinaryInterpolSolver(Enum):
    CVC5 = auto()
    MATHSAT5 = auto()
    SMTINTERPOL = auto()


class SequenceInterpolSolver(Enum):
    MATHSAT = auto()


class OMTSolver(Enum):
    CVC5 = auto()
    OPTIMATHSAT = auto()


def terminate(process, is_timeout):
    """
    Terminates a given process if it is still running.
    Args:
        process (subprocess.Popen): The process to terminate.
        is_timeout (list): A list containing a single boolean element.
                           It will be set to True if the process is terminated.
    """
    if process.poll() is None:
        try:
            process.terminate()
            is_timeout[0] = True
        except Exception as ex:
            print(f"Error while attempting to terminate the process: {ex}")
            try:
                # Attempt to forcefully kill the process as a fallback
                process.kill()
                print("Process forcefully killed.")
            except Exception as kill_ex:
                print(f"Error while attempting to kill the process: {kill_ex}")


def solve_with_bin_solver(cmd, timeout=300):
    """
    Solve a Z3 expression with an external binary solver
    """
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as p:
        is_timeout = [False]
        timer = Timer(timeout, terminate, args=[p, is_timeout])
        timer.start()
        out = p.stdout.readlines()
        out = " ".join([str(element.decode("UTF-8")) for element in out])
        timer.cancel()
        if p.poll() is None:
            p.terminate()
        if is_timeout[0]:
            return "timeout"
        return out


class Z3SolverPlus(z3.Solver):

    def __init__(self, debug=False):
        super().__init__()

        self.debug = debug

        # Get CVC5 path from global config
        cvc5_path = global_config.get_solver_path("cvc5")
        if not cvc5_path:
            logger.warning(
                "CVC5 not found in global config. " "Using 'cvc5' command from PATH."
            )
            cvc5_path = "cvc5"

        # Use CVC5 path from global_params
        # abductve inference
        self.abduction_solver = f"{cvc5_path} --produce-abducts -q"
        # quantifier elimination
        self.binary_qe_solver = f"{cvc5_path} -q --produce-models"
        # binary interpolant (NOTE: cvc5 uses the original definition of
        # interpolant)
        self.binary_interpol_solver = f"{cvc5_path} --produce-interpols=default -q"
        # sequence interpolant
        self.sequence_interpol_solver = None
        # implicant/implicate
        self.implicant_solver = None
        # model interpolant (interpolant subject to a model), a recent special feature of Yices2
        self.model_interpol_solver = None
        # sygus
        self.sygus_solver = f"{cvc5_path} -q --lang=sygus2"
        # OMT, MaxSMT
        self.omt_solver = "optimathsat -optimization=true -model_generation=true"
        # all-sat
        self.all_sat_solver = "optimathsat -model_generation=true"

    def binary_interpolant(self, pre: z3.BoolRef, post: z3.BoolRef, logic=None):
        """
        Binary interpolant
        - It seems that cvc5's interpolant follows the original definition,
          i.e., A |= I, I |= B
        - Need to use different strategies when using other solvers
          (e.g., using pysmt's APIs)

        Example from cvc5
        (set-logic LIA)
        (declare-fun a () Int)
        (assert (> a 1))
        (get-interpol A (> a 0))
        """
        # for unifying type signatures
        unify_solver = z3.Solver()
        unify_solver.add(z3.And(pre, post))
        # z3_vars = get_variables(z3.And(pre, post))
        signature = ""
        for line in unify_solver.to_smt2().split("\n"):
            if line.startswith("(as"):
                break
            signature += f"{line}\n"
        itp_cmd = f"(get-interpol A {post.sexpr()})"

        smtlib = SmtlibProc(self.binary_interpol_solver, debug=self.debug)
        smtlib.start()
        try:
            if logic:
                smtlib.send(f"(set-logic {logic})")
            smtlib.send(signature)
            smtlib.send(f"(assert {pre.sexpr()})\n")
            smtlib.send(itp_cmd)
            itp = smtlib.recv()
            if "error" in itp or "none" in itp:
                ret = z3.BoolVal(False)
            else:
                ret = z3.And(z3.parse_smt2_string(signature + itp + "\n(assert A)"))
            smtlib.stop()
            return ret
        except Exception:
            smtlib.stop()
            return False

    def sequence_interpolant(self):
        """
        Sequence interpolant
        - Need to use different strategies when using other solvers
          (e.g., using pysmt's APIs)
        """
        smtlib = SmtlibProc(self.sequence_interpol_solver, debug=self.debug)
        smtlib.start()
        # FIXME here
        smtlib.stop()

    def check_external(self, fml: z3.ExprRef):
        """
        Check-sat using an external binary solver
        """
        # assert len(self.assertions()) > 0
        self.add(fml)
        smtlib = SmtlibProc(self.binary_qe_solver, debug=self.debug)
        smtlib.start()
        smtlib.send(self.to_smt2())
        smtlib.recv()
        smtlib.send("(get-model)")
        res = smtlib.recv()
        smtlib.stop()
        return res

    def qelim(self, qfml: z3.ExprRef, logic=None):
        """
        Quantifier elimination using an external solver
        """
        self.add(qfml)
        signature = ""
        for line in self.to_smt2().split("\n"):
            if line.startswith("(as"):
                break
            signature += f"{line}\n"
        qe_cmd = f"(get-qe {qfml.sexpr()})"
        smtlib = SmtlibProc(self.binary_qe_solver, debug=self.debug)
        smtlib.start()
        if logic:
            smtlib.send(f"(set-logic {logic})")
        smtlib.send(signature)
        smtlib.send(qe_cmd)
        qe_res = smtlib.recv()
        if "error" in qe_res or "none" in qe_res:
            # FIXME: sometimes the result is also exactly False
            # we need another approach for labeling the status
            ret = z3.BoolVal(False)
        else:
            ret = z3.And(z3.parse_smt2_string(signature + f"(assert {qe_res})"))
        smtlib.stop()
        return ret

    def abduct(self, pre: z3.ExprRef, post: z3.ExprRef, logic=None):
        """
        Abduction with CVC5
        """
        # for unifying type signatures
        unify_solver = z3.Solver()
        unify_solver.add(z3.And(pre, post))
        # z3_vars = get_variables(z3.And(pre, post))
        signature = ""
        for line in unify_solver.to_smt2().split("\n"):
            if line.startswith("(as"):
                break
            signature += f"{line}\n"
        smtlib = SmtlibProc(self.abduction_solver, debug=self.debug)
        smtlib.start()
        if logic:
            smtlib.send(f"(set-logic {logic})")
        smtlib.send(signature)
        smtlib.send(f"(assert {pre.sexpr()})\n")
        abd_cmd = f"(get-abduct A {post.sexpr()})"
        smtlib.send(abd_cmd)
        abd = smtlib.recv()
        if "error" in abd or "none" in abd:
            ret = z3.BoolVal(False)
        else:
            print("abd ", abd)
            ret = z3.And(z3.parse_smt2_string(signature + abd + "\n(assert A)"))
        smtlib.stop()
        return ret

    def sygus(
        self,
        funcs: List[z3.FuncDeclRef],
        cnts: List[z3.BoolRef],
        all_vars: List[z3.ExprRef],
        grammar=None,
        logic=None,
        pbe=False,
    ):
        """
        SyGuS with CVC5
        """
        cmds = [f"(set-logic {logic})"]
        for func in funcs:
            target = f"(synth-fun {func.name()} ("
            for i in range(func.arity()):
                target += f"({all_vars[i]} {func.domain(i).sexpr()}) "
            target += f") {func.range().sexpr()})"  # return value
            cmds.append(target)
        for v in all_vars:
            cmds.append(f"(declare-var {v} {v.sort().sexpr()})")
        for c in cnts:
            cmds.append(f"(constraint {c.sexpr()})")
        cnt = "\n".join(cmds)
        sygus_cmd = self.sygus_solver
        if logic == "BV":
            sygus_cmd += " --cegqi-bv"
        if pbe:
            sygus_cmd += " --sygus-pbe"
        else:
            sygus_cmd += " --no-sygus-pbe"
        smtlib = SmtlibProc(sygus_cmd, debug=self.debug)
        smtlib.start()
        smtlib.send(cnt)
        # print(cnt)
        # TODO: strangely, seems that we need to use an independent check-synth cmd...?
        smtlib.send("(check-synth)\n")
        res = smtlib.recv()
        smtlib.stop()
        return res

    def optimize(self, fml: z3.ExprRef, obj: z3.ExprRef, minimize=False, logic=None):
        """
        Optimize one objective
        """
        s = z3.Optimize()
        s.add(fml)

        signature_vec = []
        for line in s.sexpr().split("\n"):
            if line.startswith("(as"):
                break
            signature_vec.append(line)
        if "Int" in signature_vec[0]:
            signature_vec.append("(declare-const oo Int)")
        elif "Real" in signature_vec[0]:
            signature_vec.append("(declare-const oo Real)")
            signature_vec.append("(declare-const epsilon Real)")

        if minimize:
            s.minimize(obj)
        else:
            s.maximize(obj)
        # print(s.sexpr())

        smtlib = SmtlibProc(self.omt_solver, debug=self.debug)
        smtlib.start()
        if logic:
            smtlib.send(f"(set-logic {logic})")
        smtlib.send(s.sexpr())  # interesting API sexpr()...
        smtlib.recv()
        smtlib.send("(get-objectives)")
        res = smtlib.recv()
        smtlib.stop()

        # TODO: I only tried optimathsat
        if minimize:
            cnt = f"(assert (>= {res.split(chr(10))[1][2:-1]}))"
        else:
            cnt = f"(assert (<= {res.split(chr(10))[1][2:-1]}))"

        # print(res)
        # print(z3.And(z3.parse_smt2_string("\n".join(signature_vec) + cnt)))
        return z3.And(z3.parse_smt2_string("\n".join(signature_vec) + cnt))

    def compute_min_max(
        self, fml: z3.ExprRef, minimize: List, maximize: List, logic=None
    ):
        s = z3.Optimize()
        s.add(fml)

        # for parsing the results
        signature_vec = []
        for line in s.sexpr().split("\n"):
            if line.startswith("(as"):
                break
            signature_vec.append(line)
        if "Int" in signature_vec[0]:
            signature_vec.append("(declare-const oo Int)")
        elif "Real" in signature_vec[0]:
            signature_vec.append("(declare-const oo Real)")
            signature_vec.append("(declare-const epsilon Real)")

        for e in minimize:
            s.minimize(e)
        for e in maximize:
            s.maximize(e)

        # print(s.sexpr())
        smtlib = SmtlibProc(self.omt_solver + " -opt.priority=box", debug=self.debug)
        smtlib.start()
        if logic:
            smtlib.send(f"(set-logic {logic})")
        # print(s.sexpr())
        smtlib.send(s.sexpr())
        smtlib.recv()
        smtlib.send("(get-objectives)")
        res = smtlib.recv()
        smtlib.stop()
        # to z3 expr
        # E.g., res =
        # (objectives
        #  (x (- oo))   ; min
        #  (y (- oo))   ; min
        #  (x 2)        ; max
        #  (y 7)        ; max
        # )
        # the result should be And(x >= -oo, y >= -oo, x <= 2, y <= 7)
        asserts = []
        res_values = res.split("\n")[1:-1]
        for i, res_val in enumerate(res_values):
            # TODO: I only tried optimathsat
            if i < int(len(res_values) / 2):
                asserts.append(f"(assert (>= {res_val[2:-1]}))")
            else:
                asserts.append(f"(assert (<= {res_val[2:-1]}))")
        # print(asserts)
        return z3.And(
            z3.parse_smt2_string("\n".join(signature_vec) + "\n".join(asserts))
        )

    def all_sat(self, fml: z3.ExprRef, bools: List[z3.ExprRef]):
        """
         Enumerate all the consistent assignments (i.e. solutions) for the given
        list of predicates. Notice that the arguments to check-allsat can only be
        Boolean constants. If you need to enumerate over arbitrary theory atoms,
        you can always "label" them with constants, as done above for
        "(> (+ x y) 0)", labeled by "a"
        ; from https://mathsat.fbk.eu/smt2examples.html
        (declare-fun x () Int)
        (declare-fun y () Int)

        (declare-fun a () Bool)
        (declare-fun b () Bool)
        (declare-fun c () Bool)
        (declare-fun d () Bool)

        (assert (= (> (+ x y) 0) a))
        (assert (= (< (+ (* 2 x) (* 3 y)) (- 10)) c))
        (assert (and (or a b) (or c d)))

        (check-allsat (a b))
        """
        # a trick to avoid (check-sat)
        cmd = self.all_sat_solver.split(" ")
        s = z3.Optimize()
        s.add(fml)
        cmd.append(s.sexpr())
        cmd.append(f"(check-allsat {' '.join([str(b) for b in bools])})\n")
        res = solve_with_bin_solver(cmd)
        return res
