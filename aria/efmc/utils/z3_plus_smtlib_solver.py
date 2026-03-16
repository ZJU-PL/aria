# coding: utf-8
"""
Z3 solver with extended SMT-LIB solver capabilities.

This module provides a Z3 solver wrapper that uses z3 as the default solver for
normal queries (e.g., sat, equiv, entail, etc) but delegates to specialized
external solvers for specific queries:
  - binary interpolant (cvc5, SMTInterpol, mathsat)
  - sequence interpolant (SMTInterpol)
  - non-linear interpolant (Yices2)
  - abduction (cvc5)
  - quantifier elimination (cvc5, z3,..)
  - sygus (cvc5)
  - OMT
  - MaxSMT
"""
from enum import Enum
from typing import List
import subprocess
from threading import Timer
import time
import z3

from aria.efmc.smttools.smtlib_solver import SmtlibProc


def terminate(process, is_timeout):
    """Terminate a process, with timeout flag handling."""
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
            is_timeout[0] = True
        except (OSError, subprocess.SubprocessError):
            try:
                process.kill()
            except (OSError, subprocess.SubprocessError):
                pass
            is_timeout[0] = True


def solve_with_bin_solver(cmd, timeout=300):
    """Solve using a binary solver with timeout."""
    is_timeout = [False]
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as p:
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


class BinaryInterpolSolver(Enum):
    """Enumeration of binary interpolation solvers."""

    CVC5 = 0
    MATHSAT5 = 1
    SMTINTERPOL = 2


class SequenceInterpolSolver(Enum):
    """Enumeration of sequence interpolation solvers."""

    MATHSAT = 0


class OMTSolver(Enum):
    """Enumeration of OMT (Optimization Modulo Theories) solvers."""

    CVC5 = 0
    OPTIMATHSAT = 1


# logger = logging.getLogger(__name__)
# What is making it as a subclass of z3.Solver...


class Z3SolverPlus(z3.Solver):
    """Extended Z3 solver with support for external SMT-LIB solvers."""

    def __init__(self, debug=False):
        super().__init__()

        self.debug = debug
        # abductve inference
        self.abduction_solver = "cvc5 --produce-abducts -q"
        # quantifier elimination
        self.binary_qe_solver = "cvc5 -q --produce-models"
        # binary interpolant (NOTE: cvc5 uses the original definition of interpolant)
        self.binary_interpol_solver = "cvc5 --produce-interpols=default -q"
        # sequence interpolant
        self.sequence_interpol_solver = None
        # implicant/implicate
        self.implicant_solver = None
        # model interpolant (interpolant subject to a model), a recent special feature of Yices2
        self.model_interpol_solver = None
        # sygus
        self.sygus_solver = "cvc5 -q --lang=sygus2"
        # OMT, MaxSMT
        self.omt_solver = "optimathsat -optimization=true -model_generation=true"
        # all-sat
        self.all_sat_solver = "optimathsat -model_generation=true"

    def binary_interpolant(self, pre: z3.BoolRef, post: z3.BoolRef, logic=None):
        """
        Binary interpolant
        - It seems that cvc5's interpolant follows the original definition, i.e., A |= I, I |= B
        - Need to use different strategies when using other solvers

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
        except (ValueError, z3.Z3Exception):
            smtlib.stop()
            return False

    def sequence_interpolant(self):
        """
        Sequence interpolant (not yet implemented).
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
        logic=None,
        pbe=False,
    ):
        """
        SyGuS with CVC5.

        Args:
            funcs: List of function declarations to synthesize
            cnts: List of constraints
            all_vars: List of all variables
            logic: SMT-LIB logic to use
            pbe: Whether to use PBE (Programming by Examples) mode
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
            cnt = f"(assert (>= {res.split(chr(10))[1][2:-1]}))"  # noqa: E501
        else:
            cnt = f"(assert (<= {res.split(chr(10))[1][2:-1]}))"  # noqa: E501

        # print(res)
        # print(z3.And(z3.parse_smt2_string("\n".join(signature_vec) + cnt)))
        return z3.And(z3.parse_smt2_string("\n".join(signature_vec) + cnt))

    def compute_min_max(
        self, fml: z3.ExprRef, minimize: List, maximize: List, logic=None
    ):
        """
        Compute min/max for multiple objectives.

        Args:
            fml: Formula to optimize
            minimize: List of expressions to minimize
            maximize: List of expressions to maximize
            logic: SMT-LIB logic to use
        """
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
        enumerate all the consistent assignments (i.e. solutions) for the given list of predicates.
        Notice that the arguments to check-allsat can only be Boolean constants.

        TODO: maybe we can use solve_with_bin_solver for other interfaces...
              why solve_with_
        """
        # a trick to avoid (check-sat)
        cmd = self.all_sat_solver.split(" ")
        s = z3.Optimize()
        s.add(fml)
        cmd.append(s.sexpr())
        cmd.append(f"(check-allsat {' '.join([str(b) for b in bools])})\n")
        res = solve_with_bin_solver(cmd)
        return res
