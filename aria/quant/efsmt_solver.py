# coding: utf-8
"""
The uniformed interface for solving Exists-ForAll problems
"""
import logging
from enum import Enum
from typing import List

import z3
from z3.z3util import get_vars

from aria.quant.efsmt_utils import solve_with_bin_smt
from aria.utils.z3_expr_utils import get_variables

logger = logging.getLogger(__name__)


class EFSMTStrategy(Enum):
    Z3 = 0,  # via bin solver
    CVC5 = 1,  # via bin solver
    BOOLECTOR = 2,  # via bin solver
    Yices2 = 3,  # via bin solver
    SIMPLE_CEGAR = 4,
    QBF = 5,
    SAT = 6,
    PARALLEL_CEGAR = 7


def simple_cegar_efsmt(logic: str, y: List[z3.ExprRef], phi: z3.ExprRef, maxloops=None):
    """ Solves exists x. forall y. phi(x, y) with simple CEGAR
    """
    # x = [item for item in get_vars(phi) if item not in y]   # can be slow
    x = [item for item in get_variables(phi) if item not in y]
    # set_param("verbose", 15)
    # set_param("smt.arith.solver", 3)
    if "IA" in logic:
        qf_logic = "QF_LIA"
    elif "RA" in logic:
        qf_logic = "QF_LRA"
    elif "BV" in logic:
        qf_logic = "QF_BV"
    else:
        qf_logic = ""

    if qf_logic != "":
        esolver = z3.SolverFor(qf_logic)
        fsolver = z3.SolverFor(qf_logic)
    else:
        esolver = z3.Solver()
        fsolver = z3.Solver()

    esolver.add(z3.BoolVal(True))

    loops = 0
    while maxloops is None or loops <= maxloops:
        loops += 1
        # print("round: ", loops)
        eres = esolver.check()
        if eres == z3.unsat:
            return z3.unsat
        else:
            emodel = esolver.model()
            mappings = [(var, emodel.eval(var, model_completion=True)) for var in x]
            sub_phi = z3.simplify(z3.substitute(phi, mappings))
            fsolver.push()
            fsolver.add(z3.Not(sub_phi))
            if fsolver.check() == z3.sat:
                fmodel = fsolver.model()
                y_mappings = [(var, fmodel.eval(var, model_completion=True)) for var in y]
                sub_phi = z3.simplify(z3.substitute(phi, y_mappings))
                esolver.add(sub_phi)
                fsolver.pop()
            else:
                return z3.sat
    return z3.unknown


class EFSMTSolver:
    """Solving exists forall problem"""

    def __init__(self, logic: str):
        self.tactic = EFSMTStrategy.SIMPLE_CEGAR
        self.logic = logic

    def set_tactic(self, name: str):
        raise NotImplementedError

    def solve(self, y: List[z3.ExprRef], phi: z3.ExprRef):
        """Translate EFSMT(BV) to QBF (preserve the quantifiers)
        :param y: variables to be universal quantified
        :param phi: a quantifier-free formula
        """
        if self.tactic == EFSMTStrategy.Z3 or self.tactic == EFSMTStrategy.CVC5 \
                or self.tactic == EFSMTStrategy.BOOLECTOR or self.tactic == EFSMTStrategy.Yices2:
            return self.solve_with_qsmt(y, phi)
        elif self.tactic == EFSMTStrategy.SIMPLE_CEGAR:
            return self.solve_with_simple_cegar(y, phi)
        elif self.tactic == EFSMTStrategy.PARALLEL_CEGAR:
            return self.solve_with_parallel_cegar(y, phi)
        else:
            return self.internal_solve(y, phi)

    def internal_solve(self, y: List[z3.ExprRef], phi: z3.ExprRef) -> z3.CheckSatResult:
        """Call Z3's Python API"""
        qfml = z3.ForAll(y, phi)
        s = z3.SolverFor(self.logic)  # can be very fast
        s.add(qfml)
        return s.check()

    def solve_with_simple_cegar(self, y: List[z3.ExprRef], phi: z3.ExprRef):
        """Solve with a CEGAR-style algorithm, which consists of a "forall solver" and an "exists solver"
        """
        # This can be slow (perhaps not a good idea for NRA) Maybe good for LRA or BV?
        print("Simple, sequential, CEGAR-style EFSMT!")
        z3_res, model = simple_cegar_efsmt(self.logic, y, phi)
        return z3_res, model

    def solve_with_parallel_cegar(self, y: List[z3.ExprRef], phi: z3.ExprRef):
        """This should be the focus"""
        print("Parallel EFSMT starting!!!")
        z3_res, model = simple_cegar_efsmt(self.logic, y, phi)
        return z3_res, model

    def solve_with_qsmt(self, y: List[z3.ExprRef], phi: z3.ExprRef) -> z3.CheckSatResult:
        """Solve with bin solvers"""
        if self.tactic == EFSMTStrategy.Z3:
            return solve_with_bin_smt(y, phi, self.logic, "z3")
        elif self.tactic == EFSMTStrategy.CVC5:
            return solve_with_bin_smt(y, phi, self.logic, "cvc5")
        elif self.tactic == EFSMTStrategy.BOOLECTOR:
            return solve_with_bin_smt(y, phi, self.logic, "boolector2")
        elif self.tactic == EFSMTStrategy.Yices2:
            return solve_with_bin_smt(y, phi, self.logic, "yices2")
        else:
            return self.internal_solve(y, phi)  # for special cases

    def solve_with_qbf(self, y: List[z3.ExprRef], phi: z3.ExprRef):
        assert self.logic in ("BV", "UFBV")
        raise NotImplementedError

    def solve_with_sat(self, y: List[z3.ExprRef], phi: z3.ExprRef):
        raise NotImplementedError
