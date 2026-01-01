# coding: utf-8
"""
Flattening-based QF_UFBV solver
"""
import logging

import z3
from pysat.formula import CNF
from pysat.solvers import Solver

from aria.utils import SolverResult

logger = logging.getLogger(__name__)


class QFUFBVSolver:
    sat_engine = "mgh"

    def __init__(self):
        self.fml = None
        # self.vars = []
        self.verbose = 0

    def solve_smt_file(self, filepath: str):
        fml_vec = z3.parse_smt2_file(filepath)
        return self.check_sat(z3.And(fml_vec))

    def solve_smt_string(self, smt_str: str):
        fml_vec = z3.parse_smt2_string(smt_str)
        return self.check_sat(z3.And(fml_vec))

    def solve_smt_formula(self, fml: z3.ExprRef):
        return self.check_sat(fml)

    def check_sat(self, fml):
        """Check satisfiability of an QF_FP formula"""
        if QFUFBVSolver.sat_engine == "z3":
            return self.solve_qfufbv_via_z3(fml)
        logger.debug("Start translating to CNF...")

        qfufbv_preamble = z3.AndThen(
            "simplify",
            "propagate-values",
            z3.With("solve-eqs"),
            "elim-uncnstr",
            "reduce-bv-size",
            z3.With(
                "simplify",
                som=True,
                pull_cheap_ite=True,
                push_ite_bv=False,
                local_ctx=True,
                local_ctx_limit=10000000,
            ),
            # 'max-bv-sharing',
            "ackermannize_bv",
            z3.If(z3.Probe("is-qfbv"), z3.AndThen("bit-blast", "simplify"), "simplify"),
        )

        qfufbv_prep = z3.With(qfufbv_preamble, elim_and=True, sort_store=True)

        after_simp = qfufbv_prep(fml).as_expr()

        if z3.is_false(after_simp):
            return SolverResult.UNSAT
        if z3.is_true(after_simp):
            return SolverResult.SAT

        g_probe = z3.Goal()
        g_probe.add(after_simp)
        is_bool = z3.Probe("is-propositional")
        if is_bool(g_probe) == 1.0:
            to_cnf_impl = z3.AndThen("simplify", "tseitin-cnf")
            to_cnf = z3.With(
                to_cnf_impl, elim_and=True, push_ite_bv=True, blast_distinct=True
            )
            blasted = to_cnf(after_simp).as_expr()

            if z3.is_false(blasted):
                return SolverResult.UNSAT
            if z3.is_true(blasted):
                return SolverResult.SAT

            g_to_dimacs = z3.Goal()
            g_to_dimacs.add(blasted)
            pos = CNF(from_string=g_to_dimacs.dimacs())
            # print("calling pysat")
            aux = Solver(name=QFUFBVSolver.sat_engine, bootstrap_with=pos)
            if aux.solve():
                return SolverResult.SAT
            return SolverResult.UNSAT
        # sol = z3.Tactic('smt').solver()
        return self.solve_qfufbv_via_z3(after_simp)

    def solve_qfufbv_via_z3(self, fml: z3.ExprRef):
        sol = z3.SolverFor("QF_UFBV")
        sol.add(fml)
        res = sol.check()
        if res == z3.sat:
            return SolverResult.SAT
        if res == z3.unsat:
            return SolverResult.UNSAT
        return SolverResult.UNKNOWN


def demo_qfufbv():
    # z3.set_param("verbose", 15)  #large number -> more detailed nfo
    fml_str = """
(set-logic QF_UFBV)
(declare-fun A ((_ BitVec 32)) (_ BitVec 32))
(assert (= (A (_ bv0 32)) (_ bv1 32)) )
(check-sat)

        """
    sol = QFUFBVSolver()
    print(sol.solve_smt_string(fml_str))


if __name__ == "__main__":
    demo_qfufbv()
