# coding: utf-8
"""
For solving the following formulas
- QF_FP
- QF_BVFP?
"""
import logging

import z3
from pysat.formula import CNF
from pysat.solvers import Solver

from aria.utils import SolverResult

logger = logging.getLogger(__name__)


class QFFPSolver:
    """
    This class is used to check the satisfiability of QF_FP formulas.
    It uses various tactics from Z3 to translate the formula to CNF
    and then use PySAT to solve it.
    """
    sat_engine = 'mgh'

    def __init__(self):
        self.fml = None
        # self.vars = []
        self.verbose = 0

    def solve_smt_file(self, filepath: str) -> SolverResult:
        """
        Solve an SMT problem from a file.
        Args:
            filepath (str): The path to the SMT problem file.
        Returns:
            SolverResult: The result of the solver (SAT, UNSAT, or UNKNOWN).
        """
        fml_vec = z3.parse_smt2_file(filepath)
        return self.check_sat(z3.And(fml_vec))

    def solve_smt_string(self, smt_str: str) -> SolverResult:
        """
        Solve an SMT problem from a string.
        Args:
            smt_str (str): The SMT problem as a string.
        Returns:
            SolverResult: The result of the solver (SAT, UNSAT, or UNKNOWN).
        """
        fml_vec = z3.parse_smt2_string(smt_str)
        return self.check_sat(z3.And(fml_vec))

    def solve_smt_formula(self, fml: z3.ExprRef) -> SolverResult:
        return self.check_sat(fml)

    def check_sat(self, fml) -> SolverResult:
        # z3.set_param("verbose", 15)
        """Check satisfiability of an QF_FP formula"""
        if QFFPSolver.sat_engine == 'z3':
            return self.solve_qffp_via_z3(fml)
        logger.debug("Start translating to CNF...")

        qffp_preamble = z3.AndThen(z3.With('simplify', arith_lhs=False, elim_and=True),
                                   'propagate-values',
                                   'fpa2bv',
                                   'propagate-values',
                                   # 'reduce-bv-size',   # should we add this?
                                   z3.With('simplify', arith_lhs=False, elim_and=True),
                                   'ackermannize_bv',
                                   z3.If(z3.Probe('is-qfbv'),
                                         z3.AndThen('bit-blast',
                                                    'simplify'),
                                         'simplify'),
                                   )

        try:
            # qffp_blast = z3.With(qffp_preamble, elim_and=True,
            #                      push_ite_bv=True, blast_distinct=True)
            qffp_blast = qffp_preamble
            after_simp = qffp_blast(fml).as_expr()
            if z3.is_false(after_simp):
                return SolverResult.UNSAT
            if z3.is_true(after_simp):
                return SolverResult.SAT

            g_probe = z3.Goal()
            g_probe.add(after_simp)
            is_bool = z3.Probe('is-propositional')
            if is_bool(g_probe) == 1.0:
                to_cnf_impl = z3.AndThen(
                    z3.With('simplify', local_ctx=True, flat=False,
                            flat_and_or=False),
                    'aig',
                    'tseitin-cnf')
                # o_cnf = z3.With(to_cnf_impl, elim_and=True,
                #                 push_ite_bv=True, blast_distinct=True)
                to_cnf = to_cnf_impl
                blasted = to_cnf(after_simp).as_expr()

                if z3.is_false(blasted):
                    return SolverResult.UNSAT
                if z3.is_true(blasted):
                    return SolverResult.SAT

                g_to_dimacs = z3.Goal()
                g_to_dimacs.add(blasted)
                pos = CNF(from_string=g_to_dimacs.dimacs())
                # print("Running pysat...{}".format(QFFPSolver.sat_engine))
                aux = Solver(name=QFFPSolver.sat_engine, bootstrap_with=pos)
                if aux.solve():
                    return SolverResult.SAT
                return SolverResult.UNSAT
            # sol = z3.Tactic('smt').solver()
            sol = z3.SolverFor("QF_FP")
            sol.add(after_simp)
            res = sol.check()
            if res == z3.sat:
                return SolverResult.SAT
            if res == z3.unsat:
                return SolverResult.UNSAT
            return SolverResult.UNKNOWN

        except Exception as ex:
            print("ERROR")
            # exit(0)
            print(ex)
            # sol = z3.Solver()
            # sol.add(self.fml)
            # print(sol.to_smt2())
            return SolverResult.UNKNOWN

    def solve_qffp_via_z3(self, fml: z3.ExprRef) -> SolverResult:
        sol = z3.SolverFor("QF_FP")
        sol.add(fml)
        res = sol.check()
        if res == z3.sat:
            return SolverResult.SAT
        if res == z3.unsat:
            return SolverResult.UNSAT
        return SolverResult.UNKNOWN


def demo_qffp():
    z3.set_param("verbose", 15)
    fml_str = """

        """

    sol = QFFPSolver()
    print(sol.solve_smt_string(fml_str))


if __name__ == "__main__":
    demo_qffp()
