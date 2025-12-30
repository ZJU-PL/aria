# coding: utf-8
"""
Flattening-based QF_AUFBV solver
"""
import logging

import z3
from pysat.formula import CNF
from pysat.solvers import Solver

from aria.utils import SolverResult

logger = logging.getLogger(__name__)


class QFAUFBVSolver:
    """
    A class for solving QF_AUFBV (Quantifier-Free Array Theory with Uninterpreted
    Functions and Bit-Vectors) problems.
    """
    sat_engine = 'mgh'

    def __init__(self):
        """
        Initialize the QFAUFBVSolver instance.
        """
        self.fml = None
        # self.vars = []
        self.verbose = 0

    def solve_smt_file(self, filepath: str) -> SolverResult:
        """
        Solve an SMT problem from a file.

        Args:
            filepath (str): The path to the SMT file.

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
        """
        Solve an SMT problem from a Z3 formula.

        Args:
            fml (z3.ExprRef): The Z3 formula representing the SMT problem.

        Returns:
            SolverResult: The result of the solver (SAT, UNSAT, or UNKNOWN).
        """
        return self.check_sat(fml)

    def check_sat(self, fml) -> SolverResult:
        """Check satisfiability of an formula"""
        if QFAUFBVSolver.sat_engine == 'z3':
            return self.solve_qfaufbv_via_z3(fml)
        logger.debug("Start translating to CNF...")

        qfaufbv_preamble = \
            z3.AndThen(
                'simplify',
                'propagate-values',
                'solve-eqs',
                'elim-uncnstr',
                'reduce-bv-size',
                z3.With(
                    'simplify', som=True, pull_cheap_ite=True,
                    push_ite_bv=False, local_ctx=True,
                    local_ctx_limit=10000000
                ),
                # 'bvarray2uf',  # this tactic is dangerous
                # (it only handles specific arrays)
                'max-bv-sharing',
                'ackermannize_bv',
                z3.If(z3.Probe('is-qfbv'),
                      z3.AndThen('bit-blast',
                                 z3.With('simplify', arith_lhs=False, elim_and=True)),
                      'simplify'),
            )

        qfaufbv_prep = z3.With(qfaufbv_preamble, elim_and=True, sort_store=True)

        after_simp = qfaufbv_prep(fml).as_expr()
        logger.debug("Simplified!...")
        if z3.is_false(after_simp):
            return SolverResult.UNSAT
        if z3.is_true(after_simp):
            return SolverResult.SAT

        g_probe = z3.Goal()
        g_probe.add(after_simp)
        is_bool = z3.Probe('is-propositional')
        if is_bool(g_probe) == 1.0:
            to_cnf_impl = z3.AndThen('simplify', 'tseitin-cnf')
            to_cnf = z3.With(
                to_cnf_impl, elim_and=True, push_ite_bv=True,
                blast_distinct=True
            )
            blasted = to_cnf(after_simp).as_expr()
            if z3.is_false(blasted):
                return SolverResult.UNSAT
            if z3.is_true(blasted):
                return SolverResult.SAT
            g_to_dimacs = z3.Goal()
            g_to_dimacs.add(blasted)
            pos = CNF(from_string=g_to_dimacs.dimacs())
            logger.debug("Start calling SAT solver...")
            aux = Solver(name=QFAUFBVSolver.sat_engine, bootstrap_with=pos)
            if aux.solve():
                return SolverResult.SAT
            return SolverResult.UNSAT
        # the else part
        # sol = z3.Tactic('smt').solver()
        logger.debug("Start calling Z3...")
        return self.solve_qfaufbv_via_z3(after_simp)

    def solve_qfaufbv_via_z3(self, fml: z3.ExprRef) -> SolverResult:
        sol = z3.SolverFor("QF_AUFBV")
        sol.add(fml)
        res = sol.check()
        if res == z3.sat:
            return SolverResult.SAT
        if res == z3.unsat:
            return SolverResult.UNSAT
        return SolverResult.UNKNOWN


def demo_qfaufbv():
    z3.set_param("verbose", 15)
    fml_str = """
       (set-info :smt-lib-version 2.6)
(set-logic QF_ABV)
(declare-fun size_Q_0 () (_ BitVec 32))
(declare-fun size_Q_1 () (_ BitVec 32))
(assert (= size_Q_1 (_ bv2 32)))
(declare-fun BubbleSort_Q_0_Q_j_Q_0 () (_ BitVec 32))
(declare-fun BubbleSort_Q_0_Q_j_Q_1 () (_ BitVec 32))
(assert (= BubbleSort_Q_0_Q_j_Q_1 (_ bv0 32)))
(declare-fun BubbleSort_Q_0_Q_i_Q_0 () (_ BitVec 32))
(declare-fun BubbleSort_Q_0_Q_i_Q_1 () (_ BitVec 32))
(assert (= BubbleSort_Q_0_Q_i_Q_1 (ite (bvult BubbleSort_Q_0_Q_j_Q_1 (bvsub (_ bv2 32) (_ bv1 32))) (_ bv0 32) BubbleSort_Q_0_Q_i_Q_0)))
(declare-fun array_Q_0 () (Array (_ BitVec 32) (_ BitVec 32)))
(declare-fun BubbleSort_Q_0_Q_temp_Q_0 () (_ BitVec 32))
(declare-fun BubbleSort_Q_0_Q_temp_Q_1 () (_ BitVec 32))
(assert (let ((?v_0 (select array_Q_0 BubbleSort_Q_0_Q_i_Q_1))) (= BubbleSort_Q_0_Q_temp_Q_1 (ite (and (and (bvult BubbleSort_Q_0_Q_j_Q_1 (bvsub (_ bv2 32) (_ bv1 32))) (bvult BubbleSort_Q_0_Q_i_Q_1 (bvsub (bvsub (_ bv2 32) BubbleSort_Q_0_Q_j_Q_1) (_ bv1 32)))) (bvult ?v_0 (select array_Q_0 (bvadd BubbleSort_Q_0_Q_i_Q_1 (_ bv1 32))))) ?v_0 BubbleSort_Q_0_Q_temp_Q_0))))
(declare-fun array_Q_1 () (Array (_ BitVec 32) (_ BitVec 32)))
(assert (let ((?v_0 (select array_Q_0 (bvadd BubbleSort_Q_0_Q_i_Q_1 (_ bv1 32))))) (= array_Q_1 (ite (and (and (bvult BubbleSort_Q_0_Q_j_Q_1 (bvsub (_ bv2 32) (_ bv1 32))) (bvult BubbleSort_Q_0_Q_i_Q_1 (bvsub (bvsub (_ bv2 32) BubbleSort_Q_0_Q_j_Q_1) (_ bv1 32)))) (bvult (select array_Q_0 BubbleSort_Q_0_Q_i_Q_1) ?v_0)) (store array_Q_0 BubbleSort_Q_0_Q_i_Q_1 ?v_0) array_Q_0))))
(declare-fun array_Q_2 () (Array (_ BitVec 32) (_ BitVec 32)))
(assert (let ((?v_0 (bvadd BubbleSort_Q_0_Q_i_Q_1 (_ bv1 32)))) (= array_Q_2 (ite (and (and (bvult BubbleSort_Q_0_Q_j_Q_1 (bvsub (_ bv2 32) (_ bv1 32))) (bvult BubbleSort_Q_0_Q_i_Q_1 (bvsub (bvsub (_ bv2 32) BubbleSort_Q_0_Q_j_Q_1) (_ bv1 32)))) (bvult (select array_Q_0 BubbleSort_Q_0_Q_i_Q_1) (select array_Q_0 ?v_0))) (store array_Q_1 ?v_0 BubbleSort_Q_0_Q_temp_Q_1) array_Q_1))))
(declare-fun BubbleSort_Q_0_Q_temp_Q_i_Q_0 () (_ BitVec 32))
(declare-fun BubbleSort_Q_0_Q_temp_Q_i_Q_1 () (_ BitVec 32))
(assert (= BubbleSort_Q_0_Q_temp_Q_i_Q_1 (ite (and (bvult BubbleSort_Q_0_Q_j_Q_1 (bvsub (_ bv2 32) (_ bv1 32))) (bvult BubbleSort_Q_0_Q_i_Q_1 (bvsub (bvsub (_ bv2 32) BubbleSort_Q_0_Q_j_Q_1) (_ bv1 32)))) BubbleSort_Q_0_Q_i_Q_1 BubbleSort_Q_0_Q_temp_Q_i_Q_0)))
(declare-fun BubbleSort_Q_0_Q_i_Q_2 () (_ BitVec 32))
(assert (= BubbleSort_Q_0_Q_i_Q_2 (ite (and (bvult BubbleSort_Q_0_Q_j_Q_1 (bvsub (_ bv2 32) (_ bv1 32))) (bvult BubbleSort_Q_0_Q_i_Q_1 (bvsub (bvsub (_ bv2 32) BubbleSort_Q_0_Q_j_Q_1) (_ bv1 32)))) (bvadd BubbleSort_Q_0_Q_i_Q_1 (_ bv1 32)) BubbleSort_Q_0_Q_i_Q_1)))
(declare-fun BubbleSort_Q_0_Q_temp_Q_j_Q_0 () (_ BitVec 32))
(declare-fun BubbleSort_Q_0_Q_temp_Q_j_Q_1 () (_ BitVec 32))
(assert (= BubbleSort_Q_0_Q_temp_Q_j_Q_1 (ite (bvult BubbleSort_Q_0_Q_j_Q_1 (bvsub (_ bv2 32) (_ bv1 32))) BubbleSort_Q_0_Q_j_Q_1 BubbleSort_Q_0_Q_temp_Q_j_Q_0)))
(declare-fun BubbleSort_Q_0_Q_j_Q_2 () (_ BitVec 32))
(assert (= BubbleSort_Q_0_Q_j_Q_2 (ite (bvult BubbleSort_Q_0_Q_j_Q_1 (bvsub (_ bv2 32) (_ bv1 32))) (bvadd BubbleSort_Q_0_Q_j_Q_1 (_ bv1 32)) BubbleSort_Q_0_Q_j_Q_1)))
(assert (let ((?v_9 (bvsub (_ bv2 32) (_ bv1 32)))) (let ((?v_8 (bvult BubbleSort_Q_0_Q_j_Q_1 ?v_9)) (?v_7 (bvsub (bvsub (_ bv2 32) BubbleSort_Q_0_Q_j_Q_1) (_ bv1 32)))) (let ((?v_0 (and ?v_8 (bvult BubbleSort_Q_0_Q_i_Q_1 ?v_7))) (?v_1 (bvadd BubbleSort_Q_0_Q_i_Q_1 (_ bv1 32)))) (let ((?v_4 (and ?v_0 (bvult (select array_Q_0 BubbleSort_Q_0_Q_i_Q_1) (select array_Q_0 ?v_1)))) (?v_2 (and (bvule (_ bv0 32) BubbleSort_Q_0_Q_i_Q_1) (bvult BubbleSort_Q_0_Q_i_Q_1 (_ bv2 32)))) (?v_5 (and (bvule (_ bv0 32) ?v_1) (bvult ?v_1 (_ bv2 32))))) (let ((?v_3 (=> ?v_4 ?v_2)) (?v_6 (=> ?v_4 ?v_5))) (not (and (and (and (and (and (and (and (and (and (and (=> ?v_0 ?v_2) (=> ?v_0 ?v_5)) ?v_3) ?v_3) ?v_6) ?v_6) (=> ?v_0 (not (bvult BubbleSort_Q_0_Q_i_Q_2 ?v_7)))) (=> ?v_8 (not (bvult BubbleSort_Q_0_Q_j_Q_2 ?v_9)))) (and (bvule (_ bv0 32) (_ bv0 32)) (bvult (_ bv0 32) (_ bv2 32)))) (and (bvule (_ bv0 32) (_ bv1 32)) (bvult (_ bv1 32) (_ bv2 32)))) (bvule (select array_Q_2 (_ bv0 32)) (select array_Q_2 (_ bv1 32)))))))))))
(check-sat)
        """
    sol = QFAUFBVSolver()
    print(sol.solve_smt_string(fml_str))


if __name__ == "__main__":
    demo_qfaufbv()
