# coding: utf-8
# from __future__ import print_function
"""
Wrappers for PySAT.
Currently, we hope to use this as the Boolean solver of the parallel CDCL(T) engine.
Besides, we may want to integrate some advanced facilities, such as (parallel) uniform
sampling. Or, use pyunigen package
"""
import logging
from enum import Enum
from multiprocessing import Pool
from typing import List

from pysat.formula import CNF
from pysat.solvers import Solver

from aria.utils.types import SolverResult

logger = logging.getLogger(__name__)

"""
    cadical103  = ('cd', 'cd103', 'cdl', 'cdl103', 'cadical103')
    cadical153  = ('cd15', 'cd153', 'cdl15', 'cdl153', 'cadical153')
    gluecard3   = ('gc3', 'gc30', 'gluecard3', 'gluecard30')
    gluecard4   = ('gc4', 'gc41', 'gluecard4', 'gluecard41')
    glucose3    = ('g3', 'g30', 'glucose3', 'glucose30')
    glucose4    = ('g4', 'g41', 'glucose4', 'glucose41')
    lingeling   = ('lgl', 'lingeling')
    maplechrono = ('mcb', 'chrono', 'chronobt', 'maplechrono')
    maplecm     = ('mcm', 'maplecm')
    maplesat    = ('mpl', 'maple', 'maplesat')
    mergesat3   = ('mg3', 'mgs3', 'mergesat3', 'mergesat30')
    minicard    = ('mc', 'mcard', 'minicard')
    minisat22   = ('m22', 'msat22', 'minisat22')
    minisatgh   = ('mgh', 'msat-gh', 'minisat-gh')
"""
sat_solvers = ['cd', 'cd15', 'gc3', 'gc4', 'g3',
               'g4', 'lgl', 'mcb', 'mpl', 'mg3',
               'mc', 'm22', 'msh']


class SATSolver(Enum):
    """Enumeration of SAT solvers and their aliases."""
    CADICAL103 = ('cd', 'cd103', 'cdl', 'cdl103', 'cadical103')
    CADICAL153 = ('cd15', 'cd153', 'cdl15', 'cdl153', 'cadical153')
    GLUECARD3 = ('gc3', 'gc30', 'gluecard3', 'gluecard30')
    GLUECARD4 = ('gc4', 'gc41', 'gluecard4', 'gluecard41')
    GLUCOSE3 = ('g3', 'g30', 'glucose3', 'glucose30')
    GLUCOSE4 = ('g4', 'g41', 'glucose4', 'glucose41')
    LINGELING = ('lgl', 'lingeling')
    MAPLECHRONO = ('mcb', 'chrono', 'chronobt', 'maplechrono')
    MAPLECM = ('mcm', 'maplecm')
    MAPLESAT = ('mpl', 'maple', 'maplesat')
    MERGESAT3 = ('mg3', 'mgs3', 'mergesat3', 'mergesat30')
    MINICARD = ('mc', 'mcard', 'minicard')
    MINISAT22 = ('m22', 'msat22', 'minisat22')
    MINISATGH = ('mgh', 'msat-gh', 'minisat-gh')

    @classmethod
    def get_solver_names(cls):
        """Get a list of all solver names and aliases."""
        return [alias for solver in cls for alias in solver.value]


def internal_single_solve(solver_name, clauses, assumptions):
    """Used by parallel solving"""
    solver = Solver(name=solver_name, bootstrap_with=clauses)
    ans = solver.solve(assumptions=assumptions)
    if ans:
        return ans, solver.get_model()
    return ans, solver.get_core()


class PySATSolver:
    """
    Wrapper of PySAT
    """

    def __init__(self, solver="cd"):
        self.solver_name = solver
        self._solver = Solver(name=solver)
        self._clauses = []
        self.parallel_sampling = False  # parallel sampling of satisfying assignments
        # reduce the size of each sampled model
        self.reduce_samples = True
        self.parallel_reduce = False  # parallel reduce

    def check_sat(self) -> SolverResult:
        """Check satisfiability of the current formula."""
        res = self._solver.solve()
        if res:
            return SolverResult.SAT
        return SolverResult.UNSAT

    def check_sat_assuming(self, assumptions: List[int]) -> SolverResult:
        """Check satisfiability with assumptions."""
        res = self._solver.solve(assumptions=assumptions)
        if res:
            return SolverResult.SAT
        return SolverResult.UNSAT

    def get_unsat_core(self, assumptions: List[int]) -> List[int]:
        """Get unsat core for the given assumptions."""
        res = self._solver.solve(assumptions=assumptions)
        if res:
            return self._solver.get_core()
        return []

    def add_clause(self, clause: List[int]):
        """add clause"""
        self._solver.add_clause(clause)
        self._clauses.append(clause)

    def add_clauses(self, clauses: List[List[int]]):
        """Add multiple clauses to the solver."""
        for cls in clauses:
            self._solver.add_clause(cls)
            self._clauses.append(cls)

    def add_cnf(self, cnf: CNF):
        """Add a CNF formula to the solver."""
        # self.solver.append_formula(cnf.clauses, no_return=False)
        for cls in cnf.clauses:
            self._solver.add_clause(cls)
            self._clauses.append(cls)

    def sample_models(self, to_enum: int) -> List[List[int]]:
        """
        Sample to_enum number of models
        Currently, I use the Solver::enum_models of pySAT
        TODO:
          -  Unigen and other third-party (uniform) samplers
          -  Allow for passing a set of support variables
        """
        results = []
        for i, model in enumerate(self._solver.enum_models(), 1):
            results.append(model)
            if i == to_enum:
                break
        logger.debug("Sampled models: %s", results)
        if not self.reduce_samples:
            # do not reduce the sampled models
            return results
        reduced_models = self.reduce_models(results)
        logger.debug("Reduced models: %s", reduced_models)
        # TODO: remove redundant ones in the reduced models?
        return reduced_models

    def reduce_models(self, models: List[List]) -> List[List[int]]:
        """
        http://fmv.jku.at/papers/NiemetzPreinerBiere-FMCAD14.pdf
        Consider a Boolean formula P. The model of P (given by a SAT solver)
        is not necessarily minimal. In other words, the SAT solver may assign
        truth assignments to literals irrelevant to truth of P.

        Suppose we have a model M of P. To extract a smaller assignment,
        one trick is to encode the negation of P in a separate dual SAT solver.

        We can pass M as an assumption to the dual SAT solver. (check-sat-assuming M).
        All assumptions inconsistent with -P (called the failed assumptions),
        are input assignments sufficient to falsify -P, hence sufficient to satisfy P.

        Related work
          - https://arxiv.org/pdf/2110.12924.pdf
        """
        pos = CNF(from_clauses=self._clauses)
        neg = pos.negate()
        # print(neg.clauses) print(neg.auxvars)
        if self.parallel_reduce:
            return self.internal_parallel_solve(neg, models)

        reduced_models = []  # is every query independent for the solver?
        aux_sol = Solver(name="cd", bootstrap_with=neg)
        for m in models:
            assert not aux_sol.solve(m)
            reduced_models.append(aux_sol.get_core())

        # TODO: it seems that the reduced models may also contain some redundancy
        #  E.g., it could be [[1, -3], [ 1, -3]].
        #  Perhaps we can run the CNF simplifier here
        return reduced_models

    def get_model(self) -> List[int]:
        """Get the current model if satisfiable."""
        return self._solver.get_model()

    def internal_parallel_solve(self, clauses: List[List], assumptions_lists: List[List]):
        """
        Solve clauses under a set of assumptions (deal with each one in parallel)
        TODO: - Should we enforce that clauses are satisfiable?
              - Should control size of the Pool
              - Add timeout (if timeout, use the original model?)
        """
        answers_async = [None for _ in assumptions_lists]
        with Pool(len(assumptions_lists)) as p:
            def terminate_others(val):
                if val:
                    p.terminate()

            for i, assumptions in enumerate(assumptions_lists):
                answers_async[i] = p.apply_async(
                    internal_single_solve,
                    (
                        self.solver_name,
                        clauses,
                        assumptions
                    ),
                    callback=lambda val: terminate_others(val[0]))
            p.close()
            p.join()

        answers = [answer_async.get() for answer_async in answers_async if answer_async.ready()]
        res = [pres for pans, pres in answers]
        return res


def test_pysat():
    """Test the PySAT solver functionality."""
    cnf = CNF(from_clauses=[[1, 3], [-1, 2, -4], [2, 4]])
    # solver_name = random.choice(sat_solvers)
    sol = PySATSolver()
    sol.add_cnf(cnf)
    models = sol.sample_models(10)
    print("Original models")
    print(models)
    print("Reduced models (parallel)")
    sol.parallel_reduce = True
    print(sol.reduce_models(models))

    print("Reduced models (sequential)")
    sol.parallel_reduce = False
    print(sol.reduce_models(models))
    # many reduced models are duplicate


if __name__ == "__main__":
    test_pysat()
