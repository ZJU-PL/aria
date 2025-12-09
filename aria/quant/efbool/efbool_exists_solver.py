"""
"Exists" solver for EF problems over Boolean variables
"""
import logging
import random
from typing import List, Optional

from pysat.formula import CNF
from pysat.solvers import Solver

import multiprocessing
from multiprocessing import Pool

logger = logging.getLogger(__name__)


class BoolExistsSolver(object):
    def __init__(self, exists_vars: List[int], clauses: List[List[int]], solver_name: str = "m22") -> None:
        """Initialize exists solver with configurable SAT solver

        Args:
            exists_vars: List of existential variables
            clauses: List of clauses
            solver_name: SAT solver to use (default: m22)
                Supported solvers: cd, g3, g4, gh, lgl, m22, mc, mgh, mpl
        """
        self.solver_name: str = solver_name
        self.solver = Solver(name=self.solver_name, bootstrap_with=clauses)
        self.existential_bools: List[int] = exists_vars
        self.clauses: List[List[int]] = []

    def get_random_assignment(self) -> List[int]:
        return [v if random.random() < 0.5 else -v for v in self.existential_bools]

    def get_models(self, num: int = 1) -> List[List[int]]:
        if len(self.clauses) == 0:
            return [self.get_random_assignment() for _ in range(num)]

        results: List[List[int]] = []
        if num == 1:
            if self.solver.solve():
                model = self.solver.get_model()
                existential_model: List[int] = []
                for val in model:  # project?
                    if abs(val) in self.existential_bools:
                        existential_model.append(val)
                logger.debug("model: {}".format(model))
                logger.debug("e-model: {}".format(existential_model))
                results.append(existential_model)
            else:
                logger.debug("e-solver UNSAT")
                logger.debug("clauses: {}".format(self.clauses))
        else:
            for i, model in enumerate(self.solver.enum_models(), 1):
                existential_model: List[int] = []
                for val in model:  # project?
                    if abs(val) in self.existential_bools:
                        existential_model.append(val)
                results.append(existential_model)
                if i == num:
                    break
        return results

    def get_models_parallel(self, num: int = 1, num_processes: Optional[int] = None) -> List[List[int]]:
        """Generate candidate models in parallel

        Args:
            num: Number of models to generate
            num_processes: Number of parallel processes (default: CPU count)
        Returns:
            List of candidate models
        """
        if len(self.clauses) == 0:
            return [self.get_random_assignment() for _ in range(num)]

        if num_processes is None:
            num_processes = min(num, multiprocessing.cpu_count())

        def solve_worker(solver_name: str):
            solver = Solver(name=solver_name, bootstrap_with=self.clauses)
            if solver.solve():
                model = solver.get_model()
                return [val for val in model if abs(val) in self.existential_bools]
            return None

        results: List[List[int]] = []
        with Pool(processes=num_processes) as pool:
            for model in pool.map(solve_worker, [self.solver_name] * num):
                if model is not None:
                    results.append(model)
                    if len(results) >= num:
                        break

        return results

    def add_clause(self, clause: List[int]) -> None:
        self.solver.add_clause(clause)
        self.clauses.append(clause)

    def add_clauses(self, clauses: List[List[int]]) -> None:
        """Update self.clauses
        E.g., refinement from the ForAllSolver
        """
        for cls in clauses:
            self.solver.add_clause(cls)
            self.clauses.append(cls)

    def add_cnf(self, cnf: CNF) -> None:
        """
        The add_cnf function adds a CNF object to the solver.
        It does this by adding each clause in the CNF object to the solver, and then appending that list of clauses
        to self.clauses.

        :param self: Access the attributes and methods of the class in python
        :param cnf:CNF: Add the clauses in the cnf object to the solver
        """
        # self.solver.append_formula(cnf.clauses, no_return=False)
        for cls in cnf.clauses:
            self.solver.add_clause(cls)
            self.clauses.append(cls)
