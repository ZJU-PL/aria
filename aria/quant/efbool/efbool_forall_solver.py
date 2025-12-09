"""
"Forall" solver for EF problems over Boolean variables
"""
import logging
from typing import List, Optional

from pysat.formula import CNF
from pysat.solvers import Solver
import multiprocessing
from multiprocessing import Pool

logger = logging.getLogger(__name__)


class BoolForallSolver(object):
    def __init__(self, forall_vars: List[int], exists_vars: List[int], solver_name: str = "m22") -> None:
        """Initialize forall solver with configurable SAT solver

        Args:
            forall_vars: List of universal variables
            exists_vars: List of existential variables
            solver_name: SAT solver to use (default: m22)
                Supported solvers: cd, g3, g4, gh, lgl, m22, mc, mgh, mpl
        """
        self.solver_name: str = solver_name
        self.solver = Solver(name=self.solver_name)  # seems not used
        self.universal_bools: List[int] = forall_vars
        self.existential_bools: List[int] = exists_vars
        self.clauses: List[List[int]] = []

    def reduce_counter_example(self, existential_model: List[int], existential_counter_model: List[int]) -> List[int]:
        """http://fmv.jku.at/papers/NiemetzPreinerBiere-FMCAD14.pdf
        Consider a Boolean formula P. The model of P (given by a SAT solver) is not necessarily minimal.
        In other words, the SAT solver may assign truth assignments to literals irrelevant to truth of P.

        Suppose we have a model M of P. To extract a smaller assignment, one trick is to encode the
        negation of P in a separate dual SAT solver.

        We can pass M as an assumption to the dual SAT solver. (check-sat-assuming M).
        All assumptions inconsistent with -P (called the failed assumptions),
        are input assignments sufficient to falsify -P, hence sufficient to satisfy P.

        Related work
          - https://arxiv.org/pdf/2110.12924.pdf
        """
        pos = CNF(from_clauses=self.neg_clauses)
        pos.append(existential_model)
        neg = pos.negate()

        aux_sol = Solver(name=self.solver_name, bootstrap_with=neg)
        assert not aux_sol.solve(assumptions=existential_counter_model)
        return aux_sol.get_core()

    def check_single_model(self, model_data) -> Optional[List[int]]:
        """Helper function to check a single model in parallel"""
        existential_model, neg_clauses, reduce_model, existential_bools = model_data
        solver = Solver(self.solver_name, bootstrap_with=neg_clauses)
        for v in existential_model:
            solver.add_clause([v])

        if not solver.solve():
            return None  # Model is good

        if reduce_model:
            existential_counter_model = [val for val in solver.get_model()
                                         if abs(val) in existential_bools]
            return self.reduce_counter_example(existential_model, existential_counter_model)
        return [-v for v in existential_model]

    def check_models(self, models: List[List[int]]) -> List[List[int]]:
        """Check candidates given by the exists solver in parallel"""
        if not models:
            return []

        # Prepare data for parallel processing
        model_data = [(model, self.neg_clauses, self.reduce_model, self.existential_bools)
                      for model in models]

        # Use number of CPU cores for parallelization
        num_processes = min(len(models), multiprocessing.cpu_count())

        blocking_clauses: List[List[int]] = []
        with Pool(processes=num_processes) as pool:
            results = pool.map(self.check_single_model, model_data)

        # Process results
        for result in results:
            if result is None:
                # Found a good model
                return []
            blocking_clauses.append(result)

        return blocking_clauses
