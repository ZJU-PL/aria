"""
Anytime MaxSAT solver using core-guided approach.
Based on the iterative SAT-UNSAT algorithm with cardinality constraints.
"""

from typing import List, Optional, Tuple
import time
from pysat.solvers import Solver
from pysat.card import ITotalizer


class AnytimeMaxSAT:
    """Anytime MaxSAT solver using core-guided approach"""

    def __init__(
        self,
        hard: List[List[int]],
        soft: List[List[int]],
        weights: Optional[List[int]] = None,
        solver_name: str = "glucose4",
    ) -> None:
        self.hard = hard
        self.soft = soft
        self.weights = weights if weights else [1] * len(soft)
        self.sat_engine_name = solver_name
        self.best_cost = sum(self.weights)
        self.best_model: Optional[List[int]] = None

    def _compute_cost(self, model: List[int], selector_vars: List[int]) -> int:
        """Compute the cost of a model (sum of weights of violated soft clauses)"""
        if model is None:
            return self.best_cost
        model_set = set(model)
        cost = 0
        for sel, w in zip(selector_vars, self.weights):
            if sel in model_set:
                cost += w
        return cost

    def solve(self, timeout: int = 300) -> Tuple[bool, Optional[List[int]], int]:
        """Solve MaxSAT problem using core-guided approach with iterative refinement

        Returns:
            (success, model, cost)
        """
        start_time = time.time()

        sat_oracle = Solver(name=self.sat_engine_name, bootstrap_with=self.hard, incr=True)

        max_var = max(abs(lit) for clause in self.hard + self.soft for lit in clause) if self.hard or self.soft else 0
        top_var = max_var

        selector_vars = []
        for clause in self.soft:
            top_var += 1
            sel = top_var
            selector_vars.append(sel)
            sat_oracle.add_clause(clause + [sel])

        at_most = None

        while (time.time() - start_time) < timeout:
            result = sat_oracle.solve()

            if result is False:
                break

            model = sat_oracle.get_model()
            if model is None:
                break
                
            cost = self._compute_cost(model, selector_vars)

            if cost < self.best_cost:
                self.best_cost = cost
                self.best_model = model
                if cost == 0:
                    break

            if at_most is not None and cost > 0:
                idx = min(cost, len(at_most.rhs)) - 1
                sat_oracle.add_clause([-at_most.rhs[idx]])

            unsatisfied_selectors = [sel for sel in selector_vars if sel in set(model)]

            if not unsatisfied_selectors:
                break

            at_most = ITotalizer(lits=unsatisfied_selectors, ubound=len(unsatisfied_selectors) - 1, top_id=top_var)
            top_var = at_most.top_id

            for clause in at_most.cnf.clauses:
                sat_oracle.add_clause(clause)

            sat_oracle.add_clause([-at_most.rhs[len(unsatisfied_selectors) - 1]])

        sat_oracle.delete()

        success = self.best_model is not None
        return success, self.best_model, self.best_cost

    def get_solution(self) -> Tuple[Optional[List[int]], int]:
        """Get the best solution found so far"""
        return self.best_model, self.best_cost


def solve_maxsat(
    hard: List[List[int]],
    soft: List[List[int]],
    weights: Optional[List[int]] = None,
    timeout: int = 300,
) -> Tuple[bool, Optional[List[int]], int]:
    """Convenience function to solve MaxSAT problems"""
    solver = AnytimeMaxSAT(hard, soft, weights)
    return solver.solve(timeout)
