"""
Local search algorithm for MaxSMT solving.

This module implements a local search approach for MaxSMT,
particularly suitable for SMT formulas over linear integer arithmetic.
"""

from typing import List, Optional, Tuple

import z3

from .base import MaxSMTSolverBase, logger
from aria.pyomt.result import OptimizationResult, OptimizationStatus


class LocalSearchSolver(MaxSMTSolverBase):
    """
    Local search algorithm for MaxSMT(LIA)

    Uses local search techniques to find a solution.
    Currently only implemented for SMT formulas over linear integer arithmetic.
    """

    def _candidate_neighbors(
        self, var: z3.ExprRef, model: z3.ModelRef
    ) -> Optional[List[object]]:
        """Enumerate local-search neighbors for a supported variable."""
        if z3.is_int(var):
            current_value = model.eval(var, model_completion=True).as_long()
            return [current_value + offset for offset in [-2, -1, 1, 2]]
        if z3.is_bool(var):
            current_value = z3.is_true(model.eval(var, model_completion=True))
            return [not current_value]
        return None

    def _improve_with_single_flip(
        self, current_model: z3.ModelRef, variables: List[z3.ExprRef]
    ) -> Tuple[Optional[z3.ModelRef], float]:
        """Search for the best single-variable improvement from the current model."""
        candidate_model: Optional[z3.ModelRef] = None
        candidate_cost = self._calculate_cost(current_model)

        for var in variables:
            neighbors = self._candidate_neighbors(var, current_model)
            if neighbors is None:
                continue

            try:
                for new_value in neighbors:
                    temp_solver = self._create_solver()
                    for hc in self.hard_constraints:
                        temp_solver.add(hc)

                    temp_solver.add(var == new_value)

                    for other_var in variables:
                        if other_var.eq(var):
                            continue
                        val = current_model.eval(other_var, model_completion=True)
                        temp_solver.add(other_var == val)

                    if temp_solver.check() == z3.sat:
                        new_model = temp_solver.model()
                        new_cost = self._calculate_cost(new_model)
                        if new_cost + 1e-9 < candidate_cost:
                            candidate_model = new_model
                            candidate_cost = new_cost
            except (AttributeError, z3.Z3Exception):
                continue

        return candidate_model, candidate_cost

    def solve_result(self, max_iterations: int = 1000) -> OptimizationResult:
        """Local search algorithm for MaxSMT.

        Args:
            max_iterations: Maximum number of iterations for the local search
        """
        empty_soft_result = self._default_result_for_empty_soft_constraints()
        if empty_soft_result is not None:
            return empty_soft_result

        sat, solver, current_model = self._check_hard_constraints()
        if not sat or solver is None or current_model is None:
            return OptimizationResult(
                status=OptimizationStatus.UNSAT,
                engine="local-search",
                solver=self.solver_name,
                detail="Hard constraints are unsatisfiable",
            )

        # Get all variables in the formula
        all_vars = set()
        for hc in self.hard_constraints:
            all_vars.update(self._get_variables(hc))
        for sc in self.soft_constraints:
            all_vars.update(self._get_variables(sc))

        # Convert the set to a list for iteration
        all_vars = list(all_vars)

        # Initialize best model and cost
        best_model = current_model
        best_cost = self._calculate_cost(current_model)

        if best_cost <= 0:
            return OptimizationResult(
                status=OptimizationStatus.OPTIMAL,
                model=best_model,
                cost=0.0,
                engine="local-search",
                solver=self.solver_name,
            )

        # Main local search loop
        for _ in range(max_iterations):
            candidate_model, candidate_cost = self._improve_with_single_flip(
                current_model, all_vars
            )

            # If no improvement was found, stop
            if candidate_model is None:
                break

            best_model = candidate_model
            best_cost = candidate_cost
            current_model = candidate_model

            if best_cost <= 0:
                break

        return OptimizationResult(
            status=OptimizationStatus.OPTIMAL,
            model=best_model,
            cost=best_cost,
            engine="local-search",
            solver=self.solver_name,
            detail="Heuristic local search result",
        )
