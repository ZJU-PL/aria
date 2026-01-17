"""
Local search algorithm for MaxSMT solving.

This module implements a local search approach for MaxSMT,
particularly suitable for SMT formulas over linear integer arithmetic.
"""

from typing import Tuple, Optional

import z3

from .base import MaxSMTSolverBase, logger


class LocalSearchSolver(MaxSMTSolverBase):
    """
    Local search algorithm for MaxSMT(LIA)

    Uses local search techniques to find a solution.
    Currently only implemented for SMT formulas over linear integer arithmetic.
    """

    def solve(
        self, max_iterations: int = 1000
    ) -> Tuple[bool, Optional[z3.ModelRef], float]:
        """Local search algorithm for MaxSMT

        Args:
            max_iterations: Maximum number of iterations for the local search

        Returns:
            Tuple of (sat, model, optimal_cost)
        """
        # First check if hard constraints are satisfiable
        solver = z3.Solver()
        for hc in self.hard_constraints:
            solver.add(hc)

        if solver.check() != z3.sat:
            logger.warning("Hard constraints are unsatisfiable")
            return False, None, float("inf")

        # Start with a model that satisfies hard constraints
        current_model = solver.model()

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
            return True, best_model, 0.0

        # Main local search loop
        for _ in range(max_iterations):
            candidate_model = None
            candidate_cost = best_cost

            # Try to modify each variable (steepest-descent step)
            for var in all_vars:
                try:
                    # Determine neighbor assignments for this variable
                    if z3.is_int(var):
                        current_value = current_model.eval(
                            var, model_completion=True
                        ).as_long()
                        neighbors = [current_value + offset for offset in [-2, -1, 1, 2]]
                    elif z3.is_bool(var):
                        current_value = z3.is_true(
                            current_model.eval(var, model_completion=True)
                        )
                        neighbors = [not current_value]
                    else:
                        continue  # Skip unsupported variable types

                    for new_value in neighbors:
                        temp_solver = z3.Solver()

                        # Add all hard constraints
                        for hc in self.hard_constraints:
                            temp_solver.add(hc)

                        # Fix the chosen variable to its candidate value
                        temp_solver.add(var == new_value)

                        # Keep other variables at their current values
                        for other_var in all_vars:
                            if other_var != var:
                                val = current_model.eval(
                                    other_var, model_completion=True
                                )
                                temp_solver.add(other_var == val)

                        if temp_solver.check() == z3.sat:
                            new_model = temp_solver.model()
                            new_cost = self._calculate_cost(new_model)

                            if new_cost + 1e-9 < candidate_cost:
                                candidate_model = new_model
                                candidate_cost = new_cost
                except (AttributeError, z3.Z3Exception):
                    # Skip if there are any errors (e.g., variable not in model)
                    continue

            # If no improvement was found, stop
            if candidate_model is None:
                break

            best_model = candidate_model
            best_cost = candidate_cost
            current_model = candidate_model

            if best_cost <= 0:
                break

        return True, best_model, best_cost
