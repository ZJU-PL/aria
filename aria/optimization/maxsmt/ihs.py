"""
Implicit Hitting Set algorithm for MaxSMT solving.

This module implements the Implicit Hitting Set (IHS) approach for MaxSMT solving,
based on the IJCAR'18 paper.
"""

from typing import Tuple, Optional, List

import z3

from .base import MaxSMTSolverBase, logger


class ImplicitHittingSetSolver(MaxSMTSolverBase):
    """
    Implicit Hitting Set algorithm for MaxSMT

    Iteratively finds optimal hitting sets for the collection of cores.
    """

    def solve(self) -> Tuple[bool, Optional[z3.ModelRef], float]:
        """Implicit Hitting Set algorithm for MaxSMT

        Returns:
            Tuple of (sat, model, optimal_cost)
        """
        # Check if hard constraints are satisfiable
        solver = z3.Solver()
        for hc in self.hard_constraints:
            solver.add(hc)

        if solver.check() != z3.sat:
            logger.warning("Hard constraints are unsatisfiable")
            return False, None, float("inf")

        # Create relaxation variables for soft constraints
        relax_vars = [z3.Bool(f"_relax_{i}") for i in range(len(self.soft_constraints))]

        # Solver for finding cores
        core_solver = z3.Solver()

        # Add hard constraints
        for hc in self.hard_constraints:
            core_solver.add(hc)

        # Add soft constraints with relaxation variables
        for i, sc in enumerate(self.soft_constraints):
            core_solver.add(z3.Or(sc, relax_vars[i]))

        # Current best model and cost
        best_model: Optional[z3.ModelRef] = None
        best_cost = float("inf")

        # Solver for the hitting set problem
        hs_solver = z3.Optimize()

        # Add variables for the hitting set problem
        hs_vars = [z3.Bool(f"_hs_{i}") for i in range(len(self.soft_constraints))]

        # Add objective: minimize sum of weights of selected soft constraints
        obj_terms = [
            (hs_vars[i], self.weights[i]) for i in range(len(self.soft_constraints))
        ]
        hs_solver.minimize(z3.Sum([z3.If(var, weight, 0) for var, weight in obj_terms]))

        while True:
            # Solve the hitting set problem
            if hs_solver.check() != z3.sat:
                if best_model is not None:
                    return True, best_model, best_cost
                return False, None, float("inf")

            hs_model = hs_solver.model()
            hitting_set = [
                i
                for i in range(len(self.soft_constraints))
                if z3.is_true(hs_model.evaluate(hs_vars[i], model_completion=True))
            ]

            # Check if the current hitting set gives a satisfiable formula
            assumptions = [
                z3.Not(relax_vars[i])
                for i in range(len(self.soft_constraints))
                if i not in hitting_set
            ]

            # Check satisfiability with assumptions
            result = core_solver.check(assumptions)

            if result == z3.sat:
                candidate_model = core_solver.model()

                # Compute the actual cost: weight of violated soft constraints
                violated = [
                    i
                    for i, sc in enumerate(self.soft_constraints)
                    if not z3.is_true(candidate_model.eval(sc, model_completion=True))
                ]
                current_cost = sum(self.weights[i] for i in violated)

                # Record strictly better solutions
                if current_cost + 1e-6 < best_cost:
                    best_model = candidate_model
                    best_cost = current_cost

                # Add constraint to avoid returning the same hitting set again
                if hitting_set:
                    hs_solver.add(z3.Not(z3.And([hs_vars[i] for i in hitting_set])))
                    continue

                # If no soft constraints are violated, we've found the optimal solution
                return True, best_model, best_cost

            # Extract the unsatisfiable core
            core = core_solver.unsat_core()
            core_indices = [
                i
                for i in range(len(self.soft_constraints))
                if z3.Not(relax_vars[i]) in core
            ]

            if not core_indices:
                # No soft constraints in the core
                if best_model is not None:
                    return True, best_model, best_cost
                return False, None, float("inf")

            # Add constraint: at least one variable from the core must be hit
            hs_solver.add(z3.Or([hs_vars[i] for i in core_indices]))
