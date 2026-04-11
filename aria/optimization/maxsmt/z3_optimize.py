"""
Z3 Optimize-based MaxSMT solver.

This module uses Z3's built-in Optimize engine for solving MaxSMT problems.
"""

from typing import Tuple, Optional

import z3

from .base import MaxSMTSolverBase
from aria.optimization.result import OptimizationResult, OptimizationStatus


class Z3OptimizeSolver(MaxSMTSolverBase):
    """
    Z3 Optimize-based MaxSMT solver

    Uses Z3's built-in Optimize engine to solve MaxSMT problems directly.
    """

    def solve_result(self) -> OptimizationResult:
        """Solve the MaxSMT problem using Z3's Optimize engine.

        Z3's Optimize engine can directly handle weighted MaxSMT problems.
        This method uses Z3's built-in functionality.
        """
        empty_soft_result = self._default_result_for_empty_soft_constraints()
        if empty_soft_result is not None:
            return empty_soft_result

        # Create Z3 Optimize solver
        opt = z3.Optimize()

        # Add hard constraints
        for hc in self.hard_constraints:
            opt.add(hc)

        # Create relaxation variables for soft constraints
        relax_vars = []
        for i, (sc, weight) in enumerate(zip(self.soft_constraints, self.weights)):
            # Create a relaxation variable for this soft constraint
            b = z3.Bool(f"_soft_{i}")

            # Add the implication: if b is true, then the soft constraint must hold
            opt.add(z3.Implies(b, sc))

            # Add to the list for the objective function
            relax_vars.append((b, weight))

        # Set the objective: maximize the sum of weights of satisfied soft constraints
        objective = opt.maximize(
            z3.Sum([z3.If(b, z3.RealVal(weight), z3.RealVal(0)) for b, weight in relax_vars])
        )

        # Check if the formula is satisfiable
        if opt.check() == z3.sat:
            model = opt.model()

            # Calculate the cost (sum of weights of violated constraints)
            total_weight = sum(self.weights)

            # Properly handle the objective value
            try:
                # Get the objective value
                obj_val = opt.upper(objective)

                # Convert to float - different Z3 versions may return different types
                if hasattr(obj_val, "as_decimal"):
                    satisfied_weight = float(obj_val.as_decimal(10).strip("?"))
                elif hasattr(obj_val, "as_fraction"):
                    fraction = obj_val.as_fraction()
                    satisfied_weight = float(fraction.numerator) / float(
                        fraction.denominator
                    )
                elif hasattr(obj_val, "as_long"):
                    satisfied_weight = float(obj_val.as_long())
                elif hasattr(obj_val, "as_float"):
                    satisfied_weight = obj_val.as_float()
                else:
                    # Last resort: convert to string and parse
                    satisfied_weight = float(str(obj_val).replace("?", ""))
            except (ValueError, AttributeError):
                # Alternative method: evaluate the objective expression in the model
                satisfied_weight = 0.0
                for b, weight in relax_vars:
                    if z3.is_true(model.eval(b)):
                        satisfied_weight += weight

            # Cost is the weight of violated constraints
            cost = total_weight - satisfied_weight

            return OptimizationResult(
                status=OptimizationStatus.OPTIMAL,
                model=model,
                cost=cost,
                engine="z3-opt",
                solver=self.solver_name,
            )
        # Formula is unsatisfiable
        return OptimizationResult(
            status=OptimizationStatus.UNSAT,
            engine="z3-opt",
            solver=self.solver_name,
            detail="Optimize could not satisfy hard constraints",
        )
