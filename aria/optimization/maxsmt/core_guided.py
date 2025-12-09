"""
Core-guided algorithm for MaxSMT solving.

This module implements the Fu-Malik / MSUS3 core-guided approach for MaxSMT solving.
"""

from typing import Tuple, Optional, List

import z3

from .base import MaxSMTSolverBase, logger


class CoreGuidedSolver(MaxSMTSolverBase):
    """
    Core-guided algorithm for MaxSMT (Fu-Malik/MSUS3 variant)

    Relaxes soft constraints by adding relaxation variables and uses
    the SMT solver to find unsatisfiable cores iteratively.
    """

    def __init__(self, solver_name: str = "z3") -> None:
        """Initialize the core-guided MaxSMT solver

        Args:
            solver_name: Name of the underlying SMT solver
        """
        super().__init__(solver_name)
        # Keep a copy of original weights for standardized cost calculation
        self.original_weights: List[float] = []

    def add_soft_constraint(self, constraint: z3.ExprRef, weight: float = 1.0) -> None:
        """Add a soft constraint with its weight

        Args:
            constraint: SMT formula
            weight: Weight (importance) of the constraint
        """
        super().add_soft_constraint(constraint, weight)
        # Copy original weight
        self.original_weights.append(weight)

    def add_soft_constraints(self, constraints: List[z3.ExprRef], weights: Optional[List[float]] = None) -> None:
        """Add multiple soft constraints with weights

        Args:
            constraints: List of SMT formulas
            weights: List of weights (default: all 1.0)
        """
        super().add_soft_constraints(constraints, weights)
        # Copy original weights
        if weights is None:
            weights = [1.0] * len(constraints)
        self.original_weights.extend(weights)

    def solve(self) -> Tuple[bool, Optional[z3.ModelRef], float]:
        """Core-guided algorithm for MaxSMT

        Returns:
            Tuple of (sat, model, optimal_cost)
        """
        # Create relaxation variables for soft constraints
        relax_vars: List[z3.ExprRef] = [z3.Bool(f"_relax_{i}") for i in range(len(self.soft_constraints))]

        # Add hard constraints to solver
        solver = z3.Solver()
        for hc in self.hard_constraints:
            solver.add(hc)

        # Check if hard constraints are satisfiable
        if solver.check() != z3.sat:
            logger.warning("Hard constraints are unsatisfiable")
            return False, None, float('inf')

        # Initialize with all soft constraints with relaxation variables
        relaxed_soft: List[z3.ExprRef] = []
        for i, sc in enumerate(self.soft_constraints):
            # Add relaxed version of soft constraint: sc OR relax_var
            relaxed_soft.append(z3.Or(sc, relax_vars[i]))
            solver.add(z3.Or(sc, relax_vars[i]))

        # Keep track of blocking variables and their weights
        block_vars: List[z3.ExprRef] = []
        block_weights: List[float] = []

        # Main loop: find and relax cores
        while True:
            # Assume all relaxation variables are false (soft constraints must be satisfied)
            assumptions: List[z3.ExprRef] = [z3.Not(rv) for rv in relax_vars if rv not in block_vars]

            # Check satisfiability with current assumptions
            result = solver.check(assumptions)

            if result == z3.sat:
                # All remaining soft constraints can be satisfied
                model = solver.model()

                # Calculate standardized cost (sum of weights of violated constraints)
                standardized_cost = 0.0
                for i, sc in enumerate(self.soft_constraints):
                    if not model.evaluate(sc, model_completion=True):
                        standardized_cost += self.original_weights[i]

                return True, model, standardized_cost

            # Get the unsatisfiable core
            core = solver.unsat_core()

            if not core:
                # No core found, problem is unsatisfiable
                return False, None, float('inf')

            # Find soft constraints in the core
            core_indices: List[int] = []
            for i, rv in enumerate(relax_vars):
                if z3.Not(rv) in core:
                    core_indices.append(i)

            if not core_indices:
                # No soft constraints in the core, problem is unsatisfiable
                return False, None, float('inf')

            # Find the minimum weight in the core
            min_weight = min(self.weights[i] for i in core_indices)

            # Create a blocking variable for this core
            block_var = z3.Bool(f"_block_{len(block_vars)}")
            block_vars.append(block_var)
            block_weights.append(min_weight)

            # Add cardinality constraint to allow at most one soft constraint to be relaxed in the core
            at_most_one: List[z3.ExprRef] = []
            for i in core_indices:
                # Update the weight by subtracting the minimum weight
                self.weights[i] -= min_weight

                # If weight becomes 0, move to block_vars
                if self.weights[i] < 1e-6:  # Numerical tolerance
                    if relax_vars[i] not in block_vars:
                        block_vars.append(relax_vars[i])

                # Add relaxation variable to at_most_one constraint
                at_most_one.append(relax_vars[i])

            # Add constraint: block_var OR at_most_one(relax_vars in core)
            at_most_one_constraint = z3.PbLe([(var, 1) for var in at_most_one], 1)
            solver.add(z3.Or(block_var, at_most_one_constraint))
