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

    def add_soft_constraints(
        self, constraints: List[z3.ExprRef], weights: Optional[List[float]] = None
    ) -> None:
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
        # Add hard constraints to solver
        solver = z3.Solver()
        for hc in self.hard_constraints:
            solver.add(hc)

        # Check if hard constraints are satisfiable
        if solver.check() != z3.sat:
            logger.warning("Hard constraints are unsatisfiable")
            return False, None, float("inf")

        # Build soft clauses with relaxation variables
        soft_clauses: List[dict] = []
        relax_counter = 0
        for i, sc in enumerate(self.soft_constraints):
            relax = z3.Bool(f"_relax_{relax_counter}")
            relax_counter += 1
            solver.add(z3.Or(sc, relax))
            soft_clauses.append(
                {
                    "formula": sc,
                    "relax": relax,
                    "weight": float(self.weights[i]),
                    "active": True,
                }
            )

        lower_bound = 0.0

        # Main loop: find and relax cores (weighted Fu-Malik / WPM1-style)
        while True:
            assumptions: List[z3.ExprRef] = [
                z3.Not(clause["relax"])
                for clause in soft_clauses
                if clause["active"]
            ]

            result = solver.check(assumptions)

            if result == z3.sat:
                model = solver.model()

                # Calculate standardized cost (sum of weights of violated constraints)
                standardized_cost = 0.0
                for i, sc in enumerate(self.soft_constraints):
                    if not self._evaluate(model, sc):
                        standardized_cost += self.original_weights[i]

                # Guard against tiny numerical discrepancies
                if standardized_cost + 1e-6 < lower_bound:
                    standardized_cost = lower_bound

                return True, model, standardized_cost

            core = solver.unsat_core()
            if not core:
                return False, None, float("inf")

            core_clauses = [
                clause
                for clause in soft_clauses
                if clause["active"] and z3.Not(clause["relax"]) in core
            ]
            if not core_clauses:
                return False, None, float("inf")

            min_weight = min(clause["weight"] for clause in core_clauses)
            lower_bound += min_weight

            # At least one relaxation variable in the core must be true
            solver.add(
                z3.PbGe([(clause["relax"], 1) for clause in core_clauses], 1)
            )

            # Split weights: pay min_weight now, keep residuals as new clauses
            for clause in core_clauses:
                residual = clause["weight"] - min_weight
                clause["weight"] = 0.0
                clause["active"] = False

                if residual > 1e-6:
                    relax = z3.Bool(f"_relax_{relax_counter}")
                    relax_counter += 1
                    solver.add(z3.Or(clause["formula"], relax))
                    soft_clauses.append(
                        {
                            "formula": clause["formula"],
                            "relax": relax,
                            "weight": residual,
                            "active": True,
                        }
                    )
