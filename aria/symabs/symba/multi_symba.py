"""
Multi-objective SYMBA implementation.

This module provides specialized support for multi-objective optimization
using the SYMBA algorithm.
"""

import z3
from typing import List, Dict, Tuple, Optional, Any
from .symba import SYMBA, SYMBAState


class MultiSYMBA:
    """
    Multi-objective SYMBA for optimizing multiple objective functions simultaneously.

    This class extends SYMBA to provide better support for multi-objective
    optimization scenarios where we want to find Pareto-optimal solutions.
    """

    def __init__(self, formula: z3.ExprRef, objectives: List[z3.ExprRef],
                 solver_factory=None, timeout: int = 0):
        """
        Initialize MultiSYMBA.

        Args:
            formula: The constraint formula φ
            objectives: List of objective functions T = {t₁, ..., tₙ}
            solver_factory: Factory function to create SMT solvers
            timeout: Timeout for SMT queries in milliseconds
        """
        self.formula = formula
        self.objectives = objectives
        self.solver_factory = solver_factory or (lambda: z3.Solver())
        self.timeout = timeout

        # Use the base SYMBA implementation
        self.symba = SYMBA(formula, objectives, solver_factory, timeout)

        # Pareto front - set of non-dominated solutions
        self.pareto_front: List[z3.ModelRef] = []

    def optimize(self) -> SYMBAState:
        """
        Run multi-objective optimization.

        Returns:
            The SYMBA state after optimization
        """
        # Run the base SYMBA algorithm
        state = self.symba.optimize()

        # Extract Pareto-optimal solutions from the models found
        self._extract_pareto_front()

        return state

    def _extract_pareto_front(self):
        """Extract Pareto-optimal solutions from the models found by SYMBA."""
        if not self.symba.state.M:
            return

        # Evaluate all models
        model_values = []
        for model in self.symba.state.M:
            values = []
            for obj in self.objectives:
                val = model.eval(obj, model_completion=True).as_long()
                values.append(val)
            model_values.append((model, values))

        # Find Pareto-optimal solutions
        self.pareto_front = []
        for i, (model_i, values_i) in enumerate(model_values):
            is_dominated = False

            for j, (model_j, values_j) in enumerate(model_values):
                if i != j:
                    # Check if model_i is dominated by model_j
                    # model_i is dominated if model_j is better or equal in all objectives
                    # and strictly better in at least one
                    dominated = True
                    strictly_better = False

                    for k in range(len(self.objectives)):
                        if values_j[k] < values_i[k]:  # model_j is better for objective k
                            dominated = False
                            break
                        elif values_j[k] > values_i[k]:  # model_j is worse for objective k
                            strictly_better = True

                    if dominated and strictly_better:
                        is_dominated = True
                        break

            if not is_dominated:
                self.pareto_front.append(model_i)

    def get_pareto_front(self) -> List[z3.ModelRef]:
        """
        Get the Pareto front (non-dominated solutions).

        Returns:
            List of Pareto-optimal models
        """
        return self.pareto_front

    def get_pareto_values(self) -> List[Tuple[int, ...]]:
        """
        Get the objective values of Pareto-optimal solutions.

        Returns:
            List of tuples containing objective values for each Pareto-optimal solution
        """
        values = []
        for model in self.pareto_front:
            model_values = []
            for obj in self.objectives:
                val = model.eval(obj, model_completion=True).as_long()
                model_values.append(val)
            values.append(tuple(model_values))
        return values

    def is_pareto_optimal(self, model: z3.ModelRef) -> bool:
        """
        Check if a given model is Pareto-optimal.

        Args:
            model: The model to check

        Returns:
            True if the model is Pareto-optimal, False otherwise
        """
        return model in self.pareto_front

    def dominates(self, model1: z3.ModelRef, model2: z3.ModelRef) -> bool:
        """
        Check if model1 dominates model2 in the Pareto sense.

        Args:
            model1: First model
            model2: Second model

        Returns:
            True if model1 dominates model2, False otherwise
        """
        if model1 not in self.pareto_front or model2 not in self.pareto_front:
            # Need to evaluate both models
            values1 = []
            values2 = []
            for obj in self.objectives:
                values1.append(model1.eval(obj, model_completion=True).as_long())
                values2.append(model2.eval(obj, model_completion=True).as_long())

            # Check if model1 dominates model2
            dominates = True
            strictly_better = False

            for i in range(len(self.objectives)):
                if values2[i] < values1[i]:  # model2 is better for objective i
                    dominates = False
                    break
                elif values2[i] > values1[i]:  # model1 is better for objective i
                    strictly_better = True

            return dominates and strictly_better

        # Both are in Pareto front, check their relationship
        return self._pareto_dominates(model1, model2)

    def _pareto_dominates(self, model1: z3.ModelRef, model2: z3.ModelRef) -> bool:
        """Check if model1 dominates model2 within the Pareto front."""
        # This is a simplified check - in practice, we'd need to compare their objective values
        idx1 = self.pareto_front.index(model1)
        idx2 = self.pareto_front.index(model2)

        if idx1 == -1 or idx2 == -1:
            return False

        # For now, assume no direct dominance relationship within Pareto front
        # In a more sophisticated implementation, we'd compare their objective vectors
        return False
