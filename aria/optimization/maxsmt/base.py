"""
Base class and common utilities for MaxSMT solvers.

Provides the abstract base class for MaxSMT solvers and common utility functions.
"""

import logging
from enum import Enum
from typing import List, Optional, Set, Tuple

import z3

from aria.optimization.result import OptimizationResult, OptimizationStatus

logger = logging.getLogger(__name__)


class MaxSMTAlgorithm(Enum):
    """Enumeration of available MaxSMT algorithms"""

    CORE_GUIDED = "core-guided"  # Core-guided algorithm from SAT'13
    IHS = "ihs"  # Implicit Hitting Set algorithm from IJCAR'18
    LOCAL_SEARCH = "local-search"  # Local search for MaxSMT(LIA)


class MaxSMTSolverBase:
    """
    Base class for MaxSMT solvers
    """

    def __init__(self, solver_name: str = "z3") -> None:
        """Initialize the MaxSMT solver

        Args:
            solver_name: Name of the underlying SMT solver
        """
        self.solver_name: str = solver_name
        self.hard_constraints: List[z3.ExprRef] = []
        self.soft_constraints: List[z3.ExprRef] = []
        self.weights: List[float] = []
        self.solver: Optional[z3.Solver] = None
        self._setup_solver(solver_name)

    def _setup_solver(self, solver_name: str) -> None:
        """Set up the underlying SMT solver

        Args:
            solver_name: Name of the SMT solver to use
        """
        if solver_name == "z3":
            self.solver = z3.Solver()
        else:
            # For other solvers, we could add adapter code here
            raise NotImplementedError(f"Solver {solver_name} not supported")

    def add_hard_constraint(self, constraint: z3.ExprRef) -> None:
        """Add a hard constraint (must be satisfied)

        Args:
            constraint: SMT formula that must be satisfied
        """
        self.hard_constraints.append(constraint)

    def add_hard_constraints(self, constraints: List[z3.ExprRef]) -> None:
        """Add multiple hard constraints

        Args:
            constraints: List of SMT formulas that must be satisfied
        """
        self.hard_constraints.extend(constraints)

    def add_soft_constraint(self, constraint: z3.ExprRef, weight: float = 1.0) -> None:
        """Add a soft constraint with a weight

        Args:
            constraint: SMT formula that should be satisfied if possible
            weight: Weight of the constraint (higher = more important)
        """
        self.soft_constraints.append(constraint)
        self.weights.append(weight)

    def add_soft_constraints(
        self, constraints: List[z3.ExprRef], weights: Optional[List[float]] = None
    ) -> None:
        """Add multiple soft constraints with weights

        Args:
            constraints: List of SMT formulas
            weights: List of weights (default: all 1.0)
        """
        if weights is None:
            weights = [1.0] * len(constraints)

        if len(constraints) != len(weights):
            raise ValueError("Number of constraints must match number of weights")

        self.soft_constraints.extend(constraints)
        self.weights.extend(weights)

    def solve_result(self) -> OptimizationResult:
        """Solve the MaxSMT problem and return a normalized result object."""
        raise NotImplementedError("Subclasses must implement this method")

    def solve(self) -> Tuple[bool, Optional[z3.ModelRef], float]:
        """Solve the MaxSMT problem in the legacy tuple format."""
        result = self.solve_result()
        if result.status == OptimizationStatus.OPTIMAL:
            return True, result.model, float(result.cost or 0.0)
        return False, None, float("inf")

    def _create_solver(self) -> z3.Solver:
        """Create a solver instance matching the configured backend."""
        if self.solver_name != "z3":
            raise NotImplementedError(f"Solver {self.solver_name} not supported")
        return z3.Solver()

    def _check_hard_constraints(
        self,
    ) -> Tuple[bool, Optional[z3.Solver], Optional[z3.ModelRef]]:
        """Check whether hard constraints are satisfiable."""
        solver = self._create_solver()
        for constraint in self.hard_constraints:
            solver.add(constraint)

        if solver.check() != z3.sat:
            logger.warning("Hard constraints are unsatisfiable")
            return False, None, None

        return True, solver, solver.model()

    def _get_variables(self, formula: z3.ExprRef) -> Set[z3.ExprRef]:
        """Extract variables from a Z3 formula"""
        variables: Set[z3.ExprRef] = set()

        def collect(expr: z3.ExprRef) -> None:
            if (
                z3.is_const(expr)
                and not z3.is_true(expr)
                and not z3.is_false(expr)
                and expr.decl().kind() == z3.Z3_OP_UNINTERPRETED
            ):
                variables.add(expr)
            else:
                for child in expr.children():
                    collect(child)

        collect(formula)
        return variables

    def _calculate_cost(self, model: z3.ModelRef) -> float:
        """Calculate the cost of a model (sum of weights of unsatisfied soft constraints)"""
        cost = 0.0
        for i, sc in enumerate(self.soft_constraints):
            if not self._evaluate(model, sc):
                cost += self.weights[i]
        return cost

    def _evaluate(self, model: z3.ModelRef, formula: z3.ExprRef) -> bool:
        """Evaluate a formula under a model"""
        try:
            return z3.is_true(model.eval(formula, model_completion=True))
        except (AttributeError, z3.Z3Exception):
            # For complex formulas that can't be directly evaluated
            return False

    def _satisfies_hard_constraints(self, model: z3.ModelRef) -> bool:
        """Check if a model satisfies all hard constraints"""
        for hc in self.hard_constraints:
            if not self._evaluate(model, hc):
                return False
        return True

    def _default_result_for_empty_soft_constraints(
        self,
    ) -> Optional[OptimizationResult]:
        """Return the unique result for instances without soft constraints."""
        if self.soft_constraints:
            return None

        sat, _, model = self._check_hard_constraints()
        if not sat:
            return OptimizationResult(
                status=OptimizationStatus.UNSAT,
                engine=self.__class__.__name__,
                solver=self.solver_name,
                detail="Hard constraints are unsatisfiable",
            )
        return OptimizationResult(
            status=OptimizationStatus.OPTIMAL,
            model=model,
            cost=0.0,
            engine=self.__class__.__name__,
            solver=self.solver_name,
        )
