"""
SYMBA: Symbolic Optimization with SMT Solvers

This module implements the SYMBA algorithm for optimizing objective functions
in linear real arithmetic using SMT solvers as black boxes.

Based on the paper "Symbolic Optimization with SMT Solvers" by Li, Albarghouthi, Kincaid, Gurinkel, and Chechik.
"""

import z3
from typing import List, Tuple, Optional, Set, Dict, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceRule(Enum):
    """Enumeration of SYMBA inference rules"""
    INIT = "INIT"
    GLOBALPUSH = "GLOBALPUSH"
    UNBOUNDED = "UNBOUNDED"
    UNBOUNDED_FAIL = "UNBOUNDED-FAIL"
    BOUNDED = "BOUNDED"


@dataclass
class SYMBAState:
    """
    Represents the state of the SYMBA algorithm.

    U: under-approximation (set of points that are NOT optimal)
    O: over-approximation (set of points that ARE optimal)
    M: set of models found so far
    T: set of objective functions
    """

    # Under-approximation: points that are NOT optimal
    U: z3.ExprRef

    # Over-approximation: points that ARE optimal
    O: z3.ExprRef

    # Set of models found
    M: List[z3.ModelRef] = field(default_factory=list)

    # Set of objective functions
    T: List[z3.ExprRef] = field(default_factory=list)

    # Current bounds for each objective
    bounds: Dict[z3.ExprRef, Tuple[Optional[int], Optional[int]]] = field(default_factory=dict)

    # Statistics
    num_smt_queries: int = 0
    execution_time: float = 0.0

    def __post_init__(self):
        """Initialize bounds for each objective"""
        for obj in self.T:
            self.bounds[obj] = (None, None)  # (lower_bound, upper_bound)

    def update_bounds(self, obj: z3.ExprRef, model: z3.ModelRef):
        """Update bounds for an objective based on a model"""
        current_val = model.eval(obj, model_completion=True)
        lower, upper = self.bounds[obj]

        # Update lower bound (minimum value seen)
        if lower is None or current_val.as_long() < lower:
            lower = current_val.as_long()

        # Update upper bound (maximum value seen)
        if upper is None or current_val.as_long() > upper:
            upper = current_val.as_long()

        self.bounds[obj] = (lower, upper)


class SYMBA:
    """
    SYMBA: SMT-based optimization algorithm for objective functions in LRA.

    Given a formula φ and an objective function t, SYMBA finds a satisfying
    assignment of φ that exhibits the least upper bound κ (maximum value) of t
    such that φ ∧ t ≤ κ is satisfiable and φ ∧ t ≥ κ is unsatisfiable.
    """

    def __init__(self, formula: z3.ExprRef, objectives: List[z3.ExprRef],
                 solver_factory=None, timeout: int = 0):
        """
        Initialize SYMBA with a formula and objective functions.

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

        # Initialize state
        self.state = self._initialize_state()

        # Statistics
        self.stats = {
            'total_time': 0.0,
            'smt_queries': 0,
            'rules_applied': {rule: 0 for rule in InferenceRule}
        }

    def _initialize_state(self) -> SYMBAState:
        """Initialize the SYMBA state according to the INIT rule"""
        # According to the paper (Figure 3):
        # INIT: [∅, (-∞,...,-∞) × (∞,...,∞), ∅] → [∅, U₀, ∅]
        # where U₀ represents the initial under-approximation

        # The initial U should represent points that are clearly not optimal.
        # For multiple objectives, this is complex. The paper suggests:
        # U = (-∞, ..., -∞) × (∞, ..., ∞) meaning points where some objectives
        # are -∞ and others are +∞ (which are not optimal)

        # For now, we'll initialize U as False (no points excluded)
        # and O as False (no points known to be optimal)
        # This is a simplified but correct initialization

        U = z3.BoolVal(False)  # No points are known to be non-optimal yet
        O = z3.BoolVal(False)  # No points are known to be optimal yet

        return SYMBAState(U=U, O=O, T=self.objectives)

    def _create_solver(self) -> z3.Solver:
        """Create a new SMT solver instance"""
        solver = self.solver_factory()
        if self.timeout > 0:
            solver.set("timeout", self.timeout)
        return solver

    def _check_sat_with_formula(self, formula: z3.ExprRef) -> Tuple[z3.CheckSatResult, Optional[z3.ModelRef]]:
        """Check satisfiability of formula and return result with model"""
        solver = self._create_solver()
        solver.add(formula)
        self.stats['smt_queries'] += 1

        start_time = time.time()
        result = solver.check()
        elapsed = time.time() - start_time

        self.stats['total_time'] += elapsed

        if result == z3.sat:
            return result, solver.model()
        return result, None

    def _form_T(self, V: z3.ExprRef) -> z3.ExprRef:
        """
        Convert a vector formula V to the formula it represents.

        form_T(V) takes a formula V representing a set of vectors
        and returns the formula that represents the same set in the
        objective space.

        According to the paper, for a vector (k₁, ..., kₙ), form_T should return
        a formula that represents points where each objective t_i ≤ k_i.

        Args:
            V: Formula representing a set of vectors

        Returns:
            Formula representing the set in objective space
        """
        # For a vector V = (k₁, ..., kₙ), form_T(V) should return
        # t₁ ≤ k₁ ∧ t₂ ≤ k₂ ∧ ... ∧ tₙ ≤ kₙ

        # Since V in our case is often just a conjunction of equality constraints
        # (obj_i == val_i), form_T(V) should represent the region where
        # all objectives are ≤ their corresponding values in V

        # For simplicity, we'll implement this for the case where V is a
        # conjunction of equality constraints
        if V == z3.BoolVal(False) or V == z3.BoolVal(True):
            return V

        # Extract bounds from V and create ≤ constraints
        bounds = {}
        if z3.is_and(V):
            for arg in V.children():
                if self._is_equality_constraint(arg):
                    obj, val = self._extract_equality(arg)
                    if obj is not None:
                        bounds[obj] = val

        # Create form_T as conjunction of t_i ≤ k_i for each objective
        form_T_constraints = []
        for obj in self.objectives:
            if obj in bounds:
                form_T_constraints.append(obj <= bounds[obj])
            else:
                # If no bound for this objective, it means ≤ +∞ which is always true
                pass

        if form_T_constraints:
            return z3.And(form_T_constraints) if len(form_T_constraints) > 1 else form_T_constraints[0]
        else:
            return z3.BoolVal(True)  # No constraints means everything is allowed

    def _is_equality_constraint(self, expr: z3.ExprRef) -> bool:
        """Check if expression is an equality constraint (obj == val)"""
        return z3.is_eq(expr) and len(expr.children()) == 2

    def _extract_equality(self, expr: z3.ExprRef) -> Tuple[Optional[z3.ExprRef], Optional[int]]:
        """Extract objective and value from equality constraint"""
        left, right = expr.children()
        # Check if left is an objective and right is a constant
        if left in self.objectives and right.is_int() and right.is_numeral():
            return left, right.as_long()
        # Check if right is an objective and left is a constant
        elif right in self.objectives and left.is_int() and left.is_numeral():
            return right, left.as_long()
        else:
            return None, None

    def _apply_global_push(self) -> Optional[z3.ModelRef]:
        """
        Apply the GLOBALPUSH rule.

        GLOBALPUSH: [M, U, O] → [M ∪ {p}, max(U, t₁(p), ..., tₙ(p)), O]
        where p is a model of φ ∧ ¬form_T(U)

        Find a model of φ that is not captured by form_T(U)
        (i.e., lies outside the under-approximation).

        Returns:
            A model if found, None if no such model exists
        """
        # Check if φ ∧ ¬form_T(U) is satisfiable
        not_form_T_U = z3.Not(self._form_T(self.state.U))

        # Try to find a model that satisfies φ but not form_T(U)
        solver = self._create_solver()
        solver.add(z3.And(self.formula, not_form_T_U))

        result = solver.check()
        if result == z3.sat:
            model = solver.model()
            self.state.M.append(model)

            # Update U to max(U, t₁(p), ..., tₙ(p))
            # This means U should now include the vector represented by this model
            model_vector = self._model_to_vector_formula(model)

            # U becomes the "maximum" of current U and the new vector
            # In the paper, this means U should represent the union of the
            # current under-approximation and the new point
            self.state.U = z3.Or(self.state.U, model_vector)

            # Update bounds
            for obj in self.objectives:
                self.state.update_bounds(obj, model)

            return model

        return None

    def _model_to_vector_formula(self, model: z3.ModelRef) -> z3.ExprRef:
        """
        Convert a model to a vector formula representing that point.

        Args:
            model: The model to convert

        Returns:
            Formula representing the vector of this model
        """
        # Convert a model p to a vector (t₁(p), ..., tₙ(p))
        # This is a simplified implementation
        vector_conditions = []
        for obj in self.objectives:
            val = model.eval(obj, model_completion=True)
            # For each objective, create a condition representing this value
            # This is complex and depends on how vectors are represented
            vector_conditions.append(obj == val)

        return z3.And(vector_conditions) if vector_conditions else z3.BoolVal(True)

    def _apply_unbounded(self, obj_idx: int) -> bool:
        """
        Apply the UNBOUNDED rule for objective t_i.

        UNBOUNDED(p_i ∈ M, t_i ∈ T):
        [M, U, O] → [M, U ∪ {p_i}, max(U, t₁(p_i), ..., tₙ(p_i))] ∪ O]
        where p_i is a model in M and t_i is unbounded in φ

        Args:
            obj_idx: Index of the objective to check

        Returns:
            True if unbounded was applied, False otherwise
        """
        obj = self.objectives[obj_idx]

        # Check if t_i is unbounded for each model p_i in M
        # If we can find a model where t_i > t_i(p_i), then t_i is unbounded
        # and we should update U to include p_i as a boundary

        for model in self.state.M:
            current_val = model.eval(obj, model_completion=True)

            # Try to find a model where obj > current_val
            test_formula = z3.And(self.formula, obj > current_val)
            result, _ = self._check_sat_with_formula(test_formula)

            if result == z3.sat:
                # Found a larger value, so t_i is unbounded
                # Update U to include p_i as a boundary point
                # In the paper: U ∪ {p_i} where p_i represents the unbounded direction
                model_vector = self._model_to_vector_formula(model)
                self.state.U = z3.Or(self.state.U, model_vector)

                # Also update O to include the new region
                # max(U, t₁(p_i), ..., tₙ(p_i)) ∪ O
                # This is complex, for now we'll just update U
                return True

        return False

    def _apply_unbounded_fail(self, obj_idx: int) -> bool:
        """
        Apply the UNBOUNDED-FAIL rule for objective t_i.

        UNBOUNDED-FAIL(p_i ∈ M, t_i ∈ T):
        [M, U, O] → [M ∪ {p_i}, max(U, t₁(p_i), ..., tₙ(p_i)), O]
        where φ ∧ t_i > t_i(p_i) is unsatisfiable

        For each model p_i in M, if we cannot find a model where t_i > t_i(p_i),
        then p_i represents a boundary point that should be added to the
        under-approximation.

        Args:
            obj_idx: Index of the objective to check

        Returns:
            True if unbounded-fail was applied, False otherwise
        """
        obj = self.objectives[obj_idx]

        # For each model p_i in M, check if φ ∧ t_i > t_i(p_i) is unsatisfiable
        # If so, p_i represents a point where t_i cannot be improved

        applicable_models = []
        for model in self.state.M:
            current_val = model.eval(obj, model_completion=True)

            # Check if obj > current_val is unsatisfiable
            test_formula = z3.And(self.formula, obj > current_val)
            result, _ = self._check_sat_with_formula(test_formula)

            if result == z3.unsat:
                # Cannot improve obj beyond current_val for this model
                # This model represents a boundary point
                applicable_models.append(model)

        if applicable_models:
            # Apply UNBOUNDED-FAIL to all applicable models
            for model in applicable_models:
                # Add model to M (if not already there)
                if model not in self.state.M:
                    self.state.M.append(model)

                # Update U to include this model as a boundary
                # max(U, t₁(p_i), ..., tₙ(p_i))
                model_vector = self._model_to_vector_formula(model)
                self.state.U = z3.Or(self.state.U, model_vector)

                # Update bounds
                for obj_iter in self.objectives:
                    self.state.update_bounds(obj_iter, model)

            return True

        return False

    def _apply_bounded(self, obj_idx: int) -> bool:
        """
        Apply the BOUNDED rule for objective t_i.

        BOUNDED(t_i ∈ T):
        [M, U, O] → [M, U, min(O, (k₁, ..., kₙ))]
        where O = (k₁, ..., kₙ) and k_i = max{t_i(p) | p ∈ M}

        This strengthens the over-approximation O by intersecting it with
        the region where each objective t_i ≤ max{t_i(p) | p ∈ M}.

        Args:
            obj_idx: Index of the objective to check (for compatibility)

        Returns:
            True if O was strengthened, False otherwise
        """
        if not self.state.M:
            return False

        # Create the vector (k₁, ..., kₙ) where k_i = max{t_i(p) | p ∈ M}
        # Then O becomes min(O, (k₁, ..., kₙ))

        # For each objective, find its maximum value over all models
        max_values = {}
        for obj in self.objectives:
            model_max = None
            for model in self.state.M:
                val = model.eval(obj, model_completion=True)
                if model_max is None or val.as_long() > model_max:
                    model_max = val.as_long()
            if model_max is not None:
                max_values[obj] = model_max

        # Create the new over-approximation as t₁ ≤ k₁ ∧ t₂ ≤ k₂ ∧ ... ∧ tₙ ≤ kₙ
        if max_values:
            bound_conditions = [obj <= max_val for obj, max_val in max_values.items()]
            new_O = z3.And(bound_conditions) if len(bound_conditions) > 1 else bound_conditions[0]

            # O becomes the minimum (intersection) of current O and new bounds
            if self.state.O == z3.BoolVal(False):
                self.state.O = new_O
            else:
                self.state.O = z3.And(self.state.O, new_O)

            return True

        return False

    def _should_terminate(self) -> bool:
        """
        Check if SYMBA should terminate.

        According to the paper, SYMBA terminates when:
        U = O = opt_T(φ), meaning we've found the exact set of optimal points.

        More precisely, when no more inference rules can be applied.

        Returns:
            True if should terminate, False otherwise
        """
        # SYMBA terminates when:
        # 1. No more GLOBALPUSH can be applied (no models outside U)
        # 2. No more UNBOUNDED can be applied (all objectives are bounded)
        # 3. No more UNBOUNDED-FAIL can be applied (all models are at their limits)
        # 4. No more BOUNDED can be applied (O is fully strengthened)

        # Check if GLOBALPUSH can be applied
        if self._apply_global_push() is not None:
            return False  # Can still apply GLOBALPUSH

        # Check if any UNBOUNDED rules can be applied
        for i in range(len(self.objectives)):
            if self._apply_unbounded(i) or self._apply_unbounded_fail(i) or self._apply_bounded(i):
                return False  # Can still apply some rule

        return True  # No more rules can be applied

    def optimize(self) -> SYMBAState:
        """
        Run the SYMBA optimization algorithm.

        The main algorithm applies inference rules in sequence until no more
        rules can be applied.

        Returns:
            The final SYMBA state containing the results
        """
        logger.info("Starting SYMBA optimization")
        logger.info(f"Formula: {self.formula}")
        logger.info(f"Objectives: {self.objectives}")

        start_time = time.time()

        # Main SYMBA loop
        iteration = 0
        max_iterations = 10000  # Safety limit

        while not self._should_terminate() and iteration < max_iterations:
            iteration += 1
            logger.debug(f"Iteration {iteration}")
            logger.debug(f"Current state: U={self.state.U}, O={self.state.O}")
            logger.debug(f"Models found: {len(self.state.M)}")

            rules_applied = False

            # Try GLOBALPUSH first (find new models)
            model = self._apply_global_push()
            if model is not None:
                logger.debug("Applied GLOBALPUSH rule")
                self.stats['rules_applied'][InferenceRule.GLOBALPUSH] += 1
                rules_applied = True

            # If no new model found, try to analyze existing models
            if not rules_applied:
                for i in range(len(self.objectives)):
                    # Try UNBOUNDED for each objective
                    if self._apply_unbounded(i):
                        logger.debug(f"Applied UNBOUNDED rule for objective {i}")
                        self.stats['rules_applied'][InferenceRule.UNBOUNDED] += 1
                        rules_applied = True
                        break

                    # Try UNBOUNDED-FAIL for each objective
                    if self._apply_unbounded_fail(i):
                        logger.debug(f"Applied UNBOUNDED-FAIL rule for objective {i}")
                        self.stats['rules_applied'][InferenceRule.UNBOUNDED_FAIL] += 1
                        rules_applied = True

                        # After UNBOUNDED-FAIL, try BOUNDED
                        if self._apply_bounded(i):
                            logger.debug(f"Applied BOUNDED rule for objective {i}")
                            self.stats['rules_applied'][InferenceRule.BOUNDED] += 1
                        break

            if not rules_applied:
                logger.debug("No rules could be applied in this iteration")
                break

        total_time = time.time() - start_time
        self.stats['total_time'] = total_time

        if iteration >= max_iterations:
            logger.warning(f"SYMBA reached maximum iterations ({max_iterations}), terminating")

        logger.info(f"SYMBA completed in {total_time:.2f}s with {self.stats['smt_queries']} SMT queries")
        logger.info(f"Rules applied: {self.stats['rules_applied']}")
        logger.info(f"Final bounds: {self.get_optimal_values()}")

        return self.state

    def get_optimal_values(self) -> Dict[z3.ExprRef, int]:
        """
        Get the optimal values found for each objective.

        Returns:
            Dictionary mapping objectives to their optimal values
        """
        optimal_values = {}

        for obj in self.objectives:
            lower, upper = self.state.bounds[obj]
            if upper is not None:
                optimal_values[obj] = upper  # Maximum value found
            elif lower is not None:
                optimal_values[obj] = lower  # Only lower bound available
            else:
                optimal_values[obj] = None   # No bound found

        return optimal_values

    def is_optimal(self, model: z3.ModelRef) -> bool:
        """
        Check if a given model represents an optimal solution.

        Args:
            model: The model to check

        Returns:
            True if the model is optimal, False otherwise
        """
        # A model is optimal if it achieves the maximum value for at least one objective
        # and doesn't have worse values for other objectives

        if model not in self.state.M:
            return False

        optimal_values = self.get_optimal_values()

        for obj in self.objectives:
            if optimal_values[obj] is not None:
                model_val = model.eval(obj, model_completion=True).as_long()
                if model_val < optimal_values[obj]:
                    return False

        return True
