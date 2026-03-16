"""
Base k-safety verification engine for relational verification and hyperproperty checking.

This module implements the core functionality for k-safety verification,
which handles properties that relate multiple execution traces.
"""

import logging
import time
from typing import List, Optional, Dict
import z3

from aria.efmc.sts import TransitionSystem
from aria.efmc.utils.verification_utils import VerificationResult

logger = logging.getLogger(__name__)


class BaseKSafetyProver:
    """
    Base k-safety verification engine for relational properties.

    This prover handles verification of k-safety properties by:
    1. Creating k copies of the transition system
    2. Encoding the relational property as a hyperproperty
    3. Using various verification techniques to check the property
    """

    def __init__(self, sts: TransitionSystem, k: int = 2, **kwargs):
        """Initialize the k-safety prover."""
        self.sts = sts
        self.k = k
        self.logger = logging.getLogger(__name__)

        # Configuration options
        self.timeout = kwargs.get("timeout", 300)
        self.verification_method = kwargs.get("method", "bounded_model_checking")
        self.max_bound = kwargs.get("max_bound", 10)
        self.use_induction = kwargs.get("use_induction", True)

        # State for tracking verification
        self.trace_variables = {}
        self.relational_property = None
        self.verification_formula = None

        self.logger.info("Initialized k-safety prover with k=%s", k)

    def set_relational_property(self, property_expr: z3.ExprRef):
        """Set the relational property to verify."""
        self.relational_property = property_expr
        self.logger.info("Set relational property: %s", property_expr)

    def create_trace_variables(self) -> Dict[int, List[z3.ExprRef]]:
        """Create k copies of the system variables for different execution traces."""
        trace_vars = {}

        for trace_idx in range(self.k):
            trace_vars[trace_idx] = []

            for var in self.sts.variables:
                var_name = str(var)
                var_sort = var.sort()

                # Create variable with trace index suffix
                if z3.is_bv_sort(var_sort):
                    new_var = z3.BitVec(f"{var_name}_{trace_idx}", var_sort.size())
                elif var_sort == z3.RealSort():
                    new_var = z3.Real(f"{var_name}_{trace_idx}")
                elif var_sort == z3.IntSort():
                    new_var = z3.Int(f"{var_name}_{trace_idx}")
                else:
                    new_var = z3.Bool(f"{var_name}_{trace_idx}")

                trace_vars[trace_idx].append(new_var)

        self.trace_variables = trace_vars
        return trace_vars

    def _make_step_variables(
        self, bound: int
    ) -> Dict[int, Dict[int, List[z3.ExprRef]]]:
        """Create variables indexed by trace and time step.

        Returns:
            A mapping: trace_idx -> step -> [vars_for_that_step_in_same_order_as self.sts.variables]
        """
        step_vars: Dict[int, Dict[int, List[z3.ExprRef]]] = {}
        for trace_idx in range(self.k):
            step_vars[trace_idx] = {}
            for step in range(bound + 1):
                vars_at_step: List[z3.ExprRef] = []
                for var in self.sts.variables:
                    var_name = str(var)
                    var_sort = var.sort()
                    if z3.is_bv_sort(var_sort):
                        new_var = z3.BitVec(
                            f"{var_name}_{trace_idx}_{step}", var_sort.size()
                        )
                    elif var_sort == z3.RealSort():
                        new_var = z3.Real(f"{var_name}_{trace_idx}_{step}")
                    elif var_sort == z3.IntSort():
                        new_var = z3.Int(f"{var_name}_{trace_idx}_{step}")
                    else:
                        new_var = z3.Bool(f"{var_name}_{trace_idx}_{step}")
                    vars_at_step.append(new_var)
                step_vars[trace_idx][step] = vars_at_step
        return step_vars

    def _substitute_property_at_step(
        self, step_vars: Dict[int, Dict[int, List[z3.ExprRef]]], step: int
    ) -> z3.ExprRef:
        """Return the relational property with trace variables mapped to the given time step."""
        if self.relational_property is None:
            raise ValueError("Relational property must be set before substitution")

        # Ensure we have base trace variables created to know the placeholders used in property
        base_trace_vars = self.trace_variables or self.create_trace_variables()
        subst_pairs = []
        for trace_idx in range(self.k):
            for i, _ in enumerate(self.sts.variables):
                subst_pairs.append(
                    (base_trace_vars[trace_idx][i], step_vars[trace_idx][step][i])
                )
        return z3.substitute(self.relational_property, subst_pairs)

    def create_k_safety_formula(self) -> z3.ExprRef:
        """Create the k-safety verification formula."""
        if self.relational_property is None:
            raise ValueError(
                "Relational property must be set before creating k-safety formula"
            )

        trace_vars = self.create_trace_variables()

        # Create k copies of the transition system
        init_conditions = []
        trans_conditions = []

        for trace_idx in range(self.k):
            # Create substitution for this trace
            var_substitution = [
                (var, trace_vars[trace_idx][i])
                for i, var in enumerate(self.sts.variables)
            ]

            # Substitute variables in init and trans
            init_copy = z3.substitute(self.sts.init, var_substitution)
            trans_copy = z3.substitute(self.sts.trans, var_substitution)

            init_conditions.append(init_copy)
            trans_conditions.append(trans_copy)

        # Create the k-safety formula: antecedent → relational_property
        antecedent = z3.And(*init_conditions, *trans_conditions)
        verification_formula = z3.Implies(antecedent, self.relational_property)

        self.verification_formula = verification_formula
        return verification_formula

    def create_unrolled_k_safety_formula(self, bound: int) -> z3.ExprRef:
        """Create a time-unrolled k-safety verification formula up to the given bound.

        The formula encodes:
            (for each trace: init at step 0 and transitions for steps 0..bound-1)
            -> relational_property at step = bound
        """
        if self.relational_property is None:
            raise ValueError(
                "Relational property must be set before creating k-safety formula"
            )

        # Create per-trace per-step variables
        step_vars = self._make_step_variables(bound)

        # Collect init, trans, and invariant conditions across traces/steps
        conditions = []

        for trace_idx in range(self.k):
            # init at step 0
            init_subst = [
                (self.sts.variables[i], step_vars[trace_idx][0][i])
                for i, _ in enumerate(self.sts.variables)
            ]
            conditions.append(z3.substitute(self.sts.init, init_subst))

            # invariants at step 0..bound, if any
            if getattr(self.sts, "invariants", None):
                for step in range(bound + 1):
                    inv_subst = [
                        (self.sts.variables[i], step_vars[trace_idx][step][i])
                        for i, _ in enumerate(self.sts.variables)
                    ]
                    for inv in self.sts.invariants:
                        conditions.append(z3.substitute(inv, inv_subst))

            # transitions for steps 0..bound-1
            for step in range(bound):
                trans_subst = []
                for i, _ in enumerate(self.sts.variables):
                    # current
                    trans_subst.append(
                        (self.sts.variables[i], step_vars[trace_idx][step][i])
                    )
                    # next
                    trans_subst.append(
                        (self.sts.prime_variables[i], step_vars[trace_idx][step + 1][i])
                    )
                conditions.append(z3.substitute(self.sts.trans, trans_subst))

        antecedent = z3.And(*conditions) if conditions else z3.BoolVal(True)

        # Property at step = bound
        property_at_bound = self._substitute_property_at_step(step_vars, bound)
        verification_formula = z3.Implies(antecedent, property_at_bound)
        self.verification_formula = verification_formula
        return verification_formula

    def bounded_model_checking(self, bound: int) -> VerificationResult:
        """Perform bounded model checking for k-safety verification."""
        self.logger.info("Performing BMC for k-safety with bound %s", bound)

        # Use time-unrolled encoding
        formula = self.create_unrolled_k_safety_formula(bound)
        solver = z3.Solver()
        solver.add(z3.Not(formula))  # Check for violations
        solver.set("timeout", self.timeout * 1000)

        result = solver.check()

        if result == z3.sat:
            model = solver.model()
            self.logger.info("K-safety violation found at bound %s", bound)
            return VerificationResult(False, None, model, is_unsafe=True)
        if result == z3.unsat:
            self.logger.info("No k-safety violation found at bound %s", bound)
            return VerificationResult(True, None)
        self.logger.warning("BMC result unknown at bound %s", bound)
        return VerificationResult(False, None, None, is_unknown=True, timed_out=True)

    def k_induction(self, k_bound: int) -> VerificationResult:
        """Perform k-induction for k-safety verification."""
        self.logger.info("Performing k-induction for k-safety with k=%s", k_bound)

        base_case = self._create_k_induction_base_case(k_bound)
        inductive_step = self._create_k_induction_inductive_step(k_bound)

        # Check base case
        solver_base = z3.Solver()
        solver_base.add(z3.Not(base_case))
        solver_base.set("timeout", self.timeout * 1000)

        base_result = solver_base.check()
        if base_result == z3.sat:
            model = solver_base.model()
            self.logger.info("K-induction base case violation found")
            return VerificationResult(False, None, model, is_unsafe=True)

        # Check inductive step
        solver_ind = z3.Solver()
        solver_ind.add(z3.Not(inductive_step))
        solver_ind.set("timeout", self.timeout * 1000)

        inductive_result = solver_ind.check()

        if inductive_result == z3.unsat:
            self.logger.info("K-induction successful with k=%s", k_bound)
            return VerificationResult(True, None)
        if inductive_result == z3.sat:
            model = solver_ind.model()
            self.logger.info("K-induction inductive step violation found")
            return VerificationResult(False, None, model, is_unsafe=True)
        self.logger.warning("K-induction result unknown")
        return VerificationResult(False, None, None, is_unknown=True, timed_out=True)

    def _create_k_induction_base_case(self, k_bound: int) -> z3.ExprRef:
        """Create the base case for k-induction."""
        # Base case is exactly the unrolled safety property up to k_bound
        return self.create_unrolled_k_safety_formula(k_bound)

    def _create_k_induction_inductive_step(self, k_bound: int) -> z3.ExprRef:
        """Create the inductive step for k-induction.

        Encode: from any k_bound-step path, if property holds at steps 0..k_bound-1,
        then it holds at step k_bound.
        """
        # Build step variables 0..k_bound for each trace
        step_vars = self._make_step_variables(k_bound)

        constraints = []
        for trace_idx in range(self.k):
            # link transitions 0..k_bound-1
            for step in range(k_bound):
                trans_subst = []
                for i, _ in enumerate(self.sts.variables):
                    trans_subst.append(
                        (self.sts.variables[i], step_vars[trace_idx][step][i])
                    )
                    trans_subst.append(
                        (self.sts.prime_variables[i], step_vars[trace_idx][step + 1][i])
                    )
                constraints.append(z3.substitute(self.sts.trans, trans_subst))

        # Assume property at steps 0..k_bound-1
        prop_assumptions = []
        for step in range(k_bound):
            prop_assumptions.append(self._substitute_property_at_step(step_vars, step))

        antecedent = (
            z3.And(*(constraints + prop_assumptions))
            if (constraints or prop_assumptions)
            else z3.BoolVal(True)
        )
        conseq = self._substitute_property_at_step(step_vars, k_bound)
        return z3.Implies(antecedent, conseq)

    def solve(self, timeout: Optional[int] = None) -> VerificationResult:
        """Main solving procedure for k-safety verification."""
        if timeout is not None:
            self.timeout = timeout

        if self.relational_property is None:
            raise ValueError("Relational property must be set before solving")

        self.logger.info("Starting k-safety verification with k=%s", self.k)
        start_time = time.time()

        # Try different verification methods
        if self.verification_method == "bounded_model_checking":
            for bound in range(1, self.max_bound + 1):
                if timeout and (time.time() - start_time > timeout):
                    return VerificationResult(
                        False, None, None, is_unknown=True, timed_out=True
                    )

                result = self.bounded_model_checking(bound)
                if result.is_safe or result.is_unsafe:
                    return result

        if self.verification_method == "k_induction" and self.use_induction:
            for k_bound in range(1, min(self.k, self.max_bound) + 1):
                if timeout and (time.time() - start_time > timeout):
                    return VerificationResult(
                        False, None, None, is_unknown=True, timed_out=True
                    )

                result = self.k_induction(k_bound)
                if result.is_safe or result.is_unsafe:
                    return result

        # If we reach here, we couldn't determine the result
        return VerificationResult(False, None, None, is_unknown=True)
