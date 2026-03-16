"""BDD-based Symbolic Model Checking for Boolean Programs

This prover uses Z3's Boolean reasoning capabilities to implement BDD-like
symbolic model checking for Boolean programs.

TBD: Use pysmt to access BDDs, which allows us to implement the "real"
BDD-based model checking and could be faster and more memory efficient.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import z3

from aria.efmc.sts import TransitionSystem
from aria.efmc.utils.verification_utils import VerificationResult

logger = logging.getLogger(__name__)


class BDDProver:  # pylint: disable=too-many-instance-attributes
    """
    BDD-based symbolic model checker for Boolean programs.

    This prover uses Z3's Boolean reasoning capabilities to implement BDD-like
    symbolic model checking for Boolean programs. It implements both forward and backward
    reachability analysis to verify safety properties.
    """

    def __init__(
        self,
        system: TransitionSystem,
        use_forward: bool = True,
        max_iterations: int = 1000,
    ):
        """
        Initialize the BDD prover with a transition system.

        Args:
            system: The transition system to verify
            use_forward: Whether to use forward reachability analysis (True) or
                         backward reachability analysis (False)
            max_iterations: Maximum number of iterations for fixed-point computation
        """
        self.sts = system
        self.use_forward = use_forward
        self.max_iterations = max_iterations

        # Check if the system is suitable for BDD-based verification
        if not self.sts.has_bool and not self.sts.has_bv:
            logger.warning(
                "BDD-based verification works best with Boolean or "
                "bit-vector variables"
            )

        # Create solver for fixed-point computations
        self.solver = z3.Solver()

        # Store the formulas
        self.init_formula = self.sts.init
        self.trans_formula = self.sts.trans
        self.post_formula = self.sts.post
        self.not_post_formula = z3.Not(self.sts.post)

        # Create substitution maps
        self.prime_to_current: Dict[z3.ExprRef, z3.ExprRef] = {}
        self.current_to_prime: Dict[z3.ExprRef, z3.ExprRef] = {}

        for p, c in zip(self.sts.prime_variables, self.sts.variables):
            self.prime_to_current[p] = c
            self.current_to_prime[c] = p

    def _substitute_prime_with_current(self, formula: z3.ExprRef) -> z3.ExprRef:
        """
        Substitute prime variables with current variables in a formula.

        Args:
            formula: The formula to perform substitution on

        Returns:
            A new formula with prime variables replaced by current variables
        """
        substitutions = list(self.prime_to_current.items())
        return z3.substitute(formula, substitutions)

    def _substitute_current_with_prime(self, formula: z3.ExprRef) -> z3.ExprRef:
        """
        Substitute current variables with prime variables in a formula.

        Args:
            formula: The formula to perform substitution on

        Returns:
            A new formula with current variables replaced by prime variables
        """
        substitutions = list(self.current_to_prime.items())
        return z3.substitute(formula, substitutions)

    def _image(self, states_formula: z3.ExprRef) -> z3.ExprRef:
        """
        Compute the image (successor states) of a set of states.

        Args:
            states_formula: Formula representing the set of states

        Returns:
            Formula representing the successor states
        """
        # For bit-vector systems, we need a special approach
        if self.sts.has_bv:
            return self._image_bv(states_formula)
        return self._image_bool(states_formula)

    def _image_bool(self, states_formula: z3.ExprRef) -> z3.ExprRef:
        """
        Compute the image for Boolean systems.

        Args:
            states_formula: Formula representing the set of states

        Returns:
            Formula representing the successor states
        """
        # Create a new solver for this computation
        s = z3.Solver()

        # Add the states and transition relation
        s.add(states_formula)
        s.add(self.trans_formula)

        # If the conjunction is unsatisfiable, return False
        if s.check() == z3.unsat:
            return z3.BoolVal(False)

        # Get all models that satisfy the conjunction
        result_states: List[z3.ExprRef] = []

        # Keep finding models until we can't find any more
        while s.check() == z3.sat:
            model = s.model()

            # Extract the values of prime variables
            prime_values: List[z3.ExprRef] = []
            for var in self.sts.prime_variables:
                if model.get_interp(var) is not None:
                    prime_values.append(var == model.get_interp(var))

            # Create a formula representing this state
            if prime_values:
                state_formula = z3.And(prime_values)
                # Substitute prime variables with current variables
                current_state = self._substitute_prime_with_current(state_formula)
                result_states.append(current_state)

            # Block this model
            block: List[z3.ExprRef] = []
            for var in self.sts.prime_variables:
                if model.get_interp(var) is not None:
                    block.append(var != model.get_interp(var))

            if block:
                s.add(z3.Or(block))
            else:
                break

        # If no states were found, return False
        if not result_states:
            return z3.BoolVal(False)

        # Combine all states
        return z3.Or(result_states)

    def _image_bv(self, states_formula: z3.ExprRef) -> z3.ExprRef:
        """
        Compute the image for bit-vector systems.

        Args:
            states_formula: Formula representing the set of states

        Returns:
            Formula representing the successor states
        """
        # Create a new solver for this computation
        s = z3.Solver()

        # Add the states and transition relation
        s.add(states_formula)
        s.add(self.trans_formula)

        # If the conjunction is unsatisfiable, return False
        if s.check() == z3.unsat:
            return z3.BoolVal(False)

        # Get all models that satisfy the conjunction
        result_states: List[z3.ExprRef] = []

        # Keep finding models until we can't find any more
        while s.check() == z3.sat:
            model = s.model()

            # Extract the values of prime variables
            prime_values: List[z3.ExprRef] = []
            for var in self.sts.prime_variables:
                if model.get_interp(var) is not None:
                    prime_values.append(var == model.get_interp(var))

            # Create a formula representing this state
            if prime_values:
                state_formula = z3.And(prime_values)
                # Substitute prime variables with current variables
                current_state = self._substitute_prime_with_current(state_formula)
                result_states.append(current_state)

            # Block this model
            block: List[z3.ExprRef] = []
            for var in self.sts.prime_variables:
                if model.get_interp(var) is not None:
                    block.append(var != model.get_interp(var))

            if block:
                s.add(z3.Or(block))
            else:
                break

        # If no states were found, return False
        if not result_states:
            return z3.BoolVal(False)

        # Combine all states
        return z3.Or(result_states)

    def _pre_image(self, states_formula: z3.ExprRef) -> z3.ExprRef:
        """
        Compute the pre-image (predecessor states) of a set of states.

        Args:
            states_formula: Formula representing the set of states

        Returns:
            Formula representing the predecessor states
        """
        # For bit-vector systems, we need a special approach
        if self.sts.has_bv:
            return self._pre_image_bv(states_formula)
        return self._pre_image_bool(states_formula)

    def _pre_image_bool(self, states_formula: z3.ExprRef) -> z3.ExprRef:
        """
        Compute the pre-image for Boolean systems.

        Args:
            states_formula: Formula representing the set of states

        Returns:
            Formula representing the predecessor states
        """
        # Substitute current variables with prime variables in states_formula
        prime_states_formula = self._substitute_current_with_prime(states_formula)

        # Create a new solver for this computation
        s = z3.Solver()

        # Add the prime states and transition relation
        s.add(prime_states_formula)
        s.add(self.trans_formula)

        # If the conjunction is unsatisfiable, return False
        if s.check() == z3.unsat:
            return z3.BoolVal(False)

        # Get all models that satisfy the conjunction
        result_states: List[z3.ExprRef] = []

        # Keep finding models until we can't find any more
        while s.check() == z3.sat:
            model = s.model()

            # Extract the values of current variables
            current_values: List[z3.ExprRef] = []
            for var in self.sts.variables:
                if model.get_interp(var) is not None:
                    current_values.append(var == model.get_interp(var))

            # Create a formula representing this state
            if current_values:
                state_formula = z3.And(current_values)
                result_states.append(state_formula)

            # Block this model
            block: List[z3.ExprRef] = []
            for var in self.sts.variables:
                if model.get_interp(var) is not None:
                    block.append(var != model.get_interp(var))

            if block:
                s.add(z3.Or(block))
            else:
                break

        # If no states were found, return False
        if not result_states:
            return z3.BoolVal(False)

        # Combine all states
        return z3.Or(result_states)

    def _pre_image_bv(self, states_formula: z3.ExprRef) -> z3.ExprRef:
        """
        Compute the pre-image for bit-vector systems.

        Args:
            states_formula: Formula representing the set of states

        Returns:
            Formula representing the predecessor states
        """
        # Substitute current variables with prime variables in states_formula
        prime_states_formula = self._substitute_current_with_prime(states_formula)

        # Create a new solver for this computation
        s = z3.Solver()

        # Add the prime states and transition relation
        s.add(prime_states_formula)
        s.add(self.trans_formula)

        # If the conjunction is unsatisfiable, return False
        if s.check() == z3.unsat:
            return z3.BoolVal(False)

        # Get all models that satisfy the conjunction
        result_states: List[z3.ExprRef] = []

        # Keep finding models until we can't find any more
        while s.check() == z3.sat:
            model = s.model()

            # Extract the values of current variables
            current_values: List[z3.ExprRef] = []
            for var in self.sts.variables:
                if model.get_interp(var) is not None:
                    current_values.append(var == model.get_interp(var))

            # Create a formula representing this state
            if current_values:
                state_formula = z3.And(current_values)
                result_states.append(state_formula)

            # Block this model
            block: List[z3.ExprRef] = []
            for var in self.sts.variables:
                if model.get_interp(var) is not None:
                    block.append(var != model.get_interp(var))

            if block:
                s.add(z3.Or(block))
            else:
                break

        # If no states were found, return False
        if not result_states:
            return z3.BoolVal(False)

        # Combine all states
        return z3.Or(result_states)

    def _is_contained(self, formula1: z3.ExprRef, formula2: z3.ExprRef) -> bool:
        """
        Check if formula1 implies formula2 (formula1 ⊆ formula2).

        Args:
            formula1: First formula
            formula2: Second formula

        Returns:
            True if formula1 implies formula2, False otherwise
        """
        # formula1 ⊆ formula2 iff formula1 ∧ ¬formula2 is unsatisfiable
        s = z3.Solver()
        s.add(z3.And(formula1, z3.Not(formula2)))
        return s.check() == z3.unsat

    def _is_equal(self, formula1: z3.ExprRef, formula2: z3.ExprRef) -> bool:
        """
        Check if formula1 is equivalent to formula2.

        Args:
            formula1: First formula
            formula2: Second formula

        Returns:
            True if formula1 is equivalent to formula2, False otherwise
        """
        # formula1 = formula2 iff (formula1 ⊆ formula2) and (formula2 ⊆ formula1)
        contained_1_in_2 = self._is_contained(formula1, formula2)
        # We intentionally swap arguments to check formula2 ⊆ formula1
        contained_2_in_1 = self._is_contained(
            formula2, formula1
        )  # pylint: disable=arguments-out-of-order
        return contained_1_in_2 and contained_2_in_1

    def _is_sat(self, formula: z3.ExprRef) -> bool:
        """
        Check if a formula is satisfiable.

        Args:
            formula: Formula to check

        Returns:
            True if formula is satisfiable, False otherwise
        """
        s = z3.Solver()
        s.add(formula)
        return s.check() == z3.sat

    def forward_analysis(self) -> Tuple[bool, Optional[z3.ExprRef]]:
        """
        Perform forward reachability analysis.

        Returns:
            Tuple[bool, Optional[z3.ExprRef]]: (is_safe, invariant)
                is_safe: True if the system is safe, False otherwise
                invariant: An invariant formula if the system is safe, None otherwise
        """
        logger.info("Starting forward reachability analysis")

        # Initialize reached states with initial states
        reached_formula = self.init_formula
        frontier_formula = self.init_formula
        step = 0

        # Check if initial states satisfy the property
        if self._is_sat(z3.And(reached_formula, self.not_post_formula)):
            logger.info("Initial states violate the property")
            return False, None

        # Fixed-point computation
        while True:
            step += 1
            logger.info("Forward analysis step %s", step)

            # Compute successor states
            new_frontier_formula = self._image(frontier_formula)

            # Check if new states violate the property
            if self._is_sat(z3.And(new_frontier_formula, self.not_post_formula)):
                logger.info("Property violation found at step %s", step)
                return False, None

            # Check if we've reached a fixed point
            # We need to check if new_frontier_formula is contained in reached_formula
            if self._is_contained(new_frontier_formula, reached_formula):
                # No new states, we've reached a fixed point
                logger.info("Fixed point reached after %s steps", step)
                # The invariant is the set of reached states
                return True, reached_formula

            # Add new states to reached states
            reached_formula = z3.Or(reached_formula, new_frontier_formula)
            frontier_formula = new_frontier_formula

            # Limit the number of iterations to avoid infinite loops
            if step > self.max_iterations:
                logger.warning(
                    "Reached maximum number of iterations (%s), stopping",
                    self.max_iterations,
                )
                # For safety, we assume the system is safe if we've reached the
                # maximum number of iterations and haven't found a property violation
                return True, reached_formula

    def backward_analysis(self) -> Tuple[bool, Optional[z3.ExprRef]]:
        """
        Perform backward reachability analysis.

        Returns:
            Tuple[bool, Optional[z3.ExprRef]]: (is_safe, invariant)
                is_safe: True if the system is safe, False otherwise
                invariant: An invariant formula if the system is safe, None otherwise
        """
        logger.info("Starting backward reachability analysis")

        # Initialize bad states with negation of post-condition
        bad_formula = self.not_post_formula
        frontier_formula = bad_formula
        step = 0

        # Check if initial states intersect with bad states
        if self._is_sat(z3.And(self.init_formula, bad_formula)):
            logger.info("Initial states violate the property")
            return False, None

        # Fixed-point computation
        while True:
            step += 1
            logger.info("Backward analysis step %s", step)

            # Compute predecessor states
            new_frontier_formula = self._pre_image(frontier_formula)

            # Check if new bad states intersect with initial states
            if self._is_sat(z3.And(new_frontier_formula, self.init_formula)):
                logger.info("Property violation found at step %s", step)
                return False, None

            # Check if we've reached a fixed point
            # We need to check if new_frontier_formula is contained in bad_formula
            if self._is_contained(new_frontier_formula, bad_formula):
                # No new states, we've reached a fixed point
                logger.info("Fixed point reached after %s steps", step)
                # The invariant is the negation of the bad states
                return True, z3.Not(bad_formula)

            # Add new states to bad states
            bad_formula = z3.Or(bad_formula, new_frontier_formula)
            frontier_formula = new_frontier_formula

            # Limit the number of iterations to avoid infinite loops
            if step > self.max_iterations:
                logger.warning(
                    "Reached maximum number of iterations (%s), stopping",
                    self.max_iterations,
                )
                # For safety, we assume the system is safe if we've reached the
                # maximum number of iterations and haven't found a property violation
                return True, z3.Not(bad_formula)

    def solve(self) -> VerificationResult:
        """
        Verify the safety property of the transition system.

        Returns:
            VerificationResult: The result of the verification
        """
        assert self.sts.initialized

        print("BDD-based model checking starting...")
        start = time.time()

        try:
            if self.use_forward:
                is_safe, invariant = self.forward_analysis()
            else:
                is_safe, invariant = self.backward_analysis()

            print(f"BDD verification time: {time.time() - start:.2f} seconds")

            if is_safe:
                print("Safe")
                print(f"Invariant: {invariant}")
                return VerificationResult(True, invariant)
            print("Unsafe")
            return VerificationResult(False, None, is_unsafe=True)

        except (RuntimeError, ValueError, z3.Z3Exception) as e:
            print(f"BDD verification failed: {e}")
            print(f"BDD verification time: {time.time() - start:.2f} seconds")
            print("Unknown")
            return VerificationResult(False, None, None, is_unknown=True)
