"""
Implementation of "Inductive Invariant Generation via Abductive Inference" (OOPSLA'13)

This module implements an algorithm that uses abductive inference to automatically generate
inductive invariants for transition systems by iteratively strengthening candidate invariants
through abduction until finding an inductive invariant that proves the desired safety property.
"""

import logging
from typing import Tuple, Optional, List
import time

import z3

from aria.efmc.engines.abduction.abductor.abductor import qe_abduce
from aria.efmc.sts import TransitionSystem
from aria.efmc.utils.verification_utils import (
    VerificationResult,
    SolverTimeout,
    check_entailment,
    are_expressions_equivalent,
)

logger = logging.getLogger(__name__)


class AbductionProver:
    """Implements invariant generation using abductive inference."""

    def __init__(self, system: TransitionSystem, max_iterations: int = 300):
        self.sts = system
        self.max_iterations = max_iterations
        self.var_map: List[Tuple[z3.ExprRef, z3.ExprRef]] = list(
            zip(self.sts.variables, self.sts.prime_variables)
        )
        self.var_map_rev: List[Tuple[z3.ExprRef, z3.ExprRef]] = list(
            zip(self.sts.prime_variables, self.sts.variables)
        )
        self.last_cti: Optional[z3.ModelRef] = None

    def check_inductiveness(
        self, inv: z3.ExprRef
    ) -> Tuple[bool, Optional[z3.ModelRef]]:
        """Check if the candidate invariant is inductive: I ∧ T → I'"""
        inv_prime = z3.substitute(inv, self.var_map)
        s = z3.Solver()
        s.add(inv, self.sts.trans, z3.Not(inv_prime))
        result = s.check()
        return result == z3.unsat, s.model() if result == z3.sat else None

    def strengthen_from_cti(self, inv: z3.ExprRef, cti: z3.ModelRef) -> z3.ExprRef:
        """Strengthen the invariant using a counterexample to inductiveness."""
        self.last_cti = cti

        # Extract pre-state constraints from CTI
        pre_state_constraints: List[z3.ExprRef] = [
            v == cti[v] for v in self.sts.variables if v in cti
        ]
        pre_state = (
            z3.And(*pre_state_constraints)
            if pre_state_constraints
            else z3.BoolVal(True)
        )

        # Set up abduction query
        pre_cond = z3.And(inv, pre_state, self.sts.trans)
        post_cond = z3.substitute(inv, self.var_map)

        try:
            strengthening = qe_abduce(pre_cond, post_cond)
            if strengthening is None:
                return self._generalize_strengthening(inv, pre_state, cti)

            new_inv = z3.simplify(z3.And(inv, strengthening))

            # Verify strengthening excludes CTI and isn't False
            s = z3.Solver()
            s.add(new_inv, pre_state)
            if s.check() == z3.sat or z3.is_false(new_inv):
                return self._generalize_strengthening(inv, pre_state, cti)

            return new_inv
        except Exception as e:
            logger.warning("Abduction error: %s. Using targeted exclusion.", e)
            return self._generalize_strengthening(inv, pre_state, cti)

    def _generalize_strengthening(
        self, inv: z3.ExprRef, pre_state: z3.ExprRef, cti: z3.ModelRef
    ) -> z3.ExprRef:
        """Apply targeted strengthening based on generalizing from the CTI."""
        # Try inequality constraints for numeric variables
        constraints: List[z3.ExprRef] = []
        for v in self.sts.variables:
            if v in cti and str(v) in str(inv):
                val = cti[v]
                if z3.is_int_value(val):
                    val_int = val.as_long()
                    if val_int > 0:
                        constraints.append(v <= val_int - 1)
                    elif val_int < 0:
                        constraints.append(v >= val_int + 1)
                    else:
                        constraints.extend([v > 0, v < 0])
                elif z3.is_bool(val):
                    constraints.append(z3.Not(v) if z3.is_true(val) else v)

        # Try generalized constraints
        if constraints:
            for combiner in [z3.Or, z3.And]:
                new_inv = z3.simplify(z3.And(inv, combiner(*constraints)))
                s = z3.Solver()
                s.add(new_inv, pre_state)
                if s.check() == z3.unsat and not z3.is_false(new_inv):
                    return new_inv

        # Fallback: exclude specific variable values
        exclusions: List[z3.ExprRef] = [
            v != cti[v] for v in self.sts.variables if v in cti
        ]
        if exclusions:
            return z3.simplify(z3.And(inv, z3.Or(*exclusions)))

        return inv

    def invgen(self, overall_timeout: Optional[int] = None) -> Optional[z3.ExprRef]:
        """Generate inductive invariant using abduction."""
        start_time = time.time()

        # Start with post-condition as candidate invariant
        candidate_inv = self.sts.post

        for iteration in range(self.max_iterations):
            # Check timeout
            if overall_timeout and (time.time() - start_time > overall_timeout):
                logger.warning("Abduction timeout reached")
                return None

            # Check if candidate is inductive
            is_inductive, cti = self.check_inductiveness(candidate_inv)

            if is_inductive:
                logger.info("Found inductive invariant after %d iterations", iteration)
                return candidate_inv

            if cti is None:
                logger.warning("Could not find CTI")
                return None

            # Strengthen invariant using CTI
            logger.debug("Iteration %d: strengthening invariant", iteration)
            new_inv = self.strengthen_from_cti(candidate_inv, cti)

            # Check if strengthening made invariant False
            if z3.is_false(new_inv):
                logger.warning("Strengthening resulted in False invariant")
                return None

            # Check if invariant changed
            if are_expressions_equivalent(candidate_inv, new_inv):
                logger.warning("Invariant did not change after strengthening")
                return None

            candidate_inv = new_inv

        logger.warning("Reached maximum iterations (%d)", self.max_iterations)
        return None

    def solve(self, timeout: Optional[int] = None) -> VerificationResult:
        """Solve verification problem using abduction."""
        logger.info("Starting abduction-based invariant generation")
        start_time = time.time()

        try:
            # Generate inductive invariant
            invariant = self.invgen(timeout)

            if invariant is None:
                logger.info("Could not generate inductive invariant")
                return VerificationResult(False, None, is_unknown=True)

            # Verify the invariant proves the safety property
            if check_entailment(invariant, self.sts.post):
                logger.info("Abduction successful: found inductive invariant")
                elapsed_time = time.time() - start_time
                logger.info("Abduction completed in %.2fs", elapsed_time)
                return VerificationResult(True, invariant)

            logger.warning("Generated invariant does not entail safety property")
            return VerificationResult(False, None, is_unknown=True)

        except SolverTimeout:
            logger.warning("Abduction timed out")
            return VerificationResult(False, None, is_unknown=True, timed_out=True)
        except Exception as e:
            logger.error("Abduction failed: %s", e)
            return VerificationResult(False, None, is_unknown=True)
