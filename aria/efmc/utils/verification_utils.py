"""
Common verification utilities for transition systems.

This module provides common data structures and functions for verification tasks:
- VerificationResult: Data class for storing verification results
- SolverTimeout: Exception for solver timeouts
- check_entailment: Function to check logical entailment
- are_expressions_equivalent: Function to check logical equivalence
- check_invariant: Function to verify if an invariant is correct
- get_counterexample: Function to get a counterexample for a formula
"""

import logging
from dataclasses import dataclass
from typing import Optional, Any, List, Tuple, Dict

import z3

# Remove the direct import to break the circular dependency
# from aria.efmc.sts import TransitionSystem
from aria.utils.z3.solver import is_entail

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """
    Stores the result of the verification process.

    Fields:
        is_safe: True if the system is proven safe, False otherwise
        is_unsafe: True if the system is proven unsafe (counterexample exists), False otherwise
        is_unknown: True if the safety status could not be determined, False otherwise
        invariant: The inductive invariant if the system is safe, None otherwise
        counterexample: A counterexample if the system is unsafe, None otherwise
        timed_out: True if the verification process timed out, False otherwise

    Note: At most one of is_safe, is_unsafe, and is_unknown should be True.
    """

    is_safe: bool
    invariant: Optional[z3.ExprRef]
    counterexample: Optional[z3.ModelRef] = None
    is_unknown: bool = False
    is_unsafe: bool = False
    timed_out: bool = False

    def __post_init__(self):
        """Validate the verification result."""
        # If we have a counterexample, the system should be unsafe
        if self.counterexample is not None:
            self.is_unsafe = True

        # If the verification timed out, mark it as unknown
        if self.timed_out:
            self.is_unknown = True

        # Ensure consistency of the result
        safety_flags = [self.is_safe, self.is_unsafe, self.is_unknown]
        if sum(safety_flags) != 1:
            logger.warning("Inconsistent verification result: %s", safety_flags)


class SolverTimeout(Exception):
    """Raised when the Z3 solver times out."""


def check_entailment(
    expr1: z3.ExprRef, expr2: z3.ExprRef, timeout: int = 10000
) -> bool:
    """
    Check if expr1 entails expr2 using Z3 solver.
    """
    s = z3.Solver()
    s.set("timeout", timeout)
    s.add(z3.And(expr1, z3.Not(expr2)))

    result = s.check()
    if result == z3.unknown:
        raise SolverTimeout("Solver timed out during entailment check")
    return result == z3.unsat


def are_expressions_equivalent(
    expr1: z3.ExprRef, expr2: z3.ExprRef, timeout: int = 10000
) -> bool:
    """
    Check if two Z3 expressions are logically equivalent.
    """
    s = z3.Solver()
    s.set("timeout", timeout)

    # Check if (expr1 ∧ ¬expr2) ∨ (¬expr1 ∧ expr2) is unsatisfiable
    equivalence_check = z3.Or(
        z3.And(expr1, z3.Not(expr2)), z3.And(z3.Not(expr1), expr2)
    )
    s.add(equivalence_check)

    result = s.check()
    if result == z3.unknown:
        raise SolverTimeout("Solver timed out during equivalence check")
    return result == z3.unsat


def get_counterexample(formula: z3.ExprRef):
    """
    Get a counterexample for the given formula.
    """
    s = z3.Solver()
    s.add(formula)
    if s.check() == z3.sat:
        return s.model()
    return None


def check_invariant(
    sts: Any, inv: z3.ExprRef, inv_in_prime_variables: z3.ExprRef
) -> bool:
    """
    Check whether the generated invariant is correct.

    Verifies three conditions:
    1. Initiation: init => inv
    2. Inductiveness: inv && trans => inv'
    3. Safety: inv => post

    Args:
        sts: The transition system
        inv: The invariant to check
        inv_in_prime_variables: The invariant with variables replaced by primed versions

    Returns:
        True if the invariant is correct, False otherwise
    """
    correct = True

    # 1. Check initiation: init => inv
    if not is_entail(sts.init, inv):
        correct = False
        logger.error("Initiation check failed")
        model = get_counterexample(z3.And(sts.init, z3.Not(inv)))
        if model:
            logger.error("Counterexample: %s", model)

    # 2. Check inductiveness: inv && trans => inv'
    if not is_entail(z3.And(sts.trans, inv), inv_in_prime_variables):
        correct = False
        logger.error("Inductiveness check failed")
        model = get_counterexample(
            z3.And(sts.trans, inv, z3.Not(inv_in_prime_variables))
        )
        if model:
            logger.error("Counterexample: %s", model)

    # 3. Check safety/sufficiency: inv => post
    # NOTICE: here the guard "C" in Hoare logic is "moved" to the
    # transition relation, and we can just check is_entail(inv, sts.post)
    # for correctness.
    if (
        hasattr(sts, "ignore_post_cond")
        and (not sts.ignore_post_cond)
        and (not is_entail(inv, sts.post))
    ):
        # for some applications, we may ignore the post condition
        correct = False
        logger.error("Safety check failed")
        model = get_counterexample(z3.And(inv, z3.Not(sts.post)))
        if model:
            logger.error("Counterexample: %s", model)

    if not correct:
        logger.debug("Init: %s", sts.init)
        logger.debug("Trans: %s", sts.trans)
        logger.debug("Post: %s", sts.post)
        logger.debug("Inv: %s", inv)
    else:
        logger.info("Invariant verification successful")

    return correct


def verify_invariant_with_counterexamples(
    sts: Any, inv: z3.ExprRef
) -> Tuple[bool, List[Tuple[z3.ModelRef, z3.ModelRef]]]:
    """
    Unified function for invariant verification and counterexample extraction.

    Verifies three conditions:
    1. Initiation: init => inv
    2. Inductiveness: inv && trans => inv'
    3. Safety: inv => post (if not ignored)

    Args:
        sts: The transition system
        inv: The invariant to check

    Returns:
        Tuple of (is_correct, counterexamples) where:
        - is_correct: True if the invariant is correct, False otherwise
        - counterexamples: List of (pre_state, post_state) model pairs for failed conditions
    """

    def extract_state_from_model(
        model: z3.ModelRef, use_prime: bool = False
    ) -> Dict[z3.ExprRef, Any]:
        """Extract state dictionary from model, optionally using primed variables."""
        state = {}
        for i, var in enumerate(sts.variables):
            if use_prime and i < len(sts.prime_variables):
                target_var = sts.prime_variables[i]
            else:
                target_var = var
            if target_var.decl() in model:
                state[var] = model[target_var.decl()]
        return state

    def check_condition(
        condition: z3.ExprRef,
        error_msg: str,
        counterexample_formula: z3.ExprRef,
        use_prime: bool = False,
    ) -> bool:
        """Check a condition and extract counterexample if it fails."""
        if not is_entail(condition, inv if not use_prime else primed_invariant):
            logger.error(error_msg)
            model = get_counterexample(counterexample_formula)
            if model:
                logger.error("Counterexample: %s", model)
                if use_prime:
                    pre_state = extract_state_from_model(model, False)
                    post_state = extract_state_from_model(model, True)
                    if pre_state and post_state:
                        counterexamples.append((pre_state, post_state))
                else:
                    state = extract_state_from_model(model)
                    if state:
                        counterexamples.append((state, None))
            return False
        return True

    counterexamples = []
    correct = True

    # Create primed version of invariant for inductiveness check
    substitution_map = [
        (var, sts.prime_variables[i])
        for i, var in enumerate(sts.variables)
        if i < len(sts.prime_variables)
    ]
    primed_invariant = z3.substitute(inv, substitution_map)

    # Check all three conditions
    checks = [
        (sts.init, "Initiation check failed", z3.And(sts.init, z3.Not(inv)), False),
        (
            z3.And(sts.trans, inv),
            "Inductiveness check failed",
            z3.And(sts.trans, inv, z3.Not(primed_invariant)),
            True,
        ),
    ]

    # Add safety check if not ignored
    if not (hasattr(sts, "ignore_post_cond") and sts.ignore_post_cond):
        checks.append(
            (inv, "Safety check failed", z3.And(inv, z3.Not(sts.post)), False)
        )

    for condition, error_msg, counterexample_formula, use_prime in checks:
        if not check_condition(condition, error_msg, counterexample_formula, use_prime):
            correct = False

    # Log results
    if not correct:
        logger.debug(
            "Init: %s\nTrans: %s\nPost: %s\nInv: %s", sts.init, sts.trans, sts.post, inv
        )
    else:
        logger.info("Invariant verification successful")

    return correct, counterexamples
