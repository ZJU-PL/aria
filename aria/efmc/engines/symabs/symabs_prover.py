"""
This is an "old-school" version of symbolic abstraction domain.
"""

import logging
from typing import Optional
import z3

from aria.efmc.engines.symabs.bits_symabs import (
    strongest_consequence as bits_strongest_consequence,
)
from aria.efmc.engines.symabs.bv_symabs import BVSymbolicAbstraction
from aria.efmc.sts import TransitionSystem
from aria.efmc.utils import is_valid
from aria.efmc.utils.verification_utils import VerificationResult

logger = logging.getLogger(__name__)


############################
def strongest_consequence(
    fml: z3.ExprRef, domain: str, k=None
) -> z3.ExprRef:  # pylint: disable=unused-argument
    """Compute the strongest consequence of a formula in a given domain.

    Args:
        fml: The formula to abstract
        domain: The abstract domain to use ('interval', 'bits', 'known_bits')
        k: Optional parameter for domain-specific configuration (unused)

    Returns:
        A formula representing the strongest consequence in the specified domain
    """
    if domain == "interval":
        # Create a symbolic abstraction object
        symabs = BVSymbolicAbstraction()

        # Initialize with the formula
        symabs.formula = fml
        symabs.initialized = True

        # Extract variables from the formula
        symabs.vars = list(z3.z3util.get_vars(fml))
        if not symabs.vars:
            # If no variables, just return the formula
            return fml

        # Simplify the formula
        symabs.do_simplification()

        # Compute interval abstraction
        symabs.interval_abs()

        # Return the interval abstraction as a formula
        return symabs.interval_abs_as_fml
    if domain in ["bits", "known_bits"]:
        # Map domain names
        bits_domain = domain
        if domain == "bits":
            bits_domain = "all"

        # Compute the strongest consequence using bit-level abstractions
        return bits_strongest_consequence(fml, bits_domain)
    raise NotImplementedError(f"Domain '{domain}' not implemented")


def fixpoint(old_inv: z3.ExprRef, inv: z3.ExprRef) -> bool:
    """Decide whether reaching a fixpoint or not..
    TODO: is this true? (e.g. should we check for equivalence?)"""
    return is_valid(z3.Implies(inv, old_inv))


class SymbolicAbstractionProver:
    """Symbolic abstraction prover for verification."""

    def __init__(self, system: TransitionSystem):
        self.sts = system
        """
        var_map = [(x, xp), (y, yp)]
        var_map_rev = [(xp ,x), (yp, y)]
        """
        self.var_map = []
        self.var_map_rev = []
        for i, var in enumerate(self.sts.variables):
            self.var_map.append((var, self.sts.prime_variables[i]))
            self.var_map_rev.append((self.sts.prime_variables[i], var))

        # Default domain for symbolic abstraction
        self.domain = "interval"
        # Predicates for predicate abstraction
        self.preds = []

    def solve(
        self, timeout: Optional[int] = None
    ) -> VerificationResult:  # pylint: disable=unused-argument
        """External interface for verification."""
        preds_prime = []
        for pred in self.preds:
            preds_prime.append(z3.substitute(pred, self.var_map))

        old_inv = z3.BoolVal(False)
        inv = self.sts.init
        i = 0
        while not fixpoint(old_inv, inv):
            print("\nInv at ", i, ": ", inv)
            i = i + 1

            # Compute the strongest consequence in the specified domain
            if self.sts.has_bv and self.domain in ["bits", "known_bits"]:
                # Use bit-level abstractions for bit-vector formulas
                onestep = strongest_consequence(
                    z3.And(inv, self.sts.trans), self.domain
                )
            else:
                # Use interval abstraction by default
                onestep = strongest_consequence(z3.And(inv, self.sts.trans), "interval")

            # Substitute prime variables back to non-prime variables
            onestep = z3.substitute(onestep, self.var_map_rev)
            old_inv = inv
            inv = z3.simplify(z3.Or(inv, onestep))
        # print("\n")

        # check whether the invariant is precise enough
        if is_valid(z3.Implies(inv, self.sts.post)):
            print(z3.simplify(inv))
            print(">>> SAFE\n\n")
            return VerificationResult(True, inv)
        # need refinement
        print(">>> MAYBE?!?!\n\n")
        return VerificationResult(False, None, is_unknown=True)
