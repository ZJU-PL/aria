"""
Auxiliary invariant generation for k-induction using template-based (EF) and Houdini approaches.
"""

import logging
from typing import Optional

import z3

from aria.efmc.engines.ef.ef_prover import EFProver
from aria.efmc.engines.houdini.houdini_prover import HoudiniProver
from aria.efmc.engines.houdini.houdini_templates import generate_templates
from aria.efmc.sts import TransitionSystem

logger = logging.getLogger(__name__)


class InvariantGenerator:
    """Generate auxiliary invariants for strengthening k-induction proofs."""

    def __init__(self, sts: TransitionSystem):
        self.sts = sts

    def generate_via_ef(self) -> z3.ExprRef:
        """Generate auxiliary invariant using exists-forall template-based approach."""
        ef_prover = EFProver(self.sts)
        ef_prover.ignore_post_cond = True  # Ignore post condition for aux invariant
        ef_prover.validate_invariant = False  # Skip validation for efficiency

        # Set appropriate template based on variable types
        if self.sts.has_bv:
            ef_prover.set_template("bv_interval")
        else:
            ef_prover.set_template("interval")

        if ef_prover.solve():
            inv = getattr(ef_prover, "inductive_invaraint", None)
            if inv and not z3.is_true(z3.simplify(inv)):
                logger.info("EF generated auxiliary invariant: %s", inv)
                return inv

        logger.debug("EF could not generate useful auxiliary invariant")
        return z3.BoolVal(True)

    def generate_via_houdini(self, timeout: Optional[int] = None) -> z3.ExprRef:
        """Generate auxiliary invariant using Houdini approach."""
        # Create modified STS without post condition for auxiliary invariant generation
        aux_sts = TransitionSystem()
        aux_sts.variables = self.sts.variables
        aux_sts.prime_variables = self.sts.prime_variables
        aux_sts.init = self.sts.init
        aux_sts.trans = self.sts.trans
        aux_sts.post = z3.BoolVal(True)  # Ignore original post condition

        houdini = HoudiniProver(aux_sts)

        # Generate candidate lemmas using templates
        templates = generate_templates(self.sts.variables, {"bounds", "ordering"})
        result = houdini.houdini(templates, timeout)

        if result and len(result) > 0:
            inv = z3.And(*result.values())
            if not z3.is_true(z3.simplify(inv)):
                logger.info("Houdini generated auxiliary invariant: %s", inv)
                return inv

        logger.debug("Houdini could not generate useful auxiliary invariant")
        return z3.BoolVal(True)

    def generate(self, method: str = "ef", timeout: Optional[int] = None) -> z3.ExprRef:
        """
        Generate auxiliary invariant using specified method.

        Args:
            method: "ef" for template-based, "houdini" for Houdini approach
            timeout: Timeout in seconds (applies to Houdini)
        """
        if method == "ef":
            return self.generate_via_ef()
        if method == "houdini":
            return self.generate_via_houdini(timeout)
        raise ValueError(f"Unknown method: {method}. Use 'ef' or 'houdini'")
