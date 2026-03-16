"""
Refinement checking: implementation refines specification.

Refinement here means: for the same inputs, every output allowed by spec is matched by impl
at final step (deterministic outputs, equality). This is a simple subset relation encoded
as equality under matched inputs.
"""

import logging
from typing import List

from aria.efmc.sts import TransitionSystem
from aria.efmc.utils.verification_utils import VerificationResult
from .equivalence import EquivalenceProver

logger = logging.getLogger(__name__)


class RefinementProver(EquivalenceProver):
    """
    Check that implementation refines specification by equality of outputs under same inputs.
    Currently same encoding as functional equivalence; extension points include weakening
    the consequent to allow nondeterministic spec behaviors.
    """

    def __init__(self, spec: TransitionSystem, impl: TransitionSystem, **kwargs):
        super().__init__(spec, impl, **kwargs)
        self.logger.info("Initialized refinement prover (spec -> impl)")

    def verify_refinement(
        self, input_vars: List[str], output_vars: List[str]
    ) -> VerificationResult:
        """Verify refinement property between spec and impl."""
        return self.verify_functional_equivalence(
            input_vars, output_vars, inputs_equal_all_steps=True
        )
