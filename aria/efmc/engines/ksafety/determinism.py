"""
Determinism verification for k-safety properties.

This module implements verification of determinism properties,
which ensure that for the same inputs, the system always produces the same outputs.
"""

import logging
import z3

from aria.efmc.utils.verification_utils import VerificationResult
from .base_prover import BaseKSafetyProver

logger = logging.getLogger(__name__)


class DeterminismProver(BaseKSafetyProver):
    """
    Specialized prover for determinism verification.

    Determinism states that for the same inputs, the system should
    always produce the same outputs.
    """

    def __init__(self, sts, **kwargs):
        """
        Initialize determinism prover.

        Args:
            sts: The transition system to verify
            **kwargs: Additional configuration options
        """
        # Determinism requires exactly 2 traces
        super().__init__(sts, k=2, **kwargs)
        self.logger.info("Initialized determinism prover")

    def verify_determinism(
        self, input_vars=None, output_vars=None
    ) -> VerificationResult:
        """
        Verify determinism property.

        Determinism states that for the same inputs, the system should
        always produce the same outputs.

        Returns:
            VerificationResult indicating the verification outcome
        """
        self.logger.info("Verifying determinism property")

        # Create the determinism property
        # For determinism: if inputs are the same, then outputs should be the same
        trace_vars = self.create_trace_variables()

        # If input/output sets are provided, restrict equality to those
        if input_vars is not None:
            input_set = set(input_vars)
        else:
            input_set = {str(v) for v in self.sts.variables}

        if output_vars is not None:
            output_set = set(output_vars)
        else:
            output_set = {str(v) for v in self.sts.variables}

        input_eq_terms = []
        output_eq_terms = []
        for i, var in enumerate(self.sts.variables):
            if str(var) in input_set:
                input_eq_terms.append(trace_vars[0][i] == trace_vars[1][i])
            if str(var) in output_set:
                output_eq_terms.append(trace_vars[0][i] == trace_vars[1][i])

        input_equality = z3.And(*input_eq_terms) if input_eq_terms else z3.BoolVal(True)
        output_equality = (
            z3.And(*output_eq_terms) if output_eq_terms else z3.BoolVal(True)
        )

        determinism_property = z3.Implies(input_equality, output_equality)

        self.set_relational_property(determinism_property)
        return self.solve()
