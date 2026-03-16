"""
Non-interference verification for k-safety properties.

This module implements verification of non-interference properties,
which ensure that low-security outputs do not depend on high-security inputs.
"""

import logging
from typing import List
import z3

from aria.efmc.utils.verification_utils import VerificationResult
from .base_prover import BaseKSafetyProver

logger = logging.getLogger(__name__)


class NonInterferenceProver(BaseKSafetyProver):
    """
    Specialized prover for non-interference verification.

    Non-interference states that low-security outputs should not depend
    on high-security inputs.
    """

    def __init__(self, sts, **kwargs):
        """
        Initialize non-interference prover.

        Args:
            sts: The transition system to verify
            **kwargs: Additional configuration options
        """
        # Non-interference requires exactly 2 traces
        super().__init__(sts, k=2, **kwargs)
        self.logger.info("Initialized non-interference prover")

    def verify_non_interference(
        self, high_vars: List[str], low_vars: List[str]
    ) -> VerificationResult:
        """
        Verify non-interference property.

        Non-interference states that low-security outputs should not depend
        on high-security inputs.

        Args:
            high_vars: List of high-security variable names
            low_vars: List of low-security variable names

        Returns:
            VerificationResult indicating the verification outcome
        """
        self.logger.info("Verifying non-interference property")

        # Create the non-interference property
        # For non-interference: if high inputs are the same, then low outputs should be the same
        trace_vars = self.create_trace_variables()

        high_set = set(high_vars)
        low_set = set(low_vars)

        high_eq_terms = []
        low_eq_terms = []
        for i, var in enumerate(self.sts.variables):
            name = str(var)
            if name in high_set:
                high_eq_terms.append(trace_vars[0][i] == trace_vars[1][i])
            if name in low_set:
                low_eq_terms.append(trace_vars[0][i] == trace_vars[1][i])

        high_equality = z3.And(*high_eq_terms) if high_eq_terms else z3.BoolVal(True)
        low_equality = z3.And(*low_eq_terms) if low_eq_terms else z3.BoolVal(True)

        non_interference_property = z3.Implies(high_equality, low_equality)

        self.set_relational_property(non_interference_property)
        return self.solve()
