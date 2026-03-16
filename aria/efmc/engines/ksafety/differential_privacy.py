"""
Approximate differential privacy-style relational property (deterministic skeleton).

We encode an epsilon-Lipschitz-like constraint for outputs with respect to inputs:
If two inputs are adjacent (within a given metric threshold), then outputs are within epsilon.

Note: True differential privacy is probabilistic. This module encodes a deterministic
analogue appropriate for numeric systems, suitable as a starting point.
"""

import logging
from typing import Dict, List
import z3

from aria.efmc.utils.verification_utils import VerificationResult
from .base_prover import BaseKSafetyProver

logger = logging.getLogger(__name__)


def _abs(expr: z3.ArithRef) -> z3.ArithRef:
    return z3.If(expr >= 0, expr, -expr)


class DifferentialPrivacyProver(BaseKSafetyProver):
    """
    Prover for epsilon-sensitivity (deterministic DP analogue) via 2-safety.
    """

    def __init__(self, sts, **kwargs):
        super().__init__(sts, k=2, **kwargs)
        self.logger.info("Initialized differential privacy prover")

    def verify_epsilon_sensitivity(
        self,
        input_vars: List[str],
        output_vars: List[str],
        adjacency_bounds: Dict[str, float],
        epsilon: float,
    ) -> VerificationResult:
        """
        Check that small changes in inputs lead to bounded changes in outputs.

        Args:
            input_vars: names of input variables considered for adjacency
            output_vars: names of output variables to be bounded
            adjacency_bounds: per-input maximum absolute difference allowed between traces
            epsilon: bound on maximum absolute difference for each output
        """
        trace_vars = self.create_trace_variables()

        # Build adjacency antecedent over inputs
        antecedent_terms: List[z3.ExprRef] = []
        for i, var in enumerate(self.sts.variables):
            name = str(var)
            if name in input_vars:
                # Only numeric sorts supported here
                v0 = trace_vars[0][i]
                v1 = trace_vars[1][i]
                bound = adjacency_bounds.get(name, 0.0)
                if var.sort() == z3.IntSort():
                    antecedent_terms.append(
                        _abs(z3.ToReal(v0 - v1)) <= z3.RealVal(bound)
                    )
                elif var.sort() == z3.RealSort():
                    antecedent_terms.append(_abs(v0 - v1) <= z3.RealVal(bound))
                else:
                    raise NotImplementedError("DP prover supports int/real inputs only")

        antecedent = z3.And(*antecedent_terms) if antecedent_terms else z3.BoolVal(True)

        # Build epsilon-bound consequent over outputs
        consequent_terms: List[z3.ExprRef] = []
        for i, var in enumerate(self.sts.variables):
            name = str(var)
            if name in output_vars:
                v0 = trace_vars[0][i]
                v1 = trace_vars[1][i]
                if var.sort() == z3.IntSort():
                    consequent_terms.append(
                        _abs(z3.ToReal(v0 - v1)) <= z3.RealVal(epsilon)
                    )
                elif var.sort() == z3.RealSort():
                    consequent_terms.append(_abs(v0 - v1) <= z3.RealVal(epsilon))
                else:
                    raise NotImplementedError(
                        "DP prover supports int/real outputs only"
                    )

        consequent = z3.And(*consequent_terms) if consequent_terms else z3.BoolVal(True)

        self.set_relational_property(z3.Implies(antecedent, consequent))
        return self.solve()
