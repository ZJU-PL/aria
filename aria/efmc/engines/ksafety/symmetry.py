"""
Symmetry verification for k-safety properties.

This prover checks input permutation symmetry: under a permutation of designated
input variables in one trace, outputs are permuted accordingly and remain equal
under the permutation mapping. This captures relational symmetry properties.
"""

import logging
from typing import Dict, List
import z3

from aria.efmc.utils.verification_utils import VerificationResult
from .base_prover import BaseKSafetyProver

logger = logging.getLogger(__name__)


class SymmetryProver(BaseKSafetyProver):
    """
    Prover for symmetry properties via 2-safety.

    We support verifying that under a given permutation of input variables,
    the outputs are permuted accordingly.
    """

    def __init__(self, sts, **kwargs):
        super().__init__(sts, k=2, **kwargs)
        self.logger.info("Initialized symmetry prover")

    def verify_input_permutation_symmetry(
        self, input_perm: Dict[str, str], output_perm: Dict[str, str]
    ) -> VerificationResult:
        """
        Check that applying input permutation on trace 1 yields output
        permutation equivalence on trace 2.

        Args:
            input_perm: mapping from input var name to its permuted name
            output_perm: mapping from output var name to its permuted name
        """
        trace_vars = self.create_trace_variables()

        # For variables in input_perm, equate v_0 with perm(v)_1
        input_eq_terms: List[z3.ExprRef] = []
        for i, var in enumerate(self.sts.variables):
            name = str(var)
            if name in input_perm:
                # find index of permuted variable
                perm_name = input_perm[name]
                try:
                    j = next(
                        idx
                        for idx, v in enumerate(self.sts.variables)
                        if str(v) == perm_name
                    )
                except StopIteration as exc:
                    raise ValueError(
                        f"Permutation target '{perm_name}' not found among variables"
                    ) from exc
                input_eq_terms.append(trace_vars[0][i] == trace_vars[1][j])

        # For outputs, require v_0 equals perm(v)_1
        output_eq_terms: List[z3.ExprRef] = []
        for i, var in enumerate(self.sts.variables):
            name = str(var)
            if name in output_perm:
                perm_name = output_perm[name]
                try:
                    j = next(
                        idx
                        for idx, v in enumerate(self.sts.variables)
                        if str(v) == perm_name
                    )
                except StopIteration as exc:
                    raise ValueError(
                        f"Permutation target '{perm_name}' not found among variables"
                    ) from exc
                output_eq_terms.append(trace_vars[0][i] == trace_vars[1][j])

        antecedent = z3.And(*input_eq_terms) if input_eq_terms else z3.BoolVal(True)
        consequent = z3.And(*output_eq_terms) if output_eq_terms else z3.BoolVal(True)
        self.set_relational_property(z3.Implies(antecedent, consequent))
        return self.solve()
