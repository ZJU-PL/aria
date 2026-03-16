"""
Program equivalence/refinement across two transition systems via relational BMC.

We support functional equivalence: for the same designated inputs, outputs match.
This compares two TransitionSystem instances potentially with different transitions.
"""

import logging
from typing import Dict, List, Optional
import z3

from aria.efmc.sts import TransitionSystem
from aria.efmc.utils.verification_utils import VerificationResult
from .base_prover import BaseKSafetyProver

logger = logging.getLogger(__name__)


class EquivalenceProver(BaseKSafetyProver):
    """
    Verify functional equivalence/refinement between two systems using 2-safety.
    """

    def __init__(self, sts_a: TransitionSystem, sts_b: TransitionSystem, **kwargs):
        # Initialize base with one system to reuse options; store both
        super().__init__(sts_a, k=2, **kwargs)
        self.sts_a = sts_a
        self.sts_b = sts_b
        self.logger.info("Initialized equivalence prover for two systems")

    def _make_step_variables_for(
        self, sts: TransitionSystem, trace_idx: int, bound: int
    ):
        step_vars: Dict[int, List[z3.ExprRef]] = {}
        for step in range(bound + 1):
            vars_at_step: List[z3.ExprRef] = []
            for var in sts.variables:
                name = str(var)
                sort = var.sort()
                if z3.is_bv_sort(sort):
                    new_var = z3.BitVec(f"{name}_{trace_idx}_{step}", sort.size())
                elif sort == z3.RealSort():
                    new_var = z3.Real(f"{name}_{trace_idx}_{step}")
                elif sort == z3.IntSort():
                    new_var = z3.Int(f"{name}_{trace_idx}_{step}")
                else:
                    new_var = z3.Bool(f"{name}_{trace_idx}_{step}")
                vars_at_step.append(new_var)
            step_vars[step] = vars_at_step
        return step_vars

    def _find_index(self, sts: TransitionSystem, var_name: str) -> Optional[int]:
        for i, v in enumerate(sts.variables):
            if str(v) == var_name:
                return i
        return None

    def _bmc_two_systems(
        self,
        bound: int,
        input_vars: List[str],
        output_vars: List[str],
        inputs_equal_all_steps: bool,
    ) -> VerificationResult:
        sv_a = self._make_step_variables_for(self.sts_a, 0, bound)
        sv_b = self._make_step_variables_for(self.sts_b, 1, bound)

        conditions = []

        # inits
        init_subst_a = [
            (self.sts_a.variables[i], sv_a[0][i])
            for i, _ in enumerate(self.sts_a.variables)
        ]
        init_subst_b = [
            (self.sts_b.variables[i], sv_b[0][i])
            for i, _ in enumerate(self.sts_b.variables)
        ]
        conditions.append(z3.substitute(self.sts_a.init, init_subst_a))
        conditions.append(z3.substitute(self.sts_b.init, init_subst_b))

        # invariants if any (for steps 0..bound)
        if getattr(self.sts_a, "invariants", None):
            for step in range(bound + 1):
                inv_subst = [
                    (self.sts_a.variables[i], sv_a[step][i])
                    for i, _ in enumerate(self.sts_a.variables)
                ]
                for inv in self.sts_a.invariants:
                    conditions.append(z3.substitute(inv, inv_subst))
        if getattr(self.sts_b, "invariants", None):
            for step in range(bound + 1):
                inv_subst = [
                    (self.sts_b.variables[i], sv_b[step][i])
                    for i, _ in enumerate(self.sts_b.variables)
                ]
                for inv in self.sts_b.invariants:
                    conditions.append(z3.substitute(inv, inv_subst))

        # transitions 0..bound-1
        for step in range(bound):
            trans_subst_a = []
            for i, _ in enumerate(self.sts_a.variables):
                trans_subst_a.append((self.sts_a.variables[i], sv_a[step][i]))
                trans_subst_a.append((self.sts_a.prime_variables[i], sv_a[step + 1][i]))
            conditions.append(z3.substitute(self.sts_a.trans, trans_subst_a))

            trans_subst_b = []
            for i, _ in enumerate(self.sts_b.variables):
                trans_subst_b.append((self.sts_b.variables[i], sv_b[step][i]))
                trans_subst_b.append((self.sts_b.prime_variables[i], sv_b[step + 1][i]))
            conditions.append(z3.substitute(self.sts_b.trans, trans_subst_b))

        antecedent = z3.And(*conditions) if conditions else z3.BoolVal(True)

        # inputs equal
        input_terms: List[z3.ExprRef] = []
        steps_to_check = range(bound + 1) if inputs_equal_all_steps else [0]
        for name in input_vars:
            idx_a = self._find_index(self.sts_a, name)
            idx_b = self._find_index(self.sts_b, name)
            if idx_a is None or idx_b is None:
                raise ValueError(f"Input var '{name}' not found in both systems")
            for step in steps_to_check:
                input_terms.append(sv_a[step][idx_a] == sv_b[step][idx_b])
        inputs_equal = z3.And(*input_terms) if input_terms else z3.BoolVal(True)

        # outputs equal at final step
        output_terms: List[z3.ExprRef] = []
        for name in output_vars:
            idx_a = self._find_index(self.sts_a, name)
            idx_b = self._find_index(self.sts_b, name)
            if idx_a is None or idx_b is None:
                raise ValueError(f"Output var '{name}' not found in both systems")
            output_terms.append(sv_a[bound][idx_a] == sv_b[bound][idx_b])
        outputs_equal = z3.And(*output_terms) if output_terms else z3.BoolVal(True)

        formula = z3.Implies(z3.And(antecedent, inputs_equal), outputs_equal)

        solver = z3.Solver()
        solver.add(z3.Not(formula))
        solver.set("timeout", self.timeout * 1000)
        res = solver.check()
        if res == z3.sat:
            return VerificationResult(False, None, solver.model(), is_unsafe=True)
        if res == z3.unsat:
            return VerificationResult(True, None)
        return VerificationResult(False, None, None, is_unknown=True, timed_out=True)

    def verify_functional_equivalence(
        self,
        input_vars: List[str],
        output_vars: List[str],
        inputs_equal_all_steps: bool = True,
    ) -> VerificationResult:
        """Verify functional equivalence between two systems."""
        self.logger.info("Verifying functional equivalence between two systems")
        if self.verification_method == "bounded_model_checking":
            for b in range(1, self.max_bound + 1):
                result = self._bmc_two_systems(
                    b, input_vars, output_vars, inputs_equal_all_steps
                )
                if result.is_safe or result.is_unsafe:
                    return result
            return VerificationResult(False, None, None, is_unknown=True)
        # Only BMC supported for two-system equivalence in this version
        return VerificationResult(False, None, None, is_unknown=True)
