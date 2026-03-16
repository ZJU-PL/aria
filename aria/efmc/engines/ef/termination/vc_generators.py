"""
Verification condition generators for termination verification.
"""

import logging
from typing import Tuple, List
import z3

from aria.efmc.sts import TransitionSystem
from aria.efmc.utils.bv_utils import Signedness
from aria.efmc.utils.z3_expr_utils import big_and

logger = logging.getLogger(__name__)


class RankingVCGenerator:
    """Generates verification conditions for ranking function synthesis."""

    def __init__(self, sts: TransitionSystem):
        self.sts = sts

    def generate_vc(self, ranking_template) -> z3.ExprRef:
        """Generate verification condition for ranking function synthesis."""
        constraints = []

        # Build ranking function expressions
        rank_current, rank_next = self._build_expressions(ranking_template)

        # Extract guard from transition relation
        guard = self.sts.trans if self.sts.trans is not None else z3.BoolVal(True)

        # Non-negativity constraint
        if ranking_template.signedness != Signedness.UNSIGNED:
            constraints.append(z3.Implies(guard, rank_current >= 0))

        # Decrease constraint
        if ranking_template.signedness == Signedness.UNSIGNED:
            decrease_constraint = z3.Implies(guard, z3.UGT(rank_current, rank_next))
        else:
            decrease_constraint = z3.Implies(guard, rank_current > rank_next)
        constraints.append(decrease_constraint)

        # Initial state constraint
        if (
            self.sts.init is not None
            and ranking_template.signedness != Signedness.UNSIGNED
        ):
            constraints.append(z3.Implies(self.sts.init, rank_current >= 0))

        # Combine constraints
        vc = big_and(constraints)

        # Add template-specific constraints
        for attr in ["template_cnt_init_and_post", "template_cnt_trans"]:
            if hasattr(ranking_template, attr):
                vc = z3.And(vc, getattr(ranking_template, attr))

        return vc

    def _build_expressions(self, template) -> Tuple[z3.ExprRef, z3.ExprRef]:
        """Build ranking function expressions for current and next states."""
        # Try different methods to build expressions
        for method_name in ["_build_ranking_expr", "build_invariant_expr"]:
            if hasattr(template, method_name):
                method = getattr(template, method_name)
                if method_name == "build_invariant_expr":
                    dummy_model = self._create_dummy_model()
                    if dummy_model:
                        return (
                            method(dummy_model, use_prime_variables=False),
                            method(dummy_model, use_prime_variables=True),
                        )
                else:
                    return method(self.sts.variables), method(self.sts.prime_variables)

        # Fallback
        return z3.BitVecVal(0, 32), z3.BitVecVal(0, 32)

    def _create_dummy_model(self):
        """Create a dummy model for testing purposes."""
        solver = z3.Solver()
        solver.add(z3.BoolVal(True))
        return solver.model() if solver.check() == z3.sat else None


class RecurrenceVCGenerator:
    """Generates verification conditions for recurrence set synthesis."""

    def __init__(self, sts: TransitionSystem):
        self.sts = sts

    def generate_vc(self, recurrence_template) -> z3.ExprRef:
        """Generate verification condition for recurrence set synthesis."""
        constraints = []

        # Build recurrence set expressions
        recur_current = self._build_expression(recurrence_template, self.sts.variables)
        recur_next = self._build_expression(
            recurrence_template, self.sts.prime_variables
        )

        guard = self.sts.trans if self.sts.trans is not None else z3.BoolVal(True)

        # Reachability constraint
        if self.sts.init is not None:
            constraints.append(z3.And(self.sts.init, recur_current))
        else:
            constraints.append(recur_current)

        # Closure constraint
        constraints.append(z3.Implies(z3.And(recur_current, guard), recur_next))

        # Guard satisfaction constraint
        constraints.append(z3.Implies(recur_current, guard))

        # Combine constraints
        vc = big_and(constraints)

        # Add template-specific constraints
        for attr in ["template_cnt_init_and_post", "template_cnt_trans"]:
            if hasattr(recurrence_template, attr):
                vc = z3.And(vc, getattr(recurrence_template, attr))

        return vc

    def _build_expression(self, template, variables: List[z3.ExprRef]) -> z3.ExprRef:
        """Build recurrence set expression for given variables."""
        if hasattr(template, "_build_recurrence_expr"):
            # pylint: disable=protected-access
            return template._build_recurrence_expr(variables)
        if hasattr(template, "build_invariant_expr"):
            dummy_model = self._create_dummy_model()
            if dummy_model:
                use_prime = variables == self.sts.prime_variables
                return template.build_invariant_expr(
                    dummy_model, use_prime_variables=use_prime
                )
        return z3.BoolVal(True)

    def _create_dummy_model(self):
        """Create a dummy model for testing purposes."""
        solver = z3.Solver()
        solver.add(z3.BoolVal(True))
        return solver.model() if solver.check() == z3.sat else None
