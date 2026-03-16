"""
Validators for synthesized ranking functions and recurrence sets.
"""

import logging
import z3

from aria.efmc.sts import TransitionSystem
from aria.efmc.utils.bv_utils import Signedness

logger = logging.getLogger(__name__)


class RankingFunctionValidator:
    """Validates synthesized ranking functions."""

    def __init__(self, sts: TransitionSystem):
        self.sts = sts

    def validate_ranking_function(
        self, ranking_function: z3.ExprRef, signedness: Signedness
    ) -> bool:
        """
        Validate that the synthesized ranking function is correct.

        Returns:
            True if the ranking function is valid, False otherwise
        """
        if ranking_function is None:
            return False

        # Create solver for validation
        solver = z3.Solver()

        # Check non-negativity
        if signedness == Signedness.SIGNED:
            # For signed bit-vectors, check that R(x) >= 0 when guard holds
            if self.sts.trans:
                non_neg_check = z3.Implies(self.sts.trans, ranking_function >= 0)
            else:
                non_neg_check = ranking_function >= 0

            solver.push()
            solver.add(z3.Not(non_neg_check))
            if solver.check() == z3.sat:
                logger.warning("Ranking function non-negativity validation failed")
                solver.pop()
                return False
            solver.pop()

        # Check decrease property
        rank_next = z3.substitute(
            ranking_function, list(zip(self.sts.variables, self.sts.prime_variables))
        )

        if self.sts.trans:
            if signedness == Signedness.UNSIGNED:
                decrease_check = z3.Implies(
                    self.sts.trans, z3.UGT(ranking_function, rank_next)
                )
            else:
                decrease_check = z3.Implies(
                    self.sts.trans, ranking_function > rank_next
                )
        else:
            if signedness == Signedness.UNSIGNED:
                decrease_check = z3.UGT(ranking_function, rank_next)
            else:
                decrease_check = ranking_function > rank_next

        solver.push()
        solver.add(z3.Not(decrease_check))
        if solver.check() == z3.sat:
            logger.warning("Ranking function decrease validation failed")
            solver.pop()
            return False
        solver.pop()

        return True


class RecurrenceSetValidator:
    """Validates synthesized recurrence sets."""

    def __init__(self, sts: TransitionSystem):
        self.sts = sts

    def validate_recurrence_set(self, recurrence_set: z3.ExprRef) -> bool:
        """
        Validate that the synthesized recurrence set is correct.

        Returns:
            True if the recurrence set is valid, False otherwise
        """
        if recurrence_set is None:
            return False

        # Create solver for validation
        solver = z3.Solver()

        # Check reachability: ∃x. init(x) ∧ S(x)
        if self.sts.init is not None:
            reachability_check = z3.And(self.sts.init, recurrence_set)
            solver.push()
            solver.add(reachability_check)
            if solver.check() != z3.sat:
                logger.warning("Recurrence set reachability validation failed")
                solver.pop()
                return False
            solver.pop()

        # Check closure: ∀x,x'. S(x) ∧ guard(x) ∧ trans(x,x') ⇒ S(x')
        recur_next = z3.substitute(
            recurrence_set, list(zip(self.sts.variables, self.sts.prime_variables))
        )

        if self.sts.trans:
            closure_check = z3.Implies(
                z3.And(recurrence_set, self.sts.trans), recur_next
            )
        else:
            closure_check = z3.Implies(recurrence_set, recur_next)

        solver.push()
        solver.add(z3.Not(closure_check))
        if solver.check() == z3.sat:
            logger.warning("Recurrence set closure validation failed")
            solver.pop()
            return False
        solver.pop()

        # Check guard satisfaction: ∀x. S(x) ⇒ guard(x)
        if self.sts.trans:
            guard_check = z3.Implies(recurrence_set, self.sts.trans)
            solver.push()
            solver.add(z3.Not(guard_check))
            if solver.check() == z3.sat:
                logger.warning("Recurrence set guard satisfaction validation failed")
                solver.pop()
                return False
            solver.pop()

        return True
