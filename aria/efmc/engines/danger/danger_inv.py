"""
Danger invariant checker for transition systems
===============================================

This module offers a Z3-based checker that validates whether a *danger
invariant* D(x) together with a *ranking function* R(x) forms a correct
unsafety witness for a `TransitionSystem`.

Conditions (matching the second-order danger harness):

  1. Base / Reachability
        ∃x.  Init(x) ∧ D(x)

  2. Progress
        ∀x.  D(x) ∧ G(x) ∧ Post(x)
              ⇒  ∃x'.  Trans(x,x') ∧ D(x') ∧ R(x') < R(x)
        +   (optional)  ∀x.  D(x) ⇒ R(x) > 0           # positivity

  3. Exit-Violation (only when a guard G is provided)
        ∀x.  D(x) ∧ ¬G(x) ⇒ ¬Post(x)

If no guard is supplied, G is taken to be `true` and the exit clause is
omitted.

The checker returns a `VerificationResult` whose fields carry the following
meaning for *this* prover:

  is_safe      –  True   : the provided witness is *invalid* (no bug proved)
               –  False  : the witness is *valid*      (bug proved)

  is_unknown   –  True   : the solver answered `unknown`, or the witness
                           was rejected but no safety proof has been given
"""

from __future__ import annotations

import logging
from typing import Optional

import z3

from aria.efmc.sts import TransitionSystem
from aria.efmc.utils.verification_utils import VerificationResult

__all__ = ["DangerInvariantProver"]

logger = logging.getLogger(__name__)


class DangerInvariantProver:  # pylint: disable=too-few-public-methods
    """
    Validate danger invariants and ranking functions for a `TransitionSystem`.

    Usage
    -----
    >>> prover = DangerInvariantProver(ts)
    >>> res    = prover.verify(inv, rank, guard=guard)
    >>> assert res.is_safe is False            # bug witnessed
    """

    # --------------------------------------------------------------------- #
    # construction helpers
    # --------------------------------------------------------------------- #

    def __init__(self, system: TransitionSystem):
        self.sts = system

    def _prime(self, expr: z3.ExprRef) -> z3.ExprRef:
        """Return *expr* with current variables replaced by the primed ones."""
        return z3.substitute(
            expr, list(zip(self.sts.variables, self.sts.prime_variables))
        )

    def _rank_cmp(self, r_cur: z3.ExprRef, r_nxt: z3.ExprRef) -> z3.ExprRef:
        """Strict order for the ranking sort, handling signed/unsigned BVs."""
        if self.sts.has_bv:
            signed = getattr(self.sts, "signedness", "unsigned") == "signed"
            return r_cur > r_nxt if signed else z3.UGT(r_cur, r_nxt)
        return r_cur > r_nxt  # Int / Real

    # --------------------------------------------------------------------- #
    # public API
    # --------------------------------------------------------------------- #

    def verify(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        inv: z3.ExprRef,
        rank: z3.ExprRef,
        *,
        guard: Optional[z3.ExprRef] = None,
        require_rank_positive: bool = True,
        solver_timeout_ms: Optional[int] = None,
    ) -> VerificationResult:
        """
        Check that *(inv, rank [, guard])* is a valid danger witness.

        Parameters
        ----------
        inv  : z3.ExprRef
            Danger predicate  D(x)  over current variables.
        rank : z3.ExprRef
            Ranking function R(x).  Must have an ordered sort.
        guard: z3.ExprRef | None, default None
            Optional guard predicate G(x).  See module docstring.
        require_rank_positive: bool, default True
            Whether to demand   ∀x∈D. R(x) > 0.
        solver_timeout_ms: int | None
            Optional timeout passed to Z3.

        Returns
        -------
        VerificationResult
            • is_safe  = False  ⇢ witness accepted   (program UNSAFE)
            • is_safe  = True   ⇢ witness rejected   (no conclusion)
            • is_unknown = True ⇢ Z3 returned `unknown`
        """
        # -------------------------------------------------------------- #
        # sanity checks
        # -------------------------------------------------------------- #
        if not (
            self.sts.variables
            and self.sts.prime_variables
            and self.sts.init is not None
            and self.sts.trans is not None
            and self.sts.post is not None
        ):
            logger.error(
                "TransitionSystem not fully initialised "
                "(vars/trans/init/post required)"
            )
            return VerificationResult(is_safe=False, invariant=None, is_unknown=True)

        # -------------------------------------------------------------- #
        # 1. Base / Reachability
        # -------------------------------------------------------------- #
        base_formula = z3.Exists(self.sts.variables, z3.And(self.sts.init, inv))

        # -------------------------------------------------------------- #
        # 2. Progress
        # -------------------------------------------------------------- #
        inv_prime = self._prime(inv)
        rank_cur, rank_nxt = rank, self._prime(rank)
        rank_decrease = self._rank_cmp(rank_cur, rank_nxt)

        unsigned_bv = (
            self.sts.has_bv
            and getattr(self.sts, "signedness", "unsigned") == "unsigned"
        )

        rank_positive_atom = z3.BoolVal(True)
        if require_rank_positive and not unsigned_bv:
            rank_positive_atom = rank_cur > 0

        progress_guard = guard if guard is not None else z3.BoolVal(True)
        progress_ante = z3.And(inv, progress_guard, self.sts.post)

        # ∃x'. Trans ∧ D(x') ∧ R' < R
        progress_succ = z3.Exists(
            self.sts.prime_variables, z3.And(self.sts.trans, inv_prime, rank_decrease)
        )

        # Full progress clause: antecedent ⇒ (rank_positive ∧ succ)
        progress_formula = z3.ForAll(
            self.sts.variables,
            z3.Implies(progress_ante, z3.And(rank_positive_atom, progress_succ)),
        )

        # Optional global positivity:  ∀x. D(x) ⇒ R(x) > 0
        positivity_formula = z3.BoolVal(True)
        if require_rank_positive and not unsigned_bv:
            positivity_formula = z3.ForAll(
                self.sts.variables, z3.Implies(inv, rank_cur > 0)
            )

        # -------------------------------------------------------------- #
        # 3. Exit-Violation
        # -------------------------------------------------------------- #
        exit_formula = z3.BoolVal(True)
        if guard is not None:
            exit_formula = z3.ForAll(
                self.sts.variables,
                z3.Implies(z3.And(inv, z3.Not(guard)), z3.Not(self.sts.post)),
            )

        # -------------------------------------------------------------- #
        # 4. Solve
        # -------------------------------------------------------------- #
        s = z3.Solver()
        if solver_timeout_ms is not None:
            s.set(timeout=solver_timeout_ms)

        s.add(base_formula, progress_formula, positivity_formula, exit_formula)

        res = s.check()
        if res == z3.sat:
            # Witness holds ⇒ program is UNSAFE
            return VerificationResult(
                is_safe=False, invariant=inv, counterexample=s.model()
            )
        if res == z3.unknown:
            logger.warning("Danger invariant check: Z3 returned unknown")
            return VerificationResult(is_safe=False, invariant=None, is_unknown=True)

        # UNSAT: witness rejected; we cannot conclude safety
        return VerificationResult(is_safe=True, invariant=None, is_unknown=True)
