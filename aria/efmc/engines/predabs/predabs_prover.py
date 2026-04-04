"""
This is an "old-school" version of the predicate abstraction domain.

- A set of predicates is given prior. The element in the domain is the
  Boolean combination of those predicates
- Compute the strongest post operation(similar to symbolic abstraction)
- Compute inductive invariant expressed in the element of the domain
Currently we do not explore
- Lazy abstraction, lazy abstraction with interpolation, etc...

TODO: use more "advanced" CEGAR engines
"""

import logging
from itertools import chain
from itertools import combinations
from typing import List, Optional

import z3

from aria.efmc.sts import TransitionSystem
from aria.efmc.utils import negate, is_sat, is_valid
from aria.utils.z3_solver_utils import fixpoint
from aria.efmc.utils.verification_utils import VerificationResult

logger = logging.getLogger(__name__)


def eval_preds(m: z3.ModelRef, predicates: List[z3.BoolRef]) -> List:
    """Let m be a model of a formula phi. preds be a set of predicates"""
    res = []
    for p in predicates:
        if z3.is_true(m.eval(p)):
            res.append(p)
        elif z3.is_false(m.eval(p)):
            res.append(negate(p))
        else:
            pass
    return res


############################
# predicate abstraction
def strongest_consequence(fml: z3.ExprRef, predicates: List) -> z3.ExprRef:
    """
    Compute the strongest necessary condition of fml that is the Boolean combination of preds
    Following CAV'06 paper "SMT Techniques for Fast Predicate Abstraction"
    """
    s = z3.Solver()
    s.add(fml)
    res = []
    while s.check() == z3.sat:
        m = s.model()
        # "generalizes" from the current model????
        # TODO: use prime implicate (i.e., throwing away some p in preds)
        proj = z3.And(eval_preds(m, predicates))
        res.append(proj)
        # block the current one
        s.add(negate(proj))

    return z3.simplify(z3.Or(res))


def weakest_sufficient_condition(
    fml: z3.ExprRef, predicates: List[z3.ExprRef]
) -> z3.ExprRef:
    """Compute WSC using the duality between SNC(Strongest Necessary Condition)"""
    sc = strongest_consequence(negate(fml), predicates)
    return z3.simplify(z3.Not(sc))


def stronget_consequence_simple(
    phi: z3.ExprRef, predicates: List[z3.ExprRef]
) -> z3.ExprRef:
    """Is this correct?"""
    res = z3.And(False)
    neg_preds = [z3.Not(xx) for xx in predicates]

    for ps in combinations(
        chain(predicates, neg_preds), len(list(predicates))
    ):  # for py3?
        if is_sat(z3.And(phi, *ps)):
            res = z3.Or(res, z3.And(*ps))

    return z3.simplify(res)


class PredicateAbstractionProver:
    """Predicate abstraction prover for transition systems."""

    def __init__(self, system: TransitionSystem):
        self.sts = system
        self.preds = []

        """
        var_map = [(x, xp), (y, yp)]
        var_map_rev = [(xp ,x), (yp, y)]
        """
        self.var_map = []
        self.var_map_rev = []
        for i, var in enumerate(self.sts.variables):
            self.var_map.append((var, self.sts.prime_variables[i]))
            self.var_map_rev.append((self.sts.prime_variables[i], var))

    def set_predicates(self, predicates: List[z3.ExprRef]):
        """The element in the domain is the Boolean combinations of a set of predicates"""
        self.preds = predicates

    def solve(
        self, timeout: Optional[int] = None
    ) -> VerificationResult:  # pylint: disable=unused-argument
        """External interface for verifying"""
        preds_prime = []
        for pred in self.preds:
            preds_prime.append(z3.substitute(pred, self.var_map))

        old_inv = False
        inv = self.sts.init
        i = 0
        while not fixpoint(old_inv, inv):
            print("\nInv at ", i, ": ", inv)
            i = i + 1
            # onestep = strongest_consequence_simple(And(inv, trans), preds_prime) # another imple
            # Transfer function: Given abstract state Abs(X) and transition formula P(X, X'),
            # compute the new state Abs(X') that is expressive in the predicate abstraction
            # domain. To preserve precision, we use the following function to compute the
            # most precise transfer function
            onestep = strongest_consequence(z3.And(inv, self.sts.trans), preds_prime)
            onestep = z3.substitute(onestep, self.var_map_rev)
            old_inv = inv  # Is this correct?
            inv = z3.simplify(z3.Or(inv, onestep))
        # print("\n")

        # check whether the invariant is precise enough
        if is_valid(z3.Implies(inv, self.sts.post)):
            print(z3.simplify(inv))
            print(">>> SAFE\n\n")
            return VerificationResult(True, inv)
        # need refinement
        print(">>> MAYBE?!?!\n\n")
        return VerificationResult(False, None, None, is_unknown=True)
