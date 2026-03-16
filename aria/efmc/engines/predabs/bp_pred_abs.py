"""
Verifying Boolean Programs using "Predicate abstraction"

 Given a set of Boolean variables P and a transition system S,
 it finds the strongest inductive invariant expressible as the Boolean combination of P.
"""

from typing import List, Optional

import z3

from aria.efmc.utils import negate, is_valid, ctx_simplify, eval_predicates
from aria.efmc.sts import TransitionSystem


def strongest_consequence(fml: z3.ExprRef, predicates: List[z3.ExprRef]):
    """
    Compute the strongest necessary condition of fml that is the Boolean combination of preds
    Following CAV'06 paper "SMT Techniques for Fast Predicate Abstraction"
    """
    s = z3.Solver()
    s.add(fml)
    res = []
    while s.check() == z3.sat:
        m = s.model()
        # each proj is a sufficient (or necessary?) condition that
        # "generalizes" from the current model????
        # TODO: use prime implicate (i.e., throwing away some p in preds)
        proj = z3.And(eval_predicates(m, predicates))
        res.append(proj)
        # block the current one
        s.add(negate(proj))

    return z3.simplify(z3.Or(res))


def weakest_sufficient_condition(
    fml: z3.ExprRef, predicates: List[z3.ExprRef]
) -> z3.ExprRef:
    """Compute WSC using the duality between SNC(Strongest Necessary Condition)."""
    notfml = z3.Not(fml)
    sc = strongest_consequence(notfml, predicates)
    return z3.simplify(z3.Not(sc))


def fixpoint(old_inv: z3.ExprRef, inv: z3.ExprRef) -> bool:
    """Check if inv implies old_inv (fixpoint reached)."""
    return is_valid(z3.Implies(inv, old_inv))


class PredicateAbstractionProver:
    """Predicate abstraction prover for Boolean programs."""

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

    def set_predicates(self, predicates):
        """The element in the domain is the Boolean combinations of a set of predicates"""
        self.preds = predicates

    def solve(self, timeout: Optional[int] = None):  # pylint: disable=unused-argument
        """Solve the verification problem using predicate abstraction."""
        preds_prime = []
        for pred in self.preds:
            preds_prime.append(z3.substitute(pred, self.var_map))

        old_inv = False
        inv = self.sts.init
        i = 0
        while not fixpoint(old_inv, inv):
            print("\nInv at ", i, ": ", inv)
            i = i + 1
            # onestep = stronget_consequence_simple(And(inv, trans), preds_prime) # another imple
            onestep = strongest_consequence(z3.And(inv, self.sts.trans), preds_prime)
            onestep = z3.substitute(onestep, self.var_map_rev)
            old_inv = inv  # Is this correct?
            inv = z3.simplify(z3.Or(inv, onestep))
        # print("\n")

        if is_valid(z3.Implies(inv, self.sts.post)):
            print(ctx_simplify(inv))
            print(">>> SAFE\n\n")
        else:
            print(">>> MAYBE?!?!\n\n")
