"""
Basic example showing usage of binary functions and pre-fixpoint formula generation for doing inductive proofs.
"""

# Only importing this for writing this file as a test
import unittest

from z3 import And, Or, Not, Implies, If
from z3 import IsSubset, Union, SetIntersect, SetComplement, EmptySet

from aria.quant.fossil.naturalproofs.uct import (
    fgsort,
    fgsetsort,
    intsort,
    intsetsort,
    boolsort,
)
from aria.quant.fossil.naturalproofs.decl_api import (
    Const,
    Consts,
    Function,
    RecFunction,
    AddRecDefinition,
    AddAxiom,
)
from aria.quant.fossil.naturalproofs.pfp import make_pfp_formula
from aria.quant.fossil.naturalproofs.prover import NPSolver
import aria.quant.fossil.naturalproofs.proveroptions as proveroptions

# Declarations
a, zero = Consts("a zero", fgsort)
suc = Function("suc", fgsort, fgsort)
pred = Function("pred", fgsort, fgsort)
AddAxiom(a, pred(suc(a)) == a)
nat = RecFunction("nat", fgsort, boolsort)
even = RecFunction("even", fgsort, boolsort)
odd = RecFunction("odd", fgsort, boolsort)
AddRecDefinition(nat, a, If(a == zero, True, nat(pred(a))))
AddRecDefinition(even, a, If(a == zero, True, odd(pred(a))))
AddRecDefinition(odd, a, If(a == zero, False, If(a == suc(zero), True, even(pred(a)))))
# Problem parameters
goal = Implies(nat(a), Or(even(a), odd(a)))
pfp_of_goal = make_pfp_formula(goal)
# Call a natural proofs solver
npsolver = NPSolver()
# Ask for proof
npsolution = npsolver.solve(pfp_of_goal)


class EvenOddTest(unittest.TestCase):
    def test1(self):
        self.assertFalse(npsolution.if_sat)


if __name__ == "__main__":
    unittest.main()
