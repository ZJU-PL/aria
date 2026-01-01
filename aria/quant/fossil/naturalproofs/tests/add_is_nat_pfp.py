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
a, b, c, zero = Consts("a b c zero", fgsort)
suc = Function("suc", fgsort, fgsort)
pred = Function("pred", fgsort, fgsort)
AddAxiom(a, pred(suc(a)) == a)
nat = RecFunction("nat", fgsort, boolsort)
add = RecFunction("add", fgsort, fgsort, fgsort, boolsort)
AddRecDefinition(nat, a, If(a == zero, True, nat(pred(a))))
AddRecDefinition(add, (a, b, c), If(a == zero, b == c, add(pred(a), suc(b), c)))
# Problem parameters
goal = Implies(add(a, b, c), Implies(And(nat(a), nat(b)), nat(c)))
pfp_of_goal = make_pfp_formula(goal)
# Call a natural proofs solver
npsolver = NPSolver()
# Configure the solver
npsolver.options.instantiation_mode = proveroptions.manual_instantiation
npsolver.options.terms_to_instantiate = {a, b, c, zero, pred(b), suc(b)}
# Ask for proof
npsolution = npsolver.solve(pfp_of_goal)


class AddIsNatTest(unittest.TestCase):
    def test1(self):
        self.assertFalse(npsolution.if_sat)


if __name__ == "__main__":
    unittest.main()
