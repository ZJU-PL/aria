"""
Basic example showing how extract a finite model on a finite sub-universe of the foreground sort given a satisfying
smt model.
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
from aria.quant.fossil.naturalproofs.prover import NPSolver
import aria.quant.fossil.naturalproofs.proveroptions as proveroptions
from aria.quant.fossil.naturalproofs.extensions.finitemodel import (
    extract_finite_model,
    add_fg_element_offset,
    get_fg_elements,
)

# Declarations
x, y, nil = Consts("x y nil", fgsort)
nxt = Function("nxt", fgsort, fgsort)
lst = RecFunction("lst", fgsort, boolsort)
ls = RecFunction("ls", fgsort, fgsort, boolsort)
AddRecDefinition(lst, x, If(x == nil, True, lst(nxt(x))))
AddRecDefinition(ls, (x, y), If(x == nil, True, ls(nxt(x), y)))
# Problem parameters
goal = Implies(ls(x, nil), lst(x))
lemma1_params = (x, y)
lemma1_body = Implies(And(ls(x, y), lst(y)), lst(x))
lemmas = {(lemma1_params, lemma1_body)}
# Call a natural proofs solver
npsolver = NPSolver()
# Configure the solver
npsolver.options.instantiation_mode = proveroptions.manual_instantiation
npsolver.options.terms_to_instantiate = {x, y, nil}


def extract_model():
    npsolution = npsolver.solve(goal)
    smtmodel = npsolution.model
    terms = npsolution.fg_terms
    finite_model = extract_finite_model(smtmodel, terms)
    return finite_model


# Uncomment the statements below to see the extracted model and some other functions that can be performed on it.
# finite_model = extract_model()
# print(finite_model)
# fg_universe = get_fg_elements(finite_model)
# print(fg_universe)
# transformed_finite_model = add_fg_element_offset(finite_model, 5)
# print(transformed_finite_model)
# new_fg_universe = get_fg_elements(transformed_finite_model)
# print(new_fg_universe)


class LsListModelExtractionTest(unittest.TestCase):
    def test_extract_model(self):
        try:
            # without_lemma does not raise any exceptions
            finite_model = extract_model()
        except Exception:
            self.fail("Finite model extraction failed.")

    def test_transform_model(self):
        finite_model = extract_model()
        fg_universe = get_fg_elements(finite_model)
        transformed_finite_model = add_fg_element_offset(finite_model, 5)
        new_fg_universe = get_fg_elements(transformed_finite_model)
        self.assertTrue(new_fg_universe == {x + 5 for x in fg_universe})


if __name__ == "__main__":
    unittest.main()
