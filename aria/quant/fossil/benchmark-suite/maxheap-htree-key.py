import importlib_resources

import z3
from z3 import And, Or, Not, Implies, If
from z3 import (
    IsMember,
    IsSubset,
    SetUnion,
    SetIntersect,
    SetComplement,
    EmptySet,
    SetAdd,
)

from aria.quant.fossil.naturalproofs.prover import NPSolver
import aria.quant.fossil.naturalproofs.proveroptions as proveroptions
from aria.quant.fossil.naturalproofs.uct import (
    fgsort,
    fgsetsort,
    intsort,
    intsetsort,
    boolsort,
    min_intsort,
    max_intsort,
)
from aria.quant.fossil.naturalproofs.decl_api import (
    Const,
    Consts,
    Var,
    Vars,
    Function,
    RecFunction,
    AddRecDefinition,
    AddAxiom,
)
from aria.quant.fossil.naturalproofs.pfp import make_pfp_formula

from aria.quant.fossil.lemsynth.lemsynth_engine import solveProblem


def notInChildren(x):
    return And(Not(IsMember(x, htree(lft(x)))), Not(IsMember(x, htree(rght(x)))))


# declarations
x, y = Vars("x y", fgsort)
nil, ret = Consts("nil ret", fgsort)
k = Const("k", intsort)
key = Function("key", fgsort, intsort)
lft = Function("lft", fgsort, fgsort)
rght = Function("rght", fgsort, fgsort)
maxheap = RecFunction("maxheap", fgsort, boolsort)
htree = RecFunction("htree", fgsort, fgsetsort)
AddRecDefinition(
    maxheap,
    x,
    If(
        x == nil,
        True,
        And(
            notInChildren(x),
            And(
                SetIntersect(htree(lft(x)), htree(rght(x))) == fgsetsort.lattice_bottom,
                And(
                    maxheap(lft(x)),
                    And(
                        maxheap(rght(x)),
                        And(key(lft(x)) <= key(x), key(rght(x)) <= key(x)),
                    ),
                ),
            ),
        ),
    ),
)
AddRecDefinition(
    htree,
    x,
    If(
        x == nil,
        fgsetsort.lattice_bottom,
        SetAdd(SetUnion(htree(lft(x)), htree(rght(x))), x),
    ),
)
AddAxiom((), lft(nil) == nil)
AddAxiom((), rght(nil) == nil)

# vc
goal = Implies(
    maxheap(x), Implies(key(x) != k, Implies(IsMember(y, htree(x)), key(y) <= key(x)))
)

# check validity with natural proof solver and no hardcoded lemmas
np_solver = NPSolver()
solution = np_solver.solve(make_pfp_formula(goal))
if not solution.if_sat:
    print("goal (no lemmas) is valid")
else:
    print("goal (no lemmas) is invalid")

# hardcoded lemmas
lemma_params = (x, y)
lemma_body = Implies(maxheap(x), Implies(IsMember(y, htree(x)), key(y) <= key(x)))
lemmas = {(lemma_params, lemma_body)}

# check validity of lemmas
solution = np_solver.solve(make_pfp_formula(lemma_body))
if not solution.if_sat:
    print("lemma is valid")
else:
    print("lemma is invalid")

# check validity with natural proof solver and hardcoded lemmas
solution = np_solver.solve(goal, lemmas)
if not solution.if_sat:
    print("goal (with lemmas) is valid")
else:
    print("goal (with lemmas) is invalid")

# lemma synthesis
v1, v2 = Vars("v1 v2", fgsort)
lemma_grammar_args = [v1, v2, k, nil]
lemma_grammar_terms = {
    v1,
    v2,
    lft(v1),
    lft(v2),
    rght(lft(v1)),
    rght(rght(v1)),
    rght(v2),
}

name = "maxheap-htree-key"
# name = 'maxheap-htree-key-lvl0'
grammar_string = importlib_resources.read_text("grammars", "grammar_{}.sy".format(name))

solveProblem(lemma_grammar_args, lemma_grammar_terms, goal, name, grammar_string)
