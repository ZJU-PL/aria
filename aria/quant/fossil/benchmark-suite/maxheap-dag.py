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
tree = RecFunction("tree", fgsort, boolsort)
dag = RecFunction("dag", fgsort, boolsort)
maxheap = RecFunction("maxheap", fgsort, boolsort)
htree = RecFunction("htree", fgsort, fgsetsort)
AddRecDefinition(
    dag, x, If(x == nil, True, And(notInChildren(x), And(dag(lft(x)), dag(rght(x)))))
)
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
pgm = If(x == nil, ret == nil, ret == lft(x))
goal = Implies(maxheap(x), Implies(pgm, dag(ret)))

# check validity with natural proof solver and no hardcoded lemmas
np_solver = NPSolver()
solution = np_solver.solve(make_pfp_formula(goal))
if not solution.if_sat:
    print("goal (no lemmas) is valid")
else:
    print("goal (no lemmas) is invalid")

# hardcoded lemmas
lemma_params = (x,)
lemma_body = Implies(maxheap(x), dag(x))
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
v = Var("v", fgsort)
lemma_grammar_args = [v, nil]
lemma_grammar_terms = {v, nil, rght(v), lft(v)}

name = "maxheap-dag"
grammar_string = importlib_resources.read_text("grammars", "grammar_{}.sy".format(name))

solveProblem(lemma_grammar_args, lemma_grammar_terms, goal, name, grammar_string)
