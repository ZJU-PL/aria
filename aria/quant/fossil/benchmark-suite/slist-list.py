import importlib_resources

import z3
from z3 import And, Or, Not, Implies, If
from z3 import IsSubset, Union, SetIntersect, SetComplement, EmptySet

from aria.quant.fossil.naturalproofs.prover import NPSolver
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
    Var,
    Vars,
    Function,
    RecFunction,
    AddRecDefinition,
    AddAxiom,
)
from aria.quant.fossil.naturalproofs.pfp import make_pfp_formula

from aria.quant.fossil.lemsynth.lemsynth_engine import solveProblem

# declarations
x = Var("x", fgsort)
nil, ret = Consts("nil ret", fgsort)
nxt = Function("nxt", fgsort, fgsort)
key = Function("key", fgsort, intsort)
lst = RecFunction("lst", fgsort, boolsort)
slst = RecFunction("slst", fgsort, boolsort)
AddRecDefinition(lst, x, If(x == nil, True, lst(nxt(x))))
AddRecDefinition(
    slst,
    x,
    If(
        x == nil,
        True,
        If(nxt(x) == nil, True, And(key(x) <= key(nxt(x)), slst(nxt(x)))),
    ),
)
AddAxiom((), nxt(nil) == nil)

# vc
pgm = If(x == nil, ret == nil, ret == nxt(x))
goal = Implies(slst(x), Implies(pgm, lst(ret)))

# check validity with natural proof solver and no hardcoded lemmas
np_solver = NPSolver()
solution = np_solver.solve(make_pfp_formula(goal))
if not solution.if_sat:
    print("goal (no lemmas) is valid")
else:
    print("goal (no lemmas) is invalid")

# hardcoded lemma
lemma_params = (x,)
lemma_body = Implies(slst(x), lst(x))
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
lemma_grammar_terms = {v, nxt(v), nxt(nil)}

name = "slist-list"
grammar_string = importlib_resources.read_text("grammars", "grammar_{}.sy".format(name))

solveProblem(lemma_grammar_args, lemma_grammar_terms, goal, name, grammar_string)
