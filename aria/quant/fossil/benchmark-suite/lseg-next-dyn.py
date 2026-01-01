import importlib_resources

import z3
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
    Var,
    Vars,
    Function,
    RecFunction,
    AddRecDefinition,
    AddAxiom,
)
from aria.quant.fossil.naturalproofs.prover import NPSolver
import aria.quant.fossil.naturalproofs.proveroptions as proveroptions
from aria.quant.fossil.naturalproofs.pfp import make_pfp_formula

from aria.quant.fossil.lemsynth.lemsynth_engine import solveProblem

# declarations
x = Var("x", fgsort)
nil, yc, zc = Consts("nil yc zc", fgsort)
k = Const("k", intsort)
nxt = Function("nxt", fgsort, fgsort)
lsegy = RecFunction("lsegy", fgsort, boolsort)
lsegz = RecFunction("lsegz", fgsort, boolsort)
lsegy_p = RecFunction("lsegy_p", fgsort, boolsort)
lsegz_p = RecFunction("lsegz_p", fgsort, boolsort)
key = Function("key", fgsort, intsort)
AddRecDefinition(lsegy, x, If(x == yc, True, lsegy(nxt(x))))
AddRecDefinition(lsegz, x, If(x == zc, True, lsegz(nxt(x))))
# lsegy_p, lsegz_p defs reflect change of nxt(y) == z
AddRecDefinition(lsegy_p, x, If(x == yc, True, lsegy_p(If(x == yc, zc, nxt(x)))))
AddRecDefinition(lsegz_p, x, If(x == zc, True, lsegz_p(If(x == yc, zc, nxt(x)))))
AddAxiom((), nxt(nil) == nil)

# vc
goal = Implies(lsegy(x), Implies(key(x) != k, lsegz_p(x)))

# check validity with natural proof solver and no hardcoded lemmas
np_solver = NPSolver()
solution = np_solver.solve(make_pfp_formula(goal))
if not solution.if_sat:
    print("goal (no lemmas) is valid")
else:
    print("goal (no lemmas) is invalid")

# hardcoded lemma
lemma_params = (x,)
lemma_body = Implies(lsegy(x), lsegz_p(x))
lemmas = {(lemma_params, lemma_body)}

# check validity of lemmas
np_solver.options.instantiation_mode = proveroptions.bounded_depth
np_solver.options.depth = 2
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
v1 = Var("v1", fgsort)
lemma_grammar_args = [v1, yc, zc, k, nil]
lemma_grammar_terms = {v1, yc, zc, k, nil, nxt(yc), nxt(nxt(v1)), nxt(zc)}

name = "lseg-next-dynamic"
grammar_string = importlib_resources.read_text("grammars", "grammar_{}.sy".format(name))

solveProblem(lemma_grammar_args, lemma_grammar_terms, goal, name, grammar_string)
