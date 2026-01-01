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
nil, ret = Consts("nil ret", fgsort)
nxt = Function("nxt", fgsort, fgsort)
lst = RecFunction("lst", fgsort, boolsort)
odd_lst = RecFunction("odd_lst", fgsort, boolsort)
even_lst = RecFunction("even_lst", fgsort, boolsort)
AddRecDefinition(lst, x, If(x == nil, True, lst(nxt(x))))
AddRecDefinition(
    even_lst, x, If(x == nil, True, If(nxt(x) == nil, False, even_lst(nxt(nxt(x)))))
)
AddRecDefinition(
    odd_lst, x, If(x == nil, False, If(nxt(x) == nil, True, odd_lst(nxt(nxt(x)))))
)
AddAxiom((), nxt(nil) == nil)

# vc
pgm = If(x == nil, ret == nil, ret == nxt(x))
goal = Implies(lst(x), Implies(pgm, Or(even_lst(x), odd_lst(x))))

# check validity with natural proof solver and no hardcoded lemmas
np_solver = NPSolver()
solution = np_solver.solve(make_pfp_formula(goal))
if not solution.if_sat:
    print("goal (no lemmas) is valid")
else:
    print("goal (no lemmas) is invalid")

# hardcoded lemma
lemma_params = (x,)
lemma1_body = Implies(
    lst(x), Implies(Implies(even_lst(nxt(x)), odd_lst(nil)), even_lst(x))
)
lemma2_body = Implies(even_lst(x), Implies(x != nil, odd_lst(nxt(x))))

lemmas = {(lemma_params, lemma1_body), (lemma_params, lemma2_body)}

# check validity of lemmas
solution = np_solver.solve(make_pfp_formula(lemma1_body))
if not solution.if_sat:
    print("lemma 1 is valid")
else:
    print("lemma 1 is invalid")
solution = np_solver.solve(make_pfp_formula(lemma2_body))
if not solution.if_sat:
    print("lemma 2 is valid")
else:
    print("lemma 2 is invalid")

# check validity with natural proof solver and hardcoded lemmas
solution = np_solver.solve(goal, lemmas)
if not solution.if_sat:
    print("goal (with lemmas) is valid")
else:
    print("goal (with lemmas) is invalid")

# lemma synthesis
v = Var("v", fgsort)
lemma_grammar_args = [v, nil]
lemma_grammar_terms = {v, nil, nxt(nxt(v)), nxt(nil)}

name = "list-even-or-odd"
# name = 'list-even-or-odd-lvl0'
grammar_string = importlib_resources.read_text("grammars", "grammar_{}.sy".format(name))

solveProblem(lemma_grammar_args, lemma_grammar_terms, goal, name, grammar_string)
