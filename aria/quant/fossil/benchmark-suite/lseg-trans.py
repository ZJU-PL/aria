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
x, y = Vars("x y", fgsort)
nil, c, z = Consts("nil c z", fgsort)
nxt = Function("nxt", fgsort, fgsort)
lst = RecFunction("lst", fgsort, boolsort)
lseg = RecFunction("lseg", fgsort, fgsort, boolsort)
AddRecDefinition(lst, x, If(x == nil, True, lst(nxt(x))))
AddRecDefinition(lseg, (x, y), If(x == y, True, lseg(nxt(x), y)))
AddAxiom((), nxt(nil) == nil)

# vc
rhs = Implies(And(lst(y), And(x != c, nxt(y) == z)), And(lseg(x, z), lst(z)))
goal = Implies(lseg(x, y), rhs)

# check validity with natural proof solver and no hardcoded lemmas
np_solver = NPSolver()
solution = np_solver.solve(make_pfp_formula(goal))
if not solution.if_sat:
    print("goal (no lemmas) is valid")
else:
    print("goal (no lemmas) is invalid")

# hardcoded lemma
lemma_params = (x, y)
lemma_body = Implies(lseg(x, y), Implies(lseg(y, z), lseg(x, z)))
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
lemma_grammar_args = [v1, v2, z, c, nil]
lemma_grammar_terms = {
    z,
    nxt(z),
    c,
    v1,
    nil,
    nxt(nil),
    v2,
    nxt(v2),
    nxt(v1),
    nxt(nxt(v1)),
    nxt(nxt(nxt(v1))),
}

name = "lseg-trans"
grammar_string = importlib_resources.read_text("grammars", "grammar_{}.sy".format(name))

solveProblem(lemma_grammar_args, lemma_grammar_terms, goal, name, grammar_string)
