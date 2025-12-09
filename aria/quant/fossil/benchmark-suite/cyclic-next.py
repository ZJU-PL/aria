import importlib_resources

import z3
from z3 import And, Or, Not, Implies, If
from z3 import IsSubset, Union, SetIntersect, SetComplement, EmptySet

from aria.fossil.naturalproofs.uct import fgsort, fgsetsort, intsort, intsetsort, boolsort
from aria.fossil.naturalproofs.decl_api import Const, Consts, Var, Vars, Function, RecFunction, AddRecDefinition, AddAxiom
from aria.fossil.naturalproofs.prover import NPSolver
import aria.fossil.naturalproofs.proveroptions as proveroptions
from aria.fossil.naturalproofs.pfp import make_pfp_formula

from aria.fossil.lemsynth.lemsynth_engine import solveProblem

# Declarations
x, y = Vars('x y', fgsort)
nil = Const('nil', fgsort)
nxt = Function('nxt', fgsort, fgsort)
lseg = RecFunction('lseg', fgsort, fgsort, boolsort)
cyclic = RecFunction('cyclic', fgsort, boolsort)
AddRecDefinition(lseg, (x, y) , If(x == y, True,
                                   If(x == nil, False,
                                      lseg(nxt(x), y))))
AddRecDefinition(cyclic, x, And(x != nil, lseg(nxt(x), x)))
AddAxiom((), nxt(nil) == nil)

# vc
goal = Implies(cyclic(x), cyclic(nxt(x)))

# check validity with natural proof solver and no hardcoded lemmas
np_solver = NPSolver()
solution = np_solver.solve(make_pfp_formula(goal))
if not solution.if_sat:
    print('goal (no lemmas) is valid')
else:
    print('goal (no lemmas) is invalid')

# hardcoded lemmas
lemma_params = (x,y)
lemma_body = Implies(lseg(x,y), Implies(y != nil, lseg(x, nxt(y))))
lemmas = {(lemma_params, lemma_body)}

# check validity of lemmas
solution = np_solver.solve(make_pfp_formula(lemma_body))
if not solution.if_sat:
    print('lemma is valid')
else:
    print('lemma is invalid')

# check validity with natural proof solver and hardcoded lemmas
np_solver.options.instantiation_mode = proveroptions.depth_one_stratified_instantiation
solution = np_solver.solve(goal, lemmas)
if not solution.if_sat:
    print('goal (with lemmas) is valid')
else:
    print('goal (with lemmas) is invalid')
    
# lemma synthesis
v1, v2 = Vars('v1 v2', fgsort)
lemma_grammar_args = [v1, v2, nil]
lemma_grammar_terms = {v1, v2, nil, nxt(v2), nxt(nxt(v1)), nxt(nil)}

name = 'cyclic-next'
# name = 'cyclic-next-lvl0'
grammar_string = importlib_resources.read_text('grammars', 'grammar_{}.sy'.format(name))

solveProblem(lemma_grammar_args, lemma_grammar_terms, goal, name, grammar_string)
