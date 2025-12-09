import importlib_resources

import z3
from z3 import And, Or, Not, Implies, If
from z3 import IsSubset, Union, SetIntersect, SetComplement, EmptySet

from aria.quant.fossil.naturalproofs.prover import NPSolver
from aria.quant.fossil.naturalproofs.uct import fgsort, fgsetsort, intsort, intsetsort, boolsort
from aria.quant.fossil.naturalproofs.decl_api import Const, Consts, Var, Vars, Function, RecFunction, AddRecDefinition, AddAxiom
from aria.quant.fossil.naturalproofs.pfp import make_pfp_formula

from aria.quant.fossil.lemsynth.lemsynth_engine import solveProblem

# declarations
x = Var('x', fgsort)
c, s, nil = Consts('c s nil', fgsort)
v1 = Function('v1', fgsort, fgsort)
v2 = Function('v2', fgsort, fgsort)
p = Function('p', fgsort, fgsort)
n = Function('n', fgsort, fgsort)

reach_pgm = RecFunction('reach_pgm', fgsort, boolsort)

# precondition
AddAxiom((), v1(s) == n(v2(s)))

cond = v1(p(x)) != nil
assign1 = v1(x) == n(v1(p(x)))
assign2 = v2(x) == n(v2(p(x)))
assign = And(assign1, assign2)
AddRecDefinition(reach_pgm, x, If(x == s, True, And(reach_pgm(p(x)), And(cond, assign))))

# vc
lhs = v1(x) == nil
rhs = n(v2(x)) == nil
goal = Implies(reach_pgm(x), Implies(lhs, rhs))

# check validity with natural proof solver and no hardcoded lemmas
np_solver = NPSolver()
solution = np_solver.solve(make_pfp_formula(goal))
if not solution.if_sat:
    print('goal (no lemmas) is valid')
else:
    print('goal (no lemmas) is invalid')

# hardcoded lemma
lemma_params = (x,)
lemma_body = Implies(reach_pgm(x), n(v2(x)) == v1(x))
lemmas = {(lemma_params, lemma_body)}

# check validity of lemmas
solution = np_solver.solve(make_pfp_formula(lemma_body))
if not solution.if_sat:
    print('lemma is valid')
else:
    print('lemma is invalid')

# check validity with natural proof solver and hardcoded lemmas
solution = np_solver.solve(goal, lemmas)
if not solution.if_sat:
    print('goal (with lemmas) is valid')
else:
    print('goal (with lemmas) is invalid')

# lemma synthesis
z = Var('z', fgsort)
lemma_grammar_args = [z, nil, s, c]
lemma_grammar_terms = {z, nil, s, c, v1(z), v1(p(z)), p(z), v2(z), v2(p(z)), n(v2(p(z))), n(v1(p(z))), n(c), n(nil), n(n(nil)), v1(z), n(v2(z)), n(n(c)), n(v1(z)), n(n(n(nil))), n(n(n(c)))}

name = 'reachability6'
grammar_string = importlib_resources.read_text('grammars', 'grammar_{}.sy'.format(name))

solveProblem(lemma_grammar_args, lemma_grammar_terms, goal, name, grammar_string)
