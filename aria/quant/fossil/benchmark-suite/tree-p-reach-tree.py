import importlib_resources

import z3
from z3 import And, Or, Not, Implies, If
from z3 import IsMember, IsSubset, SetUnion, SetIntersect, SetComplement, EmptySet, SetAdd

from aria.fossil.naturalproofs.prover import NPSolver
from aria.fossil.naturalproofs.uct import fgsort, fgsetsort, intsort, intsetsort, boolsort, min_intsort, max_intsort
from aria.fossil.naturalproofs.decl_api import Const, Consts, Var, Vars, Function, RecFunction, AddRecDefinition, AddAxiom
from aria.fossil.naturalproofs.pfp import make_pfp_formula

from aria.fossil.lemsynth.lemsynth_engine import solveProblem

def notInChildren(x):
    return And(Not(IsMember(x, htree(lft(x)))), Not(IsMember(x, htree(rght(x)))))

# declarations
x, y = Vars('x y', fgsort)
nil, ret = Consts('nil ret', fgsort)
k = Const('k', intsort)
key = Function('key', fgsort, intsort)
lft = Function('lft', fgsort, fgsort)
rght = Function('rght', fgsort, fgsort)
parent = Function('parent', fgsort, fgsort)
tree = RecFunction('tree', fgsort, boolsort)
tree_p = RecFunction('tree_p', fgsort, boolsort)
htree = RecFunction('htree', fgsort, fgsetsort)
reach_lr = RecFunction('reach_lr', fgsort, fgsort, boolsort)
AddRecDefinition(tree, x, If(x == nil, True,
                             And(notInChildren(x),
                                 And(SetIntersect(htree(lft(x)), htree(rght(x)))
                                     == fgsetsort.lattice_bottom,
                                     And(tree(lft(x)), tree(rght(x)))))))
AddRecDefinition(tree_p, x, If(x == nil, True,
                               And(notInChildren(x),
                                   And(SetIntersect(htree(lft(x)), htree(rght(x)))
                                       == fgsetsort.lattice_bottom,
                                       And(And(parent(lft(x)) == x, parent(rght(x)) == x),
                                           And(tree_p(lft(x)), tree_p(rght(x))))))))
AddRecDefinition(htree, x, If(x == nil, fgsetsort.lattice_bottom,
                              SetAdd(SetUnion(htree(lft(x)), htree(rght(x))), x)))
AddRecDefinition(reach_lr, (x, y), If(x == y, True,
                                   Or(reach_lr(lft(x), y), reach_lr(rght(x), y))))
AddAxiom((), lft(nil) == nil)
AddAxiom((), rght(nil) == nil)

# vc
goal = Implies(tree_p(x), Implies(parent(x) == nil, Implies(reach_lr(x,y), tree(y))))

# check validity with natural proof solver and no hardcoded lemmas
np_solver = NPSolver()
solution = np_solver.solve(make_pfp_formula(goal))
if not solution.if_sat:
    print('goal (no lemmas) is valid')
else:
    print('goal (no lemmas) is invalid')

# hardcoded lemmas
lemma1_params = (x,y)
lemma1_body = Implies(reach_lr(x, y), Implies(tree_p(x), tree_p(y)))
lemma2_params = (x,)
lemma2_body = Implies(tree_p(x), tree(x))
lemmas = {(lemma1_params, lemma1_body), (lemma1_params, lemma2_body)}

# check validity of lemmas
solution = np_solver.solve(make_pfp_formula(lemma1_body))
if not solution.if_sat:
    print('lemma 1 is valid')
else:
    print('lemma 1 is invalid')
solution = np_solver.solve(make_pfp_formula(lemma2_body))
if not solution.if_sat:
    print('lemma 2 is valid')
else:
    print('lemma 2 is invalid')

# check validity with natural proof solver and hardcoded lemmas
solution = np_solver.solve(goal, lemmas)
if not solution.if_sat:
    print('goal (with lemmas) is valid')
else:
    print('goal (with lemmas) is invalid')

# lemma synthesis
v1, v2 = Vars('v1 v2', fgsort)
lemma_grammar_args = [v1, v2, nil]
lemma_grammar_terms = {v1, v2, nil}

name = 'tree-parent-reach-tree'
grammar_string = importlib_resources.read_text('grammars', 'grammar_{}.sy'.format(name))

solveProblem(lemma_grammar_args, lemma_grammar_terms, goal, name, grammar_string)
