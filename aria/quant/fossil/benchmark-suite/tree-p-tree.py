import importlib_resources

import z3
from z3 import And, Or, Not, Implies, If
from z3 import IsMember, IsSubset, SetUnion, SetIntersect, SetComplement, EmptySet, SetAdd

from aria.quant.fossil.naturalproofs.prover import NPSolver
from aria.quant.fossil.naturalproofs.uct import fgsort, fgsetsort, intsort, intsetsort, boolsort, min_intsort, max_intsort
from aria.quant.fossil.naturalproofs.decl_api import Const, Consts, Var, Vars, Function, RecFunction, AddRecDefinition, AddAxiom
from aria.quant.fossil.naturalproofs.pfp import make_pfp_formula

from aria.quant.fossil.lemsynth.lemsynth_engine import solveProblem

def notInChildren(x):
    return And(Not(IsMember(x, htree(lft(x)))), Not(IsMember(x, htree(rght(x)))))

# declarations
x = Var('x', fgsort)
nil, ret = Consts('nil ret', fgsort)
lft = Function('lft', fgsort, fgsort)
rght = Function('rght', fgsort, fgsort)
parent = Function('parent', fgsort, fgsort)
tree = RecFunction('tree', fgsort, boolsort)
tree_p = RecFunction('tree_p', fgsort, boolsort)
htree = RecFunction('htree', fgsort, fgsetsort)
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
AddAxiom((), lft(nil) == nil)
AddAxiom((), rght(nil) == nil)

# vc
goal = Implies(tree_p(x), Implies(parent(x) == nil, tree(x)))

# check validity with natural proof solver and no hardcoded lemmas
np_solver = NPSolver()
solution = np_solver.solve(make_pfp_formula(goal))
if not solution.if_sat:
    print('goal (no lemmas) is valid')
else:
    print('goal (no lemmas) is invalid')

# hardcoded lemmas
lemma_params = (x,)
lemma_body = Implies(tree_p(x), tree(x))
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
v1 = Var('v1', fgsort)
lemma_grammar_args = [v1, nil]
lemma_grammar_terms = {v1, nil}

name = 'tree-parent-tree'
grammar_string = importlib_resources.read_text('grammars', 'grammar_{}.sy'.format(name))

solveProblem(lemma_grammar_args, lemma_grammar_terms, goal, name, grammar_string)
