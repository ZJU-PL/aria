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
    return And(Not(IsMember(x, hbst(lft(x)))), Not(IsMember(x, hbst(rght(x)))))

# declarations
x = Var('x', fgsort)
y, nil = Consts('y nil', fgsort)
k = Const('k', intsort)
key = Function('key', fgsort, intsort)
lft = Function('lft', fgsort, fgsort)
rght = Function('rght', fgsort, fgsort)
minr = Function('minr', fgsort, intsort)
maxr = Function('maxr', fgsort, intsort)
bst = RecFunction('bst', fgsort, boolsort)
hbst = RecFunction('hbst', fgsort, fgsetsort)
leftmost = RecFunction('leftmost', fgsort, fgsort)
AddRecDefinition(minr, x, If(x == nil, 100, min_intsort(key(x), minr(lft(x)), minr(rght(x)))))
AddRecDefinition(maxr, x, If(x == nil, -1, max_intsort(key(x), maxr(lft(x)), maxr(rght(x)))))
AddRecDefinition(bst, x, If(x == nil, True,
                            And(0 < key(x),
                                And(key(x) < 100,
                                    And(bst(lft(x)),
                                        And(bst(rght(x)),
                                            And(maxr(lft(x)) <= key(x),
                                                And(key(x) <= minr(rght(x)),
                                                    And(notInChildren(x),
                                                        SetIntersect(hbst(lft(x)), hbst(rght(x)))
                                                        == fgsetsort.lattice_bottom)))))))))
AddRecDefinition(hbst, x, If(x == nil, fgsetsort.lattice_bottom,
                             SetAdd(SetUnion(hbst(lft(x)), hbst(rght(x))), x)))
AddRecDefinition(leftmost, x, If(x == nil, x,
                                 If(lft(x) == nil, x, leftmost(lft(x)))))
AddAxiom((), lft(nil) == nil)
AddAxiom((), rght(nil) == nil)

# vc
goal = Implies(bst(x), Implies(And(x != nil, key(x) != k),
                                   key(leftmost(x)) == minr(x)))

# check validity with natural proof solver and no hardcoded lemmas
np_solver = NPSolver()
solution = np_solver.solve(make_pfp_formula(goal))
if not solution.if_sat:
    print('goal (no lemmas) is valid')
else:
    print('goal (no lemmas) is invalid')

# hardcoded lemmas
lemma_params = (x,)
lemma_body = Implies(bst(x), Implies(x != nil, key(leftmost(x)) == minr(x)))
lemmas = {(lemma_params, lemma_body)}

# check validity of lemmas
solution = np_solver.solve(make_pfp_formula(lemma_body))
if not solution.if_sat:
    print('lemma is valid')
else:
    print('lemma is invalid')

# check validity with natural proof solver
np_solver = NPSolver()
solution = np_solver.solve(goal, lemmas)
if not solution.if_sat:
    print('goal (with lemmas) is valid')
else:
    print('goal (with lemmas) is invalid')

# lemma synthesis
v = Var('v', fgsort)
lemma_grammar_args = [v, k, nil]
lemma_grammar_terms = {v, leftmost(lft(v)), leftmost(v), leftmost(rght(v))}

name = 'bst-leftmost'
# name = 'bst-leftmost-lvl0'
grammar_string = importlib_resources.read_text('grammars', 'grammar_{}.sy'.format(name))

solveProblem(lemma_grammar_args, lemma_grammar_terms, goal, name, grammar_string)
