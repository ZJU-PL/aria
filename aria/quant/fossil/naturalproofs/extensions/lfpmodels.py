# import z3
from z3 import *
import itertools

from aria.quant.fossil.naturalproofs.uct import fgsort, boolsort, intsort
from aria.quant.fossil.naturalproofs.decl_api import get_vocabulary, get_uct_signature, get_recursive_definition, get_all_axioms
from aria.quant.fossil.naturalproofs.utils import apply_bound_formula
from aria.quant.fossil.naturalproofs.prover_utils import instantiate, make_recdef_unfoldings

from aria.quant.fossil.naturalproofs.extensions.finitemodel import FiniteModel


def rank_fcts():
    x, y, nil = Ints('x y nil')

    # List
    nxt = Function('nxt', IntSort(), IntSort())
    lst = Function('lst', IntSort(), BoolSort())
    lst_rank = Function('lst_rank', IntSort(), IntSort())
    lst_recdef = lst(x) == If(x == nil, True, lst(nxt(x)))
    lst_rankdef = If(x == nil, True, lst(x) == (lst_rank(nxt(x)) < lst_rank(x)))
    lst_def_body = And(lst_recdef, lst_rankdef)

    # List segment
    lseg = Function('lseg', IntSort(), IntSort(), BoolSort())
    lseg_rank = Function('lseg_rank', IntSort(), IntSort(), IntSort())
    lseg_recdef = lseg(x, y) == If(x == y, True,
                                   If(x == nil, False,
                                      lseg(nxt(x), y)))
    lseg_rankdef = If(Or(x == y, x == nil), True, lseg(x, y) == (lseg_rank(nxt(x), y) < lseg_rank(x, y)))
    lseg_def_body = And(lseg_recdef, lseg_rankdef)

    # Binary tree
    lft = Function('lft', IntSort(), IntSort())
    rght = Function('rght', IntSort(), IntSort())
    tree = Function('tree', IntSort(), BoolSort())
    tree_rank = Function('tree_rank', IntSort(), IntSort())
    tree_recdef = tree(x) == If(x == nil, True,
                                And(tree(lft(x)), tree(rght(x))))
    tree_rankdef = If(x == nil, True, tree(x) == And(tree_rank(lft(x)) < tree_rank(x),
                                                     tree_rank(rght(x)) < tree_rank(x)))
    tree_def_body = And(tree_recdef, tree_rankdef)

    # Binary search tree
    minr = Function('minr', IntSort(), IntSort())
    maxr = Function('maxr', IntSort(), IntSort())
    key = Function('key', IntSort(), IntSort())
    minr_recdef = If(x == nil, minr(x) == 100,
                     If(And(key(x) <= minr(lft(x)), key(x) <= minr(rght(x))), minr(x) == key(x),
                        If(minr(lft(x)) <= minr(rght(x)), minr(x) == minr(lft(x)),
                           minr(x) == minr(rght(x)))))
    maxr_recdef = If(x == nil, maxr(x) == -1,
                     If(And(key(x) >= maxr(lft(x)), key(x) >= maxr(rght(x))), maxr(x) == key(x),
                        If(maxr(lft(x)) >= maxr(rght(x)), maxr(x) == maxr(lft(x)),
                           maxr(x) == maxr(rght(x)))))
    bst = Function('bst', IntSort(), BoolSort())
    bst_rank = Function('bst_rank', IntSort(), IntSort())
    bst_recdef = bst(x) == If(x == nil, True,
                              And(0 < key(x), key(x) < 100,
                                  bst(lft(x)), bst(rght(x)),
                                  maxr(lft(x)) <= key(x),
                                  key(x) <= minr(rght(x))))
    bst_rankdef = If(x == nil, True, bst(x) == And(bst_rank(lft(x)) < bst_rank(x),
                                                   bst_rank(rght(x)) < bst_rank(x)))
    bst_def_body = And(minr_recdef, maxr_recdef, bst_recdef, bst_rankdef)

    # Cyclic
    cyclic = Function('cyclic', IntSort(), BoolSort())
    cyclic_rank = Function('cyclic_rank', IntSort(), IntSort())
    cyclic_recdef = cyclic(x) == If(x == nil, False,
                                    lseg(nxt(x), x))
    cyclic_rankdef = If(x == nil, True, cyclic(x) == And(lseg_rank(nxt(x), x) < cyclic_rank(x),
                                                         cyclic_rank(nxt(x)) == cyclic_rank(x)))
    cyclic_def_body = And(cyclic_recdef, cyclic_rankdef)

    # Directed acyclic graph
    dag = Function('dag', IntSort(), BoolSort())
    dag_rank = Function('dag_rank', IntSort(), IntSort())
    dag_recdef = dag(x) == If(x == nil, True,
                              And(dag(lft(x)), dag(rght(x))))
    dag_rankdef = If(x == nil, True, dag(x) == And(dag_rank(lft(x)) < dag_rank(x),
                                                   dag_rank(rght(x)) < dag_rank(x)))
    dag_def_body = And(dag_recdef, dag_rankdef)

    # Reachability
    # reach = Function('reach', IntSort(), IntSort(), BoolSort())
    # reach_rank = Function('reach_rank', IntSort(), IntSort(), IntSort())
    # reach_recdef = reach(x, y) == If(x == y, True,
    #                                  Or(reach(lft(x), y), reach(rght(x), y)))
    # reach_rankdef = If(x == y, True,
    #                    And(reach(lft(x), y) == (reach_rank(lft(x), y) < reach_rank(x, y)),
    #                        reach(rght(x), y) == (reach_rank(rght(x), y) < reach_rank(x, y))))
    # reach_def_body = And(reach_recdef, reach_rankdef)

    # Directed list
    prv = Function('prv', IntSort(), IntSort())
    dlst = Function('dlst', IntSort(), BoolSort())
    dlst_rank = Function('dlst_rank', IntSort(), IntSort())
    dlst_recdef = dlst(x) == If(Or(x == nil, nxt(x) == nil), True, And(prv(nxt(x)) == x, dlst(nxt(x))))
    dlst_rankdef = If(x == nil, True, dlst(x) == (dlst_rank(nxt(x)) < dlst_rank(x)))
    dlst_def_body = And(dlst_recdef, dlst_rankdef)

    # Even list
    even_lst = Function('even_lst', IntSort(), BoolSort())
    even_lst_rank = Function('even_lst_rank', IntSort(), IntSort())
    even_lst_recdef = even_lst(x) == If(x == nil, True,
                                        even_lst(nxt(nxt(x))))
    even_lst_rankdef = If(x == nil, True, even_lst(x) == (even_lst_rank(nxt(nxt(x))) < even_lst_rank(x)))
    even_lst_def_body = And(even_lst_recdef, even_lst_rankdef)

    # Odd list
    odd_lst = Function('odd_lst', IntSort(), BoolSort())
    odd_lst_rank = Function('odd_lst_rank', IntSort(), IntSort())
    odd_lst_recdef = odd_lst(x) == If(x == nil, False,
                                      If(nxt(x) == nil, True,
                                         odd_lst(nxt(nxt(x)))))
    odd_lst_rankdef = If(nxt(x) == nil, True, odd_lst(x) == (odd_lst_rank(nxt(nxt(x))) < odd_lst_rank(x)))
    odd_lst_def_body = And(odd_lst_recdef, odd_lst_rankdef)

    # Sorted list
    slst = Function('slst', IntSort(), BoolSort())
    slst_rank = Function('slst_rank', IntSort(), IntSort())
    slst_recdef = slst(x) == If(Or(x == nil, nxt(x) == nil), True,
                                And(key(x) <= key(nxt(x)),
                                    slst(nxt(x))))
    slst_rankdef = If(Or(x == nil, nxt(x) == nil), True,
                      slst(x) == (slst_rank(nxt(x)) < slst_rank(x)))
    slst_def_body = And(slst_recdef, slst_rankdef)

    # Sorted list segment
    slseg = Function('slseg', IntSort(), IntSort(), BoolSort())
    slseg_rank = Function('slseg_rank', IntSort(), IntSort(), IntSort())
    slseg_recdef = slseg(x, y) == If(x == y, True,
                                     And(key(x) <= key(nxt(x)),
                                         slseg(nxt(x), y)))
    slseg_rankdef = If(x == y, True, slseg(x, y) == (slseg_rank(nxt(x), y) < slseg_rank(x, y)))
    slseg_def_body = And(slseg_recdef, slseg_rankdef)

    return {
        'lst': ((x,), lst_def_body),
        'lseg': ((x, y,), lseg_def_body),
        'tree': ((x,), tree_def_body),
        'bst': ((x,), bst_def_body),
        'cyclic': ((x,), cyclic_def_body),
        'dag': ((x,), dag_def_body),
        # 'reach': ((x, y,), reach_def_body),
        'dlst': ((x,), dlst_def_body),
        'even_lst': ((x,), even_lst_def_body),
        'odd_lst': ((x,), odd_lst_def_body),
        'slst': ((x,), slst_def_body),
        'slseg': ((x,), slseg_def_body)
    }


def rank_fcts_lightweight():
    x, y, z, nil = z3.Consts('x y z nil', fgsort.z3sort)
    nxt = z3.Function('nxt', fgsort.z3sort, fgsort.z3sort)
    lft = z3.Function('lft', fgsort.z3sort, fgsort.z3sort)
    rght = z3.Function('rght', fgsort.z3sort, fgsort.z3sort)
    # 'Starting' configuration in reachability benchmarks
    s = z3.Const('s', fgsort.z3sort)
    # 'Previous' configuration in reachability benchmarks
    p = z3.Function('p', fgsort.z3sort, fgsort.z3sort)

    # List
    lst = z3.Function('lst', fgsort.z3sort, boolsort.z3sort)
    lst_rank = z3.Function('lst_rank', fgsort.z3sort, intsort.z3sort)
    lst_rank_def = ((x,), If(x == nil, lst_rank(x) == 0,
                             If(lst(x), lst_rank(x) > lst_rank(nxt(x)),
                                lst_rank(x) == -1)))

    # lsegy
    lsegy = z3.Function('lsegy', fgsort.z3sort, boolsort.z3sort)
    lsegy_rank = z3.Function('lsegy_rank', fgsort.z3sort, intsort.z3sort)
    lsegy_rank_def = ((x,), If(x == y, lsegy_rank(x) == 0,
                               If(lsegy(x), lsegy_rank(x) > lsegy_rank(nxt(x)),
                                  lsegy_rank(x) == -1)))

    # lsegz
    lsegz = z3.Function('lsegz', fgsort.z3sort, boolsort.z3sort)
    lsegz_rank = z3.Function('lsegz_rank', fgsort.z3sort, intsort.z3sort)
    lsegz_rank_def = ((x,), If(x == y, lsegz_rank(x) == 0,
                               If(lsegz(x), lsegz_rank(x) > lsegz_rank(nxt(x)),
                                  lsegz_rank(x) == -1)))

    # List segment
    lseg = z3.Function('lseg', fgsort.z3sort, fgsort.z3sort, boolsort.z3sort)
    lseg_rank = z3.Function('lseg_rank', fgsort.z3sort, fgsort.z3sort, intsort.z3sort)
    lseg_rank_def = ((x, y), If(x == y, lseg_rank(x, y) == 0,
                                If(lseg(x, y), lseg_rank(x, y) > lseg_rank(nxt(x), y),
                                   lseg_rank(x, y) == -1)))

    # List variants: sorted list or sorted lseg, and doubly linked lists/sorted doubly linked lists
    slst = z3.Function('slst', fgsort.z3sort, boolsort.z3sort)
    slst_rank = z3.Function('slst_rank', fgsort.z3sort, intsort.z3sort)
    slst_rank_def = ((x,), If(x == nil, slst_rank(x) == 0,
                              If(slst(x), slst_rank(x) > slst_rank(nxt(x)),
                                 slst_rank(x) == -1)))
    slseg = z3.Function('slseg', fgsort.z3sort, fgsort.z3sort, boolsort.z3sort)
    slseg_rank = z3.Function('slseg_rank', fgsort.z3sort, fgsort.z3sort, intsort.z3sort)
    slseg_rank_def = ((x, y), If(x == y, slseg_rank(x, y) == 0,
                                 If(slseg(x, y), slseg_rank(x, y) > slseg_rank(nxt(x), y),
                                    slseg_rank(x, y) == -1)))
    dlst = z3.Function('dlst', fgsort.z3sort, boolsort.z3sort)
    dlst_rank = z3.Function('dlst_rank', fgsort.z3sort, intsort.z3sort)
    dlst_rank_def = ((x,), If(x == nil, dlst_rank(x) == 0,
                              If(dlst(x), dlst_rank(x) > dlst_rank(nxt(x)),
                                 dlst_rank(x) == -1)))
    sdlst = z3.Function('sdlst', fgsort.z3sort, boolsort.z3sort)
    sdlst_rank = z3.Function('sdlst_rank', fgsort.z3sort, intsort.z3sort)
    sdlst_rank_def = ((x,), If(x == nil, sdlst_rank(x) == 0,
                               If(sdlst(x), sdlst_rank(x) > sdlst_rank(nxt(x)),
                                  sdlst_rank(x) == -1)))
    # Record list
    rlst = z3.Function('rlst', fgsort.z3sort, boolsort.z3sort)
    rlst_rank = z3.Function('rlst_rank', fgsort.z3sort, intsort.z3sort)
    rlst_rank_def = ((x,), If(x == nil, rlst_rank(x) == 0,
                              If(rlst(x), rlst_rank(x) > rlst_rank(nxt(x)),
                                 rlst_rank(x) == -1)))
    # 'Odd' and 'Even' lists
    even_lst = z3.Function('even_lst', fgsort.z3sort, boolsort.z3sort)
    even_lst_rank = z3.Function('even_lst_rank', fgsort.z3sort, intsort.z3sort)
    even_lst_rank_def = ((x,), If(x == nil, even_lst_rank(x) == 0,
                                  If(nxt(x) == nil, even_lst_rank(x) == -1,
                                     If(even_lst(x), even_lst_rank(x) > even_lst_rank(nxt(nxt(x))),
                                        even_lst_rank(x) == -1))))
    odd_lst = z3.Function('odd_lst', fgsort.z3sort, boolsort.z3sort)
    odd_lst_rank = z3.Function('odd_lst_rank', fgsort.z3sort, intsort.z3sort)
    odd_lst_rank_def = ((x,), If(x == nil, odd_lst_rank(x) == -1,
                                 If(nxt(x) == nil, odd_lst_rank(x) == 0,
                                    If(odd_lst(x), odd_lst_rank(x) > odd_lst_rank(nxt(nxt(x))),
                                       odd_lst_rank(x) == -1))))

    # Binary tree
    tree = z3.Function('tree', fgsort.z3sort, boolsort.z3sort)
    tree_rank = z3.Function('tree_rank', fgsort.z3sort, intsort.z3sort)
    tree_rank_def = ((x,), If(x == nil, tree_rank(x) == 0,
                              If(tree(x), And(tree_rank(x) > tree_rank(lft(x)),
                                              tree_rank(x) > tree_rank(rght(x))),
                                 tree_rank(x) == -1)))

    # Binary search tree
    bst = z3.Function('bst', fgsort.z3sort, boolsort.z3sort)
    bst_rank = z3.Function('bst_rank', fgsort.z3sort, intsort.z3sort)
    bst_rank_def = ((x,), If(x == nil, bst_rank(x) == 0,
                             If(bst(x), And(bst_rank(x) > bst_rank(lft(x)),
                                            bst_rank(x) > bst_rank(rght(x))),
                                bst_rank(x) == -1)))

    # Leftmost node in a bst
    # leftmost = z3.Function('leftmost', fgsort.z3sort, fgsort.z3sort)
    # leftmost_rank = z3.Function('leftmost_rank', fgsort.z3sort, intsort.z3sort)
    # leftmost_rank_def = ((x,), If(x == nil, leftmost_rank(x) == 0,
    #                               If(bst(x), leftmost_rank(x) > leftmost_rank(lft(x)),
    #                                  leftmost_rank(x) == -1)))

    # Maxheap
    maxheap = z3.Function('maxheap', fgsort.z3sort, boolsort.z3sort)
    maxheap_rank = z3.Function('maxheap_rank', fgsort.z3sort, intsort.z3sort)
    maxheap_rank_def = ((x,), If(x == nil, maxheap_rank(x) == 0,
                                 If(maxheap(x), And(maxheap_rank(x) > maxheap_rank(lft(x)),
                                                    maxheap_rank(x) > maxheap_rank(rght(x))),
                                    maxheap_rank(x) == -1)))

    # DAG
    dag = z3.Function('dag', fgsort.z3sort, boolsort.z3sort)
    dag_rank = z3.Function('dag_rank', fgsort.z3sort, intsort.z3sort)
    dag_rank_def = ((x,), If(x == nil, dag_rank(x) == 0,
                             If(dag(x), And(dag_rank(x) > dag_rank(lft(x)),
                                            dag_rank(x) > dag_rank(rght(x))),
                                dag_rank(x) == -1)))

    # tree with parent pointer
    tree_p = z3.Function('tree_p', fgsort.z3sort, boolsort.z3sort)
    tree_p_rank = z3.Function('tree_p_rank', fgsort.z3sort, intsort.z3sort)
    tree_p_rank_def = ((x,), If(x == nil, tree_p_rank(x) == 0,
                                If(tree_p(x), And(tree_p_rank(x) > tree_p_rank(lft(x)),
                                                  tree_p_rank(x) > tree_p_rank(rght(x))),
                                   tree_p_rank(x) == -1)))

    # Reach by either using left or right pointers
    reach_lr = z3.Function('reach_lr', fgsort.z3sort, fgsort.z3sort, boolsort.z3sort)
    reach_lr_rank = z3.Function('reach_lr_rank', fgsort.z3sort, fgsort.z3sort, intsort.z3sort)
    reach_lr_rank_def = ((x, y), If(x == y, reach_lr_rank(x, y) == 0,
                                    If(reach_lr(x, y), And(reach_lr_rank(x, y) > reach_lr_rank(lft(x), y),
                                                           reach_lr_rank(x, y) > reach_lr_rank(rght(x), y)),
                                       reach_lr_rank(x, y) == -1)))

    # Reachability benchmarks (loop invariant encoding)
    reach_pgm = z3.Function('reach_pgm', fgsort.z3sort, boolsort.z3sort)
    reach_pgm_rank = z3.Function('reach_pgm_rank', fgsort.z3sort, intsort.z3sort)
    reach_pgm_rank_def = ((x,), If(x == s, reach_pgm_rank(x) == 0,
                                   If(reach_pgm(x), reach_pgm_rank(x) > reach_pgm_rank(p(x)),
                                      reach_pgm_rank(x) == -1)))

    return {
        'lst': lst_rank_def,
        'lseg': lseg_rank_def,
        'lsegy': lsegy_rank_def,
        'lsegz': lsegz_rank_def,
        'slst': slst_rank_def,
        'slseg': slseg_rank_def,
        'dlst': dlst_rank_def,
        'sdlst': sdlst_rank_def,
        'rlst': rlst_rank_def,
        'even_lst': even_lst_rank_def,
        'odd_lst': odd_lst_rank_def,
        'tree': tree_rank_def,
        'bst': bst_rank_def,
        # 'leftmost': leftmost_rank_def,
        'maxheap': maxheap_rank_def,
        'dag': dag_rank_def,
        'tree_p': tree_p_rank_def,
        'reach_lr': reach_lr_rank_def,
        'reach_pgm': reach_pgm_rank_def
    }


# Dictionary of rank constraints corresponding to recursive definitions
# rank_defs_dict = rank_fcts()
# To use lightweight rank functions comment out the line above and uncomment the line below
rank_defs_dict = rank_fcts_lightweight()


def gen_lfp_model(size, annctx, invalid_formula=None):
    """
    Generate a finite model of the theory given by annctx of specified size where the valuation
    of recursive definitions respects LFP semantics. Returns None if model with specified conditions does not exist.
    Optional argument invalid_formula is a bound formula that must be falsified by the returned model.
    :param size: int
    :param annctx: naturalproofs.AnnotatedContext.AnnotatedContext
    :param invalid_formula: pair (tuple of z3.ExprRef, z3.BoolRef)
    :return: pair (z3.ModelRef, set)
    """
    # Establish underlying universe
    universe = {z3.IntVal(i) for i in range(size)}
    constraints = []

    vocabulary = get_vocabulary(annctx=annctx)
    # Closure for vocabluary that is over the foreground sort
    # If signature guidelines follow naturalproofs.uct then range can only be fgsort if the domain args are all fgsort
    foreground_vocabluary = {funcdecl for funcdecl in vocabulary
                             if all(srt == fgsort for srt in get_uct_signature(funcdecl, annctx=annctx))}
    for funcdecl in foreground_vocabluary:
        argslist = itertools.product(universe, repeat=funcdecl.arity())
        constraints.append(And([Or([funcdecl(*args) == elem for elem in universe]) for args in argslist]))

    # Recursive definitions and ranks
    recdefs = get_recursive_definition(None, alldefs=True, annctx=annctx)
    recdef_unfoldings = make_recdef_unfoldings(recdefs)
    untagged_unfoldings = set(recdef_unfoldings.values())
    rank_formulas = {recdef.name(): rank_defs_dict.get(recdef.name(), None) for recdef, _, _ in recdefs}
    if all(value is None for value in rank_formulas.values()):
        raise Exception('Rank definitions must be given at least for the underlying datastructure definition. '
                        'This should be one among: {}'
                        .format(', '.join(key for key, value in rank_formulas.items() if value is None)))
    # Axioms
    axioms = get_all_axioms(annctx=annctx)
    structural_constraints = untagged_unfoldings | set(
        rankdef for rankdef in rank_formulas.values() if rankdef is not None) | axioms
    constraints.extend(instantiate(structural_constraints, universe))

    # Bound formula to negate
    if invalid_formula is not None:
        formal_vars, body = invalid_formula
        constraints.append(Or([Not(apply_bound_formula(invalid_formula, args))
                               for args in itertools.product(universe, repeat=len(formal_vars))]))

    z3.set_param('smt.random_seed', 0)
    z3.set_param('sat.random_seed', 0)
    sol = z3.Solver()
    sol.add(constraints)
    if sol.check() == z3.sat:
        lfp_model = sol.model()
        # Project model onto the numbers corresponding to the foreground universe
        finite_lfp_model = FiniteModel(lfp_model, universe, annctx=annctx)
        return finite_lfp_model, universe
    else:
        return None, None
