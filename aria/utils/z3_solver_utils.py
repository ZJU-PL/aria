"""
A few APIs that (typically) require SMT solving
- **is_valid**: decide validity of phi
- **is_entail**: decide whether a entails b or not (i.e., b is a consequence of a)
- **is_sat**: decide satisfiability of phi
- **is_equiv**: decide equivalence between a and b
- **compact_check**: given a precond G and a set of cnts: f1, f2,..., fn, decide if the following cnts are satisfiable: And(G, f1), And(G, f2), ..., And(G, fn)
- **prime_implicant**: find a subset ps' of ps such that /\\ ps => e
- **to_dnf**: convert a formula to DNF
- **exclusive_to_dnf**: convert a formula to DNF
- **get_models**: return the first k models satisfiying f
"""

from typing import List

import z3

from .z3_expr_utils import get_atoms, big_and, eval_predicates


def is_valid(phi: z3.ExprRef) -> bool:
    """Decide validity of phi"""
    s = z3.Solver()
    s.add(z3.Not(phi))
    if s.check() == z3.sat:
        return False
    else:
        return True


def is_entail(a: z3.ExprRef, b: z3.ExprRef) -> bool:
    """Decide whether a entails b or not (i.e., b is a consequence of a)"""
    s = z3.Solver()
    s.add(z3.Not(z3.Implies(a, b)))
    if s.check() == z3.sat:
        return False
    else:
        return True


def is_sat(phi: z3.ExprRef) -> bool:
    """Decide satisfiability of phi"""
    s = z3.Solver()
    s.add(phi)
    return s.check() == z3.sat


def is_equiv(a: z3.ExprRef, b: z3.ExprRef) -> bool:
    """Decide equivalence between a and b"""
    s = z3.Solver()
    s.add(a != b)
    return s.check() == z3.unsat


def compact_check_misc(precond, cnt_list, res_label):
    """
    TODO: In our settings, as long as there is one unsat, we can stop
      However, this algorithm stops until "all the remaining ones are UNSAT (which can have one of more instances)"
    """
    f = z3.BoolVal(False)
    for i in range(len(res_label)):
        if res_label[i] == 2:
            f = z3.Or(f, cnt_list[i])
    if z3.is_false(f):
        return

    # sol = z3.SolverFor("QF_BV")
    sol = z3.Solver()
    g = z3.And(precond, f)
    sol.add(g)
    s_res = sol.check()
    if s_res == z3.unsat:
        for i in range(len(res_label)):
            if res_label[i] == 2:
                res_label[i] = 0
    elif s_res == z3.sat:
        m = sol.model()
        # models.append(m)  # counterexample
        for i in range(len(res_label)):
            if res_label[i] == 2 and z3.is_true(m.eval(cnt_list[i], True)):
                res_label[i] = 1
    else:
        return
    compact_check_misc(precond, cnt_list, res_label)


def compact_check(precond: z3.BoolRef, cnt_list: List[z3.BoolRef]) -> List[int]:
    """
    Given a precond G and a set of cnts: f1, f2,..., fn
    Decis if the following cnts are satisfiable:
           And(G, f1), And(G, f2), ..., And(G, fn)
    Examples:
    >>> from z3 import *
    >>> x, y = Reals('x y')
    >>> pre = x > 100
    >>> f1 = x > 0; f2 = And(x > y, x < y); f3 = x < 3
    >>> cnts = [f1, f2, f3]
    >>> assert compact_check(pre, cnts) == [1, 0, 0]
    """
    res = []
    for _ in cnt_list:
        res.append(2)
    compact_check_misc(precond, cnt_list, res)
    return res


def prime_implicant(ps: List[z3.ExprRef], expr: z3.ExprRef):
    """
    TODO: this function may have flaws (need to figure why the assertion below fails
    """
    s = z3.Solver()
    # we want to find a subset ps' of ps such that /\ ps => e
    s.add(z3.Not(expr))
    # holds temp bool vars for unsat core
    bs = []
    # map from temp vars to predicates
    btop = {}
    i = 0
    for p in ps:
        bp = z3.Bool("b" + str(i))
        btop[bp] = p
        bs.append(bp)
        s.add(z3.Implies(bp, p))
        i = i + 1
    assert (s.check(bs) == z3.unsat)
    # only take predicates in unsat core
    res = [btop[x] for x in s.unsat_core()]
    return res


def to_dnf(phi: z3.BoolRef, maxlen=None):
    s = z3.Solver()
    s.add(phi)
    # get all predicates in phi
    preds = get_atoms(phi)
    # disjuncts
    # phiprime = phi
    res = []

    while s.check() == z3.sat:
        # get model
        m = s.model()
        # evaluate model --> get disjunct
        dd = eval_predicates(m, list(preds))
        # print("size before", len(d))

        # get prime implicant of disjunct
        dd = prime_implicant(dd, phi)
        # print("size after", len(d))

        # asserthe negation of disjunct to avoid getting it again
        s.add(z3.Not(big_and(dd)))
        # phiprime = And(phiprime,Not(bigAnd(d)))

        res = res + [big_and(dd)]
        if maxlen is not None:
            if len(res) > maxlen:
                return []
    # NOTE: sanity checking code, ensures DNF is equivalent to phi
    # resphi = Or(*res)
    # assert is_equivalent(resphi, phi)

    # return dnf as list
    return res


def exclusive_to_dnf(phi: z3.BoolRef, maxlen=None):
    s = z3.Solver()
    s.add(phi)
    # get all predicates in phi
    preds = get_atoms(phi)
    # disjuncts
    phiprime = phi
    res = []
    while s.check() == z3.sat:
        # get model
        m = s.model()
        # evaluate model --> get disjunct
        dd = eval_predicates(m, list(preds))
        # print("size before", len(d))
        # get prime implicant of disjunct
        dd = prime_implicant(dd, phiprime)
        # print("size after", len(d))
        # asserthe negation of disjunct to avoid getting it again
        s.add(z3.Not(big_and(dd)))
        phiprime = z3.And(phiprime, z3.Not(big_and(dd)))
        new_entry = big_and(dd)
        res = res + [new_entry]
        if maxlen is not None:
            if len(res) > maxlen:
                return []

    # NOTE: sanity checking code, ensures DNF is equivalent to phi
    # resphi = Or(*res)
    # assert is_equivalent(resphi, phi)
    return res
    # return dnf as list


def get_models(f: z3.BoolRef, k: int):
    """
    Returns the first k models satisfiying f.
    If f is not satisfiable, returns False.
    If f cannot be solved, returns None
    If f is satisfiable, returns the first k models
    Note that if f is a tautology, e.g. True, then the result is []
    Based on http://stackoverflow.com/questions/11867611/z3py-checking-all-solutions-for-equation
    EXAMPLES:
    >>> from z3 import *
    >>> x, y = Ints('x y')
    >>> len(get_models(And(0<=x,x <= 4),k=11))
    5
    >>> get_models(And(0<=x**y,x <= 1),k=2) is None
    True
    >>> get_models(And(0<=x,x <= -1),k=2)
    False
    >>> len(get_models(x+y==7,5))
    5
    >>> len(get_models(And(x<=5,x>=1),7))
    5
    >>> get_models(And(x<=0,x>=5),7)
    False
    >>> x = Bool('x')
    >>> get_models(And(x,Not(x)),k=1)
    False
    >>> get_models(Implies(x,x),k=1)
    []
    """

    assert z3.is_expr(f), f
    assert k >= 1, k

    s = z3.Solver()
    s.add(f)
    models = []
    i = 0
    while s.check() == z3.sat and i < k:
        i = i + 1
        m = s.model()

        if not m:  # if m == []
            break
        models.append(m)
        # create new constraint to block the current model
        block = z3.Not(z3.And([var() == m[var] for var in m]))
        s.add(block)

    if s.check() == z3.unknown:
        return None
    elif s.check() == z3.unsat and i == 0:
        return False
    else:
        return models


if __name__ == "__main__":
    import doctest

    doctest.testmod()
