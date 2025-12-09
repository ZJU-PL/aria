"""Quantifier Elimination via Lazy Model Enumeration (LME-QE)"""

from typing import List

import z3

from aria.utils.z3_expr_utils import negate, get_atoms


def eval_predicates(m, preds):
    res = []
    for p in preds:
        if z3.is_true(m.eval(p)):
            res.append(p)
        elif z3.is_false(m.eval(p)):
            res.append(negate(p))
    return res


def process_model(phi, qvars, preds, shared_models):
    s = z3.Solver(); s.add(phi)
    for model in shared_models:
        s.add(negate(model))
    if s.check() == z3.sat:
        m = s.model()
        minterm = z3.And(eval_predicates(m, preds))
        proj = z3.Tactic('qe2')(z3.Exists(qvars, minterm)).as_expr()
        return proj
    return None


def qelim_exists_lme(phi, qvars):
    s = z3.Solver(); s.add(phi)
    res = []
    preds = get_atoms(phi)
    qe_for_conjunction = z3.Tactic('qe2')
    while s.check() == z3.sat:
        m = s.model()
        minterm = z3.And(eval_predicates(m, preds))
        proj = qe_for_conjunction(z3.Exists(qvars, minterm)).as_expr()
        res.append(proj)
        s.add(negate(proj))
    return z3.simplify(z3.Or(res))


def test_qe():
    # x, y, z = z3.BitVecs("x y z", 16)
    x, y, z = z3.Reals("x y z")
    fml = z3.And(z3.Or(x > 2, x < y + 3), z3.Or(x - z > 3, z < 10))  # x: 4, y: 1
    qf = qelim_exists_lme(fml, [x, y])
    print(qf)


if __name__ == "__main__":
    test_qe()
