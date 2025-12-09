#!/usr/bin/env python3
# coding: utf-8
#
# SMT version of SearchTreeSample procedure from 
# "Uniform Solution Sampling Using a Constraint Solver As an Oracle" 
# (Ermon et al UAI 2012)
# 
# Using the Python interface to the Z3 theorem prover
# http://research.microsoft.com/en-us/um/redmond/projects/z3/documentation.html
#
from z3 import *
from random import randint

from aria.utils.z3_expr_utils import get_variables

g_number_smtcall = 0


def uniform_select(xs):
    """Uniform sampling from a list."""
    n = len(xs) - 1
    i = randint(0, n)
    return xs[i]


def sample_without_replacement(k, xsc):
    """Samples K elements form XSC without replacement."""
    xs = list(xsc)
    ans = []

    while (k > 0) and (xsc != []):
        i = randint(0, len(xsc) - 1)
        ans.append(xsc.pop(i))
        k -= 1

    return ans


def permutation(xs):
    return sample_without_replacement(len(xs), xs)


def findall_var(formula, variable):
    """Enumerate models of FORMULA covering all possible assignments
    to VARIABLE."""
    global g_number_smtcall
    res = []
    s = Solver()
    s.add(formula)
    while True:
        g_number_smtcall += 1
        if s.check() == sat:
            m = s.model()
            res.append(m)
            value = m[variable]
            if value is None:
                return res
            s.add(variable != value)
        else:
            return res


def project_soln(variables, model):
    """Given a list of VARIABLES that occur in MODEL,
    produce a conjunction restricting those variables
    to their values in the MODEL."""
    if not variables:
        return True
    res = []
    for variable in variables:
        res.append(variable == model[variable])
    return And(*res)


def black_box_sample(formula, prev_solns, samples, vars_used, next_var):
    """Samples approximately uniformly from the set of solutions to FORMULA
    projected down to [NEXT_VAR] + VARS_USED, 
    given a list of PREV_SOLNS. 
    
    SAMPLES is a parameter controlling the uniformity."""

    num_to_sample = min(samples, len(prev_solns))
    ancestors = sample_without_replacement(num_to_sample, prev_solns)

    res = []
    for soln in ancestors:
        ancestor_constraint = project_soln(vars_used, soln)
        res.extend(findall_var(And(formula, ancestor_constraint), next_var))
    # print(res)
    return res


def search_tree_sample(variables, formula, samples):
    # print("I am called!!")
    """Produes approximately uniform samples from the set of solutions to FORMULA
    (with VARIALBES). 
    
    SAMPLES is a parameter controlling the uniformity."""

    to_use = list(variables)
    used_vars = []
    solns = [None]

    while to_use:
        next_var = to_use[0]
        to_use = to_use[1:]

        solns = black_box_sample(formula, solns, samples, used_vars, next_var)
        used_vars.append(next_var)

    # print("Solutions: ")
    # print(solns)

    return uniform_select(solns)


def histogram(f, samples=2):
    res = {}
    i = 0
    total = 0
    while i < samples:
        v = f()
        i += 1

        res[v] = res.get(v, 0) + 1
        total += 1.0
    for k in res:
        res[k] /= total

    return res


def test():
    x = Int('x')
    y = Int('y')
    formula = And(x > 0, x < 10, y > 0, y < 10)

    # x = BitVec('x', 8)
    # y = BitVec('y', 8)
    # formula = And(UGT(x, 0), ULT(x, 10), UGT(y, 0), ULT(y, 10))

    def sampler():
        # 2 is the parameter controlling uniformanity
        smp = search_tree_sample([x, y], formula, 2)
        res = tuple(map(lambda v: (v.name(), str(smp[v])), smp))
        return res

    print(histogram(sampler))
    print("Num. SMT calls: ", g_number_smtcall)


def searchtree_sampler_for_file(fname):
    try:
        fvec = parse_smt2_file(fname)
        formula = And(fvec)
        vars = []
        for v in get_variables(formula):
            vars.append(v)
        print("start")
        # 2 is the parameter controlling uniformanity
        smp = search_tree_sample(vars, formula, 2)
        res = tuple(map(lambda v: (v.name(), str(smp[v])), smp))
        print(res)

    except Z3Exception as e:
        print(e)

    return


if __name__ == '__main__':
    # searchtree_sampler_for_file('../test/t1.smt2')
    # searchtree_sampler_for_file(("../../smt-benchmark/sampling/qsym/nm/case1798378065.smt2"))
    test()
