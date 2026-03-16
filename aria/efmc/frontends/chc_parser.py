"""
Parser for Constraint Horn Clause (CHC) files based on Z3's Python API

NOTE: this file provides a few other functionalities. For example,
    we also support replacing "inv" by a function body
"""

from __future__ import print_function

import z3


def ground_quantifier(qexpr):
    """
    Seems this can only handle exists x . fml, or forall x.fml?
    """
    body = qexpr.body()
    var_list = list()
    for i in range(qexpr.num_vars()):
        vi_name = qexpr.var_name(i)
        # print(vi_name)
        vi_sort = qexpr.var_sort(i)
        vi = z3.Const(vi_name, vi_sort)
        var_list.append(vi)

    list.reverse(var_list)  # TODO: is this correct??
    # print(var_list)
    body = z3.substitute_vars(body, *var_list)
    # print(body, var_list)
    return body, var_list


class CHCParser:
    """
    NOTE: we assume that the input strictly follows the SyGuS(invariant) style
        The first assertion: pre
        The second assertion: trans
        The third assertion (or "goal"): post
     Besides, the goal should be in the form of
         inv => post
     Instead of
         inv and P => post (this can be transformed to the above form)
    """

    fmls = []

    def __init__(self, inputs: str, to_real: bool):
        if to_real:
            self.to_real = True
        else:
            self.to_real = False
        self.parse_chc_string(inputs)

    def parse_chc_string(self, chc: str):
        self.fmls = z3.parse_smt2_string(chc)
        assert len(self.fmls) == 3

    def solve_with_pdr(self):
        # for debugging
        sol = z3.SolverFor("HORN")
        sol.add(self.fmls)
        print(sol)
        print("Solving with CHC")
        print(sol.check())

    def get_transition_system(self):
        # TODO: remove "inv"? (Besides, ground_quantifier may change the var order...)
        #  Also, ground_quantifier can be slow
        # from z3.z3util import get_vars
        init, _ = ground_quantifier(self.fmls[0])
        trans, vars_trans = ground_quantifier(self.fmls[1])
        post, _ = ground_quantifier(self.fmls[2])

        pure_init = init.children()[0]
        pure_trans = z3.And(trans.children()[0].children()[1:])
        if z3.is_and(post.children()[0]):
            # e.g., in the form of Implies(And(inv(i), i >= 10), i == 10)
            pure_post = z3.simplify(
                z3.Implies(
                    z3.And(post.children()[0].children()[1:]), post.children()[1]
                )
            )
        else:
            # e.g., in the form of Implies(inv(i), i == 10)
            pure_post = z3.simplify(post.children()[1])
        # print(pure_init, "\n", pure_trans, "\n", pure_post)
        # all_vars = get_vars(trans)
        # TODO: in some cases, vars_trans may not use all the variables?
        return vars_trans, pure_init, pure_trans, pure_post


def test_parse():
    s = z3.SolverFor("HORN")
    inv = z3.Function("inv", z3.BitVecSort(8), z3.BoolSort())
    i, ip = z3.BitVecs("i ip", 8)
    # init
    s.add(z3.ForAll([i], z3.Implies(i == 0, inv(i))))
    # inductive
    s.add(z3.ForAll([i, ip], z3.Implies(z3.And(inv(i), i < 10, ip == i + 1), inv(ip))))
    # sufficient

    s.add(z3.ForAll([i], z3.Implies(inv(i), z3.Implies(i >= 10, i == 10))))

    print(s.to_smt2())


def test_parse2():
    fml = """
(declare-fun inv ((_ BitVec 8)) Bool)
(assert
 (forall ((i (_ BitVec 8)) )(let (($x12 (inv i)))
 (let (($x13 (= i (_ bv0 8))))
 (=> $x13 $x12))))
 )
(assert
 (forall ((i (_ BitVec 8)) (ip (_ BitVec 8)) )(let (($x12 (inv ip)))
 (=> (and (inv i) (bvslt i (_ bv10 8)) (= ip (bvadd i (_ bv1 8)))) $x12)))
 )
(assert
 (forall ((i (_ BitVec 8)) )(let (($x12 (inv i)))
 (=> (and $x12 (bvsge i (_ bv10 8))) (= i (_ bv10 8)))))
 )
(check-sat)
    """
    chc = CHCParser(inputs=fml, to_real=False)
    chc.parse_chc_string(fml)
    chc.get_transition_system()


def test_parse3(filename):
    with open(filename, "r", encoding="utf-8") as f:
        res = f.read()
        ss = CHCParser(res, to_real=False)
        vars_trans, init, trans, post = ss.get_transition_system()
        print(vars_trans)
        # ss.solve_with_pdr()


if __name__ == "__main__":
    # test_parse()
    test_parse3("../../data/Regression/chc_bv/bv/array_max-1.smt2")
    # test_replace()
