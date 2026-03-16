# coding: utf-8
import logging
import z3

from aria.efmc.tests import TestCase, main
from aria.efmc.engines.predabs import PredicateAbstractionProver
from aria.efmc.sts import TransitionSystem


class TestPredicateAbstraction(TestCase):

    def test_bool_program(self):
        # return
        x, y, xp, yp = z3.Bools("x y x! y!")
        init = z3.And(x, y)
        trans = z3.And(
            z3.Implies(y, xp == y, yp == y),
            z3.Implies(z3.Not(y), xp == z3.Not(y), yp == y),
        )
        post = x
        preds = [x, y]

        all_vars = [x, y, xp, yp]
        sts = TransitionSystem()
        sts.from_z3_cnts([all_vars, init, trans, post])

        pp = PredicateAbstractionProver(sts)
        pp.set_predicates(preds)
        pp.solve()

    def test_arith_program(self):
        x, y, z, xp, yp, zp = z3.Reals("x y z x! y! z!")
        init = x == 0
        trans = z3.And(x < 10, xp == x + 1)
        post = z3.Implies(x >= 10, x == 10)
        # preds = [x == 10, x > 10]
        # preds = [x >= 0, x == y]
        preds = [x < 10, x == 10]
        all_vars = [x, xp]
        sts = TransitionSystem()
        sts.from_z3_cnts([all_vars, init, trans, post])

        pp = PredicateAbstractionProver(sts)
        pp.set_predicates(preds)
        pp.solve()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
