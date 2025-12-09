#!/usr/bin/env python3
# coding: utf-8

from z3 import *
# from z3.z3util import get_vars
from aria.utils.z3_expr_utils import get_variables
import random


# from random import *


class RegionSampler:
    def __init__(self):
        self.formula = []
        self.vars = []
        self.inputs = []
        self.valid = 0
        self.unique = 0

        self.lower_bounds = []
        self.upper_bounds = []

    def parse_and_init(self, fname):
        try:
            self.formula = parse_smt2_file(fname)
        except Z3Exception as e:
            print(e)
            return None

        self.vars = get_variables(self.formula)
        for i in range(len(self.vars)):
            self.lower_bounds.append(0)
            self.upper_bounds.append(255)

    def check_model(self, canidate):
        m = Model()

        for i in range(len(self.vars)):
            # seems some versions of Z3 do not support this API
            m.add_const_interp(self.vars[i], BitVecVal(canidate[i], 8))

        if is_true(m.eval(self.formula)):
            return True
        return False

    def compute_bounds(self):
        ##TODO: use multi-obj optimization
        for i in range(len(self.vars)):
            sol = Optimize()
            sol.add(self.formula)
            sol.minimize(self.vars[i])
            sol.check()
            m = sol.model()
            self.lower_bounds[i] = m.eval(self.vars[i]).as_long()

            sol2 = Optimize()
            sol2.add(self.formula)
            sol2.maximize(self.vars[i])
            sol2.check()
            m = sol2.model()
            self.upper_bounds[i] = m.eval(self.vars[i]).as_long()
            print(self.vars[i], "[", self.lower_bounds[i], ", ", \
                  self.upper_bounds[i], "]")

    def gen_candidate(self):
        canidate = []
        for i in range(len(self.vars)):
            r = random.randint(self.lower_bounds[i], self.upper_bounds[i])
            canidate.append(r)

        print(canidate)
        return canidate

    def feat_test(self):
        x = BitVec("x", 8)
        y = BitVec("y", 8)
        self.formula = And(x > 0, y > 0, x < 10, y < 10)
        self.vars = get_variables(self.formula)
        for i in range(len(self.vars)):
            self.lower_bounds.append(0)
            self.upper_bounds.append(255)

        self.compute_bounds()

        print(self.check_model(self.gen_candidate()))


tt = RegionSampler()
tt.feat_test()
