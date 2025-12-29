#!/usr/bin/env python3
# coding: utf-8
"""
Region-based sampler for bit-vector formulas.

This module provides a sampler that uses region-based sampling by computing
bounds for variables and sampling within those bounds.
"""

import random

from z3 import (
    parse_smt2_file, Z3Exception, Model, BitVecVal, Optimize,
    is_true, And
)

from aria.utils.z3_expr_utils import get_variables


# from random import *


class RegionSampler:
    """Region-based sampler for bit-vector formulas."""

    def __init__(self):
        self.formula = []
        self.vars = []
        self.inputs = []
        self.valid = 0
        self.unique = 0

        self.lower_bounds = []
        self.upper_bounds = []

    def parse_and_init(self, fname):
        """Parse and initialize from an SMT2 file."""
        try:
            self.formula = parse_smt2_file(fname)
        except Z3Exception as exc:
            print(exc)
            return None

        self.vars = get_variables(self.formula)
        for _ in self.vars:
            self.lower_bounds.append(0)
            self.upper_bounds.append(255)

    def check_model(self, candidate):
        """Check if a candidate model satisfies the formula."""
        m = Model()

        for idx, var in enumerate(self.vars):
            # Note: add_const_interp may not be supported in all Z3 versions
            # This is a placeholder for the intended API
            # m.add_const_interp(var, BitVecVal(candidate[idx], 8))
            pass

        if is_true(m.eval(self.formula)):
            return True
        return None

    def compute_bounds(self):
        """Compute bounds for all variables using optimization."""
        # TODO: use multi-obj optimization
        for idx, var in enumerate(self.vars):
            sol = Optimize()
            sol.add(self.formula)
            sol.minimize(var)
            sol.check()
            m = sol.model()
            self.lower_bounds[idx] = m.eval(var).as_long()

            sol2 = Optimize()
            sol2.add(self.formula)
            sol2.maximize(var)
            sol2.check()
            m2 = sol2.model()
            self.upper_bounds[idx] = m2.eval(var).as_long()
            print(var, "[", self.lower_bounds[idx], ", ",
                  self.upper_bounds[idx], "]")

    def gen_candidate(self):
        """Generate a random candidate within the computed bounds."""
        candidate = []
        for idx in range(len(self.vars)):
            r = random.randint(self.lower_bounds[idx], self.upper_bounds[idx])
            candidate.append(r)

        print(candidate)
        return candidate

    def feat_test(self):
        """Feature test method."""
        from z3 import BitVec  # pylint: disable=import-outside-toplevel
        x = BitVec("x", 8)
        y = BitVec("y", 8)
        self.formula = And(x > 0, y > 0, x < 10, y < 10)
        self.vars = get_variables(self.formula)
        for _ in self.vars:
            self.lower_bounds.append(0)
            self.upper_bounds.append(255)

        self.compute_bounds()

        print(self.check_model(self.gen_candidate()))


tt = RegionSampler()
tt.feat_test()
