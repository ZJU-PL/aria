"""
Randomly generate a formula using z3's Python APIs

NOTE: This file is a quite simplified implementation
      For generating more diverse and complex queries,
      please refer to grammar_gene.py
"""
import random

import z3


class FormulaGenerator:
    """A class for generating formulas"""

    def __init__(self, init_vars, bv_signed=True,
                 bv_no_overflow=False, bv_no_underflow=False):
        self.bools = []
        self.use_int = False
        self.ints = []
        self.use_real = False
        self.reals = []

        self.use_bv = False
        self.bvs = []
        # hard_bools are the cnts that must enforced
        # e.g., to enforce the absence of overflow and underflow!
        self.hard_bools = []
        self.bv_signed = bv_signed
        self.bv_no_overflow = bv_no_overflow
        self.bv_no_underflow = bv_no_underflow

        var_lists = [(z3.is_int, self.ints, self, 'use_int'),
                      (z3.is_real, self.reals, self, 'use_real'),
                      (z3.is_bv, self.bvs, self, 'use_bv')]

        for var in init_vars:
            for check, lst, obj, flag in var_lists:
                if check(var):
                    lst.append(var)
                    setattr(obj, flag, True)
                    break

        if self.use_int:
            self.ints.extend([FormulaGenerator.random_int() for _ in range(random.randint(3, 6))])

        if self.use_real:
            self.reals.extend([FormulaGenerator.random_real() for _ in range(random.randint(3, 6))])

        if self.use_bv:
            bvsort = self.bvs[0].sort()
            self.bvs.extend([z3.BitVecVal(random.randint(1, 100), bvsort.size())
                           for _ in range(random.randint(3, 6))])

    @staticmethod
    def random_int():
        return z3.IntVal(random.randint(-100, 100))

    @staticmethod
    def random_real():
        return z3.RealVal(random.randint(-100, 100))

    def _arith_from_vars(self, vars, ops):
        if len(vars) >= 2:
            v1, v2 = random.sample(vars, 2)
            vars.append(random.choice(ops)(v1, v2))

    def int_from_int(self):
        ops = [lambda a, b: a + b, lambda a, b: a - b, lambda a, b: a * b,
               lambda a, b: a / b, lambda a, b: a % b]
        self._arith_from_vars(self.ints, ops)

    def real_from_real(self):
        ops = [lambda a, b: a + b, lambda a, b: a - b, lambda a, b: a * b, lambda a, b: a / b]
        self._arith_from_vars(self.reals, ops)

    def bv_from_bv(self):
        """
        TODO: More bit-vec operations!!
        """
        if len(self.bvs) >= 2:
            data = random.sample(self.bvs, 2)
            r1 = data[0]
            r2 = data[1]
            # [+, -, *, /]
            prob = random.random()
            if prob <= 0.25:
                self.bvs.append(r1 + r2)
                if self.bv_no_overflow:
                    self.hard_bools.append(z3.BVAddNoOverflow(r1, r2, signed=self.bv_signed))
                if self.bv_no_underflow:
                    self.hard_bools.append(z3.BVAddNoUnderflow(r1, r2))
            elif prob <= 0.5:
                self.bvs.append(r1 - r2)
                if self.bv_no_underflow:
                    self.hard_bools.append(z3.BVSubNoOverflow(r1, r2))
                if self.bv_no_underflow:
                    self.hard_bools.append(z3.BVSubNoUnderflow(r1, r2, signed=self.bv_signed))
            elif prob <= 0.75:
                self.bvs.append(r1 * r2)
                if self.bv_no_underflow:
                    self.hard_bools.append(z3.BVMulNoOverflow(r1, r2, signed=self.bv_signed))
                if self.bv_no_underflow:
                    self.hard_bools.append(z3.BVMulNoUnderflow(r1, r2))
            else:
                self.bvs.append(r1 / r2)
                if self.bv_signed:
                    self.hard_bools.append(z3.BVSDivNoOverflow(r1, r2))

    def _bool_from_vars(self, vars):
        if len(vars) >= 2:
            v1, v2 = random.sample(vars, 2)
            ops = [v1 < v2, v1 <= v2, v1 == v2, v1 > v2, v1 >= v2, v1 != v2]
            self.bools.append(random.choice(ops))

    def bool_from_int(self):
        self._bool_from_vars(self.ints)

    def bool_from_real(self):
        self._bool_from_vars(self.reals)

    def bool_from_bv(self):
        if len(self.bvs) >= 2:
            bv1, bv2 = random.sample(self.bvs, 2)
            if not self.bv_signed:
                ops = [z3.ULT(bv1, bv2), z3.ULE(bv1, bv2), bv1 == bv2,
                       z3.UGT(bv1, bv2), z3.UGE(bv1, bv2), bv1 != bv2]
            else:
                ops = [bv1 < bv2, bv1 <= bv2, bv1 == bv2,
                       bv1 > bv2, bv1 >= bv2, bv1 != bv2]
            self.bools.append(random.choice(ops))

    def bool_from_bool(self):
        if len(self.bools) >= 2:
            if random.random() < 0.22:
                self.bools.append(z3.Not(random.choice(self.bools)))
                return

            b1, b2 = random.sample(self.bools, 2)
            ops = [z3.And, z3.Or, z3.Xor, z3.Implies]
            self.bools.append(random.choice(ops)(b1, b2))

    def generate_formula(self):
        type_methods = [(self.use_int, self.bool_from_int),
                       (self.use_real, self.bool_from_real),
                       (self.use_bv, self.bool_from_bv)]

        # Generate initial boolean expressions
        for _ in range(random.randint(3, 8)):
            for use, method in type_methods:
                if use: method()

        # Generate more complex formulas
        for _ in range(8):
            if random.random() < 0.33:
                for use, method in [(self.use_int, self.int_from_int),
                                   (self.use_real, self.real_from_real),
                                   (self.use_bv, self.bv_from_bv)]:
                    if use: method()

            if random.random() < 0.33:
                for use, method in type_methods:
                    if use: method()

            if random.random() < 0.33:
                self.bool_from_bool()

        # Generate random clauses
        max_assert = random.randint(5, 30)
        res = []
        assert len(self.bools) >= 1

        for _ in range(max_assert):
            clen = random.randint(1, 8)
            sample_size = min(len(self.bools), clen)
            cls = random.choice(self.bools) if clen == 1 else z3.Or(random.sample(self.bools, sample_size))
            res.append(cls)

        res.extend(self.hard_bools)
        return res[0] if len(res) == 1 else z3.And(res)

    def generate_formula_as_str(self):
        mutant = self.generate_formula()
        sol = z3.Solver()
        sol.add(mutant)
        smt2_string = sol.to_smt2()
        return smt2_string

    def get_preds(self, k):
        """"""
        res = []
        for _ in range(k):
            res.append(random.choice(self.bools))
        return res


if __name__ == "__main__":
    w, x, y, z = z3.Ints("w x y z")
    # r = Real("r")
    test = FormulaGenerator([w, x, y, z])
    # x, y, z = BitVecs("x y z", 16)
    # test = FormulaGenerator([x, y, z])
    print(test.generate_formula_as_str())
