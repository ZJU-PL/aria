"""
Randomly generate a formula using z3's Python APIs

NOTE: This file is a quite simplified implementation
      For generating more diverse and complex queries,
      please refer to grammar_gene.py
"""
import random

import z3


class FormulaGenerator:
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

        for var in init_vars:
            if z3.is_int(var):
                self.ints.append(var)
            elif z3.is_real(var):
                self.reals.append(var)
            elif z3.is_bv(var):
                self.bvs.append(var)

        if len(self.ints) > 0:
            self.use_int = True
            for _ in range(random.randint(3, 6)):
                self.ints.append(FormulaGenerator.random_int())

        if len(self.reals) > 0:
            self.use_real = True
            for _ in range(random.randint(3, 6)):
                self.reals.append(FormulaGenerator.random_real())

        if len(self.bvs) > 0:
            self.use_bv = True
            bvsort = self.bvs[0].sort()
            for _ in range(random.randint(3, 6)):
                self.bvs.append(z3.BitVecVal(random.randint(1, 100), bvsort.size()))

    @staticmethod
    def random_int():
        return z3.IntVal(random.randint(-100, 100))

    @staticmethod
    def random_real():
        return z3.IntVal(random.randint(-100, 100))

    def int_from_int(self):
        # TODO: also use constant
        if len(self.ints) >= 2:
            # while True:
            #     data = random.sample(self.ints, 2)
            #     i1 = data[0]
            #     i2 = data[1]
            #     if not (z3.is_int_value(i1) and z3.is_int_value(i2)):
            #         break
            data = random.sample(self.ints, 2)
            i1 = data[0]
            i2 = data[1]
            # [+, -, *, /, mod]
            prob = random.random()
            if prob <= 0.2:
                self.ints.append(i1 + i2)
            elif prob <= 0.4:
                self.ints.append(i1 - i2)
            elif prob <= 0.6:
                self.ints.append(i1 * i2)
            elif prob <= 0.8:
                self.ints.append(i1 / i2)
            else:
                # is this OK?
                self.ints.append(i1 % i2)

    def real_from_real(self):
        if len(self.reals) >= 2:
            data = random.sample(self.reals, 2)
            r1 = data[0]
            r2 = data[1]
            # [+, -, *, /]
            prob = random.random()
            if prob <= 0.25:
                self.reals.append(r1 + r2)
            elif prob <= 0.5:
                self.reals.append(r1 - r2)
            elif prob <= 0.75:
                self.reals.append(r1 * r2)
            else:
                self.reals.append(r1 / r2)

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

    def bool_from_int(self):
        if len(self.ints) >= 2:
            # while True:
            #     data = random.sample(self.ints, 2)
            #     i1 = data[0]
            #     i2 = data[1]
            #     if not (z3.is_int_value(i1) and z3.is_int_value(i2)):
            #         break
            data = random.sample(self.ints, 2)
            i1 = data[0]
            i2 = data[1]
            # [<, <=, ==,  >, >=, !=]
            prob = random.random()
            if prob <= 0.16:
                new_bool = i1 < i2
            elif prob <= 0.32:
                new_bool = i1 <= i2
            elif prob <= 0.48:
                new_bool = i1 == i2
            elif prob <= 0.62:
                new_bool = i1 > i2
            elif prob <= 0.78:
                new_bool = i1 >= i2
            else:
                new_bool = i1 != i2
            self.bools.append(new_bool)

    def bool_from_real(self):
        if len(self.reals) >= 2:
            data = random.sample(self.reals, 2)
            i1 = data[0]
            i2 = data[1]
            # [<, <=, ==,  >, >=, !=]
            prob = random.random()
            if prob <= 0.16:
                new_bool = i1 < i2
            elif prob <= 0.32:
                new_bool = i1 <= i2
            elif prob <= 0.48:
                new_bool = i1 == i2
            elif prob <= 0.62:
                new_bool = i1 > i2
            elif prob <= 0.78:
                new_bool = i1 >= i2
            else:
                new_bool = i1 != i2
            self.bools.append(new_bool)

    def bool_from_bv(self):
        unsigned = not self.bv_signed
        if len(self.bvs) >= 2:
            data = random.sample(self.bvs, 2)
            bv1 = data[0]
            bv2 = data[1]
            prob = random.random()
            # print(bv1.sort(), bv2.sort())
            if prob <= 0.16:
                if unsigned:
                    new_bv = z3.ULT(bv1, bv2)
                else:
                    new_bv = bv1 < bv2
            elif prob <= 0.32:
                if unsigned:
                    new_bv = z3.ULE(bv1, bv2)
                else:
                    new_bv = bv1 <= bv2
            elif prob <= 0.48:
                new_bv = bv1 == bv2
            elif prob <= 0.62:
                if unsigned:
                    new_bv = z3.UGT(bv1, bv2)
                else:
                    new_bv = bv1 > bv2
            elif prob <= 0.78:
                if unsigned:
                    new_bv = z3.UGE(bv1, bv2)
                else:
                    new_bv = bv1 >= bv2
            else:
                new_bv = bv1 != bv2
            self.bools.append(new_bv)

    def bool_from_bool(self):
        if len(self.bools) >= 2:
            if random.random() < 0.22:
                b = random.choice(self.bools)
                self.bools.append(z3.Not(b))
                return

            data = random.sample(self.bools, 2)
            b1 = data[0]
            b2 = data[1]
            # [and, or, xor, implies]
            prob = random.random()
            if prob <= 0.25:
                self.bools.append(z3.And(b1, b2))
            elif prob <= 0.5:
                self.bools.append(z3.Or(b1, b2))
            elif prob <= 0.75:
                self.bools.append(z3.Xor(b1, b2))
            else:
                self.bools.append(z3.Implies(b1, b2))

    def generate_formula(self):

        for i in range(random.randint(3, 8)):
            if self.use_int:
                self.bool_from_int()
            if self.use_real:
                self.bool_from_real()
            if self.use_bv:
                self.bool_from_bv()

        for i in range(8):
            if random.random() < 0.33:
                if self.use_int:
                    self.int_from_int()
                if self.use_real:
                    self.real_from_real()
                if self.use_bv:
                    self.bv_from_bv()

            if random.random() < 0.33:
                if self.use_int:
                    self.bool_from_int()
                if self.use_real:
                    self.bool_from_real()
                if self.use_bv:
                    self.bool_from_bv()

            if random.random() < 0.33:
                self.bool_from_bool()

        max_assert = random.randint(5, 30)
        res = []
        assert (len(self.bools) >= 1)
        for _ in range(max_assert):
            clen = random.randint(1, 8)  # clause length
            if clen == 1:
                cls = random.choice(self.bools)
            else:
                cls = z3.Or(random.sample(self.bools, min(len(self.bools), clen)))
            res.append(cls)

        if len(self.hard_bools) > 1:
            res += self.hard_bools

        if len(res) == 1:
            return res[0]
        else:
            return z3.And(res)

    def generate_formula_as_str(self):
        mutant = self.generate_formula()
        sol = z3.Solver()
        sol.add(mutant)
        smt2_string = sol.to_smt2()
        return smt2_string

    def get_preds(self, k):
        res = []
        for _ in range(k):
            res.append(random.choice(self.bools))
        return res


# if __name__ == "__main__":
#     w, x, y, z = z3.Ints("w x y z")
#     test = FormulaGenerator([w, x, y, z])
#     print(test.generate_formula())
#     print(test.get_preds(random.randint(50, 150)))
