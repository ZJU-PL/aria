from z3 import *
import numpy as np
from functools import reduce
import psitip


def GenerateFuncInput(x, y, z):
    assert x.size() == y.size() == z.size()
    return Concat(z, y, x)


def Decompose(rep):
    assert rep.size() % 3 == 0
    return ExtractX(rep), ExtractY(rep), ExtractZ(rep)


def ExtractX(rep):
    assert rep.size() % 3 == 0
    var_num = rep.size() // 3
    return Extract(var_num - 1, 0, rep)


def ExtractY(rep):
    assert rep.size() % 3 == 0
    var_num = rep.size() // 3
    return Extract(2 * var_num - 1, var_num, rep)


def ExtractZ(rep):
    assert rep.size() % 3 == 0
    var_num = rep.size() // 3
    return Extract(3 * var_num - 1, 2 * var_num, rep)


def BV_Union(x, y):
    assert x.size() == y.size()
    return x | y


def BV_Intersection(x, y, z=None):
    assert x.size() == y.size()
    if z == None:
        return x & y
    else:
        return (x & y) & z


def BV_Subset(x, y):
    assert x.size() == y.size()
    # return the condition that x is a subset of y
    return BV_Intersection(x, y) == x


def BV_IsSingle(x):
    var_num = x.size()
    val_size = int(np.log2(var_num))
    bits = [Extract(i, i, x) for i in range(var_num)]
    bvs = [Concat(BitVecVal(0, val_size), b) for b in bits]
    return Sum(bvs) == BitVecVal(1, val_size + 1)


def BV_Membership(x, y):
    # y must be a single element set
    return And([BV_IsSingle(y), BV_Intersection(y, x)])


def BV_GetEmptySet(size):
    return BitVecVal(0, size)


def IsEmpty(x):
    return x == BitVecVal(0, x.size())


def GenerateBitVal(x: set, var_num: int):
    bv_x = ["0"] * var_num
    for v in x:
        bv_x[v] = "1"
    bv_x_str = "".join(bv_x)
    return BitVecVal(str(int(bv_x_str, base=2)), var_num)


class CIStatement:
    def __init__(self, x: set, y: set, z: set, ci: bool) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.ci = ci
        assert (
            len(self.x) > 0 and len(self.y) > 0
        ), "CIStatement must have non-empty x and y"  # ToM: Right?

    def is_marginal(self):
        return len(self.z) == 0

    def generate_constraint(self, ci_euf: FuncDeclRef, var_num: int):
        val = BitVecVal(1, 2) if self.ci else BitVecVal(0, 2)
        return (
            ci_euf(
                GenerateBitVal(self.x, var_num),
                GenerateBitVal(self.y, var_num),
                GenerateBitVal(self.z, var_num),
            )
            == val
        )

    def within(self, nodes: set):
        return self.x in nodes or self.y in nodes or self.z in nodes

    def has_overlap(self, ci):
        ci: CIStatement
        return not (
            self.x.union(self.y).union(self.z).isdisjoint(ci.x.union(ci.y).union(ci.z))
        )

    def get_negation(self):
        return CIStatement(self.x, self.y, self.z, not self.ci)

    def is_isomorphic(self, ci: "CIStatement"):
        # ci:CIStatement
        return self.x == ci.x and self.y == ci.y and self.z == ci.z

    def is_form_equal(self, ci: "CIStatement"):
        if self.z == ci.z:
            if self.x == ci.x and self.y == ci.y:
                return True
            elif self.x == ci.y and self.y == ci.x:
                return True
        return False

    def is_equal(self, ci: "CIStatement"):
        return self.ci == ci.ci and self.is_form_equal(ci)

    def is_negation(self, ci: "CIStatement"):
        return self.ci != ci.ci and self.is_form_equal(ci)

    def graphoid_expr(self, graphoid_variables: list):

        if self.ci:
            return self.graphoid_term(graphoid_variables) == 0
        else:
            return self.graphoid_term(graphoid_variables) < 0

    def graphoid_term(self, graphoid_variables: list):
        x_list = list(self.x)
        if len(self.x) == 1:
            g_x = graphoid_variables[list(self.x)[0]]
        else:
            g_x = graphoid_variables[list(self.x)[0]]
            for i in range(1, len(self.x)):
                g_x = g_x + graphoid_variables[x_list[i]]

        y_list = list(self.y)
        if len(self.y) == 1:
            g_y = graphoid_variables[list(self.y)[0]]
        else:
            g_y = graphoid_variables[list(self.y)[0]]
            for i in range(1, len(self.y)):
                g_y = g_y + graphoid_variables[y_list[i]]

        if len(self.z) == 0:
            return psitip.I(g_x & g_y)
        else:
            z_list = list(self.z)
            if len(self.z) == 1:
                g_z = graphoid_variables[z_list[0]]
            else:
                g_z = graphoid_variables[z_list[0]]
                for i in range(1, len(self.z)):
                    g_z = g_z + graphoid_variables[z_list[i]]

            return psitip.I(g_x & g_y | g_z)

    def __str__(self) -> str:
        if self.ci:
            return f"{self.x} _|_ {self.y} | {self.z}"
        else:
            return f"{self.x} ~ {self.y} | {self.z}"

    @staticmethod
    def create(fact: tuple):
        x, y, z, ci = fact
        if isinstance(x, int):
            x = {x}
        if isinstance(y, int):
            y = {y}
        if isinstance(z, int):
            z = {z}
        return CIStatement(x, y, z, ci)

    @staticmethod
    def createByXYZ(x: int, y: int, z: set, ci: bool):
        if isinstance(x, int):
            x = {x}
        if isinstance(y, int):
            y = {y}
        if isinstance(z, int):
            z = {z}
        return CIStatement(x, y, z, ci)
