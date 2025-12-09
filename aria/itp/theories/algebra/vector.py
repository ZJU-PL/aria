import aria.itp as itp
import aria.itp.smt as smt
import aria.itp.property as prop


# dispatching on the first argument isn't great
# smul = itp.notation.SortDispatch(name="smul")


class VectorSpace(prop.TypeClass):
    key: smt.SortRef
    scalar: smt.SortRef

    add_assoc: itp.Proof
    add_comm: itp.Proof
    zero: smt.ExprRef
    add_zero: itp.Proof
    add_inv: itp.Proof

    smul: smt.FuncDeclRef

    smul_distrib: itp.Proof
    smul_assoc: itp.Proof

    def check(self, T):
        x, y, z = smt.Consts("x y z", T)
        a, b = smt.Consts("a b", self.scalar)
        smul = self.smul
        assert T in itp.notation.add
        assert self.assert_eq(
            self.add_assoc.thm, smt.ForAll([x, y], x + (y + z) == (x + y) + z)
        )
        assert self.assert_eq(self.add_comm.thm, smt.ForAll([x, y], x + y == y + x))
        assert self.assert_eq(self.add_zero.thm, smt.ForAll([x], x + self.zero == x))
        assert self.assert_eq(self.add_inv.thm, smt.ForAll([x], x + -x == self.zero))
        assert self.assert_eq(
            self.smul_distrib.thm, smt.ForAll([a, y, z], a * (y + z) == a * y + a * z)
        )
        assert self.assert_eq(
            self.smul_assoc.thm,
            smt.ForAll([a, b, z], smul(a, smul(b, z)) == smul(a * b, z)),
        )

        # assert self.scalar in itp.notation.mul and self.scalar in itp.notation.add


norm2 = itp.notation.SortDispatch(name="norm2")
norm = itp.notation.SortDispatch(name="norm")
dot = itp.notation.SortDispatch(name="dot")


class Normed(prop.TypeClass):
    """
    https://en.wikipedia.org/wiki/Normed_vector_space
    """

    key: smt.SortRef

    norm: smt.FuncDeclRef

    norm_nonneg: itp.Proof
    norm_zero: itp.Proof
    norm_homog: itp.Proof
    norm_triangle: itp.Proof

    def check(self, T):
        x, y = smt.Consts("x y", T)
        V = VectorSpace(T)
        a = smt.Const("a", V.scalar)
        assert T in itp.notation.add
        assert T in itp.notation.mul
        assert self.assert_eq(self.norm_nonneg.thm, smt.ForAll([x], self.norm(x) >= 0))
        assert self.assert_eq(
            self.norm_zero.thm, smt.ForAll([x], self.norm(x) == 0 == (x == V.zero))
        )
        assert self.assert_eq(
            self.norm_homog.thm,
            smt.ForAll([a, x], self.norm(V.smul(a, x)) == smt.Abs(a) * self.norm(x)),
        )
        assert self.assert_eq(
            self.norm_triangle.thm,
            smt.ForAll([x, y], self.norm(x + y) <= self.norm(x) + self.norm(y)),
        )

        # assert self.scalar in itp.notation.mul and self.scalar in itp.notation.add


# https://en.wikipedia.org/wiki/Vector_space

V = smt.DeclareSort("V")
u, v, w = smt.Consts("u v w", V)

add = smt.Function("vadd", V, V, V)
itp.notation.add.register(V, add)

add_comm = itp.axiom(smt.ForAll([u, v], u + v == v + u))
add_assoc = itp.axiom(smt.ForAll([u, v, w], u + (v + w) == (u + v) + w))

zero = smt.Const("zero", V)

add_zero = itp.axiom(smt.ForAll([u], u + zero == u))
zero_add = itp.prove(smt.ForAll([u], zero + u == u), by=[add_comm, add_zero])

"""
V.smul = smt.Function("smul", V, smt.RealSort(), V)
itp.notation.mul.register(V, smul)
x, y = smt.Reals("x y")

smul_one = itp.axiom(smt.ForAll([u], u * 1 == u))


# Possible design for theories.
vadd = itp.notation.SortDispatch()
vadd_assoc = {V: add_assoc}
vadd_comm = {V: add_comm}

vzero = {V: zero}
vadd_zero = {V: add_zero}
"""

"""
class VectorTheory:
    def __init__(self, T):
        self.T = T
        self.vadd = vadd[T]
        self.vadd_assoc = vadd_assoc[V]
        self.vadd_comm = vadd_comm[V]



class VecTheory:
    def __init__(self, V):
        self.V = V
        add = itp.notation.add[V]
        sub = itp.notation.sub[V]
        neg = itp.notation.neg[V]
        u, v, w = smt.Consts("u v w", V)
        self.add_comm = itp.prove(u + v == v + u, by=[add.defn])
        self.add_assoc = itp.prove((u + v) + w == u + (v + w), by=[add.defn])
        # self.add_zero = itp.prove(u + V.zero == u, by=[add.defn])
        # self.add_neg = itp.prove(u + -u == V.zero, by=[add.defn])
        self.add_neg = itp.prove(u - v == u + -v, by=[add.defn, neg.defn, sub.defn])
"""
