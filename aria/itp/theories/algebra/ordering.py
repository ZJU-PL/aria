import aria.itp as itp
import aria.itp.smt as smt
import aria.itp.property as prop


class PreOrder(prop.TypeClass):
    key: smt.SortRef
    refl: itp.Proof
    trans: itp.Proof
    less_le_not_le = itp.Proof

    def check(self, T):
        x, y, z = smt.Consts("x y z", T)
        assert itp.utils.alpha_eq(self.refl.thm, smt.ForAll([x], x <= x))
        assert itp.utils.alpha_eq(
            self.trans.thm, itp.QForAll([x, y, z], x <= y, y <= z, x <= z)
        )
        assert itp.utils.alpha_eq(
            self.less_le_not_le.thm,
            itp.QForAll([x, y], (x < y) == smt.And(x <= y, smt.Not(y <= x))),
        )


n, m, k = smt.Ints("n m k")
PreOrder.register(
    smt.IntSort(),
    refl=itp.prove(smt.ForAll([n], n <= n)),
    trans=itp.prove(itp.QForAll([n, m, k], n <= m, m <= k, n <= k)),
    less_le_not_le=itp.prove(
        itp.QForAll([n, m], (n < m) == smt.And(n <= m, smt.Not(m <= n)))
    ),
)
