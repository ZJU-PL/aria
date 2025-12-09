import aria.itp.theories.algebra.group as group
import aria.itp as itp
import aria.itp.smt as smt
import aria.itp.property as prop


# https://isabelle.in.tum.de/library/HOL/HOL/Lattices.html


class SemiLattice(prop.TypeClass):
    key: smt.SortRef
    idem: itp.Proof

    def check(self, T):
        self.Group = group.AbelSemiGroup(T)
        x, y, z = smt.Consts("x y z", T)
        assert itp.utils.alpha_eq(self.idem.thm, smt.ForAll([x], x * x == x))
        self.left_idem = itp.prove(
            smt.ForAll([x, y], x * (x * y) == x * y), by=[self.idem, self.Group.assoc]
        )
        self.right_idem = itp.prove(
            smt.ForAll([x, y], (y * x) * x == y * x), by=[self.idem, self.Group.assoc]
        )


L = smt.DeclareSort("AbstractLattice")
x, y, z = smt.Consts("x y z", L)
mul = smt.Function("mul", L, L, L)
itp.notation.mul.register(L, mul)
assoc = itp.axiom(smt.ForAll([x, y, z], x * (y * z) == (x * y) * z))
comm = itp.axiom(smt.ForAll([x, y], x * y == y * x))
idem = itp.axiom(smt.ForAll([x], x * x == x))
group.Semigroup.register(L, assoc=assoc)
group.AbelSemiGroup.register(L, comm=comm)
SemiLattice.register(L, idem=idem)
