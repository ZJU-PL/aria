import aria.itp as itp
import aria.itp.smt as smt
import aria.itp.theories.set as set_
import aria.itp.property as prop

# https://leanprover-community.github.io/mathematics_in_lean/C10_Topology.html
# https://isabelle.in.tum.de/library/HOL/HOL/Topological_Spaces.html

open = itp.notation.SortDispatch(name="open")


class Topology(prop.TypeClass):
    key: smt.SortRef
    open_UNIV: itp.Proof
    open_Int: itp.Proof
    open_Union: itp.Proof

    def check(self, T):
        self.Set = set_.Set(T)
        SetSet = set_.Set(self.Set)
        A, B = smt.Consts("A B", self.Set)
        K = smt.Const("K", SetSet)
        print(self.open_UNIV.thm.eq(open(self.Set.full)))
        self.assert_eq(self.open_UNIV.thm, open(self.Set.full))
        self.assert_eq(
            self.open_Int.thm,
            itp.QForAll([A, B], open(A), open(B), open(A & B)),
        )
        self.assert_eq(
            self.open_Union.thm,
            itp.QForAll([K], itp.QForAll([A], K[A], open(A)), open(set_.BigUnion(K))),
        )
        self.closed = itp.define("closed", [A], open(~A))


# https://en.wikipedia.org/wiki/Sierpi%C5%84ski_space
Sierpinski = itp.Inductive("Sierpinski")
Sierpinski.declare("S0")
Sierpinski.declare("S1")
Sierp = Sierpinski.create()
SierpSet = set_.Set(Sierp)
A, B = smt.Consts("A B", SierpSet)
K = smt.Const("K", set_.Set(SierpSet))
Sierp.open = open.define(
    [A],
    smt.Or(
        A == SierpSet.empty,  # {}
        A == SierpSet.full,  # {0,1}
        A == smt.Store(SierpSet.empty, Sierp.S1, True),  # {1}
    ),
)
Sierp.open_UNIV = itp.prove(open(SierpSet.full), by=[Sierp.open.defn])
Sierp.open_Int = itp.prove(
    itp.QForAll([A, B], open(A), open(B), open(A & B)), by=[Sierp.open.defn]
)
Sierp.open_Union = itp.prove(
    itp.QForAll([K], itp.QForAll([A], K[A], open(A)), open(set_.BigUnion(K))),
    by=[Sierp.open.defn],
)
Topology.register(
    Sierp,
    open_UNIV=Sierp.open_UNIV,
    open_Int=Sierp.open_Int,
    open_Union=Sierp.open_Union,
)
