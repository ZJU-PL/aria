import aria.itp as itp
import aria.itp.smt as smt
import aria.itp.theories.real as real

C = itp.Struct("C", ("re", smt.RealSort()), ("im", smt.RealSort()))

z, w, u, z1, z2 = smt.Consts("z w u z1 z2", C)
add = itp.define("add", [z1, z2], C.C(z1.re + z2.re, z1.im + z2.im))
itp.notation.add.register(C, add)
mul = itp.define(
    "mul", [z1, z2], C.C(z1.re * z2.re - z1.im * z2.im, z1.re * z2.im + z1.im * z2.re)
)
itp.notation.mul.register(C, mul)
conj = itp.define("conj", [z], C.C(z.re, -z.im))

div = itp.notation.div.define(
    [z1, z2],
    C.C(
        (z1.re * z2.re + z1.im * z2.im) / (z2.re ** 2 + z2.im ** 2),
        (z1.im * z2.re - z1.re * z2.im) / (z2.re ** 2 + z2.im ** 2),
    ),
)

C0 = C.C(0, 0)
C1 = C.C(1, 0)
Ci = C.C(0, 1)

add_zero = itp.prove(smt.ForAll([z], z + C0 == z), by=[add.defn])
mul_zero = itp.prove(smt.ForAll([z], z * C0 == C0), by=[mul.defn])
mul_one = itp.prove(smt.ForAll([z], z * C1 == z), by=[mul.defn])
add_comm = itp.prove(smt.ForAll([z, w], z + w == w + z), by=[add.defn])
add_assoc = itp.prove(
    smt.ForAll([z, w, u], (z + (w + u)) == ((z + w) + u)), by=[add.defn]
)
mul_comm = itp.prove(smt.ForAll([z, w], z * w == w * z), by=[mul.defn])

# unstable perfoamnce.
# mul_div = itp.prove(ForAll([z,w], Implies(w != C0, z == z * w / w)), by=[div.defn, mul.defn], timeout=1000)
##mul_div = Calc()
# div_one = itp.prove(smt.ForAll([z], z / C1 == z), by=[div.defn])
# div_inv = itp.prove(
#    smt.ForAll([z], smt.Implies(z != C0, z / z == C1)), by=[div.defn], admit=True
# )

# inv = itp.define("inv", [z], )

# conjugate
# polar
norm2 = itp.define("norm2", [z], z * conj(z))

t, s = smt.Reals("t s")
expi = itp.define("expi", [t], C.C(real.cos(t), real.sin(t)))

# expi_mul = itp.prove(
#   smt.ForAll([t, s], expi(t) * expi(s) == expi(t + s)),
#    by=[expi.defn, mul.defn, real.sin_add, real.cos_add],
# )

c = itp.tactics.Calc([t, s], expi(t) * expi(s))
c.eq(C.C(real.cos(t), real.sin(t)) * C.C(real.cos(s), real.sin(s)), by=[expi.defn])
c.eq(
    C.C(
        real.cos(t) * real.cos(s) - real.sin(t) * real.sin(s),
        real.cos(t) * real.sin(s) + real.sin(t) * real.cos(s),
    ),
    by=[mul.defn],
)
_1 = c.qed()
# c.eq(C.C(real.cos(t + s), real.sin(t + s)), by=[real.cos_add, real.sin_add])
c = itp.tactics.Calc([t, s], C.C(real.cos(t + s), real.sin(t + s)))
c.eq(expi(t + s), by=[expi.defn])
expi_mul = c.qed()
_2 = c.qed()

_3 = itp.prove(
    itp.QForAll(
        [t, s],
        C.C(
            real.cos(t) * real.cos(s) - real.sin(t) * real.sin(s),
            real.cos(t) * real.sin(s) + real.sin(t) * real.cos(s),
        )
        == C.C(real.cos(t + s), real.sin(t + s)),
    ),
    by=[real.cos_add, real.sin_add],
)
expi_mul = itp.prove(
    itp.QForAll([t, s], expi(t) * expi(s) == expi(t + s)), by=[_1, _2, _3], admit=True
)

RSphere = itp.Inductive("RSphere")
RSphere.declare("Finite", ("val", C))
RSphere.declare("Inf")
RSphere.create()

# mobius
# conformal
