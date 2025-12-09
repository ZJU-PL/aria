import aria.itp as itp
import aria.itp.theories.real as R
import aria.itp.smt as smt

"""
Interval arithmetic. Intervals are a record of hi and lo bounds.
"""
Interval = itp.Struct("Interval", ("lo", itp.R), ("hi", itp.R))
x, y, z = smt.Reals("x y z")
i, j, k = smt.Consts("i j k", Interval)

setof = itp.define("setof", [i], smt.Lambda([x], smt.And(i.lo <= x, x <= i.hi)))

meet = itp.define("meet", [i, j], Interval.mk(R.max(i.lo, j.lo), R.min(i.hi, j.hi)))
meet_intersect = itp.prove(
    smt.ForAll([i, j], smt.SetIntersect(setof(i), setof(j)) == setof(meet(i, j))),
    by=[setof.defn, meet.defn, R.min.defn, R.max.defn],
)

join = itp.define("join", [i, j], Interval.mk(R.min(i.lo, j.lo), R.max(i.hi, j.hi)))
join_union = itp.prove(
    smt.ForAll(
        [i, j], smt.IsSubset(smt.SetUnion(setof(i), setof(j)), setof(join(i, j)))
    ),
    by=[setof.defn, join.defn, R.min.defn, R.max.defn],
)

width = itp.define("width", [i], i.hi - i.lo)
mid = itp.define("mid", [i], (i.lo + i.hi) / 2)

add = itp.notation.add.define([i, j], Interval.mk(i.lo + j.lo, i.hi + j.hi))
add_set = itp.prove(
    smt.ForAll(
        [x, y, i, j], smt.Implies(setof(i)[x] & setof(j)[y], setof(i + j)[x + y])
    ),
    by=[add.defn, setof.defn],
)

sub = itp.notation.sub.define([i, j], Interval.mk(i.lo - j.hi, i.hi - j.lo))
