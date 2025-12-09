import aria.itp as itp
import aria.itp.smt as smt

# import itprag.theories.real as real
import aria.itp.theories.real.vec as vec
import aria.itp.theories.set as set_

Point2D = vec.Vec2
p, q, a, b, c = smt.Consts("p q a b c", Point2D)

r = smt.Real("r")
circle = itp.define("circle", [c, r], smt.Lambda([p], vec.norm2(p - c) == r * r))

Shape = set_.Set(Point2D)

A, B, C = smt.Consts("A B C", Shape)

is_circle = itp.define("is_circle", [A], smt.Exists([c, r], circle(c, r) == A))

# convex = itp.define("convex", [A], itp.QForAll([p, q], A(p), A(q), A(vec.Vec2(smul(0.5, p + q)))))
