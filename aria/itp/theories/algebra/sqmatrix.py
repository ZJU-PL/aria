import aria.itp as itp
import aria.itp.smt as smt

"""
Square matrices of a single dimension
"""

SqMat = smt.DeclareSort("SqMat")
A, B, C = smt.Consts("A B C", SqMat)

add = smt.Function("add", SqMat, SqMat, SqMat)
itp.notation.add.register(SqMat, add)
add_assoc = itp.axiom(smt.ForAll([A, B, C], A + (B + C) == (A + B) + C))
add_comm = itp.axiom(smt.ForAll([A, B], A + B == B + A))

zero = smt.Const("zero", SqMat)
add_zero = itp.axiom(smt.ForAll([A], A + zero == A))
zero_add = itp.prove(smt.ForAll([A], zero + A == A), by=[add_comm, add_zero])

smul = smt.Function("smul", smt.RealSort(), SqMat, SqMat)
itp.notation.mul.register(SqMat, lambda x, y: smul(y, x))
itp.notation.rmul.register(SqMat, lambda x, y: smul(y, x))
r, s = smt.Reals("r s")
smul_zero = itp.axiom(smt.ForAll([A], 0 * A == zero))  # Is this a lemma?
smul_assoc = itp.axiom(smt.ForAll([A, r, s], smul(s, smul(r, A)) == smul(s * r, A)))
smul_comm = itp.prove(
    smt.ForAll([A, s, r], smul(r, smul(s, A)) == smul(s, smul(r, A))), by=[smul_assoc]
)

matmul = smt.Function("mul", SqMat, SqMat, SqMat)
itp.notation.matmul.register(SqMat, matmul)
mul_assoc = itp.axiom(smt.ForAll([A, B, C], A @ (B @ C) == (A @ B) @ C))

mul_left_dist = itp.axiom(smt.ForAll([A, B, C], A @ (B + C) == A @ B + A @ C))
mul_right_dist = itp.axiom(smt.ForAll([A, B, C], (A + B) @ C == A @ C + B @ C))

I = smt.Const("I", SqMat)
id_mul = itp.axiom(smt.ForAll([A], I @ A == A))
mul_id = itp.axiom(smt.ForAll([A], A @ I == A))

trans = smt.Function("trans", SqMat, SqMat)
trans_idem = itp.axiom(smt.ForAll([A], trans(trans(A)) == A))

inv = smt.Function("inv", SqMat, SqMat)
is_inv = smt.Function("is_inv", SqMat, smt.BoolSort())

inv_is_inv = itp.axiom(smt.ForAll([A], is_inv(A) == is_inv(inv(A))))

inv_left = itp.axiom(itp.QForAll([A], is_inv(A), inv(A) @ A == I))
inv_right = itp.axiom(itp.QForAll([A], is_inv(A), A @ inv(A) == I))

# inv_unique

is_symm = itp.define("is_symm", [A], trans(A) == A)
is_orth = itp.define("is_orth", [A], A @ trans(A) == I)

is_idem = itp.define(
    "is_idem", [A], A @ A == A
)  # projection matrices https://en.wikipedia.org/wiki/Projection_(linear_algebra)
# orth_trans_inv = itp.prove(smt.QForAll([A], is_orth(A), inv(A) == trans(A)))


is_diag = smt.Function("is_diag", SqMat, smt.BoolSort())

trace = smt.Function("trace", SqMat, smt.RealSort())
