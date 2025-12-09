import aria.itp as itp
import aria.itp.smt as smt

B = smt.BoolSort()
p, q, r = smt.Bools("p q r")

or_comm = itp.prove(smt.ForAll([p, q], p | q == q | p))
or_assoc = itp.prove(smt.ForAll([p, q, r], (p | q) | r == p | (q | r)))
or_true = itp.prove(smt.ForAll([p], p | smt.BoolVal(True) == smt.BoolVal(True)))
or_false = itp.prove(smt.ForAll([p], p | smt.BoolVal(False) == p))

or_idem = itp.prove(smt.ForAll([p], p | p == p))
or_absorb = itp.prove(smt.ForAll([p, q], p | (p & q) == p))
or_distr = itp.prove(smt.ForAll([p, q, r], p | (q & r) == (p | q) & (p | r)))
or_compl = itp.prove(smt.ForAll([p], p | ~p == smt.BoolVal(True)))

and_comm = itp.prove(smt.ForAll([p, q], p & q == q & p))
and_assoc = itp.prove(smt.ForAll([p, q, r], (p & q) & r == p & (q & r)))
and_true = itp.prove(smt.ForAll([p], p & smt.BoolVal(True) == p))
and_false = itp.prove(smt.ForAll([p], p & smt.BoolVal(False) == smt.BoolVal(False)))
and_absorb = itp.prove(smt.ForAll([p, q], (p & (p | q)) == p))
and_compl = itp.prove(smt.ForAll([p], p & ~p == smt.BoolVal(False)))

and_idem = itp.prove(smt.ForAll([p], p & p == p))

imp_true = itp.prove(smt.ForAll([p], smt.Implies(smt.BoolVal(True), p) == p))
imp_false = itp.prove(
    smt.ForAll([p], smt.Implies(smt.BoolVal(False), p) == smt.BoolVal(True))
)
imp_idem = itp.prove(smt.ForAll([p], smt.Implies(p, p) == smt.BoolVal(True)))
imp_refl = itp.prove(smt.ForAll([p], smt.Implies(p, p)))
imp_trans = itp.prove(
    itp.QForAll([p, q, r], smt.Implies(p, q), smt.Implies(q, r), smt.Implies(p, r))
)

not_true = itp.prove(~smt.BoolVal(True) == smt.BoolVal(False))
not_false = itp.prove(~smt.BoolVal(False) == smt.BoolVal(True))
not_inv = itp.prove(smt.ForAll([p], ~~p == p))  # Double negation elimination
not_and = itp.prove(smt.ForAll([p, q], ~(p & q) == (~p | ~q)))  # De Morgan’s Law
not_or = itp.prove(smt.ForAll([p, q], ~(p | q) == (~p & ~q)))  # De Morgan’s Law

xor_neq = itp.prove(smt.ForAll([p, q], (p ^ q) == (p != q)))
xor_comm = itp.prove(smt.ForAll([p, q], p ^ q == q ^ p))
xor_assoc = itp.prove(smt.ForAll([p, q, r], (p ^ q) ^ r == p ^ (q ^ r)))
xor_idem = itp.prove(smt.ForAll([p], p ^ p == smt.BoolVal(False)))
xor_true = itp.prove(smt.ForAll([p], p ^ smt.BoolVal(True) == ~p))
xor_false = itp.prove(smt.ForAll([p], p ^ smt.BoolVal(False) == p))

T = smt.DeclareTypeVar("T")
a, b = smt.Consts("a b", T)
if_true = itp.prove(smt.ForAll([a, b], smt.If(True, a, b) == a))
if_false = itp.prove(smt.ForAll([a, b], smt.If(False, a, b) == b))

rws = [
    or_true,
    or_false,
    or_idem,
    or_absorb,
    or_distr,
    or_compl,
    and_comm,
    and_assoc,
    and_true,
    and_false,
    and_absorb,
    and_compl,
    imp_true,
    imp_false,
    imp_idem,
    imp_refl,
    xor_idem,
    xor_true,
    xor_false,
    if_true,
    if_false,
]
