"""
Built in smtlib theory of finite sequences.
"""

import aria.itp as itp
import aria.itp.smt as smt


# induct_list List style induction
# induct_snoc
# concat induction
# Strong induction


def induct_list(x: smt.SeqRef, P):
    """

    >>> x = smt.Const("x", Seq(smt.IntSort()))
    >>> P = smt.Function("P", Seq(smt.IntSort()), smt.BoolSort())
    >>> induct_list(x, P)
    |- Implies(And(P(Empty(Seq(Int))),
                ForAll([hd!..., tl!...],
                      Implies(P(tl!...),
                              P(Concat(Unit(hd!...), tl!...))))),
            P(x))
    """
    assert isinstance(x, smt.SeqRef)
    hd = smt.FreshConst(x.sort().basis(), prefix="hd")
    tl = smt.FreshConst(x.sort(), prefix="tl")
    return itp.axiom(
        smt.Implies(
            smt.And(
                P(smt.Empty(x.sort())),
                itp.QForAll([hd, tl], P(tl), P(smt.Unit(hd) + tl)),
            ),
            P(x),
        )
    )


def induct(T: smt.SortRef, P) -> itp.kernel.Proof:
    z = smt.FreshConst(T, prefix="z")
    sort = smt.SeqSort(T)
    x, y = smt.FreshConst(sort), smt.FreshConst(sort)
    return itp.axiom(
        smt.And(
            P(smt.Empty(sort)),
            itp.QForAll([z], P(smt.Unit(z))),
            itp.QForAll([x, y], P(x), P(y), P(smt.Concat(x, y))),
        )  # -------------------------------------------------
        == itp.QForAll([x], P(x))
    )


def seq(*args):
    """
    Helper to construct sequences.
    >>> seq(1, 2, 3)
    Concat(Unit(1), Concat(Unit(2), Unit(3)))
    >>> seq(1)
    Unit(1)
    """
    if len(args) == 0:
        raise ValueError(
            "seq() requires at least one argument. use smt.Empty(sort) instead."
        )
    elif len(args) == 1:
        return smt.Unit(smt._py2expr(args[0]))
    else:
        return smt.Concat(*[smt.Unit(smt._py2expr(a)) for a in args])


def Seq(T: smt.SortRef) -> smt.SeqSortRef:
    """
    Make sort of Sequences and prove useful lemmas.

    >>> BoolSeq = Seq(smt.BoolSort())
    >>> x,y,z = smt.Consts("x y z", BoolSeq)
    >>> x + y
    Concat(x, y)
    """
    S = smt.SeqSort(T)
    x, y, z = smt.Consts("x y z", S)
    empty = smt.Empty(S)
    S.empty = empty

    S.concat_empty = itp.prove(itp.QForAll([x], empty + x == x))
    S.empty_concat = itp.prove(itp.QForAll([x], x + empty == x))
    S.concat_assoc = itp.prove(
        itp.QForAll(
            [x, y, z],
            (x + y) + z == x + (y + z),
        )
    )

    S.length_empty = itp.prove(itp.QForAll([x], smt.Length(empty) == 0))
    S.length_unit = itp.prove(itp.QForAll([x], smt.Length(smt.Unit(x)) == 1))
    S.concat_length = itp.prove(
        itp.QForAll([x, y], smt.Length(x + y) == smt.Length(x) + smt.Length(y))
    )
    S.length_zero_unique = itp.prove(
        itp.QForAll([x], (smt.Length(x) == 0) == (x == empty))
    )
    S.concat_head = itp.prove(
        itp.QForAll(
            [x],
            smt.Length(x) > 0,
            smt.Unit(x[0]) + smt.SubSeq(x, 1, smt.Length(x) - 1) == x,
        )
    )

    n, m = smt.Ints("n m")

    S.subseq_zero = itp.prove(itp.QForAll([x, n], smt.SubSeq(x, n, 0) == empty))
    S.subseq_length = itp.prove(
        itp.QForAll(
            [x, n, m],
            m >= 0,
            n >= 0,
            n + m <= smt.Length(x),
            smt.Length(smt.SubSeq(x, n, m)) == m,
        )
    )
    S.subseq_all = itp.prove(itp.QForAll([x], smt.SubSeq(x, 0, smt.Length(x)) == x))
    S.subseq_concat = itp.prove(
        itp.QForAll(
            [x, n],
            n >= 0,
            n <= smt.Length(x),
            smt.SubSeq(x, 0, n) + smt.SubSeq(x, n, smt.Length(x) - n) == x,
        )
    )

    # S.head_length = itp.prove(
    #    itp.QForAll(
    #        [x], smt.Length(x) != 0, x[0] + smt.SubSeq(x, 1, smt.Length(x) - 1) == x
    #    )
    # )
    S.contains_empty = itp.prove(smt.ForAll([x], smt.Contains(x, empty)))
    S.contains_self = itp.prove(smt.ForAll([x], smt.Contains(x, x)))
    S.contains_concat_left = itp.prove(
        itp.QForAll([x, y, z], smt.Contains(x, z), smt.Contains(x + y, z))
    )
    S.contains_concat_right = itp.prove(
        itp.QForAll([x, y, z], smt.Contains(y, z), smt.Contains(x + y, z))
    )
    S.contains_subseq = itp.prove(smt.ForAll([x], smt.Contains(x, smt.SubSeq(x, n, m))))
    # self.contains_unit = itp.kernel.prove(
    #    itp.QForAll([x, y], smt.Contains(smt.Unit(x), y) == (smt.Unit(x) == y))
    # )
    S.empty_contains = itp.kernel.prove(
        itp.QForAll([x], smt.Contains(empty, x), (x == empty))
    )

    # InRe, Extract, IndexOf, LastIndexOf, prefixof, replace, suffixof
    # SeqMap, SeqMapI, SeqFoldLeft, SeqFoldLeftI
    # Contains
    return S


def Unit(x: smt.ExprRef) -> smt.SeqRef:
    """
    Construct a sequence of length 1.
    """
    return smt.Unit(x)
