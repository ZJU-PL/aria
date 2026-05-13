"""
Fourier-Motzkin variable elimination for linear constraints.

Operates on ``Lincons0`` / ``Linexpr0`` objects from ``wedge.py``.
"""

from __future__ import annotations
from typing import List, Tuple, Set, Optional
from fractions import Fraction


def _coeff_of(lin, dim):
    for c, d in lin.coeffs:
        if d == dim:
            return c
    return Fraction(0)


def _drop_dim(lin, dim):
    out = [(c, d) for c, d in lin.coeffs if d != dim]
    cst = lin.cst if lin.cst is not None else Fraction(0)
    return out, cst


def _make_lin(coeffs, cst, Linx):
    return Linx(coeffs, cst if cst != 0 else None)


def _scale_lin(lin, factor, Linx):
    if factor == 1:
        return lin
    cst = (lin.cst if lin.cst is not None else Fraction(0)) * factor
    return Linx([(c * factor, d) for c, d in lin.coeffs], cst if cst != 0 else None)


def _add_lin(a, b, Linx):
    coeffs = {}
    for c, d in a.coeffs:
        coeffs[d] = coeffs.get(d, Fraction(0)) + c
    for c, d in b.coeffs:
        coeffs[d] = coeffs.get(d, Fraction(0)) + c
    acst = a.cst if a.cst is not None else Fraction(0)
    bcst = b.cst if b.cst is not None else Fraction(0)
    cst = acst + bcst
    out_coeffs = [(c, d) for d, c in coeffs.items() if c != 0]
    return Linx(out_coeffs, cst if cst != 0 else None)


def eliminate(dims, constraints, Linx, Linc):
    if not dims or not constraints:
        return list(constraints)

    remaining = list(constraints)
    to_elim = set(dims)

    for k in sorted(to_elim):
        if not remaining:
            break

        pos = []
        neg = []
        zero = []
        eq_with_k = None

        for lc in remaining:
            c = _coeff_of(lc.linexpr0, k)
            if lc.typ == Linc.EQ and c != 0:
                eq_with_k = lc
            elif c > 0:
                pos.append(lc)
            elif c < 0:
                neg.append(lc)
            else:
                zero.append(lc)

        if eq_with_k is not None:
            ck = _coeff_of(eq_with_k.linexpr0, k)
            rest, rcst = _drop_dim(eq_with_k.linexpr0, k)
            xk_expr = _make_lin([(-c, d) for c, d in rest], -rcst, Linx)
            xk_expr = _scale_lin(xk_expr, Fraction(1, ck), Linx)

            next_remaining = []
            for lc in remaining:
                if lc is eq_with_k:
                    continue
                c = _coeff_of(lc.linexpr0, k)
                if c == 0:
                    next_remaining.append(lc)
                else:
                    rest_lc, rcst_lc = _drop_dim(lc.linexpr0, k)
                    scaled = _scale_lin(xk_expr, c, Linx)
                    new_lin = _add_lin(
                        _make_lin(rest_lc, rcst_lc, Linx), scaled, Linx)
                    next_remaining.append(Linc.make(new_lin, lc.typ))
            remaining = next_remaining
            continue

        for pi in pos:
            pc = _coeff_of(pi.linexpr0, k)
            p_rest = [(c, d) for c, d in pi.linexpr0.coeffs if d != k]
            p_cst = pi.linexpr0.cst if pi.linexpr0.cst is not None else Fraction(0)
            p_norm = Linx([(c / pc, d) for c, d in p_rest],
                          (p_cst / pc) if p_cst != 0 else None)

            for ni in neg:
                nc = _coeff_of(ni.linexpr0, k)
                n_rest = [(c, d) for c, d in ni.linexpr0.coeffs if d != k]
                n_cst = ni.linexpr0.cst if ni.linexpr0.cst is not None else Fraction(0)
                n_norm = Linx([(c / -nc, d) for c, d in n_rest],
                              (n_cst / -nc) if n_cst != 0 else None)

                merged = _add_lin(
                    Linx([(-c, d) for c, d in p_norm.coeffs],
                         (-p_norm.cst) if p_norm.cst is not None else None),
                    n_norm, Linx)
                comb_typ = Linc.SUP if (pi.typ == Linc.SUP or ni.typ == Linc.SUP) else Linc.SUPEQ
                zero.append(Linc.make(merged, comb_typ))

        remaining = zero

    return remaining
