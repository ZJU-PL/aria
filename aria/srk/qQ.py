"""
Rational number operations for SRK.

This module provides rational number arithmetic operations,
equivalent to the OCaml qQ.ml module but using Python's
fractions.Fraction class for arbitrary precision rationals.
"""

from __future__ import annotations
from typing import Optional, Tuple
from fractions import Fraction
import math

# Type alias for clarity
t = Fraction

# QQ helper with OCaml-like static constructors used across the codebase
class QQ:
    """Static helpers for rational numbers backed by fractions.Fraction."""

    @staticmethod
    def one() -> Fraction:
        return Fraction(1)

    @staticmethod
    def zero() -> Fraction:
        return Fraction(0)

    @staticmethod
    def of_string(s: str) -> Fraction:
        return Fraction(s)

    @staticmethod
    def of_int(x: int) -> Fraction:
        return Fraction(x)

    @staticmethod
    def denominator(x: Fraction) -> int:
        return x.denominator

# Default accuracy for approximations
opt_default_accuracy: int = -1

def pp(formatter, x: Fraction) -> None:
    """Pretty print a rational number."""
    print(x, file=formatter, end="")

def show(x: Fraction) -> str:
    """Convert rational to string representation."""
    return str(x)

def compare(x: Fraction, y: Fraction) -> int:
    """Compare two rationals: returns -1, 0, or 1."""
    if x < y:
        return -1
    elif x > y:
        return 1
    else:
        return 0

def equal(x: Fraction, y: Fraction) -> bool:
    """Check if two rationals are equal."""
    return x == y

def leq(x: Fraction, y: Fraction) -> bool:
    """Check if x <= y."""
    return x <= y

def lt(x: Fraction, y: Fraction) -> bool:
    """Check if x < y."""
    return x < y

def add(x: Fraction, y: Fraction) -> Fraction:
    """Add two rationals."""
    return x + y

def sub(x: Fraction, y: Fraction) -> Fraction:
    """Subtract two rationals."""
    return x - y

def mul(x: Fraction, y: Fraction) -> Fraction:
    """Multiply two rationals."""
    return x * y

def zero() -> Fraction:
    """Return the rational 0."""
    return Fraction(0)

def one() -> Fraction:
    """Return the rational 1."""
    return Fraction(1)

def div(x: Fraction, y: Fraction) -> Fraction:
    """Divide two rationals."""
    if y == 0:
        raise ValueError("QQ.div: divide by zero")
    return x / y

def negate(x: Fraction) -> Fraction:
    """Negate a rational."""
    return -x

def inverse(x: Fraction) -> Fraction:
    """Multiplicative inverse of a rational."""
    if x == 0:
        raise ValueError("QQ.inverse: inverse of zero")
    return Fraction(1) / x

def of_int(x: int) -> Fraction:
    """Convert int to rational."""
    return Fraction(x)

def of_zz(x: int) -> Fraction:
    """Convert arbitrary precision integer to rational."""
    return Fraction(x)

def of_frac(num: int, den: int) -> Fraction:
    """Create rational from numerator and denominator."""
    return Fraction(num, den)

def of_zzfrac(num: int, den: int) -> Fraction:
    """Create rational from arbitrary precision integers."""
    return Fraction(num, den)

def of_float(x: float) -> Fraction:
    """Convert float to rational (approximate)."""
    return Fraction(x).limit_denominator()

def of_string(s: str) -> Fraction:
    """Parse string as rational."""
    return Fraction(s)

def to_zzfrac(x: Fraction) -> Tuple[int, int]:
    """Convert rational to (numerator, denominator) pair."""
    return (x.numerator, x.denominator)

def to_zz(x: Fraction) -> Optional[int]:
    """Convert rational to integer if denominator is 1."""
    if x.denominator == 1:
        return x.numerator
    else:
        return None

def to_int(x: Fraction) -> Optional[int]:
    """Convert rational to int if possible."""
    if x.denominator == 1:
        # Check if it fits in a regular int
        try:
            if -2**63 <= x.numerator <= 2**63 - 1:
                return int(x.numerator)
            else:
                return None
        except OverflowError:
            return None
    else:
        return None

def to_float(x: Fraction) -> float:
    """Convert rational to float."""
    return float(x)

def numerator(x: Fraction) -> int:
    """Get numerator."""
    return x.numerator

def denominator(x: Fraction) -> int:
    """Get denominator."""
    return x.denominator

def hash(x: Fraction) -> int:
    """Hash function for rationals."""
    return hash((x.numerator, x.denominator))

def exp(x: Fraction, k: int) -> Fraction:
    """Compute x^k for rational x and integer k."""
    if k == 0:
        return one()
    elif k == 1:
        return x
    else:
        y = exp(x, k // 2)
        y2 = mul(y, y)
        if k % 2 == 0:
            return y2
        else:
            return mul(x, y2)

def floor(x: Fraction) -> int:
    """Floor of rational (towards negative infinity)."""
    return x.numerator // x.denominator

def ceiling(x: Fraction) -> int:
    """Ceiling of rational (towards positive infinity)."""
    num, den = x.numerator, x.denominator
    return (num + den - 1) // den

def min(x: Fraction, y: Fraction) -> Fraction:
    """Return the minimum of two rationals."""
    return min(x, y)

def max(x: Fraction, y: Fraction) -> Fraction:
    """Return the maximum of two rationals."""
    return max(x, y)

def abs(x: Fraction) -> Fraction:
    """Absolute value of rational."""
    return abs(x)

def nudge(x: Fraction, accuracy: Optional[int] = None) -> Tuple[Fraction, Fraction]:
    """Compute interval approximation of rational using continued fractions."""
    if accuracy is None:
        accuracy = opt_default_accuracy

    if accuracy < 0:
        return (x, x)

    num, den = to_zzfrac(x)
    q, r = divmod(num, den)

    if accuracy == 0:
        return (of_zz(q), of_zz(q + 1))
    elif r == 0:
        return (of_zz(q), of_zz(q))
    else:
        lo, hi = nudge(of_zzfrac(den, r), accuracy - 1)
        return (add(of_zz(q), inverse(hi)),
                add(of_zz(q), inverse(lo)))

def nudge_down(x: Fraction, accuracy: Optional[int] = None) -> Fraction:
    """Lower approximation using continued fractions."""
    if accuracy is None:
        accuracy = opt_default_accuracy

    if accuracy < 0:
        return x

    num, den = to_zzfrac(x)
    q, r = divmod(num, den)

    if accuracy == 0:
        return of_zz(q)
    elif r == 0:
        return of_zz(q)
    else:
        hi = nudge_up(of_zzfrac(den, r), accuracy - 1)
        return add(of_zz(q), inverse(hi))

def nudge_up(x: Fraction, accuracy: Optional[int] = None) -> Fraction:
    """Upper approximation using continued fractions."""
    if accuracy is None:
        accuracy = opt_default_accuracy

    if accuracy < 0:
        return x

    num, den = to_zzfrac(x)
    q, r = divmod(num, den)

    if accuracy == 0:
        return of_zz(q + 1)
    elif r == 0:
        return of_zz(q)
    else:
        lo = nudge_down(of_zzfrac(den, r), accuracy - 1)
        return add(of_zz(q), inverse(lo))

def idiv(x: Fraction, y: Fraction) -> int:
    """Integer division of rationals."""
    if y == 0:
        raise ValueError("QQ.idiv: divide by zero")

    xnum, xden = to_zzfrac(x)
    ynum, yden = to_zzfrac(y)

    # Compute (xnum * yden) / (ynum * xden)
    from .zZ import div as zz_div, mul as zz_mul
    return zz_div(zz_mul(xnum, yden), zz_mul(ynum, xden))

def modulo(x: Fraction, y: Fraction) -> Fraction:
    """Modulo operation for rationals."""
    if y == 0:
        raise ValueError("QQ.modulo: divide by zero")

    xnum, xden = to_zzfrac(x)
    ynum, yden = to_zzfrac(y)

    from .zZ import modulo as zz_modulo, mul as zz_mul
    mod_result = zz_modulo(zz_mul(xnum, yden), zz_mul(ynum, xden))
    return of_zzfrac(mod_result, zz_mul(xden, yden))

def gcd(x: Fraction, y: Fraction) -> Fraction:
    """GCD of two rationals."""
    xnum, xden = to_zzfrac(x)
    ynum, yden = to_zzfrac(y)

    from .zZ import gcd as zz_gcd, lcm as zz_lcm
    gcd_num = zz_gcd(xnum, ynum)
    lcm_den = zz_lcm(xden, yden)
    return of_zzfrac(gcd_num, lcm_den)

def lcm(x: Fraction, y: Fraction) -> Fraction:
    """LCM of two rationals."""
    xnum, xden = to_zzfrac(x)
    ynum, yden = to_zzfrac(y)

    from .zZ import lcm as zz_lcm, gcd as zz_gcd
    lcm_num = zz_lcm(xnum, ynum)
    gcd_den = zz_gcd(xden, yden)
    return of_zzfrac(lcm_num, gcd_den)
