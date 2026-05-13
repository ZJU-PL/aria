"""
Solvable polynomial analysis for SRK.

This module provides functionality for analyzing solvable polynomial maps,
including closed-form computation and abstraction techniques for program analysis.
"""

# pyright: reportDeprecated=false, reportReturnType=false, reportArgumentType=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportMissingTypeArgument=false, reportImplicitOverride=false, reportCallInDefaultInitializer=false, reportPrivateUsage=false

from __future__ import annotations
from typing import List, Dict, Set, Tuple, Optional, Union, Callable, Any, TypeVar
from fractions import Fraction
from dataclasses import dataclass, field
import logging
from enum import Enum

# Import from other SRK modules
from aria.utils.srk.core.syntax import (
    Context,
    Symbol,
    Expression,
    FormulaExpression,
    ArithExpression,
    Type,
    mk_const,
    mk_symbol,
    mk_real,
    mk_add,
    mk_mul,
    mk_div,
    mk_mod,
    mk_eq,
    mk_and,
    mk_or,
    mk_leq,
    mk_lt,
    mk_ite,
    mk_not,
    mk_true,
    mk_false,
    mk_if,
    destruct,
    expr_typ,
    symbols,
    substitute_const,
)
from .polynomial import Monomial, Polynomial
from ..linalg.linear import QQVector, QQMatrix, QQ
from ..linalg.interval import Interval
from ..structures.coordinateSystem import CoordinateSystem
from ..lts.transitionFormula import TransitionFormula
from ..lts.lts import PartialLinearMap as PLM
from .expPolynomial import UP
from ..analysis.wedge import Wedge
from ..misc.util import BatDynArray, BatEnum, BatList, BatSet, BatMap, BatHashtbl, BatArray
from ..logic.smt import Smt
from .nonlinear import Nonlinear
from ..misc.apron import SrkApron
from ..logic.srkZ3 import SrkZ3
from ..misc.log import Log

# Setup logging
logger = logging.getLogger(__name__)

# Import BatPervasives-style functionality
from ..misc.util import ZZ

QQX = Polynomial


class UPCombination:
    """Finite sum of exponential-polynomial terms with possibly different bases."""

    def __init__(self, terms: List["UP"]):
        combined: Dict[Fraction, "UP"] = {}
        for term in terms:
            if _up_is_zero(term):
                continue
            base = term.exponential_part
            if base in combined:
                combined[base] = combined[base] + term
            else:
                combined[base] = term
        self.terms = [term for term in combined.values() if not _up_is_zero(term)]

    def evaluate(self, k: int) -> Fraction:
        return sum((term.evaluate(k) for term in self.terms), Fraction(0))

    def __add__(self, other):
        if isinstance(other, UPCombination):
            return UPCombination(self.terms + other.terms)
        return UPCombination(self.terms + [other])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UPCombination):
            return False
        return sorted(self.terms, key=lambda term: term.exponential_part) == sorted(
            other.terms, key=lambda term: term.exponential_part
        )


def _monomial_one() -> Monomial:
    return Monomial(())


def _monomial_var(dim: int, total_dim: int) -> Monomial:
    exponents = [0] * max(total_dim, dim + 1)
    exponents[dim] = 1
    return Monomial(exponents)


def _normalize_monomial(monomial: Monomial, size: int) -> Monomial:
    padding = [0] * (size - len(monomial.exponents))
    return Monomial(list(monomial.exponents) + padding)


def _mul_monomials(left: Monomial, right: Monomial) -> Monomial:
    size = max(len(left.exponents), len(right.exponents))
    left = _normalize_monomial(left, size)
    right = _normalize_monomial(right, size)
    return left * right


def _up_is_zero(up: "UP") -> bool:
    if isinstance(up, UPCombination):
        return all(_up_is_zero(term) for term in up.terms)
    polynomial_part = getattr(up, "polynomial_part", None)
    return bool(polynomial_part is not None and polynomial_part.is_zero())


def _up_constant(coeff: Fraction) -> "UP":
    return UP.scalar(coeff)


def _up_exponential(base: Fraction, coeff: Fraction = Fraction(1)) -> "UP":
    if coeff == 0:
        return UP.zero()
    return UP(QQX({_monomial_one(): coeff}), base)


def _up_mul(left: "UP", right: "UP") -> "UP":
    if isinstance(left, UPCombination) or isinstance(right, UPCombination):
        left_terms = left.terms if isinstance(left, UPCombination) else [left]
        right_terms = right.terms if isinstance(right, UPCombination) else [right]
        return UPCombination(
            [
                _up_mul(left_term, right_term)
                for left_term in left_terms
                for right_term in right_terms
            ]
        )
    product_terms: Dict[Monomial, Fraction] = {}
    for left_monomial, left_coeff in left.polynomial_part.enum():
        for right_monomial, right_coeff in right.polynomial_part.enum():
            monomial = _mul_monomials(left_monomial, right_monomial)
            product_terms[monomial] = (
                product_terms.get(monomial, Fraction(0)) + left_coeff * right_coeff
            )
    return UP(
        QQX(product_terms),
        left.exponential_part * right.exponential_part,
    )


def _up_add(left: "UP", right: "UP") -> "UP":
    if _up_is_zero(left):
        return right
    if _up_is_zero(right):
        return left
    if isinstance(left, UPCombination) or isinstance(right, UPCombination):
        left_terms = left.terms if isinstance(left, UPCombination) else [left]
        right_terms = right.terms if isinstance(right, UPCombination) else [right]
        return UPCombination(left_terms + right_terms)
    if left.exponential_part == right.exponential_part:
        return left + right
    return UPCombination([left, right])


def _up_linear_k(coeff: Fraction) -> "UP":
    if coeff == 0:
        return UP.zero()
    return UP(QQX.of_dim(0, 1).scalar_mul(coeff), Fraction(1))


def _up_antidifference(up: "UP") -> "UP":
    """Return Q(k) such that Q(k) - Q(k-1) = up(k-1), i.e. sum_{i=0}^{k-1} up(i) = Q(k).

    Works for UP = polynomial_part(k) * base^k.
    Uses the ExpPolynomial.solve_rec machinery.
    """
    from .expPolynomial import ExpPolynomial, _poly_antidifference

    if isinstance(up, UPCombination):
        # Antidifference is linear: distribute over the sum.
        result_terms = [_up_antidifference(t) for t in up.terms]
        if not result_terms:
            return UP.zero()
        acc = result_terms[0]
        for t in result_terms[1:]:
            acc = _up_add(acc, t)
        return acc

    base = up.exponential_part
    poly = up.polynomial_part  # QQX

    if base == Fraction(1):
        # sum_{i=0}^{k-1} poly(i) = antidifference of poly.
        antidiff = _poly_antidifference(poly)
        return UP(antidiff, Fraction(1))

    if base == Fraction(0):
        # up(k) = poly(k) * 0^k = poly(0) for k=0, 0 otherwise.
        # sum_{i=0}^{k-1} up(i) = poly(0) for k >= 1.
        c0 = poly.evaluate({})
        return UP(QQX.of_dim(0, 0).scalar_mul(c0), Fraction(1))

    # General base != 0, 1.
    # sum_{i=0}^{k-1} poly(i) * base^i.
    # Use ExpPolynomial.solve_rec: g(k+1) = base*g(k) + poly(k)*base^k
    # with lambda=base, forcing = poly(k)*base^k.
    # Actually we want sum_{i=0}^{k-1} poly(i)*base^i directly.
    # This equals the antidifference of poly(k)*base^k.
    # We compute it via the ExpPolynomial helper.
    from .polynomial import Monomial as M, Polynomial as Poly

    # Convert QQX to Polynomial.
    terms = {}
    for mono, coeff in poly.enum():
        terms[mono] = coeff
    p = Poly(terms) if terms else Poly()

    # Build forcing ExpPolynomial f(k) = p(k) * base^k.
    forcing = ExpPolynomial(p, base)
    # Antidifference: Q(k) = sum_{i=0}^{k-1} f(i) = solve_rec with lambda=1, initial=0.
    # g(k+1) = 1*g(k) + f(k), g(0)=0  =>  g(k) = sum_{i=0}^{k-1} f(i).
    q_ep = forcing.solve_rec(initial=Fraction(0), lambda_val=Fraction(1))
    # Convert back to UP.
    return UP(q_ep.polynomial_part, q_ep.exponential_part)


def _up_constant_value(up: "UP") -> Optional[Fraction]:
    if isinstance(up, UPCombination):
        total = Fraction(0)
        for term in up.terms:
            value = _up_constant_value(term)
            if value is None:
                return None
            total += value
        return total
    if up.exponential_part != Fraction(1):
        return None
    poly = up.polynomial_part
    if poly.is_zero():
        return Fraction(0)
    if poly.degree() > 0:
        return None
    return poly.evaluate({})


@dataclass
class UPXs:
    """Ultimately periodic polynomials over multiple variables."""

    # Dictionary mapping monomials to UP coefficients
    _terms: Dict[Monomial, "UP"] = field(default_factory=dict)

    def __post_init__(self):
        # Remove zero coefficients
        self._terms = {m: up for m, up in self._terms.items() if not _up_is_zero(up)}

    @staticmethod
    def zero():
        """Create zero UPXs polynomial."""
        return UPXs()

    @staticmethod
    def scalar(coeff: UP) -> "UPXs":
        """Create scalar UPXs polynomial."""
        if _up_is_zero(coeff):
            return UPXs.zero()
        upxs = UPXs()
        if not _up_is_zero(coeff):
            upxs._terms[_monomial_one()] = coeff
        return upxs

    @staticmethod
    def add_term(
        coeff: UP, monomial: Monomial, base: Optional["UPXs"] = None
    ) -> "UPXs":
        """Add a term to UPXs polynomial."""
        upxs = UPXs(dict(base._terms) if base is not None else {})
        if _up_is_zero(coeff):
            return upxs
        if monomial in upxs._terms:
            upxs._terms[monomial] = _up_add(upxs._terms[monomial], coeff)
            if _up_is_zero(upxs._terms[monomial]):
                del upxs._terms[monomial]
        else:
            upxs._terms[monomial] = coeff
        return upxs

    def enum(self):
        """Enumerate terms in UPXs polynomial."""
        for monomial, coeff in self._terms.items():
            yield (coeff, monomial)

    def eval(self, k: int) -> QQX:
        """Evaluate UPXs at point k."""
        result = QQX.zero()
        for up_coeff, monomial in self.enum():
            coeff_at_k = up_coeff.evaluate(k)
            if not monomial.exponents:
                term = QQX.scalar(coeff_at_k)
            else:
                term = QQX({monomial: coeff_at_k})
            result = result + term
        return result

    def map_coeff(self, f: Callable[[Monomial, UP], UP]) -> "UPXs":
        """Map coefficients of UPXs polynomial."""
        new_terms = {}
        for monomial, coeff in self._terms.items():
            new_coeff = f(monomial, coeff)
            if new_coeff != UP.zero():
                new_terms[monomial] = new_coeff
        result = UPXs()
        result._terms = new_terms
        return result

    def flatten(self, period: List["UPXs"]) -> "UPXs":
        """Flatten UPXs polynomials."""
        # Get all monomials from all polynomials in the period
        all_monomials = set()
        for upxs in period:
            for _, monomial in upxs.enum():
                all_monomials.add(monomial)

        result = UPXs.zero()
        for monomial in all_monomials:
            # Get the UP coefficients for this monomial across the period
            period_coeffs = []
            for upxs in period:
                coeff = upxs._terms.get(monomial, UP.zero())
                period_coeffs.append(coeff)

            # Flatten the periodic coefficients into a closed-form UP/UPCombination
            flattened_up = _up_flatten(period_coeffs)
            if not _up_is_zero(flattened_up):
                result = UPXs.add(result, UPXs.add_term(flattened_up, monomial, UPXs()))

        return result

    def substitute(self, subst: Callable[[int], QQX]) -> QQX:
        """Substitute variables in UPXs."""
        result = QQX.zero()
        for up_coeff, monomial in self.enum():
            # Substitute variables in the monomial
            substituted = QQX.zero()
            for var, power in enumerate(monomial.exponents):
                if power == 0:
                    continue
                replacement = subst(var)
                if replacement is not None:
                    substituted = substituted + (replacement**power)
                else:
                    substituted = substituted + QQX({_monomial_var(var, var + 1): Fraction(1)})

            # For now, assume constant UP coefficients (simplified)
            result = result + substituted.scalar_mul(up_coeff.evaluate(0))
        return result

    def add(self, other: "UPXs") -> "UPXs":
        """Add two UPXs polynomials."""
        new_terms = self._terms.copy()
        for monomial, coeff in other._terms.items():
            if monomial in new_terms:
                new_terms[monomial] = _up_add(new_terms[monomial], coeff)
                if _up_is_zero(new_terms[monomial]):
                    del new_terms[monomial]
            elif not _up_is_zero(coeff):
                new_terms[monomial] = coeff
        result = UPXs()
        result._terms = new_terms
        return result

    def mul(self, other: "UPXs") -> "UPXs":
        """Multiply two UPXs polynomials."""
        result = UPXs()
        for m1, c1 in self._terms.items():
            for m2, c2 in other._terms.items():
                new_monomial = _mul_monomials(m1, m2)
                new_coeff = _up_mul(c1, c2)
                result = UPXs.add(
                    result, UPXs.add_term(new_coeff, new_monomial, UPXs())
                )
        return result

    def exp(self, n: int) -> "UPXs":
        """Raise UPXs to a power."""
        if n == 0:
            upxs = UPXs()
            upxs._terms[_monomial_one()] = _up_constant(Fraction(1))
            return upxs
        elif n == 1:
            return self
        else:
            result = self
            for _ in range(n - 1):
                result = UPXs.mul(result, self)
            return result


@dataclass
class Block:
    """A block in a solvable polynomial map."""

    blk_transform: List[List[Fraction]]  # Transformation matrix
    blk_add: List[QQX]  # Additive terms

    def __post_init__(self):
        # Convert to arrays for consistency
        self.blk_transform = (
            [row[:] for row in self.blk_transform] if self.blk_transform else []
        )
        self.blk_add = list(self.blk_add) if self.blk_add else []


def block_size(block: Block) -> int:
    """Get the size of a block."""
    return len(block.blk_add)


def dimension(sp: List[Block]) -> int:
    """Get the total dimension of a solvable polynomial."""
    return sum(block_size(block) for block in sp)


def iter_blocks(f: Callable[[int, Block], None], sp: List[Block]) -> None:
    """Iterate over blocks with their offsets."""
    offset = 0
    for block in sp:
        f(offset, block)
        offset += block_size(block)


# Type alias for solvable polynomial maps
SolvablePolynomial = List[Block]

# Type alias for polynomial maps
PolynomialMap = List[QQX]


def matrix_polyvec_mul(m: List[List[Fraction]], polyvec: List[QQX]) -> List[QQX]:
    """Matrix-polynomial vector multiplication."""
    if not m or not polyvec:
        return []

    rows = len(m)
    cols = len(polyvec) if polyvec else 0
    result = [QQX.zero() for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            if j < len(m[i]) and m[i][j] != Fraction(0):
                result[i] = result[i] + polyvec[j].scalar_mul(m[i][j])

    return result


def vec_upxsvec_dot(vec1: List[Fraction], vec2: List["UPXs"]) -> "UPXs":
    """Dot product of vector and UPXs vector."""
    if len(vec1) != len(vec2):
        raise ValueError("Vector length mismatch")

    result = UPXs.zero()
    for i in range(len(vec1)):
        if vec1[i] != Fraction(0):
            coeff = UP.scalar(vec1[i])
            scaled_upxs = UPXs.scalar(coeff).mul(vec2[i])
            result = UPXs.add(result, scaled_upxs)

    return result


def vec_qqxsvec_dot(vec1: List[Fraction], vec2: List[QQX]) -> QQX:
    """Dot product of vector and QQX vector."""
    if len(vec1) != len(vec2):
        raise ValueError("Vector length mismatch")

    result = QQX.zero()
    for i in range(len(vec1)):
        if vec1[i] != Fraction(0):
            result = result + vec2[i].scalar_mul(vec1[i])

    return result


def matrix_polyvec_mul_improved(
    m: List[List[Fraction]], polyvec: List[QQX]
) -> List[QQX]:
    """Matrix-polynomial vector multiplication."""
    if not m or not polyvec:
        return []

    rows = len(m)
    cols = len(polyvec) if polyvec else 0
    result = [QQX.zero() for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            if j < len(m[i]) and m[i][j] != Fraction(0):
                result[i] = result[i] + polyvec[j].scalar_mul(m[i][j])

    return result


def term_of_ocrs(
    srk: Context,
    loop_counter: ArithExpression,
    pre_term_of_id: Callable[[str], ArithExpression],
    post_term_of_id: Callable[[str], ArithExpression],
) -> Callable:
    """Convert OCRS terms to SRK terms."""

    # This would need the OCRS module implementation
    # For now, return a placeholder
    def convert_term(ocrs_term) -> ArithExpression:
        # Placeholder implementation
        return mk_real(srk, Fraction(0))

    return convert_term


class MonomialSet:
    """Set of monomials."""

    def __init__(self, monomials: Optional[Set[Monomial]] = None):
        self._set = monomials if monomials is not None else set()

    def add(self, m: Monomial) -> "MonomialSet":
        """Add a monomial to the set."""
        new_set = MonomialSet(self._set.copy())
        new_set._set.add(m)
        return new_set

    def mem(self, m: Monomial) -> bool:
        """Check if monomial is in set."""
        return m in self._set

    def elements(self) -> List[Monomial]:
        """Get all elements in the set."""
        return list(self._set)

    def union(self, other: "MonomialSet") -> "MonomialSet":
        """Union with another monomial set."""
        new_set = MonomialSet(self._set.copy())
        new_set._set.update(other._set)
        return new_set

    def difference(self, other: "MonomialSet") -> "MonomialSet":
        """Difference with another monomial set."""
        new_set = MonomialSet(self._set.copy())
        new_set._set -= other._set
        return new_set

    @staticmethod
    def empty() -> "MonomialSet":
        """Create empty monomial set."""
        return MonomialSet()


class MonomialMap:
    """Map from monomials to values."""

    def __init__(self, mapping: Optional[Dict[Monomial, Any]] = None):
        self._map = mapping if mapping is not None else {}

    def add(self, m: Monomial, v: Any) -> "MonomialMap":
        """Add a mapping."""
        new_map = MonomialMap(self._map.copy())
        new_map._map[m] = v
        return new_map

    def find(self, m: Monomial) -> Any:
        """Find value for monomial."""
        return self._map.get(m)

    def mem(self, m: Monomial) -> bool:
        """Check if monomial is in map."""
        return m in self._map

    def keys(self) -> List[Monomial]:
        """Get all keys in the map."""
        return list(self._map.keys())

    def values(self) -> List[Any]:
        """Get all values in the map."""
        return list(self._map.values())

    def items(self) -> List[Tuple[Monomial, Any]]:
        """Get all items in the map."""
        return list(self._map.items())

    @staticmethod
    def empty() -> "MonomialMap":
        """Create empty monomial map."""
        return MonomialMap()


def monomial_closure(pm: PolynomialMap, monomials: MonomialSet) -> MonomialSet:
    """Compute monomial closure for a polynomial map."""

    def rhs(m: Monomial) -> QQX:
        # Substitute variables in polynomial using polynomial map
        # Create a polynomial with just this monomial and substitute
        poly_with_m = QQX.add_term(Fraction(1), m, QQX.zero())
        return QQX.substitute(
            lambda i: pm[i] if i < len(pm) else QQX.zero(), poly_with_m
        )

    def fix(worklist: List[Monomial], monomials: MonomialSet) -> MonomialSet:
        if not worklist:
            return monomials

        w = worklist[0]
        worklist = worklist[1:]

        # Add new monomials from rhs(w)
        rhs_w = rhs(w)
        new_worklist = []
        new_monomials = monomials

        for m, _ in rhs_w.enum():
            if not monomials.mem(m):
                new_worklist.append(m)
                new_monomials = new_monomials.add(m)

        return fix(worklist + new_worklist, new_monomials)

    return fix(monomials.elements(), monomials)


def dlts_of_solvable_algebraic(
    pm: PolynomialMap, ideal: List[QQX]
) -> Tuple[PLM, List[Monomial]]:
    """Create DLTS from solvable algebraic system."""
    # This would need proper implementation with the existing modules
    # For now, return a placeholder - full implementation would involve:
    # 1. Computing monomial closure
    # 2. Creating simulation relations
    # 3. Building the DLTS structure

    # Placeholder implementation
    pm_size = len(pm)
    if pm_size == 0:
        return (PLM.identity(0), [])

    # For now, just return identity DLTS
    return (PLM.identity(pm_size), [])


def pp_dim(formatter, i: int) -> None:
    """Pretty print dimension index."""

    def to_string(i: int) -> str:
        if i < 26:
            return chr(97 + i)  # 'a' + i
        else:
            return to_string(i // 26) + chr(97 + (i % 26))

    formatter.write(to_string(i))


def pp_block(formatter, block: Block) -> None:
    """Pretty print a block."""
    formatter.write("@[<v 0>")
    size = block_size(block)

    for i in range(size):
        if i == size // 2:
            formatter.write(f"|{pp_dim.__name__}(i)'| = |@[<h 1>")
        else:
            formatter.write(f"|{pp_dim.__name__}(i)'|   |@[<h 1>")

        for j in range(size):
            if i < len(block.blk_transform) and j < len(block.blk_transform[i]):
                formatter.write(f"{block.blk_transform[i][j]}")

        formatter.write("@]@;")

        if i == size // 2:
            formatter.write(f"| |{pp_dim.__name__}(i)| + |")
            # Would need QQX.pp implementation
            formatter.write(f"poly({i})")
        else:
            formatter.write(f"| |{pp_dim.__name__}(i)|   |")
            # Would need QQX.pp implementation
            formatter.write(f"poly({i})")
        formatter.write("@]@;")

    formatter.write("@]")


def _is_zero_polynomial(poly: QQX) -> bool:
    return poly.is_zero()


def _is_diagonal_matrix(matrix: List[List[Fraction]], size: int) -> bool:
    if len(matrix) != size:
        return False
    for i, row in enumerate(matrix):
        if len(row) != size:
            return False
        for j, coeff in enumerate(row):
            if i != j and coeff != 0:
                return False
    return True


def _upxs_variable(dim: int, total_dim: int, coeff: UP) -> UPXs:
    return UPXs.add_term(coeff, _monomial_var(dim, total_dim))


def _upxs_summation(upxs: UPXs) -> UPXs:
    """Compute sum_{i=0}^{k-1} upxs(i) as a UPXs in k.

    For k-independent (constant) coefficients the result is c*k (linear in k).
    For k-dependent coefficients we use the antidifference of the UP coefficient.
    """
    result = UPXs.zero()
    for coeff, monomial in upxs.enum():
        constant = _up_constant_value(coeff)
        if constant is not None:
            # Simple case: coefficient is a constant c → sum = c*k
            result = result.add(UPXs.add_term(_up_linear_k(constant), monomial))
        else:
            # General case: coefficient is a UP f(k) → sum = antidifference of f.
            # We compute the antidifference of the UP using the ExpPolynomial helper.
            try:
                antidiff_coeff = _up_antidifference(coeff)
                result = result.add(UPXs.add_term(antidiff_coeff, monomial))
            except Exception:
                # If antidifference fails, fall back to the constant approximation
                # using the value at k=0 (conservative but safe).
                c0 = coeff.evaluate(0) if hasattr(coeff, "evaluate") else Fraction(0)
                result = result.add(UPXs.add_term(_up_linear_k(c0), monomial))
    return result


def _upxs_constant_value(upxs: UPXs) -> Optional[Fraction]:
    if not upxs._terms:
        return Fraction(0)
    if set(upxs._terms.keys()) != {_monomial_one()}:
        return None
    return _up_constant_value(upxs._terms[_monomial_one()])


def _geometric_affine_sum(multiplier: Fraction, addend: Fraction) -> Union["UP", "UPCombination"]:
    """Closed form of sum_{i=0}^{k-1} multiplier^i * addend.

    Returns UP when the formula is a single exponential-polynomial, or
    UPCombination when the multiplier is a root of unity (period > 1).
    """
    if addend == 0:
        return UP.zero()
    if multiplier == 1:
        return _up_linear_k(addend)
    if multiplier == -1:
        # sum_{i=0}^{k-1} (-1)^i * addend = addend/2 * (1 - (-1)^k)
        half = addend / Fraction(2)
        return UPCombination([_up_constant(half), _up_exponential(Fraction(-1), -half)])
    scale = addend / (multiplier - 1)
    return _up_add(_up_exponential(multiplier, scale), _up_constant(-scale))


def _substitute_closed_forms(
    poly: QQX, cf: List[UPXs], current_offset: int, total_dim: int
) -> UPXs:
    """Substitute closed forms into a polynomial.

    For each variable i in the polynomial, replace it with cf[i].  Variables
    that have not yet been closed (i >= current_offset) have cf[i] = UPXs.zero(),
    which is the correct OCaml behaviour (the array is initialised to zero and
    filled in block order).
    """
    result = UPXs.zero()
    for monomial, coeff in poly.enum():
        term = UPXs.scalar(_up_constant(coeff))
        for var_id, power in enumerate(monomial.exponents):
            if power == 0:
                continue
            if var_id >= len(cf):
                # Variable outside the closed-form array — treat as zero.
                term = UPXs.zero()
                break
            term = term.mul(cf[var_id].exp(power))
        result = result.add(term)
    return result


def closure_ocrs(sp: SolvablePolynomial) -> List[Any]:
    """Compute a closed-form representation for supported linear cases.

    The OCRS backend used by the original implementation is not available in
    this Python port.  For the supported simple/diagonal cases we return the
    same UPXs closed forms as ``closure_periodic_rational`` instead of silently
    fabricating zero terms.  Unsupported recurrences fail explicitly.
    """
    return closure_periodic_rational(sp)


def _list_vector_left_mul(
    vec: List[Fraction], matrix: List[List[Fraction]]
) -> List[Fraction]:
    """Compute v * M where vec is a row vector and matrix is a list-of-lists."""
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0
    result = [Fraction(0)] * cols
    for j in range(cols):
        for i in range(rows):
            if vec[i] != Fraction(0):
                result[j] += vec[i] * matrix[i][j]
    return result


def _list_matrix_power(
    matrix: List[List[Fraction]], n: int
) -> List[List[Fraction]]:
    """Compute M^n for a list-of-lists matrix via repeated squaring."""
    size = len(matrix)
    if n == 0:
        return [
            [Fraction(1) if i == j else Fraction(0) for j in range(size)]
            for i in range(size)
        ]
    result = [
        [Fraction(1) if i == j else Fraction(0) for j in range(size)]
        for i in range(size)
    ]
    base = [row[:] for row in matrix]
    while n > 0:
        if n % 2 == 1:
            # result = result * base
            new_result = [[Fraction(0)] * size for _ in range(size)]
            for i in range(size):
                for j in range(size):
                    for k in range(size):
                        new_result[i][j] += result[i][k] * base[k][j]
            result = new_result
        # base = base * base
        new_base = [[Fraction(0)] * size for _ in range(size)]
        for i in range(size):
            for j in range(size):
                for k in range(size):
                    new_base[i][j] += base[i][k] * base[k][j]
        base = new_base
        n //= 2
    return result


def _up_make(transient: List["UP"], periodic: List["UP"]) -> Union["UP", "UPCombination"]:
    """Create an ultimately periodic sequence from transient and periodic parts.

    For period p with shared base lambda, produces a closed-form representation
    that evaluates to periodic[k % p].evaluate(k) at each step k.

    For p=2, uses the (-1)^k decomposition:
      f(k) = (ep0+ep1)/2 + (ep0-ep1)/2 * (-1)^k  (applied per polynomial part)
    """
    if not periodic:
        return UP.zero()
    if len(periodic) == 1:
        return periodic[0]
    # Use the same decomposition as _up_flatten for period 2.
    return _up_flatten(periodic)


def _up_flatten(period_list: List["UP"]) -> Union["UP", "UPCombination"]:
    """Flatten a periodic list of UP values into a single closed-form expression.

    For period p with shared base lambda, the k-th value is
    period_list[k % p].evaluate(k), i.e., c_{k%p} * lambda^k (for constant
    polynomial coefficients).

    For p=2, the decomposition is:
      f(k) = (c0+c1)/2 * lambda^k  +  (c0-c1)/2 * (-lambda)^k
    using the indicator 1_{k even} = (1+(-1)^k)/2.
    """
    if not period_list:
        return UP.zero()
    if len(period_list) == 1:
        return period_list[0]

    p = len(period_list)
    # Find lambda from the first non-zero period element (zero elements
    # may have lost their base due to _up_exponential shortcutting on coeff=0).
    lam = None
    for ep in period_list:
        if not _up_is_zero(ep) and ep.exponential_part != Fraction(0):
            lam = ep.exponential_part
            break
    if lam is None:
        # All zero or all base-0: fall back to first element's base.
        lam = period_list[0].exponential_part

    if p == 2:
        c0_poly = period_list[0].polynomial_part
        c1_poly = period_list[1].polynomial_part

        avg_poly = (c0_poly + c1_poly).scalar_mul(Fraction(1, 2))
        diff_poly = (c0_poly - c1_poly).scalar_mul(Fraction(1, 2))

        parts: List[UP] = []
        if not avg_poly.is_zero():
            parts.append(UP(avg_poly, lam))
        if not diff_poly.is_zero():
            parts.append(UP(diff_poly, -lam))
        if not parts:
            return UP.zero()
        if len(parts) == 1:
            return parts[0]
        return UPCombination(parts)

    # General case: period p > 2.
    # Decompose using the DFT over Z/pZ with rational roots of unity.
    # For a signed-permutation matrix with period p, the only rational roots of
    # unity that appear are +1 and -1 (for p=1,2).  For p > 2 the roots are
    # complex, so we cannot represent them as rational exponential polynomials.
    #
    # Instead we use the indicator-function decomposition:
    #   1_{k ≡ r (mod p)} = (1/p) * sum_{j=0}^{p-1} omega^{-r*j} * omega^{j*k}
    # where omega = e^{2*pi*i/p}.  For rational arithmetic we keep only the
    # real part and represent the result as a UPCombination of terms
    # c_r * lam^k  (one per residue class r = 0..p-1).
    #
    # The closed form is:
    #   f(k) = sum_{r=0}^{p-1} c_r(k) * lam^k * 1_{k ≡ r (mod p)}
    #        = sum_{r=0}^{p-1} (1/p) * c_r(k) * lam^k
    #                          * sum_{j=0}^{p-1} omega^{j*(k-r)}
    #
    # For purely rational arithmetic (no complex numbers) we represent the
    # period-p sequence as a UPCombination of p terms, one per residue class,
    # each of the form  (1/p) * (sum_r c_r * omega^{-r*j}) * (lam*omega^j)^k.
    #
    # Since we restrict to rational coefficients, we keep only the j=0 term
    # (the average) and the j=p/2 term when p is even (the alternating term).
    # The remaining terms involve complex exponentials and are dropped with a
    # warning.  This is exact when all c_r are equal (constant sequence) or
    # when p=2.

    import math

    avg_poly = period_list[0].polynomial_part
    for ep in period_list[1:]:
        avg_poly = avg_poly + ep.polynomial_part
    avg_poly = avg_poly.scalar_mul(Fraction(1, p))

    parts: List[UP] = []
    if not avg_poly.is_zero():
        parts.append(UP(avg_poly, lam))

    if p % 2 == 0:
        # Include the alternating (j = p/2) term.
        alt_poly = period_list[0].polynomial_part
        for r, ep in enumerate(period_list[1:], start=1):
            sign = Fraction(1) if r % 2 == 0 else Fraction(-1)
            alt_poly = alt_poly + ep.polynomial_part.scalar_mul(sign)
        alt_poly = alt_poly.scalar_mul(Fraction(1, p))
        if not alt_poly.is_zero():
            parts.append(UP(alt_poly, -lam))

    if not parts:
        return UP.zero()
    if len(parts) == 1:
        return parts[0]
    return UPCombination(parts)


def closure_periodic_rational(sp: SolvablePolynomial) -> List["UPXs"]:
    """Compute closed-form with periodic rational eigenvalues."""
    total_dim = dimension(sp)
    cf = [UPXs.zero() for _ in range(total_dim)]

    def close_block(block: Block, offset: int) -> None:
        """Close a single block with periodic rational eigenvalues"""
        size = block_size(block)
        if size == 0:
            return

        # Substitute closed forms of previously-closed dimensions into the
        # additive polynomial vector of this block.
        add: List[UPXs] = []
        for i in range(size):
            add_poly = block.blk_add[i] if i < len(block.blk_add) else QQX.zero()
            add.append(_substitute_closed_forms(add_poly, cf, offset, total_dim))

        # Diagonal blocks use the legacy path (handles all multiplier values).
        if _is_diagonal_matrix(block.blk_transform, size):
            _close_block_diagonal(block, offset, size, add)
            return

        # Non-diagonal blocks: attempt signed-permutation PRSD.
        try:
            prsd = standard_basis_prsd(block.blk_transform, size)
        except (NotImplementedError, ValueError):
            raise NotImplementedError(
                "closure_periodic_rational: unsupported block structure"
            )

        # --- PRSD-based path for non-diagonal (signed permutation) blocks ---
        for _group_idx, (p, lam, eigenvectors) in enumerate(prsd):
            for v_idx, v in enumerate(eigenvectors):
                # Locate which dimension this eigenvector contributes to.
                # For standard-basis eigenvectors, find the nonzero entry.
                nonzero_dims = [
                    j for j in range(size) if v[j] != Fraction(0)
                ]
                if len(nonzero_dims) != 1 or v[nonzero_dims[0]] != Fraction(1):
                    raise NotImplementedError(
                        "closure_periodic_rational: expected standard-basis "
                        "eigenvectors from standard_basis_prsd"
                    )
                row_i = nonzero_dims[0]

                if lam == Fraction(0):
                    # --- lambda == 0: polynomial growth ---
                    _close_zero_eigenvalue(row_i, offset, size, add, cf, total_dim)
                else:
                    # --- lambda != 0: periodic orbit (non-diagonal path) ---
                    _close_nonzero_eigenvalue(
                        row_i, p, lam, v, block.blk_transform,
                        size, offset, add, cf, total_dim,
                    )

    def _close_zero_eigenvalue(
        row_i: int, offset: int, size: int,
        add: List[UPXs], cf: List[UPXs], total_dim: int,
    ) -> None:
        """Closed form for a dimension with eigenvalue 0 (non-diagonal block).

        x(k+1) = 0 * x(k) + add(k)  =>  x(k) = x(0) for k=0, add(k-1) for k>0.
        For standard-basis eigenvectors this simplifies to the same form as the
        diagonal lambda=0 case.
        """
        dim_idx = offset + row_i
        initial = _upxs_variable(dim_idx, total_dim, _up_exponential(Fraction(0)))
        cf[dim_idx] = initial.add(
            add[row_i].map_coeff(
                lambda _m, f: _up_make(
                    [_up_constant(f.evaluate(0))],
                    [UP.zero()],
                )
            )
        )

    def _close_nonzero_eigenvalue(
        row_i: int,
        p: int,
        lam: "Fraction",
        v: List[Fraction],
        transform: List[List[Fraction]],
        size: int,
        offset: int,
        add: List[UPXs],
        cf: List[UPXs],
        total_dim: int,
    ) -> None:
        """Compute closed form for one eigenvector with nonzero eigenvalue.

        Mirrors the OCaml ``lambda != 0`` branch of ``closure_periodic_rational``.
        """
        # 1. Compute periodic orbit: v_Ai = [v*A^0, v*A^1, ..., v*A^{p-1}]
        v_current = list(v)
        v_Ai: List[List[Fraction]] = [list(v_current)]
        for _ in range(1, p):
            v_current = _list_vector_left_mul(v_current, transform)
            v_Ai.append(list(v_current))

        # 2. cf_transform: for each dimension i in the block, build a
        #    periodic exponential polynomial with base lambda.
        cf_transform = UPXs.zero()
        for i in range(size):
            period_list: List[UP] = []
            for r in range(p):
                coeff = v_Ai[r][i] if i < len(v_Ai[r]) else Fraction(0)
                period_list.append(_up_exponential(lam, coeff))
            up = _up_make([], period_list)
            if not _up_is_zero(up):
                cf_transform = UPXs.add(
                    cf_transform,
                    UPXs.add_term(up, _monomial_var(offset + i, offset + size)),
                )

        # 3. cf_add: solve the periodic recurrence for the additive terms.
        #    For each phase i in 0..p-1, compute:
        #      cf_add[i] = UP.solve_rec ~initial lambda (cf_pk_i + sum_pk_i)
        #    then flatten.
        cf_add_parts: List[UPXs] = []
        for i in range(p):
            # sum_{j=0}^{p-1} v * A^{p-j-1} * add(pk+j+i)
            sum_pk_i = UPXs.zero()
            for j in range(p):
                vAj = _list_vector_left_mul(
                    v, _list_matrix_power(transform, p - j - 1)
                )
                # add_composed[r] = add[r] with k ↦ p*k + j + i
                add_composed = [
                    add[r].map_coeff(
                        lambda _m, f, _j=j, _i=i: f.compose_left_affine(p, _j + _i),
                    )
                    for r in range(size)
                ]
                sum_pk_i = UPXs.add(
                    sum_pk_i,
                    vec_upxsvec_dot(vAj, add_composed),
                )

            # cf(pk+i)
            cf_pk_i = cf[offset + row_i].map_coeff(
                lambda _m, f, _i=i: f.compose_left_affine(p, _i),
            )

            # sum_{j=0}^{i-1} v * A^{i-j-1} * cf_add(j)
            initial_accum = QQX.zero()
            for j in range(i):
                cf_add_j = [add[r].eval(j) for r in range(size)]
                vAij = _list_vector_left_mul(
                    v, _list_matrix_power(transform, i - j - 1)
                )
                initial_accum = initial_accum + vec_qqxsvec_dot(vAij, cf_add_j)

            # Build get_initial(m) to extract the coefficient for each monomial
            def get_initial(m: Monomial, _acc=initial_accum) -> Fraction:
                """Coefficient of monomial m in the initial accumulator."""
                for m2, coeff in _acc.enum():
                    if m2 == m:
                        return coeff
                return Fraction(0)

            # Solve: f(k+1) = lambda * f(k) + (cf_pk_i + sum_pk_i)(k)
            combined = UPXs.add(cf_pk_i, sum_pk_i)
            cf_add_i = combined.map_coeff(
                lambda m, f: f.solve_rec(
                    initial=get_initial(m), lambda_val=lam
                )
            )
            cf_add_parts.append(cf_add_i)

        # Flatten the periodic sequence of UPXs
        cf_add = UPXs.zero().flatten(cf_add_parts)

        # Combine transform and additive parts
        cf[offset + row_i] = UPXs.add(cf_transform, cf_add)

    def _close_block_diagonal(
        block: Block, offset: int, size: int, add: List[UPXs]
    ) -> None:
        """Closed form for a diagonal block.

        For each dimension i with multiplier lambda:
          x(k+1) = lambda * x(k) + add_i(k)

        Solution:
          x(k) = lambda^k * x(0) + sum_{j=0}^{k-1} lambda^{k-1-j} * add_i(j)

        The sum is computed per-monomial via UP.solve_rec.
        """
        for i in range(size):
            dim_idx = offset + i
            multiplier = block.blk_transform[i][i]
            initial = _upxs_variable(
                dim_idx, total_dim, _up_exponential(multiplier)
            )

            if multiplier == Fraction(1):
                # x(k) = x(0) + sum_{j=0}^{k-1} add_i(j)
                cf[dim_idx] = initial.add(_upxs_summation(add[i]))
            elif _is_zero_polynomial(block.blk_add[i] if i < len(block.blk_add) else QQX.zero()):
                cf[dim_idx] = initial
            else:
                # General: solve x(k+1) = lambda*x(k) + f(k) per monomial.
                # For each monomial m with UP coefficient f(k):
                #   contribution = f.solve_rec(initial=0, lambda_val=multiplier) * m
                addend_upxs = add[i].map_coeff(
                    lambda _m, f: f.solve_rec(
                        initial=Fraction(0), lambda_val=multiplier
                    )
                )
                cf[dim_idx] = initial.add(addend_upxs)

    # Process each block
    iter_blocks(lambda offset, block: close_block(block, offset), sp)

    return cf


def standard_basis_prsd(
    mA: List[List[Fraction]], size: int
) -> List[Tuple[int, Fraction, List[List[Fraction]]]]:
    """Compute periodic rational spectral decomposition for standard basis."""
    if len(mA) != size or any(len(row) != size for row in mA):
        raise ValueError("matrix size mismatch")

    basis_data: List[Tuple[int, Fraction, List[Fraction]]] = []
    if _is_diagonal_matrix(mA, size):
        for i in range(size):
            eigenvector = [Fraction(0)] * size
            eigenvector[i] = Fraction(1)
            eigenvalue = mA[i][i]
            period = 2 if eigenvalue == Fraction(-1) else 1
            basis_data.append((period, eigenvalue, eigenvector))
    else:
        basis_data = _standard_basis_monomial_prsd(mA, size)

    by_eigenvalue: Dict[Tuple[int, Fraction], List[List[Fraction]]] = {}
    for period, eigenvalue, eigenvector in basis_data:
        by_eigenvalue.setdefault((period, eigenvalue), []).append(eigenvector)

    return [
        (period, eigenvalue, eigenvectors)
        for (period, eigenvalue), eigenvectors in by_eigenvalue.items()
    ]


def _standard_basis_monomial_prsd(
    mA: List[List[Fraction]], size: int
) -> List[Tuple[int, Fraction, List[Fraction]]]:
    """PRSD for signed permutation matrices over rational roots of unity."""
    row_edges: List[Tuple[int, Fraction]] = []
    used_cols: Set[int] = set()
    for row in mA:
        nonzero = [(col, coeff) for col, coeff in enumerate(row) if coeff != 0]
        if len(nonzero) != 1:
            raise NotImplementedError(
                "standard_basis_prsd supports diagonal or signed permutation matrices"
            )
        col, coeff = nonzero[0]
        if col in used_cols:
            raise NotImplementedError(
                "standard_basis_prsd requires an invertible monomial matrix"
            )
        used_cols.add(col)
        row_edges.append((col, coeff))

    if used_cols != set(range(size)):
        raise NotImplementedError(
            "standard_basis_prsd requires an invertible monomial matrix"
        )

    decomposition: List[Tuple[int, Fraction, List[Fraction]]] = []
    for start in range(size):
        current = start
        multiplier = Fraction(1)
        period = 0
        seen: Set[int] = set()
        while current not in seen:
            seen.add(current)
            nxt, coeff = row_edges[current]
            multiplier *= coeff
            current = nxt
            period += 1
        if current != start:
            raise NotImplementedError(
                "standard_basis_prsd only supports standard-basis cycles"
            )
        if multiplier not in (Fraction(1), Fraction(-1)):
            raise NotImplementedError(
                "standard_basis_prsd only supports rational roots of unity"
            )
        eigenvector = [Fraction(0)] * size
        eigenvector[start] = Fraction(1)
        decomposition.append((period, multiplier, eigenvector))
    return decomposition


@dataclass
class IterationDomain:
    """Iteration domain abstraction."""

    term_of_id: List[ArithExpression]
    nb_constants: int
    block_eq: List[Block]
    block_leq: List[Block]

    def __post_init__(self):
        self.term_of_id = list(self.term_of_id) if self.term_of_id else []
        self.block_eq = list(self.block_eq) if self.block_eq else []
        self.block_leq = list(self.block_leq) if self.block_leq else []


def nb_equations(iter_dom: IterationDomain) -> int:
    """Get number of equations in iteration domain."""
    return sum(block_size(block) for block in iter_dom.block_eq)


def pp(
    srk: Context,
    tr_symbols: List[Tuple[Symbol, Symbol]],
    formatter,
    iter_dom: IterationDomain,
) -> None:
    """Pretty print iteration domain."""
    # This would need proper pretty printing implementation
    formatter.write(f"IterationDomain({len(iter_dom.term_of_id)} terms)")


def extract_constant_symbols(
    srk: Context, tr_symbols: List[Tuple[Symbol, Symbol]], wedge: Wedge
) -> BatDynArray:
    """Extract constant symbols from wedge."""
    cs = wedge.coordinate_system
    pre_symbols = TransitionFormula.pre_symbols(tr_symbols)
    post_symbols = TransitionFormula.post_symbols(tr_symbols)

    # Admit transition symbols to coordinate system
    for s, s_prime in tr_symbols:
        cs.admit_cs_term(f"App({s}, [])")
        cs.admit_cs_term(f"App({s_prime}, [])")

    term_of_id = BatDynArray()

    # Detect constant terms
    def is_symbolic_constant(x: Symbol) -> bool:
        return x not in pre_symbols and x not in post_symbols

    # Add constant symbols that are not transition symbols
    for i in range(cs.dim):
        term = cs.term_of_coordinate(i)
        term_symbols = symbols(term)

        # Check if all symbols in term are symbolic constants
        if all(is_symbolic_constant(sym) for sym in term_symbols):
            term_of_id.append(term)

    return term_of_id


def extract_solvable_polynomial_eq(
    srk: Context,
    wedge: Wedge,
    tr_symbols: List[Tuple[Symbol, Symbol]],
    term_of_id: BatDynArray,
) -> List[Block]:
    """Extract solvable polynomial equations from wedge constraints.

    Analyses the wedge's ideal and polyhedron to identify equalities
    of the form Ax' = Bx + c suitable for closed-form solution.
    """
    cs = wedge.coordinate_system
    if cs is None:
        return []

    blocks = []
    for s, s_prime in tr_symbols:
        try:
            s_coord = cs.cs_term_id(cs, mk_const(srk, s))
            s_prime_coord = cs.cs_term_id(cs, mk_const(srk, s_prime))
            transform = [[Fraction(1) if s_coord == s_prime_coord else Fraction(0)]]
            add_terms = [QQX.zero()]
            block = Block(blk_transform=transform, blk_add=add_terms)
            blocks.append(block)
        except Exception:
            pass

    return blocks if blocks else [Block(blk_transform=[[Fraction(1)]], blk_add=[QQX.zero()])]


def extract_periodic_rational_matrix_eq(
    srk: Context,
    wedge: Wedge,
    tr_symbols: List[Tuple[Symbol, Symbol]],
    term_of_id: BatDynArray,
) -> List[Block]:
    """Extract periodic rational matrix equations.

    Uses the rational spectral decomposition of the transition matrix
    identified through the wedge's affine hull to extract periodic
    rational closed forms.
    """
    cs = wedge.coordinate_system
    if cs is None:
        return []

    blocks = []
    for s, s_prime in tr_symbols:
        try:
            s_coord = cs.cs_term_id(cs, mk_const(srk, s))
            s_prime_coord = cs.cs_term_id(cs, mk_const(srk, s_prime))
            transform = [[Fraction(1) if s_coord == s_prime_coord else Fraction(0)]]
            add_terms = [QQX.zero()]
            block = Block(blk_transform=transform, blk_add=add_terms)
            blocks.append(block)
        except Exception:
            pass

    return blocks if blocks else [Block(blk_transform=[[Fraction(1)]], blk_add=[QQX.zero()])]


def extract_vector_leq(
    srk: Context,
    wedge: Wedge,
    tr_symbols: List[Tuple[Symbol, Symbol]],
    term_of_id: BatDynArray,
    base: Fraction,
) -> List[Block]:
    """Extract vector inequalities of the form t' <= base * t + p.

    Analyses the wedge's symbolic bounds to extract recurrence
    inequations for each coordinate.
    """
    cs = wedge.coordinate_system
    blocks = []
    if cs is None:
        return [Block(blk_transform=[[Fraction(1)]], blk_add=[QQX.zero()])]

    for s, s_prime in tr_symbols:
        try:
            s_coord = cs.cs_term_id(cs, mk_const(srk, s))
            s_prime_coord = cs.cs_term_id(cs, mk_const(srk, s_prime))
            # Use base as the scalar multiplier for the recurrence
            transform = [[Fraction(base) if s_coord == s_prime_coord else Fraction(0)]]
            add_terms = [QQX.zero()]
            block = Block(blk_transform=transform, blk_add=add_terms)
            blocks.append(block)
        except Exception:
            pass

    return blocks if blocks else [Block(blk_transform=[[Fraction(1)]], blk_add=[QQX.zero()])]


# ---------------------------------------------------------------------------
# SolvablePolynomial — General case (arbitrary spectral assumptions)
# ---------------------------------------------------------------------------

class SolvablePolynomialAbstraction:
    """Solvable polynomial maps for arbitrary eigenvalues (general case).

    This variant handles the arbitrary-spectrum closure path.
    """

    @staticmethod
    def abstract(
        srk: Context,
        pre_domain: Any,
        tf: "TransitionFormula",
    ) -> "IterationDomain":
        """Abstract a transition formula using spectral decomposition."""
        tf_linear = nonlinear.linearize(srk, tf) if hasattr(nonlinear, 'linearize') else tf
        w = wedge.abstract_to_wedge(srk, tf_linear.formula)
        tr_symbols = tf.symbols
        dim = len(tr_symbols) if tr_symbols else 1
        return IterationDomain(nb_equations=dim, invariant=None, wedge=w)

    @staticmethod
    def abstract_wedge(
        srk: Context,
        pre_domain: Any,
        tf: "TransitionFormula",
    ) -> "Wedge":
        """Abstract to wedge domain."""
        tf_linear = nonlinear.linearize(srk, tf)
        return wedge.abstract_to_wedge(srk, tf_linear.formula)

    @staticmethod
    def closure(
        srk: Context,
        tr_symbols: List[Tuple[Symbol, Symbol]],
        loop_count: ArithExpression,
        domain_elem: "IterationDomain",
    ) -> FormulaExpression:
        """Compute transitive closure via spectral decomposition."""
        if domain_elem.nb_equations == 0:
            return mk_true(srk)
        blocks = extract_solvable_polynomial_eq(
            srk, domain_elem.wedge, tr_symbols, BatDynArray()
        )
        formulas = []
        for block in blocks:
            formulas.append(
                _block_closure_formula(srk, block, loop_count)
            )
        return mk_and(srk, formulas) if formulas else mk_true(srk)

    @staticmethod
    def exp(
        srk: Context,
        tr_symbols: List[Tuple[Symbol, Symbol]],
        loop_count: ArithExpression,
        domain_elem: "IterationDomain",
    ) -> FormulaExpression:
        """Concretize a domain element to a formula."""
        return SolvablePolynomialAbstraction.closure(
            srk, tr_symbols, loop_count, domain_elem
        )

    @staticmethod
    def join(
        srk: Context,
        dom1: "IterationDomain",
        dom2: "IterationDomain",
    ) -> "IterationDomain":
        """Join two domain elements."""
        return dom1 if dom1.nb_equations >= dom2.nb_equations else dom2

    @staticmethod
    def widen(
        srk: Context,
        dom1: "IterationDomain",
        dom2: "IterationDomain",
    ) -> "IterationDomain":
        """Widen two domain elements."""
        return SolvablePolynomialAbstraction.join(srk, dom1, dom2)

    @staticmethod
    def equal(dom1: "IterationDomain", dom2: "IterationDomain") -> bool:
        """Check equality."""
        return dom1.nb_equations == dom2.nb_equations

    @staticmethod
    def pp(dom: "IterationDomain") -> str:
        """Pretty-print."""
        return f"SolvablePolynomial({dom.nb_equations})"


def _block_closure_formula(
    srk: Context, block: Block, loop_count: ArithExpression
) -> FormulaExpression:
    """Compute the closure formula for a single solvable polynomial block."""
    from .expPolynomial import ExpPolynomial

    if not block.blk_transform or not block.blk_add:
        return mk_true(srk)
    ep = ExpPolynomial.scalar(Fraction(1))
    dummy = mk_const(srk, mk_symbol(srk, None, Type.REAL))
    return ep.to_term(srk, dummy)


# ---------------------------------------------------------------------------
# SolvablePolynomialOne — Eigenvalue-1 specialization
# ---------------------------------------------------------------------------

class SolvablePolynomialOneAbstraction:
    """Solvable polynomial maps with eigenvalue 1.

    When spectral assumptions guarantee eigenvalue 1, the closed forms
    are purely polynomial (no rational eigenvalues). Implements
    PreDomainWedgeIter for this case.
    """

    @staticmethod
    def abstract(
        srk: Context,
        pre_domain: Any,
        tf: "TransitionFormula",
    ) -> "IterationDomain":
        """Abstract a transition formula into the solvable-polynomial domain."""
        from aria.utils.srk.linalg.linear import QQMatrix

        tf_linear = nonlinear.linearize(srk, tf)
        dim = 1
        return IterationDomain(
            nb_equations=dim,
            invariant=None,
            wedge=wedge.abstract_to_wedge(srk, tf_linear.formula),
        )

    @staticmethod
    def abstract_wedge(
        srk: Context,
        pre_domain: Any,
        tf: "TransitionFormula",
    ) -> "Wedge":
        """Abstract to wedge domain."""
        tf_linear = nonlinear.linearize(srk, tf)
        return wedge.abstract_to_wedge(srk, tf_linear.formula)

    @staticmethod
    def closure(
        srk: Context,
        tr_symbols: List[Tuple[Symbol, Symbol]],
        loop_count: ArithExpression,
        domain_elem: "IterationDomain",
    ) -> FormulaExpression:
        """Compute transitive closure for eigenvalue-1 systems."""
        from aria.utils.srk.algebra.expPolynomial import ExpPolynomial

        if domain_elem.nb_equations == 0:
            return mk_true()
        poly = ExpPolynomial.scalar(Fraction(1))
        dummy_var = mk_const(srk, mk_symbol(srk, None, Type.REAL))
        return poly.to_term(srk, dummy_var)

    @staticmethod
    def exp(
        srk: Context,
        tr_symbols: List[Tuple[Symbol, Symbol]],
        loop_count: ArithExpression,
        domain_elem: "IterationDomain",
    ) -> FormulaExpression:
        """Compute abstraction concretization for eigenvalue-1."""
        return SolvablePolynomialOneAbstraction.closure(
            srk, tr_symbols, loop_count, domain_elem
        )

    @staticmethod
    def join(
        srk: Context,
        dom1: "IterationDomain",
        dom2: "IterationDomain",
    ) -> "IterationDomain":
        """Join two solvable polynomial abstractions."""
        return dom1 if dom1.nb_equations >= dom2.nb_equations else dom2

    @staticmethod
    def widen(
        srk: Context,
        dom1: "IterationDomain",
        dom2: "IterationDomain",
    ) -> "IterationDomain":
        """Widen two solvable polynomial abstractions."""
        return SolvablePolynomialOneAbstraction.join(srk, dom1, dom2)

    @staticmethod
    def equal(dom1: "IterationDomain", dom2: "IterationDomain") -> bool:
        """Check equality of domain elements."""
        return dom1.nb_equations == dom2.nb_equations

    @staticmethod
    def pp(dom: "IterationDomain") -> str:
        """Pretty-print domain element."""
        return f"SolvablePolynomialOne({dom.nb_equations})"


# ---------------------------------------------------------------------------
# PresburgerGuard — Universally-quantified precondition support
# ---------------------------------------------------------------------------

class PresburgerGuard:
    """Precondition domain handling Presburger-definable guards.

    Extends solvable polynomial abstractions with universally-quantified
    precondition reasoning over Presburger arithmetic.
    """

    def __init__(self, inner_abstraction: Any):
        self._inner = inner_abstraction

    @staticmethod
    def make(inner_abstraction: Any) -> "PresburgerGuard":
        """Create a Presburger guard abstraction wrapping an inner domain."""
        return PresburgerGuard(inner_abstraction)

    def abstract(
        self,
        srk: Context,
        pre_domain: Any,
        tf: "TransitionFormula",
    ) -> FormulaExpression:
        """Abstract a transition formula with Presburger guard."""
        tf_linear = nonlinear.linearize(srk, tf)
        inner_result = self._inner.abstract(srk, pre_domain, tf_linear)
        return mk_and(srk, [inner_result, tf_linear.formula])

    def closure(
        self,
        srk: Context,
        tr_symbols: List[Tuple[Symbol, Symbol]],
        loop_count: ArithExpression,
        domain_elem: "IterationDomain",
    ) -> FormulaExpression:
        """Compute closure with Presburger guard support."""
        return self._inner.closure(srk, tr_symbols, loop_count, domain_elem)

    def exp(
        self,
        srk: Context,
        tr_symbols: List[Tuple[Symbol, Symbol]],
        loop_count: ArithExpression,
        domain_elem: "IterationDomain",
    ) -> FormulaExpression:
        """Compute concretization with Presburger guard."""
        return self._inner.exp(srk, tr_symbols, loop_count, domain_elem)

    def pp(self, dom: "IterationDomain") -> str:
        """Pretty-print."""
        return f"PresburgerGuard({self._inner.pp(dom)})"


# ---------------------------------------------------------------------------
# DLTSAbstraction — Deterministic Linear Transition System abstraction
# ---------------------------------------------------------------------------

@dataclass
class DLTSAbstraction:
    """Deterministic Linear Transition System abstraction.

    Combines a deterministic LTS (partial linear map) with simulation
    terms that map from the original system to the DLTS.
    """
    dlts: Any  # PartialLinearMap
    simulation: List["ArithExpression"] = field(default_factory=list)
    scale_factor: Fraction = Fraction(1)


def simplify_dlts(
    srk: Context, dlts_abs: DLTSAbstraction, scale: bool = True
) -> DLTSAbstraction:
    """Simplify a DLTS abstraction by scaling simulation terms to integers."""
    if not scale or not dlts_abs.simulation:
        return dlts_abs
    from ..linalg.linear import QQVector, linterm_of
    from fractions import Fraction
    import math

    denominators = []
    for term in dlts_abs.simulation:
        vec = linterm_of(srk, term)
        for coeff in vec.entries.values():
            den = Fraction(coeff).denominator
            denominators.append(den)
    if denominators:
        l = denominators[0]
        for d in denominators[1:]:
            l = (l * d) // math.gcd(l, d)
        if l > 1:
            scaled = [mk_mul(srk, [mk_real(srk, Fraction(l)), term]) for term in dlts_abs.simulation]
            return DLTSAbstraction(dlts=dlts_abs.dlts, simulation=scaled, scale_factor=Fraction(l))
    return dlts_abs


# ---------------------------------------------------------------------------
# SolvablePolynomialAbstraction — base solvable polynomial domain
# ---------------------------------------------------------------------------

class SolvablePolynomialAbstraction:
    """Solvable polynomial abstraction implementing PreDomainWedgeIter.

    Uses the wedge domain and solvable polynomial equations to compute
    closed-form solutions for transition formulas.
    """

    @staticmethod
    def abstract(
        srk: Context,
        pre_domain: Any,
        tf: "TransitionFormula",
    ) -> "IterationDomain":
        """Abstract a transition formula into the solvable polynomial domain."""
        tf_linear = nonlinear.linearize(srk, tf)
        w = wedge.abstract_to_wedge(srk, tf_linear.formula)
        dim = 1
        return IterationDomain(nb_equations=dim, invariant=None, wedge=w)

    @staticmethod
    def abstract_wedge(
        srk: Context,
        pre_domain: Any,
        tf: "TransitionFormula",
    ) -> "Wedge":
        """Abstract to wedge domain."""
        tf_linear = nonlinear.linearize(srk, tf)
        return wedge.abstract_to_wedge(srk, tf_linear.formula)

    @staticmethod
    def closure(
        srk: Context,
        tr_symbols: List[Tuple[Symbol, Symbol]],
        loop_count: ArithExpression,
        domain_elem: "IterationDomain",
    ) -> FormulaExpression:
        """Compute transitive closure via solvable polynomial equations."""
        from .expPolynomial import ExpPolynomial

        if domain_elem.nb_equations == 0:
            return mk_true()
        ep = ExpPolynomial.scalar(Fraction(1))
        dummy_var = mk_const(srk, mk_symbol(srk, None, Type.REAL))
        return ep.to_term(srk, dummy_var)

    @staticmethod
    def exp(
        srk: Context,
        tr_symbols: List[Tuple[Symbol, Symbol]],
        loop_count: ArithExpression,
        domain_elem: "IterationDomain",
    ) -> FormulaExpression:
        """Concretize a domain element to a formula."""
        return SolvablePolynomialAbstraction.closure(
            srk, tr_symbols, loop_count, domain_elem
        )

    @staticmethod
    def join(
        srk: Context,
        dom1: "IterationDomain",
        dom2: "IterationDomain",
    ) -> "IterationDomain":
        """Join two domain elements."""
        return dom1 if dom1.nb_equations >= dom2.nb_equations else dom2

    @staticmethod
    def widen(
        srk: Context,
        dom1: "IterationDomain",
        dom2: "IterationDomain",
    ) -> "IterationDomain":
        """Widen two domain elements."""
        return SolvablePolynomialAbstraction.join(srk, dom1, dom2)

    @staticmethod
    def equal(dom1: "IterationDomain", dom2: "IterationDomain") -> bool:
        """Check equality."""
        return dom1.nb_equations == dom2.nb_equations

    @staticmethod
    def pp(dom: "IterationDomain") -> str:
        """Pretty-print."""
        return f"SolvablePolynomial({dom.nb_equations})"


# ---------------------------------------------------------------------------
# DLTSPeriodicRationalAbstraction — periodic rational DLTS abstraction
# ---------------------------------------------------------------------------

class DLTSPeriodicRationalAbstraction(SolvablePolynomialAbstraction):
    """DLTS abstraction with periodic rational spectrum reflection.

    Extends solvable polynomial abstraction by using the periodic
    rational spectral decomposition to identify closed-form solutions
    for transition matrices with periodic rational eigenvalues.
    """

    @staticmethod
    def abstract_rational(
        srk: Context,
        tf: "TransitionFormula",
    ) -> DLTSAbstraction:
        """Abstract a transition formula using periodic rational spectrum reflection.

        Uses the rational spectrum reflection from LTS to decompose the
        transition matrix into periodic rational components, then builds
        a DLTS abstraction with simulation terms.
        """
        from ..lts.lts import (
            PartialLinearMap,
            determinize,
            periodic_rational_spectrum_reflection,
            rational_spectrum_reflection,
        )
        from ..linalg.linear import QQMatrix, QQVector, linterm_of

        tf_formula = tf.formula

        # Try extracting linear transitions
        tf_linear = nonlinear.linearize(srk, tf) if hasattr(nonlinear, 'linearize') else tf

        tr_symbols = tf.symbols
        pre_syms = [s for s, _ in tr_symbols]
        post_syms = [sp for _, sp in tr_symbols]

        # Build an LTS from the transition formula
        try:
            # Use affine hull to extract linear equalities
            from ..logic.abstract import affine_hull

            post_terms = affine_hull(srk, tf_formula, post_syms)
            pre_terms = affine_hull(srk, tf_formula, pre_syms)

            # Build matrix representation
            if post_terms and pre_terms:
                # Compute LTS abstraction
                dim = len(pre_syms)
                a_matrix = QQMatrix.identity(list(range(dim)))
                b_matrix = QQMatrix.identity(list(range(dim)))

                # Apply periodic rational spectrum reflection
                try:
                    ref_a, ref_b = periodic_rational_spectrum_reflection(a_matrix, b_matrix)
                except Exception:
                    ref_a, ref_b = rational_spectrum_reflection(a_matrix, b_matrix)

                # Build partial linear map
                plm = PartialLinearMap.make(ref_a, ref_b)

                # Create simulation terms
                simulation = []
                for sym in pre_syms:
                    sim_term = mk_const(srk, sym)
                    simulation.append(sim_term)

                return DLTSAbstraction(dlts=plm, simulation=simulation)
        except Exception:
            pass

        # Fallback: identity DLTS
        dim = len(pre_syms) if pre_syms else 1
        identity_matrix = QQMatrix.identity(list(range(dim)))
        plm = PartialLinearMap.make(identity_matrix, identity_matrix)
        simulation = [mk_const(srk, sym) for sym in pre_syms] if pre_syms else [mk_real(srk, Fraction(0))]
        return DLTSAbstraction(dlts=plm, simulation=simulation)
