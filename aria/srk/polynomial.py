"""
Polynomial operations and Groebner basis computation.

This module provides comprehensive polynomial functionality for symbolic computation
in the SRK system. It supports both univariate and multivariate polynomials over
the rational numbers, with various monomial orderings and advanced operations.

Key Features:
- Monomial representation with multiple ordering strategies
- Polynomial rings with rational coefficients (QQ[x1,...,xn])
- Groebner basis computation for ideal membership and elimination
- Polynomial division and reduction algorithms
- Integration with SRK's symbolic expression system
- Support for both sparse and dense polynomial representations

Example:
    >>> from aria.srk.polynomial import Polynomial, MonomialOrder
    >>> # Create polynomial: x^2 + 2*x*y + 3*y^2
    >>> p = Polynomial({(2,0): 1, (1,1): 2, (0,2): 3}, 2)  # 2 variables
    >>> print(p)  # x^2 + 2*x*y + 3*y^2
"""

from __future__ import annotations
from typing import (
    Dict,
    List,
    Set,
    Tuple,
    Optional,
    Union,
    Any,
    Iterator,
    Callable,
    overload,
)
from fractions import Fraction
from dataclasses import dataclass, field
from enum import Enum
import itertools
import math
from functools import cmp_to_key
import heapq

# Optional SymPy integration for advanced polynomial operations
try:
    import sympy as sp
    from sympy import (
        symbols,
        Poly,
        degree,
        LC,
        LT,
        gcd,
        factor,
        resultant,
        discriminant,
    )
    from sympy.polys import groebner

    HAS_SYMPY = True
except ImportError as e:
    HAS_SYMPY = False
    sp = None
except Exception as e:
    HAS_SYMPY = False
    sp = None

# Type aliases
# Using Fraction directly from fractions module


class MonomialOrder(Enum):
    """Monomial ordering strategies for multivariate polynomials.

    Different orderings affect how polynomials are compared and how Groebner
    bases are computed. The choice of ordering can significantly impact
    computational efficiency and the shape of computed bases.

    - LEX: Lexicographic ordering (dictionary order on variables)
    - DEGLEX: Degree lexicographic (total degree first, then lex)
    - DEGREVLEX: Degree reverse lexicographic (total degree first, then reverse lex)

    Example:
        >>> # For variables x > y > z, compare x^2*y vs x*y^2:
        >>> # LEX: x^2*y > x*y^2 (x^2*y comes after x*y^2 lexicographically)
        >>> # DEGLEX: x^2*y == x*y^2 (both degree 3, tie broken by lex)
    """

    LEX = "lex"  # Lexicographic
    DEGLEX = "deglex"  # Degree then lexicographic
    DEGREVLEX = "degrevlex"  # Degree then reverse lexicographic


@dataclass(frozen=True)
class Monomial:
    """Represents a monomial in a multivariate polynomial.

    A monomial is a product of variables raised to non-negative integer powers,
    such as x^2*y^3*z. This class represents the exponent tuple for n variables,
    where the monomial has n variables.

    The class is immutable (frozen) to ensure hashability for use in sets and
    dictionary keys, and to maintain consistency in polynomial operations.

    Attributes:
        exponents (Tuple[int, ...]): Tuple of non-negative integers representing
                                   the exponent of each variable in the monomial.

    Example:
        >>> # Monomial x^2*y^3 in 3 variables
        >>> m = Monomial((2, 3, 0))
        >>> print(m.exponents)  # (2, 3, 0)
    """

    exponents: Tuple[int, ...]

    def __init__(self, exponents: Union[List[int], Tuple[int, ...], Dict[int, int]]):
        """Initialize a monomial with variable exponents.

        Args:
            exponents: List, tuple, or dictionary of non-negative integers representing
                      the exponent of each variable. Lists/tuples are converted to tuples.
                      Dictionaries map variable indices to exponents.

        Raises:
            ValueError: If any exponent is negative.
        """
        if isinstance(exponents, dict):
            # Convert dictionary to tuple, finding the maximum variable index
            if not exponents:
                exponents = ()
            else:
                max_var = max(exponents.keys())
                exponents = tuple(exponents.get(i, 0) for i in range(max_var + 1))
        elif isinstance(exponents, list):
            exponents = tuple(exponents)

        # Validate non-negative exponents
        if any(exp < 0 for exp in exponents):
            raise ValueError("Monomial exponents must be non-negative")

        object.__setattr__(self, "exponents", exponents)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Monomial):
            return False
        return self.exponents == other.exponents

    def __hash__(self) -> int:
        return hash(self.exponents)

    def __mul__(self, other: Monomial) -> Monomial:
        """Multiply two monomials."""
        if len(self.exponents) != len(other.exponents):
            raise ValueError("Monomials must have same number of variables")
        return Monomial([a + b for a, b in zip(self.exponents, other.exponents)])

    def __truediv__(self, other: Monomial) -> Optional[Monomial]:
        """Divide monomials if possible."""
        if len(self.exponents) != len(other.exponents):
            return None
        new_exponents = []
        for a, b in zip(self.exponents, other.exponents):
            if a < b:
                return None
            new_exponents.append(a - b)
        return Monomial(new_exponents)

    def degree(self) -> int:
        """Total degree of the monomial."""
        return sum(self.exponents)

    def lcm(self, other: Monomial) -> Monomial:
        """Least common multiple of two monomials."""
        if len(self.exponents) != len(other.exponents):
            raise ValueError("Monomials must have same number of variables")
        return Monomial([max(a, b) for a, b in zip(self.exponents, other.exponents)])

    def gcd(self, other: Monomial) -> Monomial:
        """Greatest common divisor of two monomials."""
        if len(self.exponents) != len(other.exponents):
            raise ValueError("Monomials must have same number of variables")
        return Monomial([min(a, b) for a, b in zip(self.exponents, other.exponents)])

    def divides(self, other: Monomial) -> bool:
        """Check if this monomial divides another."""
        if len(self.exponents) != len(other.exponents):
            return False
        return all(a <= b for a, b in zip(self.exponents, other.exponents))

    def __lt__(self, other: Monomial) -> bool:
        """Less than comparison for sorting."""
        if len(self.exponents) != len(other.exponents):
            return len(self.exponents) < len(other.exponents)
        return self.compare(other, MonomialOrder.DEGLEX) < 0

    def __le__(self, other: Monomial) -> bool:
        """Less than or equal comparison."""
        return self == other or self < other

    def __gt__(self, other: Monomial) -> bool:
        """Greater than comparison."""
        return not self <= other

    def __ge__(self, other: Monomial) -> bool:
        """Greater than or equal comparison."""
        return not self < other

    def compare(self, other: Monomial, order: MonomialOrder) -> int:
        """Compare monomials according to the given order.

        Returns: -1 if self < other, 0 if equal, 1 if self > other
        """
        if len(self.exponents) != len(other.exponents):
            raise ValueError("Monomials must have same number of variables")

        if order == MonomialOrder.LEX:
            # Lexicographic order
            for a, b in zip(self.exponents, other.exponents):
                if a != b:
                    return -1 if a < b else 1
            return 0

        elif order == MonomialOrder.DEGLEX:
            # Degree then lexicographic
            self_deg = self.degree()
            other_deg = other.degree()
            if self_deg != other_deg:
                return -1 if self_deg < other_deg else 1
            # Same degree, use lex order
            for a, b in zip(self.exponents, other.exponents):
                if a != b:
                    return -1 if a < b else 1
            return 0

        elif order == MonomialOrder.DEGREVLEX:
            # Degree then reverse lexicographic
            self_deg = self.degree()
            other_deg = other.degree()
            if self_deg != other_deg:
                return -1 if self_deg < other_deg else 1
            # Same degree, use reverse lex order
            for a, b in zip(reversed(self.exponents), reversed(other.exponents)):
                if a != b:
                    return -1 if a < b else 1
            return 0

        else:
            raise ValueError(f"Unknown monomial order: {order}")

    def __str__(self) -> str:
        if not self.exponents:
            return "1"

        terms = []
        for i, exp in enumerate(self.exponents):
            if exp == 0:
                continue
            elif exp == 1:
                terms.append(f"x{i}")
            else:
                terms.append(f"x{i}^{exp}")

        return "*".join(terms) if terms else "1"

    def __repr__(self) -> str:
        return f"Monomial({list(self.exponents)})"


class MonomialOrdering:
    """Provides monomial comparison operations."""

    def __init__(
        self,
        num_vars: int,
        order: MonomialOrder,
        blocks: Optional[List[List[int]]] = None,
        block_orders: Optional[List[MonomialOrder]] = None,
    ):
        self.num_vars = num_vars
        self.order = order
        self.blocks = blocks
        self.block_orders = block_orders or []
        if self.blocks is not None:
            seen = sorted(i for block in self.blocks for i in block)
            if seen != list(range(num_vars)):
                raise ValueError("Monomial blocks must partition all variables")
            if self.block_orders and len(self.block_orders) != len(self.blocks):
                raise ValueError("block_orders must match blocks")

    def compare(self, m1: Monomial, m2: Monomial) -> int:
        """Compare two monomials."""
        if self.blocks is not None:
            for i, block in enumerate(self.blocks):
                block_order = (
                    self.block_orders[i] if self.block_orders else self.order
                )
                b1 = Monomial([m1.exponents[j] for j in block])
                b2 = Monomial([m2.exponents[j] for j in block])
                cmp = b1.compare(b2, block_order)
                if cmp != 0:
                    return cmp
            return 0
        return m1.compare(m2, self.order)

    def is_greater(self, m1: Monomial, m2: Monomial) -> bool:
        """Check if m1 > m2 according to this ordering."""
        return self.compare(m1, m2) > 0

    def is_greater_equal(self, m1: Monomial, m2: Monomial) -> bool:
        """Check if m1 >= m2 according to this ordering."""
        return self.compare(m1, m2) >= 0


@dataclass
class Polynomial:
    """Multivariate polynomial with rational coefficients."""

    terms: Dict[Monomial, Fraction]  # monomial -> coefficient

    def __init__(self, terms: Optional[Dict[Monomial, Fraction]] = None):
        self.terms = terms or {}

        # Remove zero coefficients
        zero_keys = [m for m, c in self.terms.items() if c == 0]
        for key in zero_keys:
            del self.terms[key]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Polynomial):
            return False
        return self.terms == other.terms

    def __hash__(self) -> int:
        # Create a sorted representation for consistent hashing
        items = sorted(self.terms.items(), key=lambda x: x[0])
        return hash(tuple((m, c) for m, c in items))

    def __add__(self, other: Union[Polynomial, Fraction]) -> Polynomial:
        """Add polynomial or scalar."""
        if isinstance(other, (int, Fraction)):
            # Add scalar: self + scalar
            result = Polynomial(self.terms.copy())
            zero_monom = Monomial((0,) * self.num_variables())
            result.terms[zero_monom] = result.terms.get(zero_monom, 0) + other

            # Clean up zero coefficients
            zero_keys = [m for m, c in result.terms.items() if c == 0]
            for key in zero_keys:
                del result.terms[key]

            return result
        elif isinstance(other, Polynomial):
            result = Polynomial(self.terms.copy())

            for monom, coeff in other.terms.items():
                result.terms[monom] = result.terms.get(monom, 0) + coeff

            # Clean up zero coefficients
            zero_keys = [m for m, c in result.terms.items() if c == 0]
            for key in zero_keys:
                del result.terms[key]

            return result
        else:
            raise TypeError(f"Cannot add {type(other)} to Polynomial")

    def __sub__(self, other: Union[Polynomial, Fraction]) -> Polynomial:
        """Subtract polynomial or scalar."""
        if isinstance(other, (int, Fraction)):
            # Subtract scalar: self - scalar = self + (-scalar)
            return self + Polynomial({Monomial((0,) * self.num_variables()): -other})
        elif isinstance(other, Polynomial):
            return self + (-other)
        else:
            raise TypeError(f"Cannot subtract {type(other)} from Polynomial")

    def __neg__(self) -> Polynomial:
        """Negate a polynomial."""
        return Polynomial({m: -c for m, c in self.terms.items()})

    def __rsub__(self, other: Fraction) -> Polynomial:
        """Right subtraction by scalar."""
        if isinstance(other, (int, Fraction)):
            # scalar - self = -self + scalar
            return -self + other
        else:
            raise TypeError(f"Cannot subtract Polynomial from {type(other)}")

    def __radd__(self, other: Fraction) -> Polynomial:
        """Right addition by scalar."""
        return self + other

    def __mul__(self, other: Union[Polynomial, Fraction, Monomial]) -> Polynomial:
        """Multiply polynomial by scalar, monomial, or another polynomial."""
        if isinstance(other, (int, Fraction)):
            if other == 0:
                return Polynomial()
            return Polynomial({m: c * other for m, c in self.terms.items()})
        elif isinstance(other, Monomial):
            # Multiply each term by the monomial
            result = Polynomial()
            for m, c in self.terms.items():
                new_monom = m * other
                result.terms[new_monom] = c
            return result
        elif isinstance(other, Polynomial):
            result = Polynomial()
            for m1, c1 in self.terms.items():
                for m2, c2 in other.terms.items():
                    new_monom = m1 * m2
                    new_coeff = c1 * c2
                    result.terms[new_monom] = result.terms.get(new_monom, 0) + new_coeff

            # Clean up zero coefficients
            zero_keys = [m for m, c in result.terms.items() if c == 0]
            for key in zero_keys:
                del result.terms[key]

            return result
        else:
            raise TypeError(f"Cannot multiply Polynomial by {type(other)}")

    def __rmul__(self, other: Fraction) -> Polynomial:
        """Right multiplication by scalar."""
        return self * other

    def __pow__(self, exponent: int) -> Polynomial:
        """Raise polynomial to integer power."""
        if not isinstance(exponent, int) or exponent < 0:
            raise ValueError("Exponent must be a non-negative integer")

        if exponent == 0:
            return Polynomial({Monomial((0,) * self.num_variables()): Fraction(1)})

        result = Polynomial({Monomial((0,) * self.num_variables()): Fraction(1)})
        for _ in range(exponent):
            result = result * self

        return result

    def __truediv__(
        self, other: Union[Polynomial, Fraction, Monomial]
    ) -> Optional[Polynomial]:
        """Divide polynomial by scalar, monomial, or polynomial."""
        if isinstance(other, (int, Fraction)):
            if other == 0:
                raise ZeroDivisionError("Division by zero")
            return self * (1 / other)
        elif isinstance(other, Monomial):
            return self.divide_by_monomial(other)
        elif isinstance(other, Polynomial):
            return self.divide_by_polynomial(other)
        else:
            raise TypeError(f"Cannot divide Polynomial by {type(other)}")

    def divide_by_monomial(self, monom: Monomial) -> Optional[Polynomial]:
        """Divide polynomial by a monomial if possible."""
        result = Polynomial()

        for m, c in self.terms.items():
            quotient = m / monom
            if quotient is None:
                return None  # Cannot divide
            result.terms[quotient] = c

        return result

    def divide_by_polynomial(self, other: Polynomial) -> Optional[Polynomial]:
        """Divide this polynomial by another polynomial using multivariate division."""
        if other.is_zero():
            raise ZeroDivisionError("Division by zero polynomial")

        # For now, implement simple division by monomial case
        if len(other.terms) == 1:
            monom, coeff = next(iter(other.terms.items()))
            if coeff == 1:
                return self.divide_by_monomial(monom)
            else:
                # Divide by scalar * monomial
                quotient = self.divide_by_monomial(monom)
                if quotient is not None:
                    return quotient * (1 / coeff)

        # General multivariate polynomial division (simplified implementation)
        # This implements a basic division algorithm where we find the leading term
        # of the divisor and divide by it, then recursively divide the remainder

        dividend = self
        divisor = other

        if divisor.is_zero():
            raise ZeroDivisionError("Division by zero polynomial")

        # Get leading terms
        dividend_lt = dividend.leading_term()
        divisor_lt = divisor.leading_term()

        if dividend_lt is None:
            return None  # dividend is zero

        if divisor_lt is None:
            raise ZeroDivisionError("Division by zero polynomial")

        dividend_coeff, dividend_monom = dividend_lt
        divisor_coeff, divisor_monom = divisor_lt

        # Check if leading monomial of divisor divides leading monomial of dividend
        quotient_monom = dividend_monom / divisor_monom
        if quotient_monom is None:
            return None  # Cannot divide

        # Compute quotient term: (dividend_coeff / divisor_coeff) * quotient_monom
        quotient_coeff = dividend_coeff / divisor_coeff
        quotient_term = Polynomial({quotient_monom: quotient_coeff})

        # Compute remainder: dividend - quotient_term * divisor
        quotient_times_divisor = quotient_term * divisor
        remainder = dividend - quotient_times_divisor

        # If remainder is zero, return quotient
        if remainder.is_zero():
            return quotient_term

        # Recursively divide remainder
        remainder_quotient = remainder.divide_by_polynomial(divisor)
        if remainder_quotient is None:
            return None  # Cannot divide remainder

        return quotient_term + remainder_quotient

    def degree(self) -> int:
        """Maximum total degree of any term."""
        if not self.terms:
            return -1
        return max(m.degree() for m in self.terms.keys())

    def leading_term(
        self, order: Optional[Union[MonomialOrder, MonomialOrdering]] = None
    ) -> Optional[Tuple[Fraction, Monomial]]:
        """Get the leading term according to the given order (or default order).

        Args:
            order: The monomial ordering to use. If None, uses DEGLEX.

        Returns:
            A tuple (coefficient, monomial) of the leading term, or None if polynomial is zero.
        """
        if not self.terms:
            return None

        # Get the number of variables from any monomial (they should all have the same)
        if self.terms:
            num_vars = len(list(self.terms.keys())[0].exponents)
        else:
            return None

        if order is None:
            ordering = MonomialOrdering(num_vars, MonomialOrder.DEGLEX)
        elif isinstance(order, MonomialOrdering):
            ordering = order
        else:
            ordering = MonomialOrdering(num_vars, order)

        # Find the maximum monomial according to the ordering
        max_monom = max(
            self.terms.keys(),
            key=cmp_to_key(ordering.compare),
        )

        return self.terms[max_monom], max_monom

    def leading_coefficient(
        self, order: Optional[Union[MonomialOrder, MonomialOrdering]] = None
    ) -> Fraction:
        """Get the leading coefficient according to the given order.

        Args:
            order: The monomial ordering to use. If None, uses DEGLEX.

        Returns:
            The leading coefficient.

        Raises:
            ValueError: If the polynomial is zero.
        """
        lt = self.leading_term(order)
        if lt is None:
            raise ValueError("Cannot get leading coefficient of zero polynomial")
        coeff, _ = lt
        return coeff

    def leading_monomial(
        self, order: Optional[Union[MonomialOrder, MonomialOrdering]] = None
    ) -> Monomial:
        """Get the leading monomial according to the given order.

        Args:
            order: The monomial ordering to use. If None, uses DEGLEX.

        Returns:
            The leading monomial.

        Raises:
            ValueError: If the polynomial is zero.
        """
        lt = self.leading_term(order)
        if lt is None:
            raise ValueError("Cannot get leading monomial of zero polynomial")
        _, monom = lt
        return monom

    def content(self) -> Fraction:
        """Greatest common divisor of all coefficients."""
        if not self.terms:
            return Fraction(0)

        coeffs = list(self.terms.values())
        gcd = coeffs[0]

        for coeff in coeffs[1:]:
            gcd = (
                math.gcd(gcd, coeff)
                if hasattr(math, "gcd")
                else self._gcd_rational(gcd, coeff)
            )

        return gcd

    def _gcd_rational(self, a: Fraction, b: Fraction) -> Fraction:
        """Compute GCD of two rationals."""
        # Convert to integers and compute GCD
        a_num, a_den = a.as_integer_ratio()
        b_num, b_den = b.as_integer_ratio()

        gcd_num = math.gcd(a_num, b_num)
        gcd_den = math.gcd(a_den, b_den)

        return Fraction(gcd_num, gcd_den)

    def dimensions(self) -> Set[int]:
        """Variables that appear in the polynomial."""
        dims = set()
        for monom in self.terms.keys():
            for i, exp in enumerate(monom.exponents):
                if exp > 0:
                    dims.add(i)
        return dims

    def evaluate(self, values: Dict[int, Fraction]) -> Fraction:
        """Evaluate the polynomial at given values."""
        result = Fraction(0)

        for monom, coeff in self.terms.items():
            term_value = coeff
            for i, exp in enumerate(monom.exponents):
                if exp > 0:
                    term_value *= values.get(i, Fraction(0)) ** exp
            result += term_value

        return result

    def substitute(self, substitutions: Dict[int, Polynomial]) -> Polynomial:
        """Substitute variables with polynomials."""
        result = Polynomial()

        for monom, coeff in self.terms.items():
            # Start with the constant term
            term = Polynomial({Monomial((0,) * len(monom.exponents)): coeff})

            # Get the maximum number of variables needed
            max_vars = len(monom.exponents)
            for sub_poly in substitutions.values():
                for sub_monom in sub_poly.terms.keys():
                    max_vars = max(max_vars, len(sub_monom.exponents))

            # Extend monomials to have the right number of variables
            extended_monom = Monomial(
                list(monom.exponents) + [0] * (max_vars - len(monom.exponents))
            )
            term = Polynomial({extended_monom: coeff})

            for i, exp in enumerate(monom.exponents):
                if exp > 0:
                    if i in substitutions:
                        var_poly = substitutions[i]
                        # Extend substitution polynomials to have enough variables
                        extended_subs = {}
                        for sub_monom, sub_coeff in var_poly.terms.items():
                            extended_sub_monom = Monomial(
                                list(sub_monom.exponents)
                                + [0] * (max_vars - len(sub_monom.exponents))
                            )
                            extended_subs[extended_sub_monom] = sub_coeff

                        var_poly = Polynomial(extended_subs)

                        # Multiply by variable^exp
                        var_term = var_poly
                        for _ in range(exp - 1):
                            var_term = var_term * var_poly
                        term = term * var_term
                    else:
                        # Keep variable as is (extend to max_vars)
                        var_monom = Monomial(
                            [1 if j == i else 0 for j in range(max_vars)]
                        )
                        var_term = Polynomial({var_monom: Fraction(1)})
                        for _ in range(exp - 1):
                            var_term = var_term * var_term
                        term = term * var_term

            result = result + term

        return result

    def __str__(self) -> str:
        if not self.terms:
            return "0"

        terms_str = []
        for monom, coeff in sorted(
            self.terms.items(), key=lambda x: x[0], reverse=True
        ):  # Sort by monomial for consistent output
            if coeff == 1 and monom != Monomial((0,) * len(monom.exponents)):
                terms_str.append(str(monom))
            elif coeff == -1 and monom != Monomial((0,) * len(monom.exponents)):
                terms_str.append(f"-{monom}")
            elif coeff != 0:
                terms_str.append(f"{coeff}*{monom}")

        return " + ".join(terms_str)

    def __repr__(self) -> str:
        return f"Polynomial({dict(self.terms)})"

    @staticmethod
    def scalar(k: Fraction) -> "Polynomial":
        """Create a scalar polynomial."""
        if k == 0:
            return Polynomial()
        return Polynomial({Monomial((0,)): k})

    @staticmethod
    def zero() -> "Polynomial":
        """Create zero polynomial."""
        return Polynomial()

    @staticmethod
    def one() -> "Polynomial":
        """Create one polynomial."""
        return Polynomial({Monomial((0,)): Fraction(1)})

    def add_term(
        self, coeff: Fraction, monom: Monomial, other: "Polynomial"
    ) -> "Polynomial":
        """Add a term to another polynomial."""
        result = Polynomial(other.terms.copy())
        result.terms[monom] = result.terms.get(monom, Fraction(0)) + coeff
        # Clean up zero coefficients
        zero_keys = [m for m, c in result.terms.items() if c == 0]
        for key in zero_keys:
            del result.terms[key]
        return result

    def scalar_mul(self, scalar: Fraction) -> "Polynomial":
        """Multiply polynomial by scalar."""
        if scalar == 0:
            return Polynomial()
        return Polynomial({m: c * scalar for m, c in self.terms.items()})

    def negate(self) -> "Polynomial":
        """Negate the polynomial."""
        return Polynomial({m: -c for m, c in self.terms.items()})

    @staticmethod
    def of_dim(dim: int, num_vars: int) -> "Polynomial":
        """Create a polynomial representing the variable at given dimension.

        Args:
            dim: The dimension (variable index) to create.
            num_vars: Total number of variables in the polynomial ring.

        Returns:
            A polynomial representing the variable x_dim.
        """
        exponents = [0] * num_vars
        if 0 <= dim < num_vars:
            exponents[dim] = 1
        else:
            raise ValueError(f"Dimension {dim} out of range for {num_vars} variables")
        return Polynomial({Monomial(tuple(exponents)): Fraction(1)})

    def enum(self) -> List[Tuple[Fraction, Monomial]]:
        """Enumerate terms as (coefficient, monomial) pairs."""
        return list(self.terms.items())

    def is_zero(self) -> bool:
        """Check if the polynomial is zero."""
        return len(self.terms) == 0

    def is_constant(self) -> bool:
        """Check if the polynomial is a constant."""
        return all(sum(monom.exponents) == 0 for monom in self.terms.keys())

    def is_monomial(self) -> bool:
        """Check if the polynomial is a single monomial."""
        return len(self.terms) == 1

    def to_sympy(self) -> Any:
        """Convert to SymPy polynomial if SymPy is available.

        Returns:
            A SymPy Poly object, or None if SymPy is not available or conversion fails.

        Raises:
            ImportError: If SymPy is not available.
        """
        if not HAS_SYMPY:
            raise ImportError("SymPy is not available")

        if self.is_zero():
            return sp.Poly(0, *sp.symbols(f"x:{self.num_variables()}"))

        # Get number of variables
        num_vars = self.num_variables()

        # Create SymPy symbols
        sympy_vars = sp.symbols(f"x:{num_vars}")

        # Convert terms to SymPy expression
        expr = 0
        for monom, coeff in self.terms.items():
            term = coeff
            for i, exp in enumerate(monom.exponents):
                if exp > 0:
                    term *= sympy_vars[i] ** exp
            expr += term

        return sp.Poly(expr, sympy_vars)

    @classmethod
    def from_sympy(cls, sympy_poly: Any) -> "Polynomial":
        """Create a Polynomial from a SymPy polynomial.

        Args:
            sympy_poly: A SymPy Poly object.

        Returns:
            A new Polynomial object.

        Raises:
            ImportError: If SymPy is not available.
            TypeError: If input is not a SymPy Poly.
        """
        if not HAS_SYMPY:
            raise ImportError("SymPy is not available")

        if not isinstance(sympy_poly, sp.Poly):
            raise TypeError("Input must be a SymPy Poly object")

        # Get the generators (variables)
        generators = sympy_poly.gens

        # Convert to our format
        terms = {}
        for term in sympy_poly.terms():
            # term is (exponents_tuple, coefficient)
            exponents = list(term[0])
            coeff = Fraction(term[1])

            our_monom = Monomial(exponents)
            terms[our_monom] = coeff

        return cls(terms)

    def factor(self) -> "Polynomial":
        """Factor the polynomial if SymPy is available.

        Returns:
            A new Polynomial that is the factored form, or the original if SymPy unavailable.

        Raises:
            ImportError: If SymPy is not available.
        """
        if not HAS_SYMPY:
            raise ImportError("SymPy is not available")

        try:
            sympy_poly = self.to_sympy()
            factored = sympy_poly.factor()
            return self.from_sympy(factored)
        except Exception:
            # If factoring fails, return original
            return self

    def gcd(self, other: "Polynomial") -> "Polynomial":
        """Compute GCD of two polynomials using SymPy if available.

        Args:
            other: Another polynomial.

        Returns:
            A new Polynomial representing the GCD.

        Raises:
            ImportError: If SymPy is not available.
        """
        if not HAS_SYMPY:
            raise ImportError("SymPy is not available")

        try:
            sympy_self = self.to_sympy()
            sympy_other = other.to_sympy()
            gcd_poly = sp.gcd(sympy_self, sympy_other)
            return self.from_sympy(gcd_poly)
        except Exception:
            # Fallback to simple coefficient GCD if SymPy fails
            return self._coefficient_gcd(other)

    def _coefficient_gcd(self, other: "Polynomial") -> "Polynomial":
        """Compute GCD of coefficients only."""
        if self.is_zero() or other.is_zero():
            return Polynomial()

        coeffs_self = list(self.terms.values())
        coeffs_other = list(other.terms.values())

        # Simple GCD of all coefficients
        gcd_val = coeffs_self[0]
        for c in coeffs_self[1:] + coeffs_other:
            gcd_val = self._gcd_rational(gcd_val, c)

        if gcd_val == 0:
            return Polynomial()

        return Polynomial({Monomial((0,) * self.num_variables()): gcd_val})

    def resultant(self, other: "Polynomial") -> Fraction:
        """Compute the resultant of two univariate polynomials.

        Uses SymPy when available; falls back to the Sylvester-matrix
        determinant computed with exact rational arithmetic.

        Args:
            other: Another polynomial (must be univariate).

        Returns:
            The resultant as a Fraction.
        """
        if HAS_SYMPY:
            try:
                sympy_self = self.to_sympy()
                sympy_other = other.to_sympy()
                return Fraction(resultant(sympy_self, sympy_other))
            except Exception:
                pass  # fall through to pure-Python path

        # Pure-Python fallback: Sylvester matrix determinant.
        # Works for univariate polynomials only.
        coeffs_f = self._univariate_coeffs()
        coeffs_g = other._univariate_coeffs()
        if coeffs_f is None or coeffs_g is None:
            raise ValueError(
                "resultant fallback only supports univariate polynomials"
            )
        return self._sylvester_resultant(coeffs_f, coeffs_g)

    def _univariate_coeffs(self) -> Optional[List[Fraction]]:
        """Return dense coefficient list [a0, a1, ..., an] for a univariate poly.

        Returns None if the polynomial is not univariate.
        """
        if not self.terms:
            return [Fraction(0)]
        # All monomials must have exactly one variable.
        for mono in self.terms:
            if len(mono.exponents) > 1:
                return None
        deg = max(
            (mono.exponents[0] if mono.exponents else 0) for mono in self.terms
        )
        coeffs = [Fraction(0)] * (deg + 1)
        for mono, coeff in self.terms.items():
            exp = mono.exponents[0] if mono.exponents else 0
            coeffs[exp] = Fraction(coeff)
        return coeffs

    @staticmethod
    def _sylvester_resultant(f: List[Fraction], g: List[Fraction]) -> Fraction:
        """Compute resultant via the Sylvester matrix determinant.

        f and g are dense coefficient lists [a0, a1, ..., an] (ascending degree).
        """
        # Strip leading zeros.
        while len(f) > 1 and f[-1] == 0:
            f = f[:-1]
        while len(g) > 1 and g[-1] == 0:
            g = g[:-1]
        m = len(f) - 1  # deg(f)
        n = len(g) - 1  # deg(g)
        if m == 0 and n == 0:
            return Fraction(1)
        size = m + n
        # Build Sylvester matrix (size x size).
        mat = [[Fraction(0)] * size for _ in range(size)]
        # n rows for f (shifted 0..n-1)
        for i in range(n):
            for j, c in enumerate(reversed(f)):
                mat[i][i + j] = c
        # m rows for g (shifted 0..m-1)
        for i in range(m):
            for j, c in enumerate(reversed(g)):
                mat[n + i][i + j] = c
        return Polynomial._det_rational(mat)

    @staticmethod
    def _det_rational(mat: List[List[Fraction]]) -> Fraction:
        """Compute determinant of a rational matrix via Gaussian elimination."""
        n = len(mat)
        mat = [row[:] for row in mat]  # copy
        det = Fraction(1)
        for col in range(n):
            # Find pivot.
            pivot_row = None
            for row in range(col, n):
                if mat[row][col] != 0:
                    pivot_row = row
                    break
            if pivot_row is None:
                return Fraction(0)
            if pivot_row != col:
                mat[col], mat[pivot_row] = mat[pivot_row], mat[col]
                det = -det
            pivot = mat[col][col]
            det *= pivot
            for row in range(col + 1, n):
                factor = mat[row][col] / pivot
                for c in range(col, n):
                    mat[row][c] -= factor * mat[col][c]
        return det

    def discriminant(self) -> Fraction:
        """Compute the discriminant of a univariate polynomial.

        Uses SymPy when available; falls back to the resultant-based formula:
        disc(f) = (-1)^(n*(n-1)/2) / lc(f) * res(f, f').

        Returns:
            The discriminant as a Fraction.
        """
        if HAS_SYMPY:
            try:
                sympy_poly = self.to_sympy()
                return Fraction(discriminant(sympy_poly))
            except Exception:
                pass  # fall through to pure-Python path

        # Pure-Python fallback.
        coeffs = self._univariate_coeffs()
        if coeffs is None:
            raise ValueError(
                "discriminant fallback only supports univariate polynomials"
            )
        n = len(coeffs) - 1  # degree
        if n <= 0:
            return Fraction(1)
        # Derivative coefficients.
        deriv = [Fraction(i) * coeffs[i] for i in range(1, n + 1)]
        res = self._sylvester_resultant(coeffs, deriv)
        lc = coeffs[-1]
        sign = Fraction((-1) ** (n * (n - 1) // 2))
        return sign / lc * res

    def num_variables(self) -> int:
        """Get the number of variables in the polynomial."""
        if not self.terms:
            return 0
        return len(list(self.terms.keys())[0].exponents)

    @staticmethod
    def term_of(srk, ctx_of_int, poly: "Polynomial"):
        """Convert polynomial to a term (placeholder implementation)."""
        # This is a simplified implementation
        # In practice, this would need to create actual SRK terms
        from .syntax import mk_var, Type

        return mk_var(0, Type.INT)  # Placeholder


# Compatibility alias for SRK interface
class QQ:
    """Rational numbers (fractions)."""

    @staticmethod
    def zero():
        return Fraction(0)

    @staticmethod
    def one():
        return Fraction(1)

    @staticmethod
    def equal(a, b):
        return a == b

    @staticmethod
    def add(a, b):
        return a + b

    @staticmethod
    def mul(a, b):
        return a * b

    @staticmethod
    def negate(a):
        return -a

    @staticmethod
    def lcm(a, b):
        """Least common multiple of denominators."""
        return Fraction(a * b // math.gcd(a, b) if a and b else 0)


class UnivariatePolynomial:
    """Univariate polynomial with rational coefficients."""

    def __init__(self, coeffs: List[Fraction]):
        """Initialize with coefficients from lowest to highest degree."""
        # Remove trailing zeros
        while len(coeffs) > 1 and coeffs[-1] == 0:
            coeffs.pop()

        self.coeffs = coeffs
        self.degree = len(coeffs) - 1

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, UnivariatePolynomial):
            return False
        return self.coeffs == other.coeffs

    def __hash__(self) -> int:
        return hash(tuple(self.coeffs))

    def __add__(self, other: UnivariatePolynomial) -> UnivariatePolynomial:
        """Add two univariate polynomials."""
        max_len = max(len(self.coeffs), len(other.coeffs))
        result_coeffs = [Fraction(0)] * max_len

        for i in range(len(self.coeffs)):
            result_coeffs[i] += self.coeffs[i]

        for i in range(len(other.coeffs)):
            result_coeffs[i] += other.coeffs[i]

        return UnivariatePolynomial(result_coeffs)

    def __mul__(self, other: UnivariatePolynomial) -> UnivariatePolynomial:
        """Multiply two univariate polynomials."""
        result_coeffs = [Fraction(0)] * (len(self.coeffs) + len(other.coeffs) - 1)

        for i, c1 in enumerate(self.coeffs):
            for j, c2 in enumerate(other.coeffs):
                result_coeffs[i + j] += c1 * c2

        return UnivariatePolynomial(result_coeffs)

    def evaluate(self, x: Fraction) -> Fraction:
        """Evaluate the polynomial at x."""
        result = Fraction(0)
        for i, coeff in enumerate(reversed(self.coeffs)):
            result = result * x + coeff
        return result

    def compose(self, other: UnivariatePolynomial) -> UnivariatePolynomial:
        """Compose this polynomial with another."""
        if self.degree < 0:
            return UnivariatePolynomial([Fraction(0)])

        result = UnivariatePolynomial([self.coeffs[-1]])
        x = UnivariatePolynomial([Fraction(0), Fraction(1)])  # x

        for coeff in reversed(self.coeffs[:-1]):
            result = result * other + UnivariatePolynomial([coeff])

        return result

    def derivative(self) -> UnivariatePolynomial:
        """Compute the derivative."""
        if self.degree <= 0:
            return UnivariatePolynomial([Fraction(0)])

        deriv_coeffs = []
        for i in range(1, len(self.coeffs)):
            deriv_coeffs.append(Fraction(i) * self.coeffs[i])

        return UnivariatePolynomial(deriv_coeffs)

    def __str__(self) -> str:
        if not self.coeffs:
            return "0"

        terms = []
        for i, coeff in enumerate(reversed(self.coeffs)):
            if coeff == 0:
                continue

            if i == 0:
                terms.append(str(coeff))
            elif i == 1:
                if coeff == 1:
                    terms.append("x")
                elif coeff == -1:
                    terms.append("-x")
                else:
                    terms.append(f"{coeff}*x")
            else:
                if coeff == 1:
                    terms.append(f"x^{i}")
                elif coeff == -1:
                    terms.append(f"-x^{i}")
                else:
                    terms.append(f"{coeff}*x^{i}")

        return " + ".join(terms) if terms else "0"

    def __repr__(self) -> str:
        return f"UnivariatePolynomial({self.coeffs})"


# Type aliases (defined after classes)
QQX = UnivariatePolynomial  # Univariate polynomials with rational coefficients


# Utility functions for creating polynomials
def zero() -> Polynomial:
    """Create the zero polynomial."""
    return Polynomial()


def one() -> Polynomial:
    """Create the constant 1 polynomial."""
    return Polynomial({Monomial(()): Fraction(1)})


def constant(c: Fraction) -> Polynomial:
    """Create a constant polynomial."""
    if c == 0:
        return zero()
    return Polynomial({Monomial(()): c})


def variable(index: int, num_vars: int) -> Polynomial:
    """Create a polynomial representing variable i.

    Args:
        index: The variable index.
        num_vars: Total number of variables in the polynomial ring.

    Returns:
        A polynomial representing the variable x_index.

    Raises:
        ValueError: If index is out of range.
    """
    if not 0 <= index < num_vars:
        raise ValueError(
            f"Variable index {index} out of range for {num_vars} variables"
        )
    exponents = [0] * num_vars
    exponents[index] = 1
    return Polynomial({Monomial(exponents): Fraction(1)})


def monomial(exponents: List[int], coeff: Fraction = Fraction(1)) -> Polynomial:
    """Create a monomial polynomial."""
    return Polynomial({Monomial(exponents): coeff})


def _extend_polynomial(poly: Polynomial, num_vars: int) -> Polynomial:
    """Pad monomials with trailing zero exponents."""
    if poly.is_zero():
        return Polynomial()
    if poly.num_variables() > num_vars:
        raise ValueError("Polynomial has more variables than target ring")
    return Polynomial(
        {
            Monomial(list(m.exponents) + [0] * (num_vars - len(m.exponents))): c
            for m, c in poly.terms.items()
        }
    )


def _permute_polynomial(poly: Polynomial, permutation: List[int]) -> Polynomial:
    """Return polynomial with new variable i equal to old variable permutation[i]."""
    return Polynomial(
        {
            Monomial([m.exponents[old_i] for old_i in permutation]): c
            for m, c in poly.terms.items()
        }
    )


def _inverse_permutation(permutation: List[int]) -> List[int]:
    inverse = [0] * len(permutation)
    for new_i, old_i in enumerate(permutation):
        inverse[old_i] = new_i
    return inverse


def _uses_only(monom: Monomial, variables: Set[int]) -> bool:
    return all(exp == 0 or i in variables for i, exp in enumerate(monom.exponents))


def _scale_to_monic(poly: Polynomial, order: MonomialOrder) -> Polynomial:
    if poly.is_zero():
        return poly
    return poly * (Fraction(1) / poly.leading_coefficient(order))


class Ideal:
    """Finitely generated polynomial ideal over rational coefficients."""

    def __init__(
        self,
        generators: Optional[List[Polynomial]] = None,
        order: MonomialOrder = MonomialOrder.DEGLEX,
        num_vars: Optional[int] = None,
    ):
        generators = generators or []
        self.order = order
        inferred_num_vars = max((p.num_variables() for p in generators), default=0)
        if num_vars is not None and num_vars < inferred_num_vars:
            raise ValueError("Ideal ring dimension is smaller than a generator")
        self.num_vars = num_vars if num_vars is not None else inferred_num_vars
        self.generators = [
            _extend_polynomial(p, self.num_vars) for p in generators if not p.is_zero()
        ]
        self._basis: Optional[RewriteSystem] = None

    @classmethod
    def make(
        cls,
        generators: List[Polynomial],
        order: MonomialOrder = MonomialOrder.DEGLEX,
    ) -> "Ideal":
        return cls(generators, order)

    def groebner_basis(self) -> "RewriteSystem":
        if self._basis is None:
            self._basis = groebner_basis(self.generators, self.order)
        return self._basis

    def reduce(self, poly: Polynomial) -> Polynomial:
        return self.groebner_basis().reduce(_extend_polynomial(poly, self.num_vars))

    def mem(self, poly: Polynomial) -> bool:
        """Return True when poly is in this ideal."""
        if poly.is_zero():
            return True
        return self.reduce(poly).is_zero()

    def sum(self, other: "Ideal") -> "Ideal":
        num_vars = max(self.num_vars, other.num_vars)
        return Ideal(
            [_extend_polynomial(p, num_vars) for p in self.generators]
            + [_extend_polynomial(p, num_vars) for p in other.generators],
            self.order,
            num_vars=num_vars,
        )

    def product(self, other: "Ideal") -> "Ideal":
        num_vars = max(self.num_vars, other.num_vars)
        left = [_extend_polynomial(p, num_vars) for p in self.generators]
        right = [_extend_polynomial(p, num_vars) for p in other.generators]
        return Ideal(
            [p * q for p in left for q in right],
            self.order,
            num_vars=num_vars,
        )

    def intersect(self, other: "Ideal") -> "Ideal":
        """Compute I cap J by eliminating t from tI + (1-t)J."""
        num_vars = max(self.num_vars, other.num_vars)
        t = variable(0, num_vars + 1)
        one_poly = Polynomial({Monomial((0,) * (num_vars + 1)): Fraction(1)})

        def shift(poly: Polynomial) -> Polynomial:
            poly = _extend_polynomial(poly, num_vars)
            return Polynomial(
                {Monomial((0,) + m.exponents): c for m, c in poly.terms.items()}
            )

        generators = [t * shift(p) for p in self.generators]
        generators.extend((one_poly - t) * shift(p) for p in other.generators)
        eliminated = Ideal(generators, MonomialOrder.LEX).project(
            set(range(1, num_vars + 1))
        )
        return Ideal(
            [
                Polynomial({Monomial(m.exponents[1:]): c for m, c in p.terms.items()})
                for p in eliminated.generators
            ],
            self.order,
            num_vars=num_vars,
        )

    def project(self, variables: Union[Set[int], List[int], Tuple[int, ...]]) -> "Ideal":
        """Eliminate variables not listed and return the induced ideal."""
        keep = set(variables)
        if not keep:
            keep = set()
        if any(i < 0 or i >= self.num_vars for i in keep):
            raise ValueError("Projection variable out of range")

        eliminate = [i for i in range(self.num_vars) if i not in keep]
        keep_ordered = [i for i in range(self.num_vars) if i in keep]
        permutation = eliminate + keep_ordered
        inverse = _inverse_permutation(permutation) if permutation else []
        moved = [_permute_polynomial(p, permutation) for p in self.generators]
        eliminated_count = len(eliminate)
        basis = _basis_polynomials(moved, MonomialOrder.LEX)

        projected = []
        for poly in basis:
            if all(
                all(m.exponents[i] == 0 for i in range(eliminated_count))
                for m in poly.terms
            ):
                restored = _permute_polynomial(poly, inverse)
                projected.append(
                    Polynomial(
                        {
                            m: c
                            for m, c in restored.terms.items()
                            if _uses_only(m, keep)
                        }
                    )
                )
        return Ideal(projected, self.order, num_vars=self.num_vars)


# Groebner basis computation (simplified implementation)
class RewriteRule:
    """A polynomial rewrite rule for Groebner basis computation."""

    def __init__(self, lhs: Monomial, rhs: Polynomial):
        self.lhs = lhs
        self.rhs = rhs

    def applies_to(self, monom: Monomial) -> bool:
        """Check if this rule applies to a monomial."""
        return self.lhs.divides(monom)

    def apply(self, poly: Polynomial) -> Optional[Polynomial]:
        """Apply this rule to a polynomial."""
        result = Polynomial()

        for m, c in poly.terms.items():
            quotient = m / self.lhs
            if quotient is not None:
                # Replace m with quotient * rhs
                replacement = self.rhs * quotient * c
                result = result + replacement
            else:
                result.terms[m] = c

        return result


class RewriteSystem:
    """A system of polynomial rewrite rules."""

    def __init__(self, rules: List[RewriteRule]):
        self.rules = rules

    def reduce(self, poly: Polynomial) -> Polynomial:
        """Reduce a polynomial using the rewrite rules."""
        current = poly
        changed = True

        while changed:
            changed = False
            for rule in self.rules:
                # Find terms that can be rewritten
                new_terms = {}
                rewritten = False

                for m, c in current.terms.items():
                    if rule.applies_to(m):
                        # Apply the rule
                        quotient = m / rule.lhs
                        if quotient is not None:
                            replacement = rule.rhs * c
                            # Add replacement terms
                            for rm, rc in replacement.terms.items():
                                new_monom = quotient * rm
                                new_terms[new_monom] = (
                                    new_terms.get(new_monom, Fraction(0)) + rc
                                )
                            rewritten = True
                        else:
                            new_terms[m] = c
                    else:
                        new_terms[m] = new_terms.get(m, Fraction(0)) + c

                if rewritten:
                    current = Polynomial(new_terms)
                    changed = True
                    break

        return current


def groebner_basis(
    polys: List[Polynomial], order: MonomialOrder = MonomialOrder.DEGLEX
) -> RewriteSystem:
    """Compute a Groebner basis using SymPy if available, otherwise simplified implementation.

    Args:
        polys: List of polynomials.
        order: Monomial ordering to use.

    Returns:
        A RewriteSystem containing the Groebner basis rules.

    Raises:
        ImportError: If SymPy is not available for full computation.
    """
    if not HAS_SYMPY:
        # Fallback to simplified implementation
        return _groebner_basis_simplified(polys, order)

    try:
        # Convert to SymPy polynomials
        sympy_polys = []
        for poly in polys:
            if not poly.is_zero():
                sympy_polys.append(poly.to_sympy())

        if not sympy_polys:
            return RewriteSystem([])

        # Compute Groebner basis using SymPy
        num_vars = max(p.num_variables() for p in polys)
        gb = groebner(
            sympy_polys, *sp.symbols(f"x:{num_vars}"), order=_sympy_order(order)
        )

        rules = _rewrite_rules_from_basis(
            [Polynomial.from_sympy(sp.Poly(poly, *gb.gens)) for poly in gb], order
        )

        return RewriteSystem(rules)

    except Exception:
        # Fallback to simplified implementation if SymPy fails
        return _groebner_basis_simplified(polys, order)


def _basis_polynomials(
    polys: List[Polynomial], order: MonomialOrder
) -> List[Polynomial]:
    """Compute Groebner basis polynomials and return them in local representation."""
    polys = [p for p in polys if not p.is_zero()]
    if not polys:
        return []
    if HAS_SYMPY:
        try:
            sympy_polys = [p.to_sympy() for p in polys]
            num_vars = max(p.num_variables() for p in polys)
            gb = groebner(
                sympy_polys, *sp.symbols(f"x:{num_vars}"), order=_sympy_order(order)
            )
            return [
                _scale_to_monic(Polynomial.from_sympy(sp.Poly(poly, *gb.gens)), order)
                for poly in gb
                if poly != 0
            ]
        except Exception:
            pass
    return _buchberger_basis(polys, order)


def _rewrite_rules_from_basis(
    basis: List[Polynomial], order: MonomialOrder
) -> List[RewriteRule]:
    rules = []
    for poly in basis:
        if poly.terms:
            lt_coeff, lt_monom = poly.leading_term(order)
            remainder = poly - Polynomial({lt_monom: lt_coeff})
            rules.append(
                RewriteRule(lt_monom, remainder * (Fraction(-1) / lt_coeff))
            )
    return rules


def _groebner_basis_simplified(
    polys: List[Polynomial], order: MonomialOrder
) -> RewriteSystem:
    """Simplified Groebner basis implementation using basic Buchberger's algorithm."""
    return RewriteSystem(
        _rewrite_rules_from_basis(_buchberger_basis(polys, order), order)
    )


def _buchberger_basis(polys: List[Polynomial], order: MonomialOrder) -> List[Polynomial]:
    """Compute a reduced Groebner basis using Buchberger's algorithm.

    The OCaml implementation orders pairs by the degree of the LCM of leading
    monomials and avoids relatively-prime pairs.  This fallback mirrors those
    sound parts without trying to be a full CAS replacement.
    """
    basis = _minimal_monic_basis(polys, order)
    pairs: List[Tuple[int, int, int]] = []
    queued: Set[Tuple[int, int]] = set()

    def enqueue(i: int, j: int) -> None:
        if i == j:
            return
        if i > j:
            i, j = j, i
        pair = (i, j)
        if pair in queued:
            return
        lm_i = basis[i].leading_monomial(order)
        lm_j = basis[j].leading_monomial(order)
        if lm_i.gcd(lm_j).degree() == 0:
            return
        queued.add(pair)
        heapq.heappush(pairs, (lm_i.lcm(lm_j).degree(), i, j))

    for i in range(len(basis)):
        for j in range(i + 1, len(basis)):
            enqueue(i, j)

    while pairs:
        _, i, j = heapq.heappop(pairs)
        if i >= len(basis) or j >= len(basis):
            continue
        s_poly = _s_polynomial(basis[i], basis[j], order)
        if s_poly is None or s_poly.is_zero():
            continue
        reduced = _scale_to_monic(_reduce_polynomial(s_poly, basis, order), order)
        if reduced.is_zero():
            continue
        if _leading_monomial_reducible(reduced, basis, order):
            reduced = _scale_to_monic(_reduce_polynomial(reduced, basis, order), order)
            if reduced.is_zero():
                continue
        if any(reduced == existing for existing in basis):
            continue
        new_index = len(basis)
        basis.append(reduced)
        for old_index in range(new_index):
            enqueue(old_index, new_index)

    return _self_reduce_basis(basis, order)


def _minimal_monic_basis(
    polys: List[Polynomial], order: MonomialOrder
) -> List[Polynomial]:
    """Normalize input generators and drop duplicate/reducible leading terms."""
    basis: List[Polynomial] = []
    for poly in polys:
        poly = _scale_to_monic(poly, order)
        if poly.is_zero() or any(poly == existing for existing in basis):
            continue
        if _leading_monomial_reducible(poly, basis, order):
            reduced = _scale_to_monic(_reduce_polynomial(poly, basis, order), order)
            if reduced.is_zero() or any(reduced == existing for existing in basis):
                continue
            poly = reduced
        basis = [
            existing
            for existing in basis
            if not poly.leading_monomial(order).divides(
                existing.leading_monomial(order)
            )
        ]
        basis.append(poly)
    return basis


def _leading_monomial_reducible(
    poly: Polynomial, basis: List[Polynomial], order: MonomialOrder
) -> bool:
    if poly.is_zero():
        return False
    lm = poly.leading_monomial(order)
    return any(
        not divisor.is_zero() and divisor.leading_monomial(order).divides(lm)
        for divisor in basis
    )


def _self_reduce_basis(
    basis: List[Polynomial], order: MonomialOrder
) -> List[Polynomial]:
    reduced_basis: List[Polynomial] = []
    for i, poly in enumerate(basis):
        others = basis[:i] + basis[i + 1 :]
        reduced = _scale_to_monic(_reduce_polynomial(poly, others, order), order)
        if reduced.is_zero():
            continue
        if any(reduced == existing for existing in reduced_basis):
            continue
        if _leading_monomial_reducible(reduced, reduced_basis, order):
            continue
        reduced_basis = [
            existing
            for existing in reduced_basis
            if not reduced.leading_monomial(order).divides(
                existing.leading_monomial(order)
            )
        ]
        reduced_basis.append(reduced)
    reduced_basis.sort(
        key=cmp_to_key(
            lambda p, q: -p.leading_monomial(order).compare(
                q.leading_monomial(order), order
            )
        )
    )
    return reduced_basis


def _s_polynomial(
    p1: Polynomial, p2: Polynomial, order: MonomialOrder
) -> Optional[Polynomial]:
    """Compute S-polynomial of two polynomials."""
    if p1.is_zero() or p2.is_zero():
        return None

    # Get leading terms
    lt1_coeff, lt1_monom = p1.leading_term(order)
    lt2_coeff, lt2_monom = p2.leading_term(order)

    # Compute LCM of leading monomials
    lcm_monom = lt1_monom.lcm(lt2_monom)

    # Compute S-polynomial: (lcm/lt1) * p1 - (lcm/lt2) * p2
    # Multiply polynomial first to avoid Monomial * Polynomial issue
    term1 = p1 * (lcm_monom / lt1_monom) * (Fraction(1) / lt1_coeff)
    term2 = p2 * (lcm_monom / lt2_monom) * (Fraction(1) / lt2_coeff)

    return term1 - term2


def _reduce_polynomial(
    poly: Polynomial, basis: List[Polynomial], order: MonomialOrder
) -> Polynomial:
    """Reduce polynomial with respect to a basis using multivariate division."""
    if poly.is_zero():
        return poly

    result = Polynomial()
    remainder = poly

    # Keep reducing until no more reductions possible
    changed = True
    while changed and not remainder.is_zero():
        changed = False

        for divisor in basis:
            if divisor.is_zero():
                continue

            # Try to divide remainder by divisor's leading term
            lt_div_coeff, lt_div_monom = divisor.leading_term(order)

            # Check if leading monomial of remainder is divisible by leading monomial of divisor
            rem_lt_coeff, rem_lt_monom = remainder.leading_term(order)

            quotient_monom = rem_lt_monom / lt_div_monom
            if quotient_monom is not None:
                # Compute quotient: (rem_lt_coeff / lt_div_coeff) * quotient_monom
                quotient_coeff = rem_lt_coeff / lt_div_coeff
                quotient = Polynomial({quotient_monom: quotient_coeff})

                # Subtract quotient * divisor from remainder
                remainder = remainder - quotient * divisor
                changed = True
                break

    return remainder


def _sympy_order(order: MonomialOrder) -> str:
    """Convert our monomial order to SymPy order."""
    if order == MonomialOrder.LEX:
        return "lex"
    elif order == MonomialOrder.DEGLEX:
        return "grlex"
    elif order == MonomialOrder.DEGREVLEX:
        return "grevlex"
    else:
        return "grlex"  # Default


# ---------------------------------------------------------------------------
# Missing OCaml API functions (QQXs, Rewrite, Ideal)
# ---------------------------------------------------------------------------

def of_vec(vec: "QQVector") -> Polynomial:
    """Convert a QQVector to a polynomial (one variable per dimension)."""
    from aria.srk.linear import QQVector
    terms: Dict[Monomial, Fraction] = {}
    if hasattr(vec, 'entries'):
        for dim, coeff in vec.entries.items():
            if coeff != 0:
                mon = Monomial((dim,))
                terms[mon] = coeff
    return Polynomial(terms)


def vec_of(poly: Polynomial) -> "QQVector":
    """Convert a polynomial (with one variable) to a QQVector."""
    from aria.srk.linear import QQVector
    entries: Dict[int, Fraction] = {}
    for mon, coeff in poly.terms.items():
        if len(mon.exponents) == 1:
            entries[mon.exponents[0]] = coeff
        elif len(mon.exponents) == 0:
            entries[-1] = coeff
    return QQVector(entries)


def split_linear(poly: Polynomial) -> Tuple[Polynomial, Polynomial]:
    """Split polynomial into linear and non-linear parts."""
    linear_terms: Dict[Monomial, Fraction] = {}
    nonlinear_terms: Dict[Monomial, Fraction] = {}
    for mon, coeff in poly.terms.items():
        deg = sum(abs(e) for e in mon.exponents)
        if deg <= 1:
            linear_terms[mon] = coeff
        else:
            nonlinear_terms[mon] = coeff
    return Polynomial(linear_terms), Polynomial(nonlinear_terms)


def factor_gcd(poly: Polynomial) -> Tuple[Fraction, Polynomial]:
    """Factor out the GCD of coefficients."""
    if poly.is_zero():
        return Fraction(0), poly
    coeffs = list(poly.terms.values())
    g = coeffs[0]
    for c in coeffs[1:]:
        g = _gcd_frac(g, c)
    if g == 0:
        return Fraction(0), poly
    new_terms = {mon: coeff / g for mon, coeff in poly.terms.items()}
    return g, Polynomial(new_terms)


def _gcd_frac(a: Fraction, b: Fraction) -> Fraction:
    import math
    na, da = abs(a.numerator), abs(a.denominator)
    nb, db = abs(b.numerator), abs(b.denominator)
    g_num = math.gcd(na, nb)
    g_den = da * db // math.gcd(da, db)
    return Fraction(g_num, g_den)


def split_leading(poly: Polynomial) -> Tuple[Tuple[Monomial, Fraction], Polynomial]:
    """Split off the leading term of the polynomial."""
    if poly.is_zero():
        return (Monomial(()), Fraction(0)), poly
    lt = poly.leading_term()
    lm = poly.leading_monomial()
    lc = poly.leading_coefficient()
    rest_terms = dict(poly.terms)
    del rest_terms[lm]
    return ((lm, lc), Polynomial(rest_terms))


def mk_rewrite(lhs: Polynomial, rhs: Polynomial) -> "RewriteRule":
    """Create a rewrite rule lhs -> rhs."""
    return RewriteRule(lhs, rhs)


def preduce(poly: Polynomial, rules: "RewriteSystem") -> Polynomial:
    """Reduce a polynomial using rewrite rules (alias for reduce)."""
    return rules.reduce(poly)


def add_saturate(rules: "RewriteSystem", poly: Polynomial) -> "RewriteSystem":
    """Add a polynomial to the rewrite system and saturate."""
    new_rules = list(rules.rules)
    new_rules.append(RewriteRule(poly, Polynomial.zero()))
    return RewriteSystem(new_rules)


def generators(ideal: "Ideal") -> List[Polynomial]:
    """Get the generators of an ideal."""
    return getattr(ideal, 'generators', [])


def ideal_subset(i1: "Ideal", i2: "Ideal") -> bool:
    """Check if ideal i1 is a subset of i2."""
    for g in generators(i1):
        if not i2.mem(g):
            return False
    return True


def ideal_equal(i1: "Ideal", i2: "Ideal") -> bool:
    """Check if two ideals are equal."""
    return ideal_subset(i1, i2) and ideal_subset(i2, i1)


def qq_choose(dim: int, k: int) -> Fraction:
    """Binomial coefficient as a rational."""
    import math
    if k < 0 or k > dim:
        return Fraction(0)
    return Fraction(math.comb(int(dim), int(k)), 1)
