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

    def __init__(self, num_vars: int, order: MonomialOrder):
        self.num_vars = num_vars
        self.order = order

    def compare(self, m1: Monomial, m2: Monomial) -> int:
        """Compare two monomials."""
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

        dividend_monom, dividend_coeff = dividend_lt
        divisor_monom, divisor_coeff = divisor_lt

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
        self, order: Optional[MonomialOrder] = None
    ) -> Optional[Tuple[Fraction, Monomial]]:
        """Get the leading term according to the given order (or default order).

        Args:
            order: The monomial ordering to use. If None, uses DEGLEX.

        Returns:
            A tuple (coefficient, monomial) of the leading term, or None if polynomial is zero.
        """
        if not self.terms:
            return None

        if order is None:
            order = MonomialOrder.DEGLEX

        # Get the number of variables from any monomial (they should all have the same)
        if self.terms:
            num_vars = len(list(self.terms.keys())[0].exponents)
        else:
            return None

        ordering = MonomialOrdering(num_vars, order)

        # Find the maximum monomial according to the ordering
        max_monom = max(
            self.terms.keys(),
            key=lambda m: ordering.compare(m, Monomial((0,) * num_vars)),
        )

        return self.terms[max_monom], max_monom

    def leading_coefficient(self, order: Optional[MonomialOrder] = None) -> Fraction:
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

    def leading_monomial(self, order: Optional[MonomialOrder] = None) -> Monomial:
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
        """Compute the resultant of two polynomials using SymPy if available.

        Args:
            other: Another polynomial.

        Returns:
            The resultant as a Fraction.

        Raises:
            ImportError: If SymPy is not available.
        """
        if not HAS_SYMPY:
            raise ImportError("SymPy is not available")

        try:
            sympy_self = self.to_sympy()
            sympy_other = other.to_sympy()
            return Fraction(resultant(sympy_self, sympy_other))
        except Exception:
            raise NotImplementedError("Resultant computation failed")

    def discriminant(self) -> Fraction:
        """Compute the discriminant of the polynomial using SymPy if available.

        Returns:
            The discriminant as a Fraction.

        Raises:
            ImportError: If SymPy is not available.
        """
        if not HAS_SYMPY:
            raise ImportError("SymPy is not available")

        try:
            sympy_poly = self.to_sympy()
            return Fraction(discriminant(sympy_poly))
        except Exception:
            raise NotImplementedError("Discriminant computation failed")

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


# Groebner basis computation (simplified implementation)
class RewriteRule:
    """A polynomial rewrite rule for Groebner basis computation."""

    def __init__(self, lhs: Monomial, rhs: Polynomial):
        self.lhs = lhs
        self.rhs = rhs

    def applies_to(self, monom: Monomial) -> bool:
        """Check if this rule applies to a monomial."""
        return self.lhs.divides(monom) is not None

    def apply(self, poly: Polynomial) -> Optional[Polynomial]:
        """Apply this rule to a polynomial."""
        result = Polynomial()

        for m, c in poly.terms.items():
            quotient = m / self.lhs
            if quotient is not None:
                # Replace m with quotient * rhs
                replacement = self.rhs * c
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
                        new_terms[m] = c

                if rewritten:
                    current = Polynomial(new_terms)
                    changed = True
                    break

        return current


def groebner_basis(polys: List[Polynomial], order: MonomialOrder) -> RewriteSystem:
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
        gb = groebner(sympy_polys, order=_sympy_order(order))

        # Convert back to our format
        rules = []
        for poly in gb:
            if poly != 0:
                our_poly = Polynomial.from_sympy(poly)
                if our_poly.terms:
                    lt_coeff, lt_monom = our_poly.leading_term(order)
                    remainder = our_poly - Polynomial({lt_monom: lt_coeff})
                    if remainder.terms:
                        rules.append(RewriteRule(lt_monom, remainder))

        return RewriteSystem(rules)

    except Exception:
        # Fallback to simplified implementation if SymPy fails
        return _groebner_basis_simplified(polys, order)


def _groebner_basis_simplified(
    polys: List[Polynomial], order: MonomialOrder
) -> RewriteSystem:
    """Simplified Groebner basis implementation using basic Buchberger's algorithm."""
    # This is a basic implementation of Buchberger's algorithm.
    # For production use, the SymPy implementation is recommended.

    if not polys:
        return RewriteSystem([])

    # Filter out zero polynomials
    polys = [p for p in polys if not p.is_zero()]

    if not polys:
        return RewriteSystem([])

    # Get number of variables (assume all polynomials have same number)
    num_vars = polys[0].num_variables()

    # Initialize basis with input polynomials
    basis = polys.copy()
    rules = []

    # Buchberger's algorithm
    changed = True
    while changed:
        changed = False
        new_basis = basis.copy()

        # Check all pairs of polynomials
        for i in range(len(basis)):
            for j in range(i + 1, len(basis)):
                p1 = basis[i]
                p2 = basis[j]

                # Compute S-polynomial
                s_poly = _s_polynomial(p1, p2, order)
                if s_poly is None or s_poly.is_zero():
                    continue

                # Reduce S-polynomial with respect to current basis
                reduced = _reduce_polynomial(s_poly, basis, order)
                if not reduced.is_zero():
                    # Add remainder to basis if non-zero
                    new_basis.append(reduced)
                    changed = True

        basis = new_basis

    # Convert basis to rewrite rules
    for poly in basis:
        if poly.terms:
            lt_coeff, lt_monom = poly.leading_term(order)
            remainder = poly - Polynomial({lt_monom: lt_coeff})
            if remainder.terms:
                rules.append(RewriteRule(lt_monom, remainder))

    return RewriteSystem(rules)


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
        return "deglex"
    elif order == MonomialOrder.DEGREVLEX:
        return "degrevlex"
    else:
        return "deglex"  # Default
