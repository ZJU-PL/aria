"""
Operations on exponential polynomials.

Exponential polynomials are expressions of the form:
E ::= x | λ | λ^x | E*E | E+E
where λ is a rational number.

This module provides operations for working with exponential polynomials,
which are useful for analyzing complex recurrences and program termination.
"""

from __future__ import annotations
from typing import Dict, List, Set, Tuple, Optional, Union, Any, Iterator
from dataclasses import dataclass, field
from fractions import Fraction
from abc import ABC, abstractmethod

from aria.srk.syntax import Context, ArithExpression
from aria.srk.polynomial import Polynomial, QQX
from aria.srk.linear import QQVector, QQMatrix


class ExpPolynomial:
    """Represents an exponential polynomial."""

    def __init__(self, polynomial_part: Polynomial, exponential_part: Fraction):
        """Initialize with polynomial and exponential parts."""
        self.polynomial_part = polynomial_part
        self.exponential_part = exponential_part

    @staticmethod
    def scalar(coeff: Fraction) -> "ExpPolynomial":
        """Create an exponential polynomial from a scalar coefficient."""
        from .polynomial import constant

        return ExpPolynomial(constant(coeff), Fraction(1))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExpPolynomial):
            return False
        return (
            self.polynomial_part == other.polynomial_part
            and self.exponential_part == other.exponential_part
        )

    def __hash__(self) -> int:
        return hash((self.polynomial_part, self.exponential_part))

    def __add__(self, other: ExpPolynomial) -> ExpPolynomial:
        """Add two exponential polynomials."""
        if self.exponential_part != other.exponential_part:
            raise ValueError("Cannot add exponential polynomials with different bases")

        new_poly = self.polynomial_part + other.polynomial_part
        return ExpPolynomial(new_poly, self.exponential_part)

    def __mul__(self, other: ExpPolynomial) -> ExpPolynomial:
        """Multiply two exponential polynomials."""
        # (p * λ^a) * (q * λ^b) = p*q * λ^(a+b)
        new_poly = self.polynomial_part * other.polynomial_part
        new_exp = self.exponential_part + other.exponential_part
        return ExpPolynomial(new_poly, new_exp)

    def __neg__(self) -> ExpPolynomial:
        """Negate an exponential polynomial."""
        return ExpPolynomial(-self.polynomial_part, self.exponential_part)

    def evaluate(self, x: int) -> Fraction:
        """Evaluate the exponential polynomial at integer x."""
        base_value = (
            self.exponential_part**x if self.exponential_part != 0 else Fraction(1)
        )
        poly_value = self.polynomial_part.evaluate({0: Fraction(x)})
        return base_value * poly_value

    def summation(self) -> ExpPolynomial:
        """Compute the summation of this exponential polynomial."""
        # For exponential polynomials, summation is more complex
        # This is a simplified implementation
        return self  # Placeholder

    def solve_recurrence(
        self, initial: Fraction = Fraction(0), multiplier: Fraction = Fraction(1)
    ) -> ExpPolynomial:
        """Solve the recurrence g(n+1) = multiplier * g(n) + f(n)."""
        # This would solve the recurrence relation
        # For now, return a placeholder
        return ExpPolynomial(Polynomial(), Fraction(0))

    def compose_left_affine(self, a: int, b: int) -> ExpPolynomial:
        """Compose with affine function: λx. f(ax + b)."""
        # This is a complex operation for exponential polynomials
        # Placeholder implementation
        return self

    def to_term(self, context: Context, variable: ArithExpression) -> ArithExpression:
        """Convert to an arithmetic term."""
        # Placeholder implementation
        return variable

    def __str__(self) -> str:
        if self.exponential_part == 0:
            return str(self.polynomial_part)
        elif self.polynomial_part == Polynomial():
            return f"λ^{self.exponential_part}"
        else:
            return f"({self.polynomial_part}) * λ^{self.exponential_part}"

    @staticmethod
    def zero() -> "ExpPolynomial":
        """Create zero exponential polynomial."""
        return ExpPolynomial(Polynomial(), Fraction(0))

    @staticmethod
    def one() -> "ExpPolynomial":
        """Create unit exponential polynomial."""
        return ExpPolynomial(Polynomial(), Fraction(0))

    def flatten(self, period: List["ExpPolynomial"]) -> "ExpPolynomial":
        """Flatten periodic exponential polynomials."""
        # This is a simplified implementation
        # In the OCaml version, this handles ultimately periodic sequences
        if not period:
            return ExpPolynomial.zero()

        # For now, just return the first element
        return period[0] if period else ExpPolynomial.zero()

    @staticmethod
    def eval(ep: "ExpPolynomial", k: int) -> Fraction:
        """Evaluate exponential polynomial at point k."""
        return ep.evaluate(k)

    @staticmethod
    def mul(ep1: "ExpPolynomial", ep2: "ExpPolynomial") -> "ExpPolynomial":
        """Multiply two exponential polynomials."""
        return ep1 * ep2

    @staticmethod
    def add(ep1: "ExpPolynomial", ep2: "ExpPolynomial") -> "ExpPolynomial":
        """Add two exponential polynomials."""
        return ep1 + ep2

    def make(
        transient: List[Fraction], periodic: List["ExpPolynomial"]
    ) -> "ExpPolynomial":
        """Create ultimately periodic exponential polynomial."""
        # Simplified implementation - in OCaml this creates a UP from transient and periodic parts
        if not periodic:
            return ExpPolynomial.zero()

        # For now, just return the first periodic element
        return periodic[0] if periodic else ExpPolynomial.zero()

    def period_len(self) -> int:
        """Get period length."""
        return 1  # Simplified

    def compose_left_affine(self, a: int, b: int) -> "ExpPolynomial":
        """Compose with affine function: λx. f(ax + b)."""
        # This is a complex operation for exponential polynomials
        # Placeholder implementation
        return self

    def solve_rec(
        self, initial: Fraction = Fraction(0), lambda_val: Fraction = Fraction(1)
    ) -> "ExpPolynomial":
        """Solve recurrence g(n+1) = λ*g(n) + f(n)."""
        # This would solve the recurrence relation
        # For now, return a placeholder
        return ExpPolynomial(Polynomial(), Fraction(0))


class ExpPolynomialVector:
    """Vector of exponential polynomials."""

    def __init__(self, components: Dict[int, ExpPolynomial]):
        self.components = components

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExpPolynomialVector):
            return False
        return self.components == other.components

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.components.items())))

    def __add__(self, other: ExpPolynomialVector) -> ExpPolynomialVector:
        """Add two exponential polynomial vectors."""
        result = {}
        all_dims = set(self.components.keys()) | set(other.components.keys())

        for dim in all_dims:
            comp1 = self.components.get(dim, ExpPolynomial(Polynomial(), Fraction(0)))
            comp2 = other.components.get(dim, ExpPolynomial(Polynomial(), Fraction(0)))
            result[dim] = comp1 + comp2

        return ExpPolynomialVector(result)

    def __mul__(self, scalar: Fraction) -> ExpPolynomialVector:
        """Multiply by scalar."""
        result = {}
        for dim, comp in self.components.items():
            result[dim] = ExpPolynomial(
                comp.polynomial_part * scalar, comp.exponential_part
            )

        return ExpPolynomialVector(result)

    def evaluate(self, x: int) -> Dict[int, Fraction]:
        """Evaluate all components."""
        return {dim: comp.evaluate(x) for dim, comp in self.components.items()}

    def __str__(self) -> str:
        terms = []
        for dim in sorted(self.components.keys()):
            comp = self.components[dim]
            terms.append(f"e{dim}: {comp}")

        return "{" + ", ".join(terms) + "}"


class ExpPolynomialMatrix:
    """Matrix of exponential polynomials."""

    def __init__(self, rows: List[ExpPolynomialVector]):
        self.rows = rows

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExpPolynomialMatrix):
            return False
        return self.rows == other.rows

    def __hash__(self) -> int:
        return hash(tuple(self.rows))

    def __mul__(self, other: ExpPolynomialMatrix) -> ExpPolynomialMatrix:
        """Matrix multiplication."""
        if not self.rows or not other.rows:
            return ExpPolynomialMatrix([])

        # Number of columns in self must equal number of rows in other
        if len(self.rows[0].components) != len(other.rows):
            raise ValueError("Incompatible matrix dimensions")

        result_rows = []

        for row in self.rows:
            result_row = {}
            for j in range(len(other.rows[0].components)):
                # Compute dot product of row with column j of other
                dot_product = ExpPolynomial(Polynomial(), Fraction(0))

                for i in range(len(other.rows)):
                    coeff_self = row.components.get(
                        i, ExpPolynomial(Polynomial(), Fraction(0))
                    )
                    coeff_other = other.rows[i].components.get(
                        j, ExpPolynomial(Polynomial(), Fraction(0))
                    )
                    dot_product = dot_product + (coeff_self * coeff_other)

                result_row[j] = dot_product

            result_rows.append(ExpPolynomialVector(result_row))

        return ExpPolynomialMatrix(result_rows)

    def __str__(self) -> str:
        return "\n".join(str(row) for row in self.rows)


# Factory functions
def zero_exp_polynomial() -> ExpPolynomial:
    """Create the zero exponential polynomial."""
    return ExpPolynomial(Polynomial(), Fraction(0))


def one_exp_polynomial() -> ExpPolynomial:
    """Create the constant 1 exponential polynomial."""
    from aria.srk.polynomial import Monomial

    return ExpPolynomial(Polynomial({Monomial(()): Fraction(1)}), Fraction(0))


def constant_exp_polynomial(c: Fraction) -> ExpPolynomial:
    """Create a constant exponential polynomial."""
    from aria.srk.polynomial import Monomial

    return ExpPolynomial(Polynomial({Monomial(()): c}), Fraction(0))


def variable_exp_polynomial() -> ExpPolynomial:
    """Create the variable x exponential polynomial."""
    from aria.srk.polynomial import Monomial

    return ExpPolynomial(Polynomial({Monomial([1]): Fraction(1)}), Fraction(0))


def exponential_exp_polynomial(base: Fraction) -> ExpPolynomial:
    """Create an exponential λ^x."""
    return ExpPolynomial(Polynomial(), base)


def polynomial_to_exp_polynomial(poly: Polynomial) -> ExpPolynomial:
    """Convert a polynomial to an exponential polynomial."""
    return ExpPolynomial(poly, Fraction(0))


def exp_polynomial_from_term(poly: Polynomial, base: Fraction) -> ExpPolynomial:
    """Create an exponential polynomial from a polynomial term."""
    return ExpPolynomial(poly, base)


# Operations on exponential polynomials
def exp_polynomial_add(ep1: ExpPolynomial, ep2: ExpPolynomial) -> ExpPolynomial:
    """Add two exponential polynomials."""
    return ep1 + ep2


def exp_polynomial_mul(ep1: ExpPolynomial, ep2: ExpPolynomial) -> ExpPolynomial:
    """Multiply two exponential polynomials."""
    return ep1 * ep2


def exp_polynomial_summation(ep: ExpPolynomial) -> ExpPolynomial:
    """Compute the summation of an exponential polynomial."""
    return ep.summation()


def exp_polynomial_solve_recurrence(
    ep: ExpPolynomial,
    initial: Fraction = Fraction(0),
    multiplier: Fraction = Fraction(1),
) -> ExpPolynomial:
    """Solve a recurrence relation."""
    return ep.solve_recurrence(initial, multiplier)


def exp_polynomial_compose_left_affine(
    ep: ExpPolynomial, a: int, b: int
) -> ExpPolynomial:
    """Compose with affine function."""
    return ep.compose_left_affine(a, b)


# Vector operations
def exp_polynomial_vector_from_qqvector(vec: QQVector) -> ExpPolynomialVector:
    """Convert a QQVector to an exponential polynomial vector."""
    components = {}
    for dim, coeff in vec.entries.items():
        components[dim] = ExpPolynomial(Polynomial({coeff}), Fraction(0))

    return ExpPolynomialVector(components)


def exp_polynomial_matrix_from_qqmatrix(matrix: QQMatrix) -> ExpPolynomialMatrix:
    """Convert a QQMatrix to an exponential polynomial matrix."""
    rows = []
    for row in matrix.rows:
        rows.append(exp_polynomial_vector_from_qqvector(row))

    return ExpPolynomialMatrix(rows)


def exp_polynomial_exponentiate_rational(
    matrix: QQMatrix,
) -> Optional[ExpPolynomialMatrix]:
    """Symbolically exponentiate a matrix with rational eigenvalues."""
    # This is a complex operation that would require eigenvalue computation
    # For now, return None to indicate irrational eigenvalues
    return None


# Enumeration and conversion
def exp_polynomial_enum(ep: ExpPolynomial) -> Iterator[Tuple[Polynomial, Fraction]]:
    """Enumerate the terms of an exponential polynomial."""
    # Placeholder implementation
    yield (ep.polynomial_part, ep.exponential_part)


def exp_polynomial_to_term(
    context: Context, variable: ArithExpression, ep: ExpPolynomial
) -> ArithExpression:
    """Convert exponential polynomial to arithmetic term."""
    return ep.to_term(context, variable)


# Type alias for compatibility with OCaml naming
UP = ExpPolynomial
