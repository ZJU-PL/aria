"""
Asymptotic complexity analysis (Big O notation) for SRK.

This module provides analysis of the asymptotic complexity of expressions,
classifying them into different complexity classes like polynomials, logarithms,
exponentials, and unknown cases.
"""

from __future__ import annotations
from typing import List, Union, Any
from fractions import Fraction
from enum import Enum
from dataclasses import dataclass

from fractions import Fraction as QQ
from aria.srk.syntax import Context


class ComplexityClass(Enum):
    """Enumeration of complexity classes."""
    POLYNOMIAL = "polynomial"
    LOG = "log"
    EXP = "exp"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class BigO:
    """Represents an asymptotic complexity class."""

    class_type: ComplexityClass
    degree: Union[int, QQ] = 0

    def __post_init__(self):
        if self.class_type == ComplexityClass.POLYNOMIAL:
            if not isinstance(self.degree, int):
                raise ValueError("Polynomial degree must be an integer")
        elif self.class_type == ComplexityClass.EXP:
            if not isinstance(self.degree, QQ):
                raise ValueError("Exponential base must be a rational")
        elif self.class_type in (ComplexityClass.LOG, ComplexityClass.UNKNOWN):
            if self.degree != 0:
                raise ValueError(f"{self.class_type.value} complexity should have degree 0")

    @staticmethod
    def polynomial(degree: int) -> BigO:
        """Create a polynomial complexity class."""
        return BigO(ComplexityClass.POLYNOMIAL, degree)

    @staticmethod
    def log() -> BigO:
        """Create a logarithmic complexity class."""
        return BigO(ComplexityClass.LOG, 0)

    @staticmethod
    def exp(base: QQ) -> BigO:
        """Create an exponential complexity class."""
        return BigO(ComplexityClass.EXP, base)

    @staticmethod
    def unknown() -> BigO:
        """Create an unknown complexity class."""
        return BigO(ComplexityClass.UNKNOWN, 0)

    def __str__(self) -> str:
        """String representation."""
        if self.class_type == ComplexityClass.POLYNOMIAL:
            if self.degree == 0:
                return "1"
            elif self.degree == 1:
                return "n"
            else:
                return f"n^{self.degree}"
        elif self.class_type == ComplexityClass.LOG:
            return "log(n)"
        elif self.class_type == ComplexityClass.EXP:
            return f"{self.degree}^n"
        else:
            return "??"

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"BigO({self.class_type.value}, {self.degree})"


def compare(x: BigO, y: BigO) -> str:
    """Compare two complexity classes.

    Returns:
        'eq' if equal, 'leq' if x <= y, 'geq' if x >= y, 'unknown' otherwise
    """
    # Handle Unknown cases
    if x.class_type == ComplexityClass.UNKNOWN or y.class_type == ComplexityClass.UNKNOWN:
        return "unknown"

    # Handle Log cases
    if x.class_type == ComplexityClass.LOG and y.class_type == ComplexityClass.LOG:
        return "eq"
    if x.class_type == ComplexityClass.LOG:
        return "leq"
    if y.class_type == ComplexityClass.LOG:
        return "geq"

    # Handle Polynomial cases
    if x.class_type == ComplexityClass.POLYNOMIAL and y.class_type == ComplexityClass.POLYNOMIAL:
        if x.degree == y.degree:
            return "eq"
        elif x.degree < y.degree:
            return "leq"
        else:
            return "geq"

    # Polynomial vs Exponential
    if x.class_type == ComplexityClass.POLYNOMIAL:
        return "leq"
    if y.class_type == ComplexityClass.POLYNOMIAL:
        return "geq"

    # Exponential cases
    if x.class_type == ComplexityClass.EXP and y.class_type == ComplexityClass.EXP:
        if x.degree == y.degree:
            return "eq"
        elif x.degree < y.degree:
            return "leq"
        else:
            return "geq"

    # Should not reach here
    return "unknown"


def mul(x: BigO, y: BigO) -> BigO:
    """Multiply two complexity classes."""
    # Identity cases
    if x.class_type == ComplexityClass.POLYNOMIAL and x.degree == 0:
        return y
    if y.class_type == ComplexityClass.POLYNOMIAL and y.degree == 0:
        return x

    # Both polynomials
    if (x.class_type == ComplexityClass.POLYNOMIAL and
        y.class_type == ComplexityClass.POLYNOMIAL):
        return BigO.polynomial(x.degree + y.degree)

    # Both exponentials with same base
    if (x.class_type == ComplexityClass.EXP and
        y.class_type == ComplexityClass.EXP and
        x.degree == y.degree):
        return BigO.exp(max(x.degree, y.degree))

    # Otherwise unknown
    return BigO.unknown()


def add(x: BigO, y: BigO) -> BigO:
    """Add two complexity classes (takes maximum)."""
    # Unknown cases
    if x.class_type == ComplexityClass.UNKNOWN or y.class_type == ComplexityClass.UNKNOWN:
        return BigO.unknown()

    # Identity cases
    if x.class_type == ComplexityClass.POLYNOMIAL and x.degree == 0:
        return y
    if y.class_type == ComplexityClass.POLYNOMIAL and y.degree == 0:
        return x

    # Log cases
    if x.class_type == ComplexityClass.LOG or y.class_type == ComplexityClass.LOG:
        return BigO.log()

    # Both polynomials - take maximum degree
    if (x.class_type == ComplexityClass.POLYNOMIAL and
        y.class_type == ComplexityClass.POLYNOMIAL):
        return BigO.polynomial(max(x.degree, y.degree))

    # Both exponentials - take maximum base
    if (x.class_type == ComplexityClass.EXP and
        y.class_type == ComplexityClass.EXP):
        return BigO.exp(max(x.degree, y.degree))

    # Exponential dominates
    if x.class_type == ComplexityClass.EXP:
        return x
    if y.class_type == ComplexityClass.EXP:
        return y

    # Should not reach here
    return BigO.unknown()


def minimum(x: BigO, y: BigO) -> BigO:
    """Take the minimum of two complexity classes."""
    cmp_result = compare(x, y)
    if cmp_result == "eq":
        return x
    elif cmp_result == "leq":
        return x
    elif cmp_result == "geq":
        return y
    else:
        return BigO.unknown()


def maximum(x: BigO, y: BigO) -> BigO:
    """Take the maximum of two complexity classes."""
    cmp_result = compare(x, y)
    if cmp_result == "eq":
        return x
    elif cmp_result == "leq":
        return y
    elif cmp_result == "geq":
        return x
    else:
        return BigO.unknown()


def of_arith_term(srk: Context, term: Any) -> BigO:
    """Analyze the complexity of an arithmetic term.

    This analyzes the asymptotic complexity by examining the structure
    of arithmetic expressions, identifying polynomial, logarithmic,
    exponential, and other complexity patterns.
    """
    from .syntax import destruct

    def analyze_term(t) -> BigO:
        """Recursively analyze the complexity of a term."""
        destruct_result = destruct(srk, t)

        if destruct_result[0] == 'Real':
            # Constants are O(1)
            return BigO.polynomial(0)

        elif destruct_result[0] == 'Var':
            # Single variables are O(1) in terms of complexity
            # (they represent input size)
            return BigO.polynomial(1)

        elif destruct_result[0] == 'Add':
            # Sum of terms - take the maximum complexity
            args = destruct_result[1] if len(destruct_result) > 1 else []
            complexities = [analyze_term(arg) for arg in args]
            return maximum(*complexities) if complexities else BigO.polynomial(0)

        elif destruct_result[0] == 'Mul':
            # Product of terms - add complexities
            args = destruct_result[1] if len(destruct_result) > 1 else []
            complexities = [analyze_term(arg) for arg in args]
            return sum_complexities(complexities)

        elif destruct_result[0] == 'Unop':
            op_type = destruct_result[1] if len(destruct_result) > 1 else None
            arg = destruct_result[2] if len(destruct_result) > 2 else None

            if op_type == 'Neg':
                # Negation doesn't change complexity
                return analyze_term(arg)

            elif op_type == 'Floor':
                # Floor is typically O(1) but depends on argument
                arg_complexity = analyze_term(arg)
                if arg_complexity.class_type == ComplexityClass.POLYNOMIAL:
                    return BigO.polynomial(max(0, arg_complexity.degree))
                else:
                    return arg_complexity

        elif destruct_result[0] == 'Binop':
            op_type = destruct_result[1] if len(destruct_result) > 1 else None
            left = destruct_result[2] if len(destruct_result) > 2 else None
            right = destruct_result[3] if len(destruct_result) > 3 else None

            if op_type == 'Div':
                # Division - complexity is max of numerator and denominator
                left_complexity = analyze_term(left)
                right_complexity = analyze_term(right)
                return maximum(left_complexity, right_complexity)

            elif op_type == 'Mod':
                # Modulo - typically O(1) but depends on arguments
                left_complexity = analyze_term(left)
                right_complexity = analyze_term(right)
                return maximum(left_complexity, right_complexity)

        elif destruct_result[0] == 'App':
            # Function application
            func = destruct_result[1] if len(destruct_result) > 1 else None
            args = destruct_result[2] if len(destruct_result) > 2 else []

            # Analyze function name for common complexity patterns
            func_name = str(func) if func else ""

            if 'pow' in func_name.lower() or 'power' in func_name.lower():
                # Power function - if base is constant and exponent is variable, it's exponential
                if args and len(args) >= 2:
                    base_complexity = analyze_term(args[0])
                    exp_complexity = analyze_term(args[1])

                    if (base_complexity.class_type == ComplexityClass.POLYNOMIAL and
                        base_complexity.degree == 0 and
                        exp_complexity.class_type == ComplexityClass.POLYNOMIAL and
                        exp_complexity.degree > 0):
                        # Constant base, variable exponent -> exponential
                        return BigO.exp(Fraction(2))  # Default base 2

            elif 'log' in func_name.lower() or 'logarithm' in func_name.lower():
                # Logarithmic function
                if args:
                    arg_complexity = analyze_term(args[0])
                    if arg_complexity.class_type == ComplexityClass.POLYNOMIAL:
                        return BigO.log()

            elif 'min' in func_name.lower() or 'max' in func_name.lower():
                # Min/max - complexity is max of arguments
                arg_complexities = [analyze_term(arg) for arg in args]
                return maximum(*arg_complexities) if arg_complexities else BigO.polynomial(0)

            # For other functions, analyze arguments and take maximum
            arg_complexities = [analyze_term(arg) for arg in args]
            return maximum(*arg_complexities) if arg_complexities else BigO.polynomial(0)

        elif destruct_result[0] == 'Ite':
            # Conditional - complexity is max of branches
            cond = destruct_result[1] if len(destruct_result) > 1 else None
            then_branch = destruct_result[2] if len(destruct_result) > 2 else None
            else_branch = destruct_result[3] if len(destruct_result) > 3 else None

            then_complexity = analyze_term(then_branch)
            else_complexity = analyze_term(else_branch)
            return maximum(then_complexity, else_complexity)

        # For other expression types or unrecognized patterns, return unknown
        return BigO.unknown()

    def sum_complexities(complexities: List[BigO]) -> BigO:
        """Sum complexities according to Big O rules."""
        if not complexities:
            return BigO.polynomial(0)

        # Filter out constants (degree 0 polynomials)
        non_constants = [c for c in complexities if not (c.class_type == ComplexityClass.POLYNOMIAL and c.degree == 0)]

        if not non_constants:
            return BigO.polynomial(0)

        # If any exponential, result is exponential
        exponentials = [c for c in non_constants if c.class_type == ComplexityClass.EXP]
        if exponentials:
            # Take the exponential with largest base
            return max(exponentials, key=lambda x: x.degree)

        # If any polynomial, take the one with highest degree
        polynomials = [c for c in non_constants if c.class_type == ComplexityClass.POLYNOMIAL]
        if polynomials:
            max_degree = max(c.degree for c in polynomials)
            return BigO.polynomial(max_degree)

        # If any logarithmic, result is logarithmic
        logs = [c for c in non_constants if c.class_type == ComplexityClass.LOG]
        if logs:
            return BigO.log()

        # Otherwise, return the first non-constant
        return non_constants[0]

    return analyze_term(term)
