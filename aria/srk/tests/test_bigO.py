"""
Tests for the asymptotic complexity analysis module.
"""

import unittest
from fractions import Fraction
from aria.srk.bigO import (
    ComplexityClass, BigO, compare, mul, add, minimum, maximum
)


class TestComplexityClass(unittest.TestCase):
    """Test ComplexityClass enumeration."""

    def test_enum_values(self):
        """Test that enum values are correct."""
        self.assertEqual(ComplexityClass.POLYNOMIAL.value, "polynomial")
        self.assertEqual(ComplexityClass.LOG.value, "log")
        self.assertEqual(ComplexityClass.EXP.value, "exp")
        self.assertEqual(ComplexityClass.UNKNOWN.value, "unknown")


class TestBigO(unittest.TestCase):
    """Test BigO class functionality."""

    def test_polynomial_creation(self):
        """Test polynomial complexity creation."""
        poly1 = BigO.polynomial(0)
        self.assertEqual(poly1.class_type, ComplexityClass.POLYNOMIAL)
        self.assertEqual(poly1.degree, 0)

        poly2 = BigO.polynomial(3)
        self.assertEqual(poly2.class_type, ComplexityClass.POLYNOMIAL)
        self.assertEqual(poly2.degree, 3)

    def test_log_creation(self):
        """Test logarithmic complexity creation."""
        log = BigO.log()
        self.assertEqual(log.class_type, ComplexityClass.LOG)
        self.assertEqual(log.degree, 0)

    def test_exp_creation(self):
        """Test exponential complexity creation."""
        exp = BigO.exp(Fraction(2))
        self.assertEqual(exp.class_type, ComplexityClass.EXP)
        self.assertEqual(exp.degree, Fraction(2))

    def test_unknown_creation(self):
        """Test unknown complexity creation."""
        unknown = BigO.unknown()
        self.assertEqual(unknown.class_type, ComplexityClass.UNKNOWN)
        self.assertEqual(unknown.degree, 0)

    def test_validation(self):
        """Test input validation."""
        # Polynomial degree must be integer
        with self.assertRaises(ValueError):
            BigO(ComplexityClass.POLYNOMIAL, Fraction(1, 2))

        # Exponential base must be rational
        with self.assertRaises(ValueError):
            BigO(ComplexityClass.EXP, 2)  # Should be Fraction(2)

        # Log and unknown should have degree 0
        with self.assertRaises(ValueError):
            BigO(ComplexityClass.LOG, 1)

        with self.assertRaises(ValueError):
            BigO(ComplexityClass.UNKNOWN, 1)

    def test_string_representation(self):
        """Test string representations."""
        # Polynomials
        self.assertEqual(str(BigO.polynomial(0)), "1")
        self.assertEqual(str(BigO.polynomial(1)), "n")
        self.assertEqual(str(BigO.polynomial(3)), "n^3")

        # Log
        self.assertEqual(str(BigO.log()), "log(n)")

        # Exponential
        self.assertEqual(str(BigO.exp(Fraction(2))), "2^n")

        # Unknown
        self.assertEqual(str(BigO.unknown()), "??")

    def test_repr(self):
        """Test repr for debugging."""
        poly = BigO.polynomial(2)
        self.assertEqual(repr(poly), "BigO(polynomial, 2)")

        exp = BigO.exp(Fraction(3, 2))
        self.assertEqual(repr(exp), "BigO(exp, 3/2)")


class TestCompare(unittest.TestCase):
    """Test complexity comparison function."""

    def test_polynomial_comparison(self):
        """Test polynomial degree comparison."""
        poly1 = BigO.polynomial(1)
        poly2 = BigO.polynomial(2)
        poly3 = BigO.polynomial(1)

        self.assertEqual(compare(poly1, poly2), "leq")  # n^1 <= n^2
        self.assertEqual(compare(poly2, poly1), "geq")  # n^2 >= n^1
        self.assertEqual(compare(poly1, poly3), "eq")   # n^1 == n^1

    def test_log_comparison(self):
        """Test logarithmic comparisons."""
        log = BigO.log()
        poly = BigO.polynomial(1)
        exp = BigO.exp(Fraction(2))

        self.assertEqual(compare(log, log), "eq")       # log(n) == log(n)
        self.assertEqual(compare(log, poly), "leq")     # log(n) <= n
        self.assertEqual(compare(poly, log), "geq")     # n >= log(n)
        self.assertEqual(compare(log, exp), "leq")      # log(n) <= 2^n
        self.assertEqual(compare(exp, log), "geq")      # 2^n >= log(n)

    def test_exponential_comparison(self):
        """Test exponential base comparison."""
        exp1 = BigO.exp(Fraction(2))
        exp2 = BigO.exp(Fraction(3))

        self.assertEqual(compare(exp1, exp2), "leq")   # 2^n <= 3^n
        self.assertEqual(compare(exp2, exp1), "geq")   # 3^n >= 2^n

    def test_unknown_cases(self):
        """Test comparisons involving unknowns."""
        unknown = BigO.unknown()
        poly = BigO.polynomial(1)

        self.assertEqual(compare(unknown, poly), "unknown")
        self.assertEqual(compare(poly, unknown), "unknown")
        self.assertEqual(compare(unknown, unknown), "unknown")

    def test_polynomial_vs_exponential(self):
        """Test polynomial vs exponential comparison."""
        poly = BigO.polynomial(100)
        exp = BigO.exp(Fraction(2))

        self.assertEqual(compare(poly, exp), "leq")    # n^100 <= 2^n
        self.assertEqual(compare(exp, poly), "geq")    # 2^n >= n^100


class TestMul(unittest.TestCase):
    """Test complexity multiplication."""

    def test_polynomial_multiplication(self):
        """Test polynomial degree addition."""
        poly1 = BigO.polynomial(2)
        poly2 = BigO.polynomial(3)
        result = mul(poly1, poly2)

        self.assertEqual(result.class_type, ComplexityClass.POLYNOMIAL)
        self.assertEqual(result.degree, 5)  # 2 + 3

    def test_identity_cases(self):
        """Test identity multiplication."""
        poly = BigO.polynomial(2)
        constant = BigO.polynomial(0)

        # Multiplying by constant should return original
        self.assertEqual(mul(poly, constant), poly)
        self.assertEqual(mul(constant, poly), poly)

    def test_exponential_multiplication(self):
        """Test exponential multiplication."""
        exp1 = BigO.exp(Fraction(2))
        exp2 = BigO.exp(Fraction(2))

        result = mul(exp1, exp2)
        self.assertEqual(result.class_type, ComplexityClass.EXP)
        self.assertEqual(result.degree, Fraction(2))  # Same base, take max

    def test_mixed_cases(self):
        """Test mixed complexity multiplication."""
        poly = BigO.polynomial(2)
        exp = BigO.exp(Fraction(2))

        # Should return unknown for mixed cases
        result = mul(poly, exp)
        self.assertEqual(result.class_type, ComplexityClass.UNKNOWN)


class TestAdd(unittest.TestCase):
    """Test complexity addition (maximum)."""

    def test_polynomial_addition(self):
        """Test polynomial degree maximization."""
        poly1 = BigO.polynomial(2)
        poly2 = BigO.polynomial(5)
        result = add(poly1, poly2)

        self.assertEqual(result.class_type, ComplexityClass.POLYNOMIAL)
        self.assertEqual(result.degree, 5)  # max(2, 5)

    def test_logarithmic_addition(self):
        """Test logarithmic addition."""
        log = BigO.log()
        poly = BigO.polynomial(2)

        # Log dominates
        result = add(log, poly)
        self.assertEqual(result.class_type, ComplexityClass.LOG)

    def test_exponential_dominance(self):
        """Test exponential dominance."""
        poly = BigO.polynomial(100)
        exp = BigO.exp(Fraction(2))

        # Exponential dominates
        result = add(poly, exp)
        self.assertEqual(result.class_type, ComplexityClass.EXP)
        self.assertEqual(result.degree, Fraction(2))

    def test_identity_cases(self):
        """Test identity addition."""
        poly = BigO.polynomial(3)
        constant = BigO.polynomial(0)

        # Adding constant should return original
        self.assertEqual(add(poly, constant), poly)
        self.assertEqual(add(constant, poly), poly)

    def test_unknown_cases(self):
        """Test addition with unknowns."""
        unknown = BigO.unknown()
        poly = BigO.polynomial(2)

        result = add(unknown, poly)
        self.assertEqual(result.class_type, ComplexityClass.UNKNOWN)


class TestMinimumMaximum(unittest.TestCase):
    """Test minimum and maximum operations."""

    def test_minimum(self):
        """Test minimum complexity selection."""
        poly1 = BigO.polynomial(2)
        poly2 = BigO.polynomial(5)

        result = minimum(poly1, poly2)
        self.assertEqual(result.class_type, ComplexityClass.POLYNOMIAL)
        self.assertEqual(result.degree, 2)  # min(2, 5)

        # Equal complexities
        result = minimum(poly1, BigO.polynomial(2))
        self.assertEqual(result.degree, 2)

    def test_maximum(self):
        """Test maximum complexity selection."""
        poly1 = BigO.polynomial(2)
        poly2 = BigO.polynomial(5)

        result = maximum(poly1, poly2)
        self.assertEqual(result.class_type, ComplexityClass.POLYNOMIAL)
        self.assertEqual(result.degree, 5)  # max(2, 5)

        # Equal complexities
        result = maximum(poly1, BigO.polynomial(2))
        self.assertEqual(result.degree, 2)

    def test_unknown_cases(self):
        """Test minimum/maximum with unknowns."""
        unknown = BigO.unknown()
        poly = BigO.polynomial(2)

        result = minimum(unknown, poly)
        self.assertEqual(result.class_type, ComplexityClass.UNKNOWN)

        result = maximum(unknown, poly)
        self.assertEqual(result.class_type, ComplexityClass.UNKNOWN)

    def test_cross_type_minimum(self):
        """Test minimum across different types."""
        log = BigO.log()
        poly = BigO.polynomial(2)

        # Log should be minimum compared to polynomial
        result = minimum(log, poly)
        self.assertEqual(result.class_type, ComplexityClass.LOG)

    def test_cross_type_maximum(self):
        """Test maximum across different types."""
        log = BigO.log()
        poly = BigO.polynomial(2)

        # Polynomial should be maximum compared to log
        result = maximum(log, poly)
        self.assertEqual(result.class_type, ComplexityClass.POLYNOMIAL)
        self.assertEqual(result.degree, 2)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def test_fractional_degrees(self):
        """Test fractional degrees for exponentials."""
        exp1 = BigO.exp(Fraction(1, 2))
        exp2 = BigO.exp(Fraction(3, 4))

        result = compare(exp1, exp2)
        self.assertEqual(result, "leq")  # 0.5^n <= 0.75^n

    def test_large_degrees(self):
        """Test with large polynomial degrees."""
        poly1 = BigO.polynomial(1000)
        poly2 = BigO.polynomial(100)

        result = compare(poly1, poly2)
        self.assertEqual(result, "geq")  # n^1000 >= n^100

    def test_zero_degree_polynomial(self):
        """Test zero-degree polynomial (constant)."""
        constant = BigO.polynomial(0)
        poly = BigO.polynomial(2)

        # Constants should be O(1)
        self.assertEqual(str(constant), "1")

        # Adding constant should not change result
        result = add(constant, poly)
        self.assertEqual(result, poly)

        result = mul(constant, poly)
        self.assertEqual(result, poly)


if __name__ == '__main__':
    unittest.main()
