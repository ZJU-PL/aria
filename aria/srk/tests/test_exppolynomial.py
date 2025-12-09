"""
Tests for the exponential polynomial module.
"""

import unittest
from fractions import Fraction
from aria.srk.syntax import Context, Symbol, Type
from aria.srk.expPolynomial import ExpPolynomial, ExpPolynomialVector, ExpPolynomialMatrix


class TestExpPolynomial(unittest.TestCase):
    """Test exponential polynomial functionality."""

    def setUp(self):
        self.context = Context()

    def test_zero_polynomial(self):
        """Test zero exponential polynomial."""
        from aria.srk.polynomial import zero
        from fractions import Fraction
        zero_poly = zero()
        zero = ExpPolynomial(zero_poly, Fraction(0))
        self.assertIsNotNone(zero)

    def test_constant_polynomial(self):
        """Test constant exponential polynomial."""
        from aria.srk.polynomial import constant
        from aria.srk.qQ import QQ
        from fractions import Fraction
        const_poly = constant(QQ.of_int(5))
        const = ExpPolynomial(const_poly, Fraction(1))
        self.assertIsNotNone(const)

    def test_variable_polynomial(self):
        """Test polynomial with a single variable."""
        x = self.context.mk_symbol("x", Type.INT)
        from aria.srk.polynomial import Polynomial, Monomial
        from aria.srk.qQ import QQ
        # Create a monomial for the variable
        monomial = Monomial({x.id: 1})
        var_poly = Polynomial({monomial: QQ.one()})
        var_term = ExpPolynomial(var_poly, Fraction(1))
        self.assertIsNotNone(var_term)

    def test_addition(self):
        """Test addition of exponential polynomials."""
        from aria.srk.polynomial import constant
        from aria.srk.qQ import QQ
        from fractions import Fraction

        p1_poly = constant(QQ.one())
        p2_poly = constant(QQ.of_int(2))
        p1 = ExpPolynomial(p1_poly, Fraction(1))
        p2 = ExpPolynomial(p2_poly, Fraction(1))

        result = p1 + p2
        self.assertIsNotNone(result)

    def test_multiplication(self):
        """Test multiplication of exponential polynomials."""
        from aria.srk.polynomial import constant
        from aria.srk.qQ import QQ
        from fractions import Fraction

        p1_poly = constant(QQ.of_int(2))
        p2_poly = constant(QQ.of_int(3))
        p1 = ExpPolynomial(p1_poly, Fraction(1))
        p2 = ExpPolynomial(p2_poly, Fraction(1))

        result = p1 * p2
        self.assertIsNotNone(result)


class TestExpPolynomialVector(unittest.TestCase):
    """Test exponential polynomial vector functionality."""

    def setUp(self):
        self.context = Context()

    def test_zero_vector(self):
        """Test zero exponential polynomial vector."""
        from aria.srk.polynomial import zero
        from fractions import Fraction
        zero_poly = zero()
        zero_comp = ExpPolynomial(zero_poly, Fraction(0))
        zero_vec = ExpPolynomialVector({0: zero_comp, 1: zero_comp})
        self.assertEqual(len(zero_vec.components), 2)

    def test_constant_vector(self):
        """Test constant exponential polynomial vector."""
        from aria.srk.polynomial import constant
        from aria.srk.qQ import QQ
        from fractions import Fraction

        const_poly = constant(QQ.one())
        const_comp = ExpPolynomial(const_poly, Fraction(1))
        const_vec = ExpPolynomialVector({0: const_comp, 1: const_comp})

        self.assertEqual(len(const_vec.components), 2)

    def test_vector_addition(self):
        """Test addition of exponential polynomial vectors."""
        from aria.srk.polynomial import constant
        from aria.srk.qQ import QQ
        from fractions import Fraction

        poly1 = constant(QQ.one())
        poly2 = constant(QQ.zero())
        comp1 = ExpPolynomial(poly1, Fraction(1))
        comp2 = ExpPolynomial(poly2, Fraction(1))

        v1 = ExpPolynomialVector({0: comp1, 1: comp2})
        v2 = ExpPolynomialVector({0: comp2, 1: comp1})
        result = v1 + v2

        self.assertEqual(len(result.components), 2)


class TestExpPolynomialMatrix(unittest.TestCase):
    """Test exponential polynomial matrix functionality."""

    def setUp(self):
        self.context = Context()

    def test_zero_matrix(self):
        """Test zero exponential polynomial matrix."""
        from aria.srk.polynomial import zero
        from fractions import Fraction
        zero_poly = zero()
        zero_comp = ExpPolynomial(zero_poly, Fraction(0))
        zero_vec = ExpPolynomialVector({0: zero_comp, 1: zero_comp})

        zero_mat = ExpPolynomialMatrix([zero_vec, zero_vec])
        self.assertEqual(len(zero_mat.rows), 2)

    def test_identity_matrix(self):
        """Test identity exponential polynomial matrix."""
        from aria.srk.polynomial import constant
        from aria.srk.qQ import QQ
        from fractions import Fraction

        # Create identity-like matrix (simplified)
        one_poly = constant(QQ.one())
        zero_poly = constant(QQ.zero())
        one_comp = ExpPolynomial(one_poly, Fraction(1))
        zero_comp = ExpPolynomial(zero_poly, Fraction(1))

        row1 = ExpPolynomialVector({0: one_comp, 1: zero_comp})
        row2 = ExpPolynomialVector({0: zero_comp, 1: one_comp})

        identity = ExpPolynomialMatrix([row1, row2])
        self.assertEqual(len(identity.rows), 2)

    def test_matrix_multiplication(self):
        """Test multiplication of exponential polynomial matrices."""
        # Create simple matrices for testing
        from aria.srk.polynomial import constant
        from aria.srk.qQ import QQ
        from fractions import Fraction

        # Use exponential part 0 for simplicity (constant matrices)
        one_poly = constant(QQ.one())
        zero_poly = constant(QQ.zero())
        one_comp = ExpPolynomial(one_poly, Fraction(0))
        zero_comp = ExpPolynomial(zero_poly, Fraction(0))

        row1 = ExpPolynomialVector({0: one_comp, 1: zero_comp})
        row2 = ExpPolynomialVector({0: zero_comp, 1: one_comp})

        m1 = ExpPolynomialMatrix([row1, row2])
        m2 = ExpPolynomialMatrix([row1, row2])
        result = m1 * m2

        self.assertEqual(len(result.rows), 2)


if __name__ == '__main__':
    unittest.main()
