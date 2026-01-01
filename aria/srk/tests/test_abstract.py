"""
Tests for the abstract domains module.
"""

import unittest
from fractions import Fraction
from aria.srk.syntax import Context, Symbol, Type
from aria.srk.abstract import (
    AbstractValue,
    SignDomain,
    AffineRelation,
    AffineDomain,
    ProductDomain,
    sign_domain,
    affine_domain,
    product_domain,
)


class TestSignDomain(unittest.TestCase):
    """Test sign analysis abstract domain."""

    def setUp(self):
        self.context = Context()
        self.x = self.context.mk_symbol("x", Type.INT)
        self.y = self.context.mk_symbol("y", Type.INT)

    def test_creation(self):
        """Test sign domain creation."""
        signs = {self.x: AbstractValue.POSITIVE, self.y: AbstractValue.NEGATIVE}
        domain = SignDomain(signs)

        self.assertEqual(domain.signs[self.x], AbstractValue.POSITIVE)
        self.assertEqual(domain.signs[self.y], AbstractValue.NEGATIVE)

    def test_join(self):
        """Test sign domain join operation."""
        signs1 = {self.x: AbstractValue.POSITIVE, self.y: AbstractValue.ZERO}
        signs2 = {self.x: AbstractValue.NEGATIVE, self.y: AbstractValue.NON_ZERO}

        domain1 = SignDomain(signs1)
        domain2 = SignDomain(signs2)

        joined = domain1.join(domain2)

        # x should be NON_ZERO (POSITIVE join NEGATIVE)
        # y should be NON_ZERO (ZERO join NON_ZERO)
        self.assertEqual(joined.signs[self.x], AbstractValue.NON_ZERO)
        self.assertEqual(joined.signs[self.y], AbstractValue.NON_ZERO)

    def test_meet(self):
        """Test sign domain meet operation."""
        signs1 = {self.x: AbstractValue.NON_NEGATIVE, self.y: AbstractValue.POSITIVE}
        signs2 = {self.x: AbstractValue.POSITIVE, self.y: AbstractValue.NON_NEGATIVE}

        domain1 = SignDomain(signs1)
        domain2 = SignDomain(signs2)

        met = domain1.meet(domain2)

        # x should be POSITIVE (intersection of NON_NEGATIVE and POSITIVE)
        # y should be POSITIVE (intersection of POSITIVE and NON_NEGATIVE)
        self.assertEqual(met.signs[self.x], AbstractValue.POSITIVE)
        self.assertEqual(met.signs[self.y], AbstractValue.POSITIVE)

    def test_projection(self):
        """Test sign domain projection."""
        signs = {self.x: AbstractValue.POSITIVE, self.y: AbstractValue.NEGATIVE}
        domain = SignDomain(signs)

        projected = domain.project({self.x})

        self.assertEqual(projected.signs[self.x], AbstractValue.POSITIVE)
        self.assertNotIn(self.y, projected.signs)

    def test_to_formula(self):
        """Test conversion to logical formulas."""
        signs = {self.x: AbstractValue.POSITIVE, self.y: AbstractValue.ZERO}
        domain = SignDomain(signs)

        formulas = domain.to_formula(self.context)

        # Should have formulas for x > 0 and y = 0
        self.assertEqual(len(formulas), 2)

    def test_bottom(self):
        """Test bottom element detection."""
        empty_domain = SignDomain({})
        self.assertTrue(empty_domain.is_bottom())

        non_empty_domain = SignDomain({self.x: AbstractValue.POSITIVE})
        self.assertFalse(non_empty_domain.is_bottom())


class TestAffineRelation(unittest.TestCase):
    """Test affine relations."""

    def setUp(self):
        self.context = Context()
        self.x = self.context.mk_symbol("x", Type.INT)
        self.y = self.context.mk_symbol("y", Type.INT)

    def test_creation(self):
        """Test affine relation creation."""
        coeffs = {self.x: Fraction(1), self.y: Fraction(-1)}
        relation = AffineRelation(coeffs, Fraction(5))

        self.assertEqual(relation.coefficients[self.x], Fraction(1))
        self.assertEqual(relation.coefficients[self.y], Fraction(-1))
        self.assertEqual(relation.constant, Fraction(5))

    def test_evaluation(self):
        """Test affine relation evaluation."""
        coeffs = {self.x: Fraction(2), self.y: Fraction(-1)}
        relation = AffineRelation(coeffs, Fraction(3))

        values = {self.x: Fraction(4), self.y: Fraction(1)}
        result = relation.evaluate(values)

        # 2*4 + (-1)*1 + 3 = 8 - 1 + 3 = 10
        self.assertEqual(result, Fraction(10))

    def test_string_representation(self):
        """Test string representation."""
        coeffs = {self.x: Fraction(1), self.y: Fraction(-2)}
        relation = AffineRelation(coeffs, Fraction(3))

        # Should be "x -2*y + 3 = 0"
        expected = "x -2*y + 3 = 0"
        self.assertEqual(str(relation), expected)


class TestAffineDomain(unittest.TestCase):
    """Test affine relations abstract domain."""

    def setUp(self):
        self.context = Context()
        self.x = self.context.mk_symbol("x", Type.INT)
        self.y = self.context.mk_symbol("y", Type.INT)

    def test_creation(self):
        """Test affine domain creation."""
        rel1 = AffineRelation({self.x: Fraction(1)}, Fraction(0))
        rel2 = AffineRelation({self.y: Fraction(1)}, Fraction(0))

        domain = AffineDomain([rel1, rel2])

        self.assertEqual(len(domain.relations), 2)

    def test_join(self):
        """Test affine domain join."""
        rel1 = AffineRelation({self.x: Fraction(1)}, Fraction(0))
        rel2 = AffineRelation({self.y: Fraction(1)}, Fraction(0))

        domain1 = AffineDomain([rel1])
        domain2 = AffineDomain([rel2])

        joined = domain1.join(domain2)

        # Should contain both relations
        self.assertEqual(len(joined.relations), 2)

    def test_projection(self):
        """Test affine domain projection."""
        rel1 = AffineRelation({self.x: Fraction(1)}, Fraction(0))
        rel2 = AffineRelation({self.y: Fraction(1)}, Fraction(0))

        domain = AffineDomain([rel1, rel2])
        projected = domain.project({self.x})

        # Should only contain relation for x
        self.assertEqual(len(projected.relations), 1)
        self.assertEqual(list(projected.relations[0].coefficients.keys()), [self.x])

    def test_to_formula(self):
        """Test conversion to logical formulas."""
        rel = AffineRelation({self.x: Fraction(1), self.y: Fraction(-1)}, Fraction(0))
        domain = AffineDomain([rel])

        formulas = domain.to_formula(self.context)

        # Should have one formula: x - y = 0
        self.assertEqual(len(formulas), 1)


class TestProductDomain(unittest.TestCase):
    """Test product abstract domains."""

    def setUp(self):
        self.context = Context()
        self.x = self.context.mk_symbol("x", Type.INT)

        self.sign_domain = SignDomain({self.x: AbstractValue.POSITIVE})
        self.affine_domain = AffineDomain([])

        self.product_domain = ProductDomain(self.sign_domain, self.affine_domain)

    def test_creation(self):
        """Test product domain creation."""
        self.assertEqual(self.product_domain.domain1, self.sign_domain)
        self.assertEqual(self.product_domain.domain2, self.affine_domain)

    def test_join(self):
        """Test product domain join."""
        other_sign = SignDomain({self.x: AbstractValue.NEGATIVE})
        other_product = ProductDomain(other_sign, self.affine_domain)

        joined = self.product_domain.join(other_product)

        # Sign should be joined, affine should remain the same
        self.assertEqual(joined.domain1.signs[self.x], AbstractValue.NON_ZERO)
        self.assertEqual(joined.domain2, self.affine_domain)

    def test_projection(self):
        """Test product domain projection."""
        projected = self.product_domain.project({self.x})

        # Both domains should be projected
        self.assertEqual(projected.domain1.signs[self.x], AbstractValue.POSITIVE)
        self.assertEqual(len(projected.domain2.relations), 0)

    def test_to_formula(self):
        """Test conversion to logical formulas."""
        formulas = self.product_domain.to_formula(self.context)

        # Should have formulas from sign domain
        self.assertGreater(len(formulas), 0)


if __name__ == "__main__":
    unittest.main()
