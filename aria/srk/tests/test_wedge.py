"""
Tests for the Wedge (convex polyhedron) module.
"""

import unittest
from aria.srk.syntax import Context, Symbol, Type, mk_symbol, mk_const, mk_true, mk_false, mk_eq, mk_lt, mk_leq, mk_and, mk_or, mk_not, mk_ite, mk_add, mk_mul, mk_real
from aria.srk.wedge import WedgeDomain, WedgeElement
from aria.srk.qQ import QQ


class TestWedge(unittest.TestCase):
    """Test Wedge operations."""

    def setUp(self):
        """Set up test context and symbols."""
        self.ctx = Context()
        self.x = mk_symbol(self.ctx, 'x', Type.REAL)
        self.y = mk_symbol(self.ctx, 'y', Type.REAL)
        self.z = mk_symbol(self.ctx, 'z', Type.REAL)
        self.w = mk_symbol(self.ctx, 'w', Type.REAL)
        self.r = mk_symbol(self.ctx, 'r', Type.REAL)
        self.s = mk_symbol(self.ctx, 's', Type.REAL)

    def test_wedge_creation(self):
        """Test creating wedge elements."""
        # Create a simple wedge with constraints
        constraints = [
            mk_leq(mk_const(self.x), mk_real(self.ctx, QQ.of_int(10))),
            mk_leq(mk_real(self.ctx, QQ.zero()), mk_const(self.x)),
            mk_leq(mk_const(self.y), mk_const(self.x))
        ]

        wedge = WedgeElement(self.ctx, constraints)
        self.assertIsNotNone(wedge)

    def test_wedge_join(self):
        """Test joining wedge elements."""
        wedge1 = WedgeElement(self.ctx, [
            mk_leq(mk_const(self.x), mk_real(self.ctx, QQ.of_int(5)))
        ])

        wedge2 = WedgeElement(self.ctx, [
            mk_leq(mk_real(self.ctx, QQ.of_int(3)), mk_const(self.x))
        ])

        joined = wedge1.join(wedge2)
        self.assertIsNotNone(joined)

    def test_wedge_meet(self):
        """Test meeting wedge elements."""
        wedge1 = WedgeElement(self.ctx, [
            mk_leq(mk_const(self.x), mk_real(self.ctx, QQ.of_int(10)))
        ])

        wedge2 = WedgeElement(self.ctx, [
            mk_leq(mk_const(self.x), mk_real(self.ctx, QQ.of_int(7)))
        ])

        met = wedge1.meet(wedge2)
        self.assertIsNotNone(met)

    def test_wedge_exists(self):
        """Test existential quantification."""
        wedge = WedgeElement(self.ctx, [
            mk_leq(mk_const(self.x), mk_const(self.y)),
            mk_leq(mk_real(self.ctx, QQ.zero()), mk_const(self.x)),
            mk_leq(mk_const(self.y), mk_real(self.ctx, QQ.of_int(10)))
        ])

        # Existentially quantify x
        exists_x = wedge.exists([self.x])
        self.assertIsNotNone(exists_x)

    def test_wedge_projection(self):
        """Test projecting onto variables."""
        wedge = WedgeElement(self.ctx, [
            mk_leq(mk_const(self.x), mk_const(self.y)),
            mk_leq(mk_const(self.y), mk_const(self.z)),
            mk_leq(mk_real(self.ctx, QQ.zero()), mk_const(self.x))
        ])

        # Project onto y and z
        projected = wedge.project([self.y, self.z])
        self.assertIsNotNone(projected)

    def test_wedge_strengthen(self):
        """Test strengthening with additional constraints."""
        wedge = WedgeElement(self.ctx, [
            mk_leq(mk_real(self.ctx, QQ.zero()), mk_const(self.x)),
            mk_leq(mk_const(self.x), mk_real(self.ctx, QQ.of_int(10)))
        ])

        # Strengthen with x^2 <= 100
        additional = [mk_leq(mk_mul([mk_const(self.x), mk_const(self.x)]),
                           mk_real(self.ctx, QQ.of_int(100)))]

        strengthened = wedge.strengthen(additional)
        self.assertIsNotNone(strengthened)

    def test_wedge_is_bottom(self):
        """Test checking if wedge is bottom (empty)."""
        # Empty wedge should be bottom
        empty_wedge = WedgeElement(self.ctx, [])
        self.assertTrue(empty_wedge.is_bottom())

        # Non-empty wedge should not be bottom
        non_empty_wedge = WedgeElement(self.ctx, [
            mk_leq(mk_real(self.ctx, QQ.zero()), mk_const(self.x))
        ])
        self.assertFalse(non_empty_wedge.is_bottom())

    def test_wedge_to_atoms(self):
        """Test converting wedge to atomic formulas."""
        wedge = WedgeElement(self.ctx, [
            mk_leq(mk_const(self.x), mk_const(self.y)),
            mk_leq(mk_real(self.ctx, QQ.zero()), mk_const(self.x))
        ])

        atoms = wedge.to_atoms()
        self.assertIsInstance(atoms, list)
        self.assertGreater(len(atoms), 0)


class TestWedgeDomain(unittest.TestCase):
    """Test Wedge Domain operations."""

    def setUp(self):
        """Set up test context."""
        self.ctx = Context()

    def test_domain_creation(self):
        """Test creating wedge domains."""
        domain = WedgeDomain(self.ctx)
        self.assertIsNotNone(domain)

    def test_domain_join(self):
        """Test joining wedge domains."""
        domain1 = WedgeDomain(self.ctx)
        domain2 = WedgeDomain(self.ctx)

        joined = domain1.join(domain2)
        self.assertIsNotNone(joined)

    def test_domain_meet(self):
        """Test meeting wedge domains."""
        domain1 = WedgeDomain(self.ctx)
        domain2 = WedgeDomain(self.ctx)

        met = domain1.meet(domain2)
        self.assertIsNotNone(met)


if __name__ == '__main__':
    unittest.main()
