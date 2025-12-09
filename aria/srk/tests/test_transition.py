"""
Tests for the Transition module.
"""

import unittest
from aria.srk.syntax import Context, Symbol, Type, mk_symbol, mk_const, mk_lt, mk_add, mk_real
from aria.srk.transition import Transition, TransitionSystem
from aria.srk.abstract import SignDomain, AbstractValue
from aria.srk.qQ import QQ


class TestTransition(unittest.TestCase):
    """Test Transition operations."""

    def setUp(self):
        """Set up test context and symbols."""
        self.ctx = Context()
        self.x = mk_symbol(self.ctx, 'x', Type.INT)
        self.n = mk_symbol(self.ctx, 'n', Type.INT)

    def test_transition_creation(self):
        """Test basic transition creation."""
        tr1 = Transition.assume(self.ctx, mk_lt(mk_const(self.x), mk_const(self.n)))
        tr2 = Transition.assign(self.ctx, self.x, mk_add([mk_const(self.x), mk_real(self.ctx, QQ.one())]))

        self.assertIsNotNone(tr1)
        self.assertIsNotNone(tr2)

    def test_transition_equality(self):
        """Test transition equality."""
        tr1 = Transition.assume(self.ctx, mk_lt(mk_const(self.x), mk_const(self.n)))
        tr2 = Transition.assume(self.ctx, mk_lt(mk_const(self.x), mk_const(self.n)))
        tr3 = Transition.assume(self.ctx, mk_lt(mk_const(self.n), mk_const(self.x)))

        self.assertEqual(tr1, tr2)
        self.assertNotEqual(tr1, tr3)


class TestTransitionSystem(unittest.TestCase):
    """Test Transition System operations."""

    def setUp(self):
        """Set up test context and symbols."""
        self.ctx = Context()
        self.x = mk_symbol(self.ctx, 'x', Type.INT)
        self.n = mk_symbol(self.ctx, 'n', Type.INT)

    def test_transition_system_creation(self):
        """Test basic transition system creation."""
        ts = TransitionSystem(self.ctx, [
            (0, Transition.assign(self.ctx, self.x, mk_real(self.ctx, QQ.zero())), 1),
            (1, Transition.assume(self.ctx, mk_lt(mk_const(self.x), mk_const(self.n))), 2)
        ])

        self.assertIsNotNone(ts)


class TestAbstractDomains(unittest.TestCase):
    """Test abstract domain operations."""

    def setUp(self):
        """Set up test context and symbols."""
        self.ctx = Context()
        self.x = mk_symbol(self.ctx, 'x', Type.INT)
        self.y = mk_symbol(self.ctx, 'y', Type.INT)

    def test_sign_domain(self):
        """Test sign domain operations."""
        domain = SignDomain({self.x: AbstractValue.POSITIVE, self.y: AbstractValue.NEGATIVE})

        # Test string representation
        self.assertIn("x=positive", str(domain))
        self.assertIn("y=negative", str(domain))

        # Test join
        other_domain = SignDomain({self.x: AbstractValue.ZERO})
        joined = domain.join(other_domain)
        self.assertIsInstance(joined, SignDomain)


if __name__ == '__main__':
    unittest.main()
