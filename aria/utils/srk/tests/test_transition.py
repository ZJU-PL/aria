"""
Tests for the Transition module.
"""

import unittest
from aria.utils.srk.core.syntax import (
    Context,
    Symbol,
    Type,
    mk_symbol,
    mk_const,
    mk_lt,
    mk_add,
    mk_and,
    mk_real,
)
from aria.utils.srk.lts.transition import Transition, TransitionSystem
from aria.utils.srk.logic.abstract import SignDomain, AbstractValue
from aria.utils.srk.linalg.qQ import QQ


class TestTransition(unittest.TestCase):
    """Test Transition operations."""

    def setUp(self):
        """Set up test context and symbols."""
        self.ctx = Context()
        self.x = mk_symbol(self.ctx, "x", Type.INT)
        self.n = mk_symbol(self.ctx, "n", Type.INT)

    def test_transition_creation(self):
        """Test basic transition creation."""
        tr1 = Transition.assume(self.ctx, mk_lt(mk_const(self.x), mk_const(self.n)))
        tr2 = Transition.assign(
            self.ctx, self.x, mk_add([mk_const(self.x), mk_real(self.ctx, QQ.one())])
        )

        self.assertIsNotNone(tr1)
        self.assertIsNotNone(tr2)

    def test_transition_equality(self):
        """Test transition equality."""
        tr1 = Transition.assume(self.ctx, mk_lt(mk_const(self.x), mk_const(self.n)))
        tr2 = Transition.assume(self.ctx, mk_lt(mk_const(self.x), mk_const(self.n)))
        tr3 = Transition.assume(self.ctx, mk_lt(mk_const(self.n), mk_const(self.x)))

        self.assertEqual(tr1, tr2)
        self.assertNotEqual(tr1, tr3)

    def test_sequential_composition_substitutes_second_rhs_and_guard(self):
        """Sequential composition evaluates the second transition after the first."""
        one = mk_real(self.ctx, QQ.one())
        first = Transition.assign(self.ctx, self.x, one)
        second_guard = mk_lt(mk_const(self.x), mk_const(self.n))
        second = Transition(
            transform={self.n: mk_add([mk_const(self.x), one])},
            guard=second_guard,
            context=self.ctx,
        )

        composed = first.mul(second)

        self.assertEqual(composed.transform[self.x], one)
        self.assertEqual(composed.transform[self.n], mk_add([one, one]))
        self.assertEqual(
            composed.guard,
            mk_and(self.ctx, [first.guard, mk_lt(one, mk_const(self.n))]),
        )

    def test_nondeterministic_choice_creates_skolem_choice(self):
        """Choice creates a Skolem choice variable for conflicting assignments.

        Matching the OCaml implementation, ``add`` introduces a fresh choice
        symbol ``phi_x`` and builds an OR-guard with equalities so that the
        combined transform uses ``ite(phi_x, left_val, right_val)``.
        """
        left = Transition.assign(self.ctx, self.x, mk_real(self.ctx, QQ.zero()))
        right = Transition.assign(self.ctx, self.x, mk_real(self.ctx, QQ.one()))

        choice = left.add(right)

        # The variable should still be in the transform (not dropped).
        self.assertIn(self.x, choice.transform)
        # The guard should be a disjunction (uses Unicode or ASCII).
        guard_str = str(choice.guard)
        self.assertTrue(
            "Or" in guard_str or "or" in guard_str.lower()
            or "∨" in guard_str or "||" in guard_str,
            f"Expected disjunction in guard, got: {guard_str}",
        )


class TestTransitionSystem(unittest.TestCase):
    """Test Transition System operations."""

    def setUp(self):
        """Set up test context and symbols."""
        self.ctx = Context()
        self.x = mk_symbol(self.ctx, "x", Type.INT)
        self.n = mk_symbol(self.ctx, "n", Type.INT)

    def test_transition_system_creation(self):
        """Test basic transition system creation."""
        ts = TransitionSystem(
            self.ctx,
            [
                (
                    0,
                    Transition.assign(self.ctx, self.x, mk_real(self.ctx, QQ.zero())),
                    1,
                ),
                (
                    1,
                    Transition.assume(
                        self.ctx, mk_lt(mk_const(self.x), mk_const(self.n))
                    ),
                    2,
                ),
            ],
        )

        self.assertIsNotNone(ts)


class TestAbstractDomains(unittest.TestCase):
    """Test abstract domain operations."""

    def setUp(self):
        """Set up test context and symbols."""
        self.ctx = Context()
        self.x = mk_symbol(self.ctx, "x", Type.INT)
        self.y = mk_symbol(self.ctx, "y", Type.INT)

    def test_sign_domain(self):
        """Test sign domain operations."""
        domain = SignDomain(
            {self.x: AbstractValue.POSITIVE, self.y: AbstractValue.NEGATIVE}
        )

        # Test string representation
        self.assertIn("x=positive", str(domain))
        self.assertIn("y=negative", str(domain))

        # Test join
        other_domain = SignDomain({self.x: AbstractValue.ZERO})
        joined = domain.join(other_domain)
        self.assertIsInstance(joined, SignDomain)


if __name__ == "__main__":
    unittest.main()
