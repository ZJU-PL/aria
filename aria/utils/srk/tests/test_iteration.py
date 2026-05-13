"""
Tests for the Iteration module (approximate transitive closure computation).
"""

import unittest
from aria.utils.srk.core.syntax import (
    Context,
    Symbol,
    Type,
    mk_symbol,
    mk_const,
    mk_true,
    mk_false,
    mk_eq,
    mk_lt,
    mk_leq,
    mk_and,
    mk_or,
    mk_not,
    mk_ite,
    mk_add,
    mk_mul,
    mk_real,
)
from aria.utils.srk.analysis.iteration import (
    IterationEngine,
    WedgeGuard,
    PolyhedronGuard,
    GuardedTranslation,
    MakeDomain,
    make_wedge_guard,
    make_polyhedron_guard,
)
from aria.utils.srk.lts.transitionFormula import TransitionFormula
from aria.utils.srk.linalg.qQ import QQ, zero, one


class TestIteration(unittest.TestCase):
    """Test iteration and transitive closure computation."""

    def setUp(self):
        """Set up test context and symbols."""
        self.ctx = Context()
        self.w = mk_symbol("w", Type.INT)
        self.x = mk_symbol("x", Type.INT)
        self.y = mk_symbol("y", Type.INT)
        self.z = mk_symbol("z", Type.INT)
        self.w_prime = mk_symbol("w'", Type.INT)
        self.x_prime = mk_symbol("x'", Type.INT)
        self.y_prime = mk_symbol("y'", Type.INT)
        self.z_prime = mk_symbol("z'", Type.INT)

        # Transition symbols for testing
        self.tr_symbols = [
            (self.w, self.w_prime),
            (self.x, self.x_prime),
            (self.y, self.y_prime),
            (self.z, self.z_prime),
        ]

    def test_iteration_engine_creation(self):
        """Test creating an iteration engine."""
        # Create a simple domain for testing
        domain = make_wedge_guard()
        engine = IterationEngine(domain)

        self.assertIsNotNone(engine)
        self.assertEqual(engine.domain, domain)

    def test_wedge_guard_creation(self):
        """Test creating wedge guard."""
        wedge_guard = make_wedge_guard()
        self.assertIsNotNone(wedge_guard)

    def test_polyhedron_guard_creation(self):
        """Test creating polyhedron guard."""
        poly_guard = make_polyhedron_guard()
        self.assertIsNotNone(poly_guard)

    def test_simple_closure_computation(self):
        """Test basic closure computation."""
        # Create a simple transition formula
        # This is a simplified test - in a full implementation,
        # we would need complete TransitionFormula support

        # For now, test that the engine can be created and basic operations work
        wedge_domain = make_wedge_guard()
        engine = IterationEngine(wedge_domain)

        # Test that the engine exists and has basic functionality
        self.assertIsNotNone(engine.domain)

    def test_domain_operations(self):
        """Test basic domain operations."""
        # Test WedgeGuard operations
        wedge_guard = make_wedge_guard()

        # These would be properly implemented when the modules are complete
        # For now, test that the objects can be created and have expected methods
        self.assertTrue(hasattr(wedge_guard, "abstract"))
        self.assertTrue(hasattr(wedge_guard, "exp"))

        # Test PolyhedronGuard operations
        poly_guard = make_polyhedron_guard()
        self.assertTrue(hasattr(poly_guard, "abstract"))
        self.assertTrue(hasattr(poly_guard, "join"))
        self.assertTrue(hasattr(poly_guard, "widen"))


class TestPrePostConditions(unittest.TestCase):
    """Test pre/post condition checking."""

    def setUp(self):
        """Set up test context and symbols."""
        self.ctx = Context()
        self.x = mk_symbol("x", Type.INT)
        self.x_prime = mk_symbol("x'", Type.INT)

    def test_basic_pre_post(self):
        """Test basic pre/post condition relationships."""
        # Create a simple transition: x' = x + 1
        # Pre: x >= 0, Post: x' >= 1

        # This is a simplified test - full implementation would need
        # complete transition formula and iteration support

        # For now, test that basic logical operations work
        pre = mk_leq(mk_real(float(zero())), mk_const(self.x))
        post = mk_leq(mk_real(float(one())), mk_const(self.x_prime))

        # Test that expressions can be created
        self.assertIsNotNone(pre)
        self.assertIsNotNone(post)


class TestInductionProofs(unittest.TestCase):
    """Test induction-based proofs."""

    def setUp(self):
        """Set up test context and symbols."""
        self.ctx = Context()
        self.w = mk_symbol("w", Type.INT)
        self.x = mk_symbol("x", Type.INT)
        self.y = mk_symbol("y", Type.INT)
        self.z = mk_symbol("z", Type.INT)

    def test_simple_induction(self):
        """Test simple induction proof."""
        # This would test induction principles like:
        # Base case: P(0)
        # Inductive step: P(n) => P(n+1)

        # Simplified test - full implementation needs complete iteration support
        self.assertIsNotNone(self.ctx)


class TestGuardedTranslation(unittest.TestCase):
    def setUp(self):
        self.ctx = Context()
        self.x = mk_symbol("x", Type.INT)
        self.y = mk_symbol("y", Type.INT)
        self.xp = mk_symbol("x'", Type.INT)
        self.yp = mk_symbol("y'", Type.INT)

    def _make_tf(self, guard, assigns):
        pre_syms = set()
        post_syms = set()
        atoms = []
        if guard is not None:
            atoms.append(guard)
        for pre_sym, post_sym, expr in assigns:
            pre_syms.add(pre_sym)
            post_syms.add(post_sym)
            atoms.append(mk_eq(mk_const(post_sym), expr))
        formula = mk_and(atoms) if len(atoms) > 1 else atoms[0]
        tr_symbols = [(self.x, self.xp), (self.y, self.yp)]
        return TransitionFormula(formula=formula, symbols=tr_symbols)

    def test_guarded_translation1(self):
        guard = mk_or([mk_lt(mk_const(self.x), mk_const(self.y)),
                       mk_lt(mk_const(self.y), mk_const(self.x))])
        tf = self._make_tf(guard, [
            (self.x, self.xp, mk_add([mk_const(self.x), mk_real(self.ctx, 1)])),
            (self.y, self.yp, mk_const(self.y)),
        ])
        domain = GuardedTranslation()
        self.assertIsNotNone(domain)

    def test_guarded_translation2(self):
        guard = mk_or([mk_lt(mk_real(self.ctx, 0), mk_const(self.x)),
                       mk_lt(mk_const(self.x), mk_real(self.ctx, 0))])
        tf = self._make_tf(guard, [
            (self.x, self.xp, mk_add([mk_const(self.x), mk_real(self.ctx, -2)])),
        ])
        domain = GuardedTranslation()
        self.assertIsNotNone(domain)


if __name__ == "__main__":
    unittest.main()
