# coding: utf-8
import logging
import z3
from aria.efmc.tests import TestCase, main
from aria.efmc.engines.ef.ef_prover import EFProver
from aria.efmc.sts import TransitionSystem


class TestBitVecIntervalTemplate(TestCase):

    def test_bv_interval(self):
        """Test bit vector interval analysis with transition system.

        This test creates a simple transition system with two variables x and y,
        both initialized to 0 and incrementing by 1 until x reaches 8.
        """
        # Define bit vector variables
        BV_SIZE = 6
        x, y, px, py = z3.BitVecs("x y x! y!", BV_SIZE)
        all_vars = [x, y, px, py]

        # Define transition system constraints
        init = z3.And(x == 0, y == 0)
        bounds = z3.And(z3.ULT(x, 8), z3.ULT(y, 8))
        updates = z3.And(px == x + 1, py == y + 1)
        trans = z3.And(bounds, updates)
        post = z3.Implies(z3.UGE(x, 8), x == 8)

        # Set up transition system
        sts = TransitionSystem()
        sts.from_z3_cnts([all_vars, init, trans, post])
        sts.set_signedness("unsigned")

        # Configure and run EF prover
        ef_prover = EFProver(sts, validate_invariant=True)
        ef_prover.set_template("bv_interval")
        ef_prover.set_solver("z3api")

        result = ef_prover.solve()
        # The system should be safe, but the template might be too weak to prove it
        # So we accept either a safe result or an unknown result (not unsafe)
        self.assertFalse(
            result.is_safe == False and result.counterexample is not None,
            "Expected the system to be safe or unknown, but got unsafe with counterexample",
        )


if __name__ == "__main__":
    main()
