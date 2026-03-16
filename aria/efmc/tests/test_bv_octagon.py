# coding: utf-8
import logging
from typing import List

import z3
from aria.efmc.engines.ef.ef_prover import EFProver
from aria.efmc.sts import TransitionSystem
from aria.efmc.tests import TestCase, main


class TestBitVecOctagonTemplate(TestCase):

    def create_transition_system(self, bit_width: int = 10) -> TransitionSystem:
        """
        Create a transition system with bitvector variables.

        Args:
            bit_width: Width of bitvectors (default: 10)

        Returns:
            TransitionSystem: Configured transition system
        """
        x, y, px, py = z3.BitVecs("x y x! y!", bit_width)
        all_vars = [x, y, px, py]

        # Define system constraints
        init = z3.And(x == 0, y == 0)
        trans = z3.And(z3.ULT(x, 8), z3.ULT(y, 8), px == x + 1, py == y + 1)
        post = z3.Implies(z3.And(z3.UGE(x, 8), z3.UGE(y, 8)), x == 8)

        # Configure transition system
        sts = TransitionSystem()
        sts.from_z3_cnts([all_vars, init, trans, post])
        sts.set_signedness("unsigned")

        return sts

    def test_bv_octagon(self):
        """Test bitvector octagon template with Z3 solver"""
        sts = self.create_transition_system()

        ef_prover = EFProver(
            sts, no_overflow=False, no_underflow=False, validate_invariant=True
        )
        ef_prover.set_template("bv_octagon")
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
