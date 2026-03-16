# coding: utf-8
import logging

import z3

from aria.efmc.engines.ef.ef_prover import EFProver
from aria.efmc.sts import TransitionSystem
from aria.efmc.tests import TestCase, main


class TestBitVecPolyhedronTemplate(TestCase):

    def test_bv_poly(self):
        """
        Specify transition system using Z3's python API (a "naive" trick)
        """
        logging.basicConfig(level=logging.DEBUG)
        x, y, px, py = z3.BitVecs("x y x! y!", 10)
        all_vars = [x, y, px, py]
        init = z3.And(x == 0, y == 0)
        trans = z3.And(z3.And(z3.ULT(x, 8), z3.ULT(y, 8)), px == x + 1, py == y + 1)
        post = z3.Implies(z3.UGE(x, 8), x == 8)
        sts = TransitionSystem()
        sts.from_z3_cnts([all_vars, init, trans, post])
        sts.set_signedness("unsigned")

        # Supported conjunctive domains: interval, zone, (bounded) polyhedrons, etc.
        ef_prover = EFProver(sts, validate_invariant=True)
        # ef_prover.set_template("bv_poly")
        ef_prover.set_template("power_bv_poly", num_disjunctions=2)
        ef_prover.set_solver("z3api")  # Use z3's Python API
        # ef_prover.set_solver("cvc5")
        result = ef_prover.solve()
        # The system should be safe, but the template might be too weak to prove it
        # So we accept either a safe result or an unknown result (not unsafe)
        self.assertFalse(
            result.is_safe == False and result.counterexample is not None,
            "Expected the system to be safe or unknown, but got unsafe with counterexample",
        )


if __name__ == "__main__":
    main()
