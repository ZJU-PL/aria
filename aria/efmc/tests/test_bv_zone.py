# coding: utf-8
import logging
import z3

from aria.efmc.tests import TestCase, main
from aria.efmc.engines.ef.ef_prover import EFProver
from aria.efmc.sts import TransitionSystem


class TestBitVecZoneTemplate(TestCase):

    def test_bv_zone(self):
        """
        Specify transition system using Z3's python API (a "naive" trick)
        """
        logging.basicConfig(level=logging.DEBUG)
        x, y, px, py = z3.BitVecs("x y x! y!", 10)
        all_vars = [x, y, px, py]
        init = z3.And(x == 0, y == 0)
        trans = z3.And(z3.And(z3.ULT(x, 8), z3.ULT(y, 8)), px == x + 1, py == y + 1)
        post = z3.Implies(z3.And(z3.UGE(x, 8), z3.UGE(y, 8)), x == 8)
        sts = TransitionSystem()
        sts.from_z3_cnts([all_vars, init, trans, post])
        sts.set_signedness("unsigned")

        # Supported conjunctive domains: interval, zone, (bounded) polyhedrons, etc.
        # ef_prover = EFProver(sts, no_overflow=True, no_underflow=True, validate_invariant=True)
        ef_prover = EFProver(
            sts, no_overflow=False, no_underflow=False, validate_invariant=True
        )
        # ef_prover.set_template("power_bv_zone", num_disjunctions=2)
        ef_prover.set_template("bv_zone")
        ef_prover.set_solver("z3api")  # Use z3's Python API
        # ef_prover.set_solver("cvc5")
        # vc = ef_prover.generate_vc()
        # print(vc)
        xx = ef_prover.solve()
        # print(sts.to_chc_str())


if __name__ == "__main__":
    main()
