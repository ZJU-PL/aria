# coding: utf-8
import z3

from aria.efmc.tests import TestCase, main
from aria.efmc.engines.ef.ef_prover import EFProver
from aria.efmc.sts import TransitionSystem
from aria.efmc.engines.ef.templates.bv_affine import BitVecAffineTemplate
from aria.efmc.engines.ef.templates.abstract_template import TemplateType
from aria.utils.z3.bv import Signedness


class TestBitVecAffineTemplate(TestCase):

    def create_affine_invariant_program(self, bit_width: int = 8) -> TransitionSystem:
        """
        Create a program that should be provable with affine templates.
        The invariant is: x - y = 0 (linear equality)
        """
        x, y, px, py = z3.BitVecs("x y x! y!", bit_width)
        all_vars = [x, y, px, py]

        # Program: x and y always stay equal
        init = z3.And(x == 0, y == 0)
        trans = z3.And(z3.ULT(x, 5), px == x + 1, py == y + 1)
        post = z3.Implies(z3.UGE(x, 5), x == y)  # Safety: when x >= 5, x == y

        sts = TransitionSystem()
        sts.from_z3_cnts([all_vars, init, trans, post])
        sts.set_signedness("unsigned")
        return sts

    def create_counter_program(self, bit_width: int = 8) -> TransitionSystem:
        """
        Create a counter program that increments x until it reaches a limit.
        This should be provable with affine templates.
        """
        x, y, px, py = z3.BitVecs("x y x! y!", bit_width)
        all_vars = [x, y, px, py]

        # Counter program: x increments from 0 to 10, y stays constant
        init = z3.And(x == 0, y == 5)
        trans = z3.And(z3.ULT(x, 10), px == x + 1, py == y)
        post = z3.Implies(z3.UGE(x, 10), z3.And(z3.ULE(x, 10), y == 5))

        sts = TransitionSystem()
        sts.from_z3_cnts([all_vars, init, trans, post])
        sts.set_signedness("unsigned")
        return sts

    def test_bv_affine_template_creation(self):
        """Test BitVecAffineTemplate class creation and basic properties"""
        sts = self.create_affine_invariant_program()
        template = BitVecAffineTemplate(sts)

        self.assertEqual(template.template_type, TemplateType.BV_AFFINE)
        self.assertEqual(template.arity, 2)
        self.assertEqual(template.num_templates, 1)
        self.assertEqual(len(template.template_vars), 1)
        self.assertEqual(len(template.template_vars[0]), 3)

        self.assertIsNotNone(template.template_cnt_init_and_post)
        self.assertIsNotNone(template.template_cnt_trans)

    def test_bv_affine_signedness(self):
        """Test template with different signedness"""
        sts = self.create_affine_invariant_program()

        sts.set_signedness("unsigned")
        template_unsigned = BitVecAffineTemplate(sts)
        self.assertEqual(template_unsigned.signedness, Signedness.UNSIGNED)

        sts.set_signedness("signed")
        template_signed = BitVecAffineTemplate(sts)
        self.assertEqual(template_signed.signedness, Signedness.SIGNED)

    def test_bv_affine_provable_program(self):
        """Test solving a program that should be provable with affine templates"""
        sts = self.create_affine_invariant_program()

        ef_prover = EFProver(sts, validate_invariant=True)
        ef_prover.set_template("bv_affine")
        ef_prover.set_solver("z3api")

        result = ef_prover.solve()

        # This program should be provable with affine templates
        # The invariant is x - y = 0, which is a linear equality
        self.assertTrue(
            result.is_safe or result.is_unknown,
            f"Expected safe or unknown, got: {result}",
        )

    def test_bv_affine_counter_program(self):
        """Test solving a counter program (should be unknown due to template limitations)"""
        sts = self.create_counter_program()

        ef_prover = EFProver(sts, validate_invariant=True)
        ef_prover.set_template("bv_affine")
        ef_prover.set_solver("z3api")

        result = ef_prover.solve()

        # Counter program needs inequalities (x <= 10), so affine template is too weak
        # Should get unknown, not unsafe
        self.assertTrue(
            result.is_unknown or result.is_safe,
            f"Expected unknown or safe, got: {result}",
        )


if __name__ == "__main__":
    main()
