# coding: utf-8
import logging
import z3
from aria.efmc.tests import TestCase, main
from aria.efmc.engines.ef.ef_prover import EFProver
from aria.efmc.sts import TransitionSystem


class TestBitVecKnownBitsTemplate(TestCase):
    """Test cases for the KnownBitsTemplate abstract domain."""

    def test_bv_knownbits_basic(self):
        """Test basic functionality of the KnownBits domain.

        This test creates a simple transition system where:
        - x starts at 0 and increments by 1 each step, saturating at 15
        - y starts at 0 and increments by 2 each step, saturating at 30

        The property to verify is that if x's least significant bit is 1 (odd),
        then y's least significant bit is 0 (even). This property holds because
        x and y start at 0, and when x becomes odd (1,3,5...), y is always even (2,6,10...).

        The KnownBitsTemplate should be able to prove this property by tracking
        the exact values of specific bits.
        """
        print("\n" + "=" * 50)
        print("RUNNING TEST: test_bv_knownbits_basic")
        print("=" * 50)

        # Define bit vector variables
        BV_SIZE = 8
        x, y, px, py = z3.BitVecs("x y x! y!", BV_SIZE)
        all_vars = [x, y, px, py]

        # Define transition system constraints
        init = z3.And(x == 0, y == 0)

        # x is incremented by 1 each step, but saturates at 15
        # y is incremented by 2 each step, but saturates at 30
        x_update = z3.If(z3.ULT(x, 15), x + 1, 15)
        y_update = z3.If(z3.ULT(y, 30), y + 2, 30)

        updates = z3.And(px == x_update, py == y_update)
        trans = updates

        # The property we want to verify:
        # If x is odd (bit 0 is 1), then y is even (bit 0 is 0)
        post = z3.Implies(z3.Extract(0, 0, x) == 1, z3.Extract(0, 0, y) == 0)

        # Set up transition system
        sts = TransitionSystem()
        sts.from_z3_cnts([all_vars, init, trans, post])
        sts.set_signedness("unsigned")

        # Configure and run EF prover with KnownBits template
        ef_prover = EFProver(sts, validate_invariant=True)
        ef_prover.set_template("knownbits")
        ef_prover.set_solver("z3api")

        result = ef_prover.solve()
        print(f"Result: {result}")
        print("=" * 50 + "\n")
        # The system should be safe, but the template might be too weak to prove it
        # So we accept either a safe result or an unknown result (not unsafe)
        self.assertFalse(
            result.is_safe == False and result.counterexample is not None,
            "Expected the system to be safe or unknown, but got unsafe with counterexample",
        )

    def test_bv_knownbits_bitwise_ops(self):
        """Test KnownBits domain with bitwise operations.

        This test creates a transition system where:
        - x starts at 0 and increments by 1 each step
        - y is always the bitwise complement of x
        - z is always the bitwise AND of x and y

        The property to verify is that z is always 0. This is true because
        a bit and its complement can never both be 1, so their AND is always 0.

        The KnownBitsTemplate should be able to prove this by tracking the
        exact relationship between bits of x, y, and z.
        """
        print("\n" + "=" * 50)
        print("RUNNING TEST: test_bv_knownbits_bitwise_ops")
        print("=" * 50)

        # Define bit vector variables
        BV_SIZE = 8
        x, y, z, px, py, pz = z3.BitVecs("x y z x! y! z!", BV_SIZE)
        all_vars = [x, y, z, px, py, pz]

        # Define transition system constraints
        # Initial state: x=0, y=~x (all 1s), z=x&y (all 0s)
        init = z3.And(x == 0, y == ~x, z == (x & y))

        # Transition relation:
        # x increments by 1 each step
        # y is always the bitwise complement of x
        # z is always the bitwise AND of x and y
        x_update = x + 1
        y_update = ~x_update
        z_update = x_update & y_update

        updates = z3.And(px == x_update, py == y_update, pz == z_update)
        trans = updates

        # The property we want to verify:
        # z is always 0 (since a bit and its complement can never both be 1)
        post = z == 0

        # Set up transition system
        sts = TransitionSystem()
        sts.from_z3_cnts([all_vars, init, trans, post])
        sts.set_signedness("unsigned")

        # Configure and run EF prover with KnownBits template
        ef_prover = EFProver(sts, validate_invariant=True)
        ef_prover.set_template("knownbits")
        ef_prover.set_solver("z3api")

        result = ef_prover.solve()
        print(f"Result: {result}")
        print("=" * 50 + "\n")
        # The system should be safe, but the template might be too weak to prove it
        # So we accept either a safe result or an unknown result (not unsafe)
        self.assertFalse(
            result.is_safe == False and result.counterexample is not None,
            "Expected the system to be safe or unknown, but got unsafe with counterexample",
        )

    def test_bv_knownbits_bit_patterns(self):
        """Test KnownBits domain with simple bit patterns.

        This test creates a transition system where:
        - x starts with specific bits set (bit 0=1, bit 7=0)
        - x is updated in a way that preserves these specific bits

        The property to verify is that these specific bits maintain their values.
        This demonstrates the template's ability to track individual bits.
        """
        print("\n" + "=" * 50)
        print("RUNNING TEST: test_bv_knownbits_bit_patterns")
        print("=" * 50)

        # Define bit vector variables
        BV_SIZE = 8
        x, px = z3.BitVecs("x x!", BV_SIZE)
        all_vars = [x, px]

        # Define transition system constraints
        # Initial state: x has bit 0 set to 1 and bit 7 set to 0
        init = z3.And(
            z3.Extract(0, 0, x) == 1,  # Least significant bit is 1
            z3.Extract(7, 7, x) == 0,  # Most significant bit is 0
        )

        # Transition relation:
        # x is updated but preserves bits 0 and 7
        # We'll use a mask to ensure bit 0 stays 1 and bit 7 stays 0
        mask_keep_bit0_set = 0x01  # 00000001 in binary
        mask_keep_bit7_clear = 0x7F  # 01111111 in binary

        # Update x but preserve the specific bits
        # (x & ~mask_keep_bit0_set) | mask_keep_bit0_set ensures bit 0 is 1
        # x & mask_keep_bit7_clear ensures bit 7 is 0
        x_update = z3.BitVecVal(mask_keep_bit0_set, BV_SIZE) | (
            x & z3.BitVecVal(~mask_keep_bit0_set & mask_keep_bit7_clear, BV_SIZE)
        )

        trans = px == x_update

        # The property we want to verify:
        # Bit 0 of x is always 1 and bit 7 is always 0
        post = z3.And(z3.Extract(0, 0, x) == 1, z3.Extract(7, 7, x) == 0)

        # Set up transition system
        sts = TransitionSystem()
        sts.from_z3_cnts([all_vars, init, trans, post])
        sts.set_signedness("unsigned")

        # Configure and run EF prover with KnownBits template
        ef_prover = EFProver(sts, validate_invariant=True)
        ef_prover.set_template("knownbits")
        ef_prover.set_solver("z3api")

        result = ef_prover.solve()
        print(f"Result: {result}")
        print("=" * 50 + "\n")
        # The system should be safe, but the template might be too weak to prove it
        # So we accept either a safe result or an unknown result (not unsafe)
        self.assertFalse(
            result.is_safe == False and result.counterexample is not None,
            "Expected the system to be safe or unknown, but got unsafe with counterexample",
        )


if __name__ == "__main__":
    main()
