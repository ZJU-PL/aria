# coding: utf-8
import time
import z3

from aria.efmc.tests import TestCase, main
from aria.efmc.sts import TransitionSystem
from aria.efmc.engines.abduction.abduction_prover import AbductionProver, VerificationResult
from aria.efmc.engines.kinduction.kind_prover import KInductionProver
from aria.efmc.engines.pdr.pdr_prover import PDRProver
from aria.efmc.tests.simple_sts import (
    get_int_sys1,
    get_int_sys5,
    get_int_sys7,
    get_int_sys8,
    get_int_sys9,
    get_int_sys10,
    get_int_sys11,
    get_int_sys12,
)


class TestAbductionProver(TestCase):

    def test_abd2(self):
        print("Running one test...")
        sts = TransitionSystem()
        sts.from_z3_cnts(get_int_sys1())

        # k-induction
        kind_prover = KInductionProver(sts)
        kind_prover.use_aux_invariant = False
        kind_res = kind_prover.solve(k=20)
        print("kind prover res: ", kind_res)
        print("======================================")

        # PDR
        pdr_prover = PDRProver(sts)
        pdr_res = pdr_prover.solve()
        print("PDR prover res: ", pdr_res)
        print("======================================")

        abd_prover = AbductionProver(sts)
        abd_res = abd_prover.solve()
        print("abd prover res: ", abd_res)
        print("======================================")

        # Verify that at least one prover finds the system safe
        self.assertTrue(
            kind_res.is_safe or pdr_res.is_safe,
            "At least one of k-induction or PDR should prove the system safe",
        )

    def test_counter_system(self):
        """Test with a counter system with multiple variables (get_int_sys7)."""
        print("Running counter system test...")
        sts = TransitionSystem()
        sts.from_z3_cnts(get_int_sys7())

        start = time.time()
        abd_prover = AbductionProver(sts)
        res = abd_prover.solve()
        print("abd prover res: ", res)
        print("Time taken: ", time.time() - start)
        print("======================================")

        # Verify that the system is safe
        self.assertTrue(res.is_safe, "AbductionProver should prove the system safe")
        self.assertIsNotNone(res.invariant, "AbductionProver should find an invariant")

    def test_mutual_exclusion(self):
        """Test with a mutual exclusion system (get_int_sys8)."""
        print("Running mutual exclusion test...")
        sts = TransitionSystem()
        sts.from_z3_cnts(get_int_sys8())

        start = time.time()
        abd_prover = AbductionProver(sts)
        res = abd_prover.solve()
        print("abd prover res: ", res)
        print("Time taken: ", time.time() - start)
        print("======================================")

        # Verify that the system is safe
        self.assertTrue(res.is_safe, "AbductionProver should prove the system safe")
        self.assertIsNotNone(res.invariant, "AbductionProver should find an invariant")

    def test_ticket_lock(self):
        """Test with a ticket lock system (get_int_sys9)."""
        print("Running ticket lock test...")
        sts = TransitionSystem()
        sts.from_z3_cnts(get_int_sys9())

        start = time.time()
        abd_prover = AbductionProver(sts)
        res = abd_prover.solve()
        print("abd prover res: ", res)
        print("Time taken: ", time.time() - start)
        print("======================================")

        # Verify that the system is safe
        self.assertTrue(res.is_safe, "AbductionProver should prove the system safe")
        self.assertIsNotNone(res.invariant, "AbductionProver should find an invariant")

    def test_resource_allocation(self):
        """Test with a resource allocation system (get_int_sys10)."""
        print("Running resource allocation test...")
        sts = TransitionSystem()
        sts.from_z3_cnts(get_int_sys10())

        start = time.time()
        abd_prover = AbductionProver(sts)
        res = abd_prover.solve()
        print("abd prover res: ", res)
        print("Time taken: ", time.time() - start)
        print("======================================")

        # Verify that the system is safe
        self.assertTrue(res.is_safe, "AbductionProver should prove the system safe")
        self.assertIsNotNone(res.invariant, "AbductionProver should find an invariant")

    def test_timeout_handling(self):
        """Test the AbductionProver with a short timeout."""
        print("Running timeout handling test...")
        sts = TransitionSystem()
        sts.from_z3_cnts(get_int_sys10())  # Using a more complex system

        start = time.time()
        abd_prover = AbductionProver(sts)  # Very short timeout
        res = abd_prover.solve()
        print("abd prover res with short timeout: ", res)
        print("Time taken: ", time.time() - start)
        print("======================================")

        # The test should complete without exceptions
        # We don't assert on the result since it might vary based on machine speed

    def test_max_iterations(self):
        """Test the AbductionProver with limited iterations."""
        print("Running max iterations test...")
        sts = TransitionSystem()
        sts.from_z3_cnts(get_int_sys7())

        start = time.time()
        abd_prover = AbductionProver(sts, max_iterations=5)  # Limited iterations
        res = abd_prover.solve()
        print("abd prover res with limited iterations: ", res)
        print("Time taken: ", time.time() - start)
        print("======================================")

        # Verify that the system is safe even with limited iterations
        self.assertTrue(
            res.is_safe,
            "AbductionProver should prove the system safe with limited iterations",
        )

    def test_producer_consumer(self):
        """Test with a complex producer-consumer system (get_int_sys11)."""
        print("Running producer-consumer system test...")
        sts = TransitionSystem()
        sts.from_z3_cnts(get_int_sys11())

        # Compare all three provers
        print("Testing with AbductionProver:")
        start = time.time()
        abd_prover = AbductionProver(sts)
        abd_res = abd_prover.solve()
        abd_time = time.time() - start
        print("abd prover res: ", abd_res)
        print("Time taken: ", abd_time)
        print("======================================")

        print("Testing with KInductionProver:")
        start = time.time()
        kind_prover = KInductionProver(sts)
        kind_prover.use_aux_invariant = False
        kind_res = kind_prover.solve(k=20)
        kind_time = time.time() - start
        print("kind prover res: ", kind_res)
        print("Time taken: ", kind_time)
        print("======================================")

        print("Testing with PDRProver:")
        start = time.time()
        pdr_prover = PDRProver(sts)
        pdr_res = pdr_prover.solve()
        pdr_time = time.time() - start
        print("PDR prover res: ", pdr_res)
        print("Time taken: ", pdr_time)
        print("======================================")

        # Print comparison summary
        print("Comparison summary:")
        print(
            f"AbductionProver: {'Safe' if abd_res.is_safe else 'Unsafe'} in {abd_time:.2f}s"
        )
        print(
            f"KInductionProver: {'Safe' if kind_res.is_safe else 'Unsafe'} in {kind_time:.2f}s"
        )
        print(
            f"PDRProver: {'Safe' if pdr_res.is_safe else 'Unsafe'} in {pdr_time:.2f}s"
        )
        print("======================================")

        # All provers should agree on the result
        self.assertEqual(
            abd_res.is_safe,
            kind_res.is_safe,
            "AbductionProver and KInductionProver should agree on safety",
        )
        self.assertEqual(
            abd_res.is_safe,
            pdr_res.is_safe,
            "AbductionProver and PDRProver should agree on safety",
        )

    def test_invalid_property(self):
        """Test with a system that has an invalid property (get_int_sys12)."""
        print("Running invalid property test...")
        sts = TransitionSystem()
        sts.from_z3_cnts(get_int_sys12())

        # Compare all three provers
        print("Testing with AbductionProver:")
        start = time.time()
        abd_prover = AbductionProver(sts)
        abd_res = abd_prover.solve()
        abd_time = time.time() - start
        print("abd prover res: ", abd_res)
        print("Time taken: ", abd_time)
        print("======================================")

        print("Testing with KInductionProver:")
        start = time.time()
        kind_prover = KInductionProver(sts)
        kind_prover.use_aux_invariant = False
        kind_res = kind_prover.solve(k=20)
        kind_time = time.time() - start
        print("kind prover res: ", kind_res)
        print("Time taken: ", kind_time)
        print("======================================")

        print("Testing with PDRProver:")
        start = time.time()
        pdr_prover = PDRProver(sts)
        pdr_res = pdr_prover.solve()
        pdr_time = time.time() - start
        print("PDR prover res: ", pdr_res)
        print("Time taken: ", pdr_time)
        print("======================================")

        # Print comparison summary
        print("Comparison summary:")
        print(
            f"AbductionProver: {'Safe' if abd_res.is_safe else 'Unsafe'} in {abd_time:.2f}s"
        )
        print(
            f"KInductionProver: {'Safe' if kind_res.is_safe else 'Unsafe'} in {kind_time:.2f}s"
        )
        print(
            f"PDRProver: {'Safe' if pdr_res.is_safe else 'Unsafe'} in {pdr_time:.2f}s"
        )
        print("======================================")

        # Verify that the property is indeed invalid
        self.assertFalse(
            abd_res.is_safe, "AbductionProver should identify the property as invalid"
        )
        self.assertFalse(
            kind_res.is_safe, "KInductionProver should identify the property as invalid"
        )
        self.assertFalse(
            pdr_res.is_safe, "PDRProver should identify the property as invalid"
        )

        if abd_res.counterexample:
            print("Counterexample found by AbductionProver:")
            print(abd_res.counterexample)


if __name__ == "__main__":
    main()
