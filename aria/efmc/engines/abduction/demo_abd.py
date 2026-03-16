"""
Test module for the Abductive Inference-based Invariant Generation.
Provides test cases comparing AbductionProver with K-induction verification.
"""

import logging
import time
from typing import List, Tuple

import z3

from aria.efmc.engines.abduction.abduction_prover import AbductionProver
from aria.efmc.engines.kinduction.kind_prover import KInductionProver
from aria.efmc.sts import TransitionSystem

logger = logging.getLogger(__name__)


class VerificationTester:
    """Manages verification tests comparing abduction and k-induction approaches."""

    def __init__(self, k_bound: int = 20):
        self.k_bound = k_bound

    def create_test_cases(self) -> List[Tuple[str, TransitionSystem, bool, str]]:
        """Create all test cases. Returns (name, system, expected_safe, description)."""
        test_cases = []

        # Simple loop: y = 2x
        x, y = z3.Ints("x y")
        xp, yp = z3.Ints("x! y!")
        system = TransitionSystem(
            variables=[x, y],
            prime_variables=[xp, yp],
            init=z3.And(x == 0, y == 0),
            trans=z3.And(xp == x + 1, yp == y + 2),
            post=y <= 2 * x,
        )
        test_cases.append(("Simple Loop", system, True, "Loop maintaining y = 2x"))

        # Faulty system
        a, b = z3.Ints("a b")
        ap, bp = z3.Ints("a! b!")
        system = TransitionSystem(
            variables=[a, b],
            prime_variables=[ap, bp],
            init=z3.And(a == 0, b == 0),
            trans=z3.And(ap == a + 1, bp == b + 1),
            post=b < a,
        )
        test_cases.append(
            ("Faulty System", system, False, "Parallel increments violating b < a")
        )

        # Bounded loop
        i = z3.Int("i")
        ip = z3.Int("i!")
        system = TransitionSystem(
            variables=[i],
            prime_variables=[ip],
            init=i == 0,
            trans=z3.If(i < 10, ip == i + 1, ip == i),
            post=i <= 10,
        )
        test_cases.append(("Bounded Loop", system, True, "Counter with upper bound 10"))

        # Dependent variables
        n, m = z3.Ints("n m")
        np, mp = z3.Ints("n! m!")
        system = TransitionSystem(
            variables=[n, m],
            prime_variables=[np, mp],
            init=z3.And(n == 2, m == 1),
            trans=z3.And(np == n + 1, mp == n),
            post=m <= n / 2,
        )
        test_cases.append(
            ("Dependent Variables", system, False, "m tracking n-1, violating m <= n/2")
        )

        return test_cases

    def verify_system(
        self, name: str, system: TransitionSystem
    ) -> Tuple[bool, bool, float, str]:
        """Verify system with both methods. Returns (kind_result, abd_result, time, invariant)."""
        # K-induction
        kind_prover = KInductionProver(system)
        kind_prover.use_aux_invariant = False
        kind_result = kind_prover.solve(k=self.k_bound)
        kind_safe = None if kind_result.is_unknown else kind_result.is_safe

        # Abduction
        start_time = time.time()
        abduction_prover = AbductionProver(system)

        try:
            abd_result = abduction_prover.solve()
            abd_safe = abd_result.is_safe
            invariant = str(abd_result.invariant) if abd_result.invariant else "None"
        except Exception as e:
            logger.error("Abduction error for %s: %s", name, e)
            abd_safe = False
            invariant = "Error"

        execution_time = time.time() - start_time
        return kind_safe, abd_safe, execution_time, invariant

    def run_tests(self) -> None:
        """Run all test cases and report results."""
        test_cases = self.create_test_cases()
        results = []

        print("Verification Test Results")
        print("=" * 60)

        for name, system, expected_safe, description in test_cases:
            kind_result, abd_result, exec_time, invariant = self.verify_system(
                name, system
            )

            # Verify abduction result matches expectation
            success = abd_result == expected_safe
            if not success:
                logger.error(
                    "Test %s failed: expected %s, got %s",
                    name,
                    expected_safe,
                    abd_result,
                )

            results.append(
                (name, kind_result, abd_result, success, exec_time, invariant)
            )

            # Print result
            if kind_result is None:
                kind_status = "unknown"
            else:
                kind_status = "safe" if kind_result else "unsafe"
            abd_status = "safe" if abd_result else "unsafe"
            status_mark = "✓" if success else "✗"

            print(f"\n{status_mark} {name}")
            print(f"  Description: {description}")
            print(f"  K-induction: {kind_status}")
            print(f"  Abduction: {abd_status} ({exec_time:.2f}s)")
            if invariant not in ("None", "Error"):
                print(f"  Invariant: {invariant}")

        # Summary
        total_tests = len(results)
        successful = sum(1 for _, _, _, success, _, _ in results)
        total_time = sum(exec_time for _, _, _, _, exec_time, _ in results)
        unknown_kind = sum(
            1 for _, kind_result, _, _, _, _ in results if kind_result is None
        )

        print(f"\n{'=' * 60}")
        print(f"Summary: {successful}/{total_tests} tests passed")
        print(f"Accuracy: {successful/total_tests:.1%}")
        print(f"Total time: {total_time:.2f}s (avg: {total_time/total_tests:.2f}s)")
        if unknown_kind > 0:
            print(f"K-induction unknown: {unknown_kind}/{total_tests}")


def main():
    """Run verification tests."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        tester = VerificationTester()
        tester.run_tests()
    except Exception as e:
        logger.error("Test execution failed: %s", e)
        raise


if __name__ == "__main__":
    main()
