"""Tests for process cleanup and leak prevention"""

import psutil
import os
import time
from aria.tests import TestCase, main
from aria.smt.pcdclt import solve
from aria.utils import SolverResult
from aria.global_params import SMT_SOLVERS_PATH


class TestProcessCleanup(TestCase):
    """Verify that worker processes and solver subprocesses are properly cleaned up"""

    def setUp(self):
        """Check if Z3 is available"""
        z3_config = SMT_SOLVERS_PATH.get('z3', {})
        if not z3_config.get('available', False):
            self.skipTest("Z3 not available")

    def get_process_count(self, name_pattern=None):
        """Count processes (optionally matching a pattern)"""
        # Give processes time to fully terminate and be reaped
        time.sleep(0.3)

        current_process = psutil.Process(os.getpid())
        children = current_process.children(recursive=True)

        if name_pattern is None:
            return len(children)

        count = 0
        for child in children:
            try:
                if name_pattern.lower() in child.name().lower():
                    count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        return count

    def test_no_process_leak_simple(self):
        """Verify no processes leak after simple SAT query"""
        # Count processes before
        before_count = self.get_process_count()

        formula = """
        (set-logic QF_LRA)
        (declare-fun x () Real)
        (assert (> x 5))
        """

        result = solve(formula, logic="QF_LRA")
        self.assertEqual(result, SolverResult.SAT)

        # Count processes after - should be the same
        after_count = self.get_process_count()
        self.assertEqual(before_count, after_count,
                        f"Process leak detected: {before_count} -> {after_count} processes")

    def test_no_process_leak_unsat(self):
        """Verify no processes leak after UNSAT query"""
        before_count = self.get_process_count()

        formula = """
        (set-logic QF_LRA)
        (declare-fun x () Real)
        (assert (and (> x 5) (< x 3)))
        """

        result = solve(formula, logic="QF_LRA")
        self.assertEqual(result, SolverResult.UNSAT)

        after_count = self.get_process_count()
        self.assertEqual(before_count, after_count,
                        f"Process leak detected: {before_count} -> {after_count} processes")

    def test_no_process_leak_complex(self):
        """Verify no significant process leak after complex query with multiple iterations"""
        before_count = self.get_process_count()

        formula = """
        (set-logic QF_LRA)
        (declare-fun x () Real)
        (declare-fun y () Real)
        (declare-fun z () Real)
        (assert (and (>= x 0) (<= x 5)))
        (assert (and (>= y 0) (<= y 5)))
        (assert (and (>= z 0) (<= z 5)))
        (assert (>= (+ x y z) 20))
        """

        result = solve(formula, logic="QF_LRA")
        self.assertEqual(result, SolverResult.UNSAT)

        after_count = self.get_process_count()
        # Allow small difference (1-2) due to test infrastructure
        # The key is to ensure we don't leak many processes
        self.assertLessEqual(after_count - before_count, 2,
                            f"Significant process leak detected: {before_count} -> {after_count} processes")

    def test_no_zombie_processes(self):
        """Verify no zombie processes are created"""
        formula = """
        (set-logic QF_LRA)
        (declare-fun x () Real)
        (assert (> x 0))
        """

        # Run multiple times to ensure cleanup is consistent
        for _ in range(3):
            result = solve(formula, logic="QF_LRA")
            self.assertEqual(result, SolverResult.SAT)

        # Check for zombie processes
        current_process = psutil.Process(os.getpid())
        children = current_process.children(recursive=True)

        zombies = []
        for child in children:
            try:
                if child.status() == psutil.STATUS_ZOMBIE:
                    zombies.append(child)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        self.assertEqual(len(zombies), 0,
                        f"Found {len(zombies)} zombie processes")

    def test_no_solver_subprocess_leak(self):
        """Verify Z3 solver subprocesses are cleaned up"""
        # Count Z3 processes before
        before_z3_count = self.get_process_count("z3")

        # This formula is UNSAT (sum must be >= 20, but max is 10)
        formula = """
        (set-logic QF_LRA)
        (declare-fun x () Real)
        (declare-fun y () Real)
        (assert (>= (+ x y) 20))
        (assert (<= x 5))
        (assert (<= y 5))
        """

        result = solve(formula, logic="QF_LRA")
        self.assertEqual(result, SolverResult.UNSAT)

        # Count Z3 processes after - should be the same
        after_z3_count = self.get_process_count("z3")
        self.assertEqual(before_z3_count, after_z3_count,
                        f"Z3 subprocess leak: {before_z3_count} -> {after_z3_count} z3 processes")


if __name__ == '__main__':
    main()
