"""
Property-Directed Reachability (or IC3)

Currently, we use the implementation inside Z3's CHC engine.

TODO: use other PDR engines
"""

import logging
import time
from typing import Optional

import z3

from aria.efmc.sts import TransitionSystem
from aria.efmc.utils.verification_utils import VerificationResult

logger = logging.getLogger(__name__)


class PDRProver:
    """Property-Directed Reachability (PDR) prover using Z3's CHC engine."""

    def __init__(self, system: TransitionSystem):
        self.sts = system
        self.verbose = False

    def set_verbose(self, verbose: bool) -> None:
        """Set verbose mode"""
        self.verbose = verbose

    def solve(self, timeout: Optional[int] = None) -> VerificationResult:
        """From transition system to CHC

        Args:
            timeout: Timeout in seconds (default: 60)

        Returns:
            VerificationResult: The verification result
        """
        assert self.sts.initialized

        # construct the "inv" uninterpreted function
        s = z3.SolverFor("HORN")

        # Set timeout in milliseconds
        if timeout is not None:
            s.set("timeout", timeout * 1000)  # does this work?

        inv_sig = "z3.Function('inv', "

        if self.sts.has_int:
            for _ in range(len(self.sts.variables)):
                inv_sig += "z3.IntSort(), "
        elif self.sts.has_real:
            for _ in range(len(self.sts.variables)):
                inv_sig += "z3.RealSort(), "
        elif self.sts.has_bv:
            bv_size = self.sts.variables[0].sort().size()
            for _ in range(len(self.sts.variables)):
                inv_sig += f"z3.BitVecSort({bv_size}), "
        elif self.sts.has_bool:
            for _ in range(len(self.sts.variables)):
                inv_sig += "z3.BoolSort(), "
        else:
            raise NotImplementedError

        inv_sig += "z3.BoolSort())"
        inv = eval(inv_sig)  # pylint: disable=eval-used
        # Init
        s.add(
            z3.ForAll(
                self.sts.variables, z3.Implies(self.sts.init, inv(self.sts.variables))
            )
        )
        # Inductive
        inductive_condition = z3.Implies(
            z3.And(inv(self.sts.variables), self.sts.trans),
            inv(self.sts.prime_variables),
        )
        s.add(z3.ForAll(self.sts.all_variables, inductive_condition))
        # Post
        s.add(
            z3.ForAll(
                self.sts.variables, z3.Implies(inv(self.sts.variables), self.sts.post)
            )
        )

        if self.verbose:
            print("PDR starting!!!")
            # print(s.to_smt2())

        start = time.time()
        try:
            res = s.check()
            elapsed_time = time.time() - start

            if self.verbose:
                print(f"PDR time: {elapsed_time:.2f}s")

            if res == z3.sat:
                if self.verbose:
                    print("safe")
                invariant = s.model().eval(inv(self.sts.variables))
                if self.verbose:
                    print("Invariant: ", invariant)
                return VerificationResult(True, invariant)
            if res == z3.unsat:
                if self.verbose:
                    print("unsafe")
                # PDR doesn't provide a concrete counterexample, so we mark it as unknown
                # rather than unsafe since we can't provide a concrete counterexample
                return VerificationResult(False, None, None, is_unknown=True)
            if self.verbose:
                print("unknown")
            return VerificationResult(False, None, None, is_unknown=True)

        except (z3.Z3Exception, RuntimeError, ValueError) as e:
            if self.verbose:
                print(f"PDR failed: {e}")
            return VerificationResult(False, None, None, is_unknown=True)
