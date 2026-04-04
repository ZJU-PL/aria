"""
Use quantifier elimination to compute the strongest inductive invariant?
- The abstract transformer is strongest
- The invariant is also?

TODO: this can be very slow and may not terminate （How to over-approximate qe)
"""

import logging
import time
from typing import Optional

import z3

from aria.efmc.sts import TransitionSystem
from aria.utils.z3.solver import is_valid, fixpoint
from aria.efmc.utils.verification_utils import VerificationResult

logger = logging.getLogger(__name__)


class QuantifierEliminationProver:
    """Prover using quantifier elimination to compute inductive invariants."""

    def __init__(self, system: TransitionSystem, timeout=None, verbose=True):
        self.sts = system
        self.timeout = timeout  # Timeout in seconds
        self.verbose = verbose
        """
        var_map = [(x, xp), (y, yp)]
        var_map_rev = [(xp ,x), (yp, y)]
        """
        self.var_map = []
        self.var_map_rev = []
        for i, var in enumerate(self.sts.variables):
            self.var_map.append((var, self.sts.prime_variables[i]))
            self.var_map_rev.append((self.sts.prime_variables[i], var))

    def solve(
        self, timeout: Optional[int] = None
    ) -> VerificationResult:  # pylint: disable=unused-argument
        """
        Verify the system using quantifier elimination.

        Returns:
            VerificationResult: Object containing verification result and related data
        """
        start = time.time()
        old_inv = z3.BoolVal(False)  # Use z3.BoolVal instead of Python's False
        inv = self.sts.init
        i = 0

        if self.verbose:
            print("QE-based invariant inference starting!!!")

        while not fixpoint(old_inv, inv):
            if self.verbose:
                print(f"\nIteration {i}, Inv: {inv}")

            i = i + 1

            # Check timeout
            if self.timeout and time.time() - start > self.timeout:
                print(f"Timeout reached after {time.time() - start:.2f} seconds")
                return VerificationResult(False, None, is_unknown=True, timed_out=True)

            qfml = z3.Exists(self.sts.variables, z3.And(inv, self.sts.trans))

            # Try different QE tactics if one fails
            try:
                # compute the best abstract transformer
                onestep = z3.Tactic("qe2")(qfml).as_expr()
            except z3.Z3Exception:
                try:
                    # Fall back to regular qe if qe2 fails
                    onestep = z3.Tactic("qe")(qfml).as_expr()
                except z3.Z3Exception:
                    print("QE tactics failed, using default simplification")
                    onestep = z3.simplify(qfml)

            # rename
            onestep = z3.substitute(onestep, self.var_map_rev)
            old_inv = inv
            inv = z3.simplify(z3.Or(inv, onestep))

        # Check if the invariant implies the post-condition
        if is_valid(z3.Implies(inv, self.sts.post)):
            if self.verbose:
                print(">>> SAFE\n")
                print(f"QE success time: {time.time() - start:.2f} seconds")
                print(f"Invariant: {z3.simplify(inv)}")
            return VerificationResult(True, inv)
        if self.verbose:
            print(">>> UNKNOWN (invariant too weak)\n")
            print(f"QE UNKNOWN time: {time.time() - start:.2f} seconds")
        return VerificationResult(False, None, is_unknown=True)


if __name__ == "__main__":
    x, y, z, xp, yp, zp = z3.Reals("x y z x! y! z!")

    init = x == 0
    trans = z3.And(x < 10, xp == x + 1)
    post = z3.Implies(x >= 10, x == 10)
    all_vars = [x, xp]
    sts = TransitionSystem()
    sts.from_z3_cnts([all_vars, init, trans, post])

    pp = QuantifierEliminationProver(sts, timeout=60)  # Set a 60-second timeout
    result = pp.solve()
    print(f"Final result: {result}")
