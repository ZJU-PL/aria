"""
Sampling-based CEGIS algorithm for EFSMT over QF_BV.
"""

import time
import logging
from typing import List
import z3
from pysmt.shortcuts import And, Not, Bool, get_model, Solver, BV
from pysmt.logics import QF_BV

from aria.efmc.smttools.pysmt_solver import PySMTSolver, to_pysmt_vars

logger = logging.getLogger(__name__)


class SamplingEFSMTSolver:  # pylint: disable=too-few-public-methods
    """Sampling-based CEGIS solver for exists-forall problems over QF_BV."""

    def __init__(self, **kwargs):
        self.esolver_name = kwargs.get("esolver_name", "z3")
        self.fsolver_name = kwargs.get("fsolver_name", "z3")
        self.verbose = kwargs.get("verbose", False)
        self.timeout = kwargs.get("timeout", None)
        self.sample_size = kwargs.get("sample_size", 10)
        self.max_samples = kwargs.get("max_samples", 100)

    def solve(
        self,
        evars: List[z3.ExprRef],
        uvars: List[z3.ExprRef],
        z3fml: z3.ExprRef,
        maxloops=None,
    ):
        """Solves exists x. forall y. phi(x, y) using sampling-based CEGIS
        over QF_BV."""
        start_time = time.time()

        # Convert to PySMT format
        _, phi = PySMTSolver.convert(z3fml)
        y_vars = to_pysmt_vars(uvars)
        x_vars = to_pysmt_vars(evars)

        # Initialize sample set
        y_samples = self._generate_initial_samples(y_vars, self.sample_size)

        loops = 0
        result = "unknown"

        while maxloops is None or loops < maxloops:
            if self.timeout and time.time() - start_time > self.timeout:
                if self.verbose:
                    elapsed = time.time() - start_time
                    print(f"Timeout reached after {elapsed:.2f} seconds")
                break

            loops += 1
            if self.verbose:
                print(f"CEGIS iteration {loops} with {len(y_samples)} samples")

            # Synthesize x for all sampled y
            candidate_x = self._synthesize_for_samples(x_vars, phi, y_samples)
            if candidate_x is None:
                result = "unsat"
                break

            # Verify candidate
            counterexample = self._verify_candidate(y_vars, phi, candidate_x)
            if counterexample is None:
                result = "sat"
                break
            y_samples.append(counterexample)
            if len(y_samples) >= self.max_samples:
                if self.verbose:
                    print(f"Reached maximum sample size ({self.max_samples})")
                break

        return result

    def _generate_initial_samples(self, y_vars, sample_size):
        """Generate initial samples for y variables"""
        samples = []
        for _ in range(sample_size):
            sample = {}
            for var in y_vars:
                var_type = var.symbol_type()
                # Check if it's a BVType (need to import from pysmt.typing)
                if hasattr(var_type, "width"):
                    width = var_type.width
                    # Start with 0, could be randomized
                    sample[var] = BV(0, width)
                else:
                    sample[var] = Bool(False)
            samples.append(sample)
        return samples

    def _synthesize_for_samples(self, x_vars, phi, y_samples):
        """Synthesize x that satisfies phi(x, y) for all sampled y"""
        constraints = []
        for sample in y_samples:
            phi_substituted = phi.substitute(sample).simplify()
            constraints.append(phi_substituted)

        with Solver(logic=QF_BV, name=self.esolver_name) as solver:
            solver.add_assertion(And(constraints))
            if solver.solve():
                return {var: solver.get_value(var) for var in x_vars}
            return None

    def _verify_candidate(self, y_vars, phi, candidate_x):
        """Verify if candidate_x works for all y, return counterexample if not"""
        phi_substituted = phi.substitute(candidate_x).simplify()
        counterexample = get_model(
            Not(phi_substituted), logic=QF_BV, solver_name=self.fsolver_name
        )

        if counterexample is None:
            return None
        return {var: counterexample[var] for var in y_vars}


def sampling_efsmt(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    evars: List[z3.ExprRef],
    uvars: List[z3.ExprRef],
    z3fml: z3.ExprRef,
    maxloops=None,
    esolver_name="z3",
    fsolver_name="z3",
    verbose=False,
    timeout=None,
    sample_size=10,
    max_samples=100,
):
    """Convenience function for sampling-based CEGIS over QF_BV"""
    solver = SamplingEFSMTSolver(
        esolver_name=esolver_name,
        fsolver_name=fsolver_name,
        verbose=verbose,
        timeout=timeout,
        sample_size=sample_size,
        max_samples=max_samples,
    )
    return solver.solve(evars, uvars, z3fml, maxloops)
